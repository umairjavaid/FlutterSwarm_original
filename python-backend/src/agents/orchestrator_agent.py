"""
Orchestrator Agent for FlutterSwarm Multi-Agent System.

This is the master agent that coordinates and manages all other specialized agents.
It handles task decomposition, workflow management, and inter-agent coordination
using LLM-driven reasoning and decision-making.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..agents.base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..core.llm_client import LLMClient
from ..models.agent_models import AgentMessage, TaskResult, AgentCapabilityInfo
from ..models.task_models import TaskContext, WorkflowDefinition, TaskType, TaskPriority, ExecutionStrategy
from ..models.project_models import ProjectContext
from ..models.tool_models import (
    WorkflowFeedback, AdaptationResult, PerformanceAnalysis, WorkflowImprovement,
    WorkflowSession, PerformanceBottleneck, WorkflowStepResult, AgentPerformanceMetrics,
    ToolCoordinationResult, ToolConflict, Resolution, UsagePattern, AllocationPlan,
    QueueStatus, SharedOperation, CoordinationResult
)
from ..config import get_logger

logger = get_logger("orchestrator_agent")


class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that manages the entire multi-agent system.
    
    Responsibilities:
    - Task decomposition and workflow planning
    - Agent discovery and capability management
    - Task assignment and delegation
    - Progress monitoring and coordination
    - Error handling and recovery
    - Resource optimization
    
    The orchestrator uses LLM reasoning for all decision-making processes
    and maintains no hardcoded logic for task management.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        super().__init__(config, llm_client, memory_manager, event_bus)
        
        # Agent management
        self.available_agents: Dict[str, AgentCapabilityInfo] = {}
        self.active_workflows: Dict[str, WorkflowSession] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
        # Tool coordination
        self.tool_usage_patterns: Dict[str, UsagePattern] = {}
        self.active_tool_conflicts: Dict[str, ToolConflict] = {}
        self.tool_queues: Dict[str, QueueStatus] = {}
        self.shared_operations: Dict[str, SharedOperation] = {}
        self.tool_allocations: Dict[str, str] = {}  # tool_name -> current_agent_id
        self.coordination_history: List[ToolCoordinationResult] = []
        
        # System tracking
        self.completion_stats: Dict[str, Any] = {
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tasks_delegated": 0,
            "average_completion_time": 0.0
        }
    
    async def process_task(self, task: TaskContext) -> TaskResult:
        """
        Process a high-level task by decomposing it into a workflow.
        
        This is the main entry point for the orchestrator. It uses LLM
        reasoning to understand the task and create an appropriate workflow.
        """
        try:
            logger.info(f"Orchestrator processing task: {task.task_id}")
            
            # Store task context in memory
            await self.memory_manager.store_memory(
                content=f"Processing task: {task.description}",
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value
                },
                correlation_id=task.correlation_id,
                importance=0.8,
                long_term=True
            )
            
            # Use LLM to analyze and decompose the task
            workflow = await self._decompose_task_with_llm(task)
            
            if not workflow:
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    result={},
                    error="Failed to decompose task into workflow",
                    correlation_id=task.correlation_id
                )
            
            # Execute the workflow
            workflow_result = await self._execute_workflow(workflow)
            
            # Return consolidated result
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="completed" if workflow_result["success"] else "failed",
                result=workflow_result,
                deliverables=workflow_result.get("deliverables", {}),
                error=workflow_result.get("error"),
                execution_time=workflow_result.get("execution_time"),
                metadata={"workflow_id": workflow.workflow_id},
                correlation_id=task.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                result={},
                error=str(e),
                correlation_id=task.correlation_id
            )
    
    async def _decompose_task_with_llm(self, task: TaskContext) -> Optional[WorkflowDefinition]:
        """
        Use LLM to analyze and decompose a complex task into a workflow.
        
        This method leverages the LLM's reasoning capabilities to understand
        the task requirements and create an appropriate execution plan.
        """
        try:
            # Gather context for LLM reasoning
            context = await self._gather_decomposition_context(task)
            
            # Create prompt for task decomposition
            prompt = self._create_decomposition_prompt(task, context)
            
            # Get LLM analysis
            response = await self.llm_client.generate(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=3000,
                agent_id=self.agent_id,
                correlation_id=task.correlation_id
            )
            
            # Parse LLM response to create workflow
            workflow = await self._parse_workflow_from_llm_response(response, task)
            
            if workflow:
                self.active_workflows[workflow.workflow_id] = workflow
                logger.info(f"Created workflow {workflow.workflow_id} with {len(workflow.tasks)} tasks")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error decomposing task with LLM: {e}")
            return None
    
    def _create_decomposition_prompt(self, task: TaskContext, context: Dict[str, Any]) -> str:
        """Create a detailed prompt for task decomposition."""
        
        # Get available agent capabilities
        agent_capabilities = []
        for agent_info in self.available_agents.values():
            agent_capabilities.append({
                "agent_type": agent_info.agent_type,
                "capabilities": agent_info.capabilities,
                "specializations": agent_info.specializations
            })
        
        prompt = f"""
You are the Orchestrator Agent in a multi-agent Flutter development system. Your role is to analyze complex tasks and decompose them into executable workflows.

TASK TO DECOMPOSE:
Description: {task.description}
Type: {task.task_type.value}
Priority: {task.priority.value}
Requirements: {[req.name for req in task.requirements]}
Expected Deliverables: {[deliv.name for deliv in task.expected_deliverables]}

PROJECT CONTEXT:
{json.dumps(task.project_context, indent=2)}

AVAILABLE AGENTS AND CAPABILITIES:
{json.dumps(agent_capabilities, indent=2)}

HISTORICAL CONTEXT:
{context.get('similar_tasks', 'No similar tasks found')}

INSTRUCTIONS:
1. Analyze the task and understand its complexity and requirements
2. Break it down into smaller, manageable subtasks
3. Assign each subtask to the most appropriate agent type based on capabilities
4. Define dependencies between tasks (which tasks must complete before others can start)
5. Estimate execution time for each subtask
6. Choose an appropriate execution strategy (sequential, parallel, hybrid)

RESPOND WITH A JSON STRUCTURE:
{{
    "workflow_name": "Descriptive name for the workflow",
    "description": "Brief description of the overall workflow",
    "execution_strategy": "sequential|parallel|hybrid",
    "estimated_duration_minutes": 60,
    "tasks": [
        {{
            "task_id": "unique_task_id",
            "description": "Detailed task description",
            "task_type": "task_type_enum_value",
            "assigned_agent_type": "agent_type",
            "requirements": [
                {{
                    "name": "requirement_name",
                    "description": "requirement_description",
                    "required": true,
                    "data_type": "string"
                }}
            ],
            "expected_deliverables": [
                {{
                    "name": "deliverable_name",
                    "description": "deliverable_description",
                    "type": "file|data|report",
                    "required": true
                }}
            ],
            "priority": "normal|high|critical",
            "estimated_duration_minutes": 30,
            "metadata": {{}}
        }}
    ],
    "dependencies": {{
        "task_id": ["dependency_task_id1", "dependency_task_id2"]
    }},
    "success_criteria": {{
        "all_tasks_completed": true,
        "quality_checks_passed": true,
        "deliverables_validated": true
    }}
}}

IMPORTANT GUIDELINES:
- Each task should be atomic and assignable to a single agent
- Consider dependencies carefully - some tasks require others to complete first
- Match tasks to agent capabilities appropriately
- Include quality checks and validation steps
- Consider the project context when planning
- Be specific about requirements and deliverables
- Use appropriate task types from the available enums
"""
        
        return prompt
    
    async def _gather_decomposition_context(self, task: TaskContext) -> Dict[str, Any]:
        """Gather relevant context for task decomposition."""
        context = {}
        
        try:
            # Search for similar tasks in memory
            similar_tasks = await self.memory_manager.search_memory(
                query=f"task decomposition {task.task_type.value} {task.description[:100]}",
                limit=5,
                similarity_threshold=0.6
            )
            
            if similar_tasks:
                context["similar_tasks"] = [
                    {"content": entry.content, "metadata": entry.metadata}
                    for entry, score in similar_tasks
                ]
            
            # Get current agent availability
            context["agent_availability"] = {
                agent_id: {
                    "available": info.availability,
                    "current_load": info.current_load,
                    "max_concurrent": info.max_concurrent_tasks
                }
                for agent_id, info in self.available_agents.items()
            }
            
            # Add performance history
            context["performance_stats"] = self.completion_stats
            
        except Exception as e:
            logger.warning(f"Error gathering decomposition context: {e}")
        
        return context
    
    async def _parse_workflow_from_llm_response(
        self,
        response: str,
        original_task: TaskContext
    ) -> Optional[WorkflowDefinition]:
        """Parse the LLM response to create a WorkflowDefinition."""
        try:
            # Extract JSON from response
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            workflow_data = json.loads(response_clean)
            
            # Create workflow definition
            workflow = WorkflowDefinition(
                workflow_id=str(uuid.uuid4()),
                name=workflow_data.get("workflow_name", f"Workflow for {original_task.task_id}"),
                description=workflow_data.get("description", ""),
                execution_strategy=ExecutionStrategy(workflow_data.get("execution_strategy", "hybrid")),
                success_criteria=workflow_data.get("success_criteria", {}),
                estimated_duration=timedelta(minutes=workflow_data.get("estimated_duration_minutes", 60))
            )
            
            # Create tasks from LLM response
            for task_data in workflow_data.get("tasks", []):
                task_context = self._create_task_from_llm_data(task_data, original_task)
                workflow.tasks.append(task_context)
            
            # Set dependencies
            workflow.dependencies = workflow_data.get("dependencies", {})
            
            # Validate the workflow
            if workflow.validate_dependencies():
                return workflow
            else:
                logger.error("Workflow dependency validation failed")
                return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing workflow from LLM response: {e}")
            return None
    
    def _create_task_from_llm_data(self, task_data: Dict[str, Any], original_task: TaskContext) -> TaskContext:
        """Create a TaskContext from LLM-generated task data."""
        from ..models.task_models import TaskRequirement, TaskDeliverable
        
        # Parse requirements
        requirements = []
        for req_data in task_data.get("requirements", []):
            requirements.append(TaskRequirement(
                name=req_data.get("name", ""),
                description=req_data.get("description", ""),
                required=req_data.get("required", True),
                data_type=req_data.get("data_type", "string")
            ))
        
        # Parse deliverables
        deliverables = []
        for deliv_data in task_data.get("expected_deliverables", []):
            deliverables.append(TaskDeliverable(
                name=deliv_data.get("name", ""),
                description=deliv_data.get("description", ""),
                type=deliv_data.get("type", "data"),
                required=deliv_data.get("required", True)
            ))
        
        # Create task context
        return TaskContext(
            task_id=task_data.get("task_id", str(uuid.uuid4())),
            description=task_data.get("description", ""),
            task_type=TaskType(task_data.get("task_type", "analysis")),
            requirements=requirements,
            expected_deliverables=deliverables,
            project_context=original_task.project_context,
            priority=TaskPriority[task_data.get("priority", "NORMAL").upper()],
            correlation_id=original_task.correlation_id,
            metadata={
                **task_data.get("metadata", {}),
                "assigned_agent_type": task_data.get("assigned_agent_type"),
                "estimated_duration_minutes": task_data.get("estimated_duration_minutes", 30)
            },
            estimated_duration=timedelta(minutes=task_data.get("estimated_duration_minutes", 30))
        )
    
    async def _execute_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute a workflow by coordinating task execution across agents."""
        logger.info(f"Executing workflow: {workflow.workflow_id}")
        
        start_time = datetime.utcnow()
        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        task_results: Dict[str, TaskResult] = {}
        
        try:
            while len(completed_tasks) + len(failed_tasks) < len(workflow.tasks):
                # Get tasks ready for execution
                ready_tasks = workflow.get_ready_tasks(completed_tasks)
                
                if not ready_tasks and not failed_tasks:
                    # No tasks ready and no failures - something is wrong
                    logger.error("No tasks ready for execution and no failures detected")
                    break
                
                if not ready_tasks:
                    # Only failed tasks remain
                    break
                
                # Execute ready tasks based on strategy
                execution_tasks = []
                
                if workflow.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                    # Execute one task at a time
                    ready_tasks = ready_tasks[:1]
                
                for task in ready_tasks:
                    if task.task_id not in completed_tasks and task.task_id not in failed_tasks:
                        execution_tasks.append(self._delegate_task(task))
                
                # Wait for task completions
                if execution_tasks:
                    results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(results):
                        task = ready_tasks[i]
                        
                        if isinstance(result, Exception):
                            logger.error(f"Task {task.task_id} failed with exception: {result}")
                            failed_tasks.add(task.task_id)
                        elif isinstance(result, TaskResult):
                            task_results[task.task_id] = result
                            
                            if result.is_successful():
                                completed_tasks.add(task.task_id)
                                logger.info(f"Task {task.task_id} completed successfully")
                            else:
                                failed_tasks.add(task.task_id)
                                logger.error(f"Task {task.task_id} failed: {result.error}")
                
                # Brief pause before next iteration
                await asyncio.sleep(1)
            
            # Calculate results
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            success = len(completed_tasks) == len(workflow.tasks)
            
            # Store workflow result in memory
            await self.memory_manager.store_memory(
                content=f"Workflow {workflow.workflow_id} {'completed' if success else 'failed'} "
                       f"({len(completed_tasks)}/{len(workflow.tasks)} tasks successful)",
                metadata={
                    "workflow_id": workflow.workflow_id,
                    "success": success,
                    "completed_tasks": len(completed_tasks),
                    "total_tasks": len(workflow.tasks),
                    "execution_time": execution_time
                },
                correlation_id=workflow.tasks[0].correlation_id if workflow.tasks else "",
                importance=0.9,
                long_term=True
            )
            
            # Update stats
            if success:
                self.completion_stats["successful_workflows"] += 1
            else:
                self.completion_stats["failed_workflows"] += 1
            
            self.completion_stats["total_tasks_delegated"] += len(workflow.tasks)
            
            return {
                "success": success,
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "total_tasks": len(workflow.tasks),
                "execution_time": execution_time,
                "task_results": {task_id: result.to_dict() for task_id, result in task_results.items()},
                "deliverables": self._consolidate_deliverables(task_results),
                "error": f"Failed tasks: {failed_tasks}" if failed_tasks else None
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.workflow_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _delegate_task(self, task: TaskContext) -> TaskResult:
        """Delegate a task to the most appropriate agent."""
        try:
            # Find the best agent for this task
            agent_id = await self._find_best_agent_for_task(task)
            
            if not agent_id:
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    result={},
                    error="No suitable agent found for task",
                    correlation_id=task.correlation_id
                )
            
            # Store assignment
            self.task_assignments[task.task_id] = agent_id
            
            # Send task to agent via event bus
            message = AgentMessage(
                type="task_assignment",
                source=self.agent_id,
                target=agent_id,
                payload={
                    "task": task.to_dict(),
                    "deadline": task.deadline.isoformat() if task.deadline else None
                },
                correlation_id=task.correlation_id,
                priority=task.priority.value
            )
            
            await self.event_bus.publish(f"agent.{agent_id}.tasks", message)
            
            # Wait for task completion or timeout
            timeout = task.estimated_duration.total_seconds() if task.estimated_duration else 300
            result = await self._wait_for_task_completion(task.task_id, timeout)
            
            return result
            
        except Exception as e:
            logger.error(f"Error delegating task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="failed",
                result={},
                error=str(e),
                correlation_id=task.correlation_id
            )
    
    async def _find_best_agent_for_task(self, task: TaskContext) -> Optional[str]:
        """Find the best available agent for a given task using LLM reasoning."""
        try:
            # Get required agent type from task metadata
            required_agent_type = task.metadata.get("assigned_agent_type")
            
            if required_agent_type:
                # Find agents of the specified type
                suitable_agents = [
                    (agent_id, info) for agent_id, info in self.available_agents.items()
                    if info.agent_type == required_agent_type and info.availability
                ]
            else:
                # Use LLM to determine best agent type
                suitable_agents = await self._llm_agent_selection(task)
            
            if not suitable_agents:
                return None
            
            # Select agent with lowest load
            best_agent = min(suitable_agents, key=lambda x: x[1].current_load)
            return best_agent[0]
            
        except Exception as e:
            logger.error(f"Error finding agent for task {task.task_id}: {e}")
            return None
    
    async def _llm_agent_selection(self, task: TaskContext) -> List[tuple]:
        """Use LLM to select appropriate agents for a task."""
        # This would involve calling the LLM to analyze the task and recommend agents
        # For now, return all available agents
        return [
            (agent_id, info) for agent_id, info in self.available_agents.items()
            if info.availability
        ]
    
    async def _wait_for_task_completion(self, task_id: str, timeout: float) -> TaskResult:
        """Wait for a delegated task to complete."""
        # This would involve listening for task completion events
        # For now, return a placeholder result
        await asyncio.sleep(min(timeout, 5))  # Simulate task execution
        
        return TaskResult(
            task_id=task_id,
            agent_id=self.task_assignments.get(task_id, "unknown"),
            status="completed",
            result={"simulated": True},
            correlation_id=""
        )
    
    def _consolidate_deliverables(self, task_results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Consolidate deliverables from all completed tasks."""
        deliverables = {}
        
        for task_id, result in task_results.items():
            if result.is_successful():
                for key, value in result.deliverables.items():
                    deliverables[f"{task_id}_{key}"] = value
        
        return deliverables
    
    async def register_agent(self, agent_info: AgentCapabilityInfo) -> None:
        """Register a new agent with the orchestrator."""
        self.available_agents[agent_info.agent_id] = agent_info
        logger.info(f"Registered agent {agent_info.agent_id} of type {agent_info.agent_type}")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id in self.available_agents:
            del self.available_agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this orchestrator agent."""
        return [AgentCapability.ORCHESTRATION]
    
    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the orchestrator agent."""
        return """
You are the Orchestrator Agent in the FlutterSwarm multi-agent system, responsible for coordinating all aspects of Flutter app development.

CORE RESPONSIBILITIES:
1. Task Decomposition: Break down complex development requests into manageable subtasks
2. Workflow Management: Design and execute efficient workflows across multiple agents
3. Agent Coordination: Assign tasks to the most suitable specialized agents
4. Progress Monitoring: Track task completion and handle dependencies

You must use logical reasoning to make all decisions and never rely on hardcoded rules.
"""

    # Workflow Adaptation Methods

    async def adapt_workflow(self, workflow_id: str, feedback: 'WorkflowFeedback') -> 'AdaptationResult':
        """
        Adapt workflow based on real-time feedback using LLM reasoning.
        
        This method analyzes current workflow performance and modifies the workflow
        to improve efficiency, quality, and resource utilization.
        """
        try:
            logger.info(f"Starting workflow adaptation for workflow {workflow_id}")
            
            # Analyze current workflow performance
            performance_analysis = await self._analyze_workflow_performance(workflow_id, feedback)
            
            # Identify improvement opportunities using LLM reasoning
            improvements = await self._identify_improvement_opportunities(feedback, performance_analysis)
            
            # Get current workflow session
            workflow_session = self.active_workflows.get(workflow_id)
            if not workflow_session:
                raise ValueError(f"Workflow {workflow_id} not found in active workflows")
            
            # Modify workflow steps based on improvements
            modified_workflow = await self._modify_workflow_steps(workflow_session, improvements)
            
            # Rebalance agent assignments
            new_assignments = await self._rebalance_agent_assignments(modified_workflow)
            
            # Create adaptation result
            adaptation_result = AdaptationResult(
                workflow_id=workflow_id,
                changes_made=[{
                    "type": improvement.type,
                    "description": improvement.description,
                    "target_steps": improvement.target_steps,
                    "expected_benefit": improvement.expected_benefit
                } for improvement in improvements],
                steps_modified=[step for improvement in improvements for step in improvement.target_steps],
                agents_reassigned=new_assignments,
                expected_improvements={
                    "efficiency": sum(imp.expected_benefit for imp in improvements if "efficiency" in imp.type),
                    "quality": sum(imp.expected_benefit for imp in improvements if "quality" in imp.type),
                    "time_savings": sum(imp.expected_benefit for imp in improvements if "time" in imp.type)
                },
                confidence_score=min(imp.confidence_score for imp in improvements) if improvements else 0.0,
                estimated_time_savings=sum(imp.expected_benefit for imp in improvements if "time" in imp.type),
                estimated_quality_improvement=sum(imp.expected_benefit for imp in improvements if "quality" in imp.type),
                estimated_resource_efficiency=sum(imp.expected_benefit for imp in improvements if "resource" in imp.type)
            )
            
            # Update the active workflow
            self.active_workflows[workflow_id] = modified_workflow
            
            # Store adaptation in memory
            await self.memory_manager.store_memory(
                content=f"Workflow adaptation completed for {workflow_id}",
                metadata={
                    "workflow_id": workflow_id,
                    "adaptation_id": adaptation_result.adaptation_id,
                    "changes_count": len(adaptation_result.changes_made),
                    "expected_improvements": adaptation_result.expected_improvements
                },
                correlation_id=workflow_id,
                importance=0.9,
                long_term=True
            )
            
            logger.info(f"Workflow adaptation completed: {len(improvements)} improvements applied")
            return adaptation_result
            
        except Exception as e:
            logger.error(f"Workflow adaptation failed: {e}")
            return AdaptationResult(
                workflow_id=workflow_id,
                changes_made=[],
                confidence_score=0.0
            )

    async def _analyze_workflow_performance(self, workflow_id: str, feedback: 'WorkflowFeedback') -> 'PerformanceAnalysis':
        """Analyze workflow performance to identify bottlenecks and inefficiencies."""
        try:
            # Create LLM prompt for performance analysis
            prompt = self._create_performance_analysis_prompt(workflow_id, feedback)
            
            # Get LLM analysis
            analysis_response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse LLM response into structured analysis
            analysis_data = await self._parse_performance_analysis(analysis_response)
            
            # Calculate efficiency metrics
            efficiency_score = self._calculate_efficiency_score(feedback)
            completion_rate = self._calculate_completion_rate(feedback)
            resource_utilization = self._calculate_resource_utilization(feedback)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(feedback, analysis_data)
            
            return PerformanceAnalysis(
                workflow_id=workflow_id,
                efficiency_score=efficiency_score,
                completion_rate=completion_rate,
                resource_utilization=resource_utilization,
                bottlenecks=bottlenecks,
                inefficiencies=analysis_data.get("inefficiencies", []),
                critical_path=analysis_data.get("critical_path", []),
                parallel_opportunities=analysis_data.get("parallel_opportunities", []),
                agent_utilization={
                    agent_id: metrics.resource_efficiency 
                    for agent_id, metrics in feedback.agent_performance.items()
                },
                agent_specialization_mismatches=analysis_data.get("specialization_mismatches", []),
                output_quality_distribution=self._analyze_quality_distribution(feedback),
                error_patterns=analysis_data.get("error_patterns", [])
            )
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return PerformanceAnalysis(workflow_id=workflow_id)

    async def _identify_improvement_opportunities(
        self, 
        feedback: 'WorkflowFeedback', 
        performance_analysis: 'PerformanceAnalysis'
    ) -> List['WorkflowImprovement']:
        """Identify specific improvement opportunities using LLM reasoning."""
        try:
            # Create LLM prompt for improvement identification
            prompt = self._create_improvement_identification_prompt(feedback, performance_analysis)
            
            # Get LLM recommendations
            improvements_response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.4
            )
            
            # Parse improvements from LLM response
            improvements_data = await self._parse_improvement_recommendations(improvements_response)
            
            improvements = []
            for improvement_data in improvements_data:
                improvement = WorkflowImprovement(
                    type=improvement_data.get("type", "optimize"),
                    priority=improvement_data.get("priority", "medium"),
                    target_steps=improvement_data.get("target_steps", []),
                    description=improvement_data.get("description", ""),
                    proposed_changes=improvement_data.get("proposed_changes", {}),
                    expected_benefit=improvement_data.get("expected_benefit", 0.0),
                    implementation_cost=improvement_data.get("implementation_cost", 0.0),
                    risk_level=improvement_data.get("risk_level", "low"),
                    confidence_score=improvement_data.get("confidence_score", 0.0),
                    supporting_evidence=improvement_data.get("supporting_evidence", [])
                )
                improvements.append(improvement)
            
            # Sort by priority and expected benefit
            improvements.sort(key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority],
                x.expected_benefit
            ), reverse=True)
            
            return improvements
            
        except Exception as e:
            logger.error(f"Improvement identification failed: {e}")
            return []

    async def _modify_workflow_steps(
        self, 
        workflow: 'WorkflowSession', 
        improvements: List['WorkflowImprovement']
    ) -> 'WorkflowSession':
        """Modify workflow steps based on identified improvements."""
        try:
            modified_workflow = workflow
            
            for improvement in improvements:
                if improvement.type == "reorder":
                    modified_workflow = await self._reorder_workflow_steps(modified_workflow, improvement)
                elif improvement.type == "parallelize":
                    modified_workflow = await self._parallelize_workflow_steps(modified_workflow, improvement)
                elif improvement.type == "add_step":
                    modified_workflow = await self._add_workflow_step(modified_workflow, improvement)
                elif improvement.type == "remove_step":
                    modified_workflow = await self._remove_workflow_step(modified_workflow, improvement)
                elif improvement.type == "resource_reallocation":
                    modified_workflow = await self._reallocate_resources(modified_workflow, improvement)
                
                # Record modification
                modified_workflow.modifications.append({
                    "timestamp": datetime.now(),
                    "type": improvement.type,
                    "description": improvement.description,
                    "target_steps": improvement.target_steps,
                    "changes": improvement.proposed_changes
                })
                modified_workflow.adaptation_count += 1
                modified_workflow.last_adaptation = datetime.now()
            
            return modified_workflow
            
        except Exception as e:
            logger.error(f"Workflow modification failed: {e}")
            return workflow

    async def _rebalance_agent_assignments(self, workflow: 'WorkflowSession') -> Dict[str, str]:
        """Rebalance agent assignments for optimal performance."""
        try:
            # Create LLM prompt for agent assignment optimization
            prompt = self._create_agent_assignment_prompt(workflow)
            
            # Get LLM recommendations
            assignment_response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse new assignments
            new_assignments = await self._parse_agent_assignments(assignment_response)
            
            # Update workflow with new assignments
            for step_id, agent_id in new_assignments.items():
                if step_id in workflow.agent_assignments:
                    old_agent = workflow.agent_assignments[step_id]
                    workflow.agent_assignments[step_id] = agent_id
                    logger.info(f"Reassigned step {step_id} from {old_agent} to {agent_id}")
            
            return new_assignments
            
        except Exception as e:
            logger.error(f"Agent rebalancing failed: {e}")
            return {}

    # Helper methods for workflow modification

    async def _reorder_workflow_steps(
        self, 
        workflow: 'WorkflowSession', 
        improvement: 'WorkflowImprovement'
    ) -> 'WorkflowSession':
        """Reorder workflow steps to optimize execution flow."""
        new_order = improvement.proposed_changes.get("new_order", [])
        if new_order:
            # Create new step list with optimized order
            reordered_steps = []
            step_dict = {step["step_id"]: step for step in workflow.current_steps}
            
            for step_id in new_order:
                if step_id in step_dict:
                    reordered_steps.append(step_dict[step_id])
            
            # Add any remaining steps
            for step in workflow.current_steps:
                if step["step_id"] not in new_order:
                    reordered_steps.append(step)
            
            workflow.current_steps = reordered_steps
        
        return workflow

    async def _parallelize_workflow_steps(
        self, 
        workflow: 'WorkflowSession', 
        improvement: 'WorkflowImprovement'
    ) -> 'WorkflowSession':
        """Enable parallel execution for compatible steps."""
        parallel_groups = improvement.proposed_changes.get("parallel_groups", [])
        
        for group in parallel_groups:
            # Update dependencies to enable parallel execution
            for step_id in group:
                if step_id in workflow.step_dependencies:
                    # Remove dependencies within the parallel group
                    workflow.step_dependencies[step_id] = [
                        dep for dep in workflow.step_dependencies[step_id] 
                        if dep not in group
                    ]
        
        return workflow

    async def _add_workflow_step(
        self, 
        workflow: 'WorkflowSession', 
        improvement: 'WorkflowImprovement'
    ) -> 'WorkflowSession':
        """Add new step to workflow."""
        new_step = improvement.proposed_changes.get("new_step")
        if new_step:
            workflow.current_steps.append(new_step)
            
            # Set up dependencies
            step_id = new_step["step_id"]
            dependencies = improvement.proposed_changes.get("dependencies", [])
            workflow.step_dependencies[step_id] = dependencies
        
        return workflow

    async def _remove_workflow_step(
        self, 
        workflow: 'WorkflowSession', 
        improvement: 'WorkflowImprovement'
    ) -> 'WorkflowSession':
        """Remove unnecessary step from workflow."""
        steps_to_remove = improvement.target_steps
        
        # Remove steps
        workflow.current_steps = [
            step for step in workflow.current_steps 
            if step["step_id"] not in steps_to_remove
        ]
        
        # Clean up dependencies
        for step_id in steps_to_remove:
            if step_id in workflow.step_dependencies:
                del workflow.step_dependencies[step_id]
        
        # Remove references to deleted steps from other dependencies
        for step_id, deps in workflow.step_dependencies.items():
            workflow.step_dependencies[step_id] = [
                dep for dep in deps if dep not in steps_to_remove
            ]
        
        return workflow

    async def _reallocate_resources(
        self, 
        workflow: 'WorkflowSession', 
        improvement: 'WorkflowImprovement'
    ) -> 'WorkflowSession':
        """Reallocate resources based on performance data."""
        resource_changes = improvement.proposed_changes.get("resource_allocation", {})
        
        for step_id, resources in resource_changes.items():
            workflow.resource_allocations[step_id] = resources
        
        return workflow

    # Prompt creation methods

    def _create_performance_analysis_prompt(self, workflow_id: str, feedback: 'WorkflowFeedback') -> str:
        """Create LLM prompt for workflow performance analysis."""
        return f"""
Analyze the performance of workflow {workflow_id} based on the following feedback:

WORKFLOW FEEDBACK:
- Overall completion time: {feedback.overall_completion_time}s (target: {feedback.target_completion_time}s)
- Quality score: {feedback.quality_score}/1.0
- Efficiency score: {feedback.efficiency_score}/1.0
- User satisfaction: {feedback.user_satisfaction}/1.0

STEP RESULTS:
{json.dumps([{
    "step_id": step.step_id,
    "agent_id": step.agent_id,
    "status": step.status,
    "execution_time": step.execution_time,
    "quality_score": step.quality_score,
    "errors": step.errors
} for step in feedback.step_results], indent=2)}

AGENT PERFORMANCE:
{json.dumps({
    agent_id: {
        "tasks_completed": metrics.tasks_completed,
        "tasks_failed": metrics.tasks_failed,
        "average_execution_time": metrics.average_execution_time,
        "resource_efficiency": metrics.resource_efficiency,
        "quality_average": metrics.quality_average,
        "error_rate": metrics.error_rate
    } for agent_id, metrics in feedback.agent_performance.items()
}, indent=2)}

RESOURCE USAGE:
{json.dumps(feedback.resource_usage, indent=2)}

IDENTIFIED ISSUES:
- Bottlenecks: {feedback.bottlenecks}
- Failures: {feedback.failures}
- System alerts: {feedback.system_alerts}

Please analyze this data and provide insights on:
1. Performance bottlenecks and their root causes
2. Inefficiencies in the current workflow
3. Critical path analysis and parallel execution opportunities
4. Agent specialization mismatches
5. Error patterns and quality issues
6. Resource utilization problems

Respond with a JSON object containing your analysis.
"""

    def _create_improvement_identification_prompt(
        self, 
        feedback: 'WorkflowFeedback', 
        analysis: 'PerformanceAnalysis'
    ) -> str:
        """Create LLM prompt for identifying improvement opportunities."""
        return f"""
Based on the workflow performance analysis, identify specific improvement opportunities:

PERFORMANCE ANALYSIS:
- Efficiency score: {analysis.efficiency_score}/1.0
- Completion rate: {analysis.completion_rate}%
- Resource utilization: {analysis.resource_utilization}

IDENTIFIED BOTTLENECKS:
{json.dumps([{
    "type": bottleneck.type,
    "severity": bottleneck.severity,
    "affected_steps": bottleneck.affected_steps,
    "root_cause": bottleneck.root_cause,
    "impact_score": bottleneck.impact_score
} for bottleneck in analysis.bottlenecks], indent=2)}

INEFFICIENCIES:
{analysis.inefficiencies}

PARALLEL OPPORTUNITIES:
{analysis.parallel_opportunities}

AGENT UTILIZATION:
{analysis.agent_utilization}

Please identify specific improvements with the following types:
- "reorder": Change step execution order
- "parallelize": Enable parallel execution
- "replace_agent": Assign better-suited agents
- "add_step": Add missing steps for quality/efficiency
- "remove_step": Remove unnecessary steps
- "resource_reallocation": Optimize resource distribution

For each improvement, provide:
1. Type and priority (critical/high/medium/low)
2. Target steps affected
3. Description of the change
4. Proposed changes (specific modifications)
5. Expected benefit (percentage improvement)
6. Implementation cost and risk level
7. Confidence score and supporting evidence

Respond with a JSON array of improvement recommendations.
"""

    def _create_agent_assignment_prompt(self, workflow: 'WorkflowSession') -> str:
        """Create LLM prompt for optimizing agent assignments."""
        return f"""
Optimize agent assignments for the following workflow:

CURRENT WORKFLOW STEPS:
{json.dumps([{
    "step_id": step["step_id"],
    "description": step.get("description", ""),
    "current_agent": workflow.agent_assignments.get(step["step_id"], "unassigned"),
    "estimated_duration": workflow.step_timing_estimates.get(step["step_id"], 0)
} for step in workflow.current_steps], indent=2)}

AVAILABLE AGENTS:
{json.dumps([{
    "agent_id": info.agent_id,
    "agent_type": info.agent_type,
    "capabilities": [cap.value for cap in info.capabilities],
    "current_load": info.current_load,
    "availability": info.availability
} for info in self.available_agents.values()], indent=2)}

CURRENT ASSIGNMENTS:
{json.dumps(workflow.agent_assignments, indent=2)}

AGENT AVAILABILITY:
{json.dumps(workflow.agent_availability, indent=2)}

Please provide optimized agent assignments considering:
1. Agent capabilities and specializations
2. Current workload and availability
3. Task complexity and requirements
4. Parallel execution opportunities
5. Resource constraints

Respond with a JSON object mapping step_ids to optimal agent_ids.
"""

    # Utility methods

    def _calculate_efficiency_score(self, feedback: 'WorkflowFeedback') -> float:
        """Calculate overall workflow efficiency score."""
        if feedback.target_completion_time > 0:
            time_efficiency = min(1.0, feedback.target_completion_time / max(1.0, feedback.overall_completion_time))
        else:
            time_efficiency = 0.5
        
        quality_factor = feedback.quality_score
        error_factor = 1.0 - (len(feedback.failures) / max(1, len(feedback.step_results)))
        
        return (time_efficiency + quality_factor + error_factor) / 3.0

    def _calculate_completion_rate(self, feedback: 'WorkflowFeedback') -> float:
        """Calculate task completion rate."""
        if not feedback.step_results:
            return 0.0
        
        completed = sum(1 for step in feedback.step_results if step.status == "completed")
        return (completed / len(feedback.step_results)) * 100.0

    def _calculate_resource_utilization(self, feedback: 'WorkflowFeedback') -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        return {
            "cpu": feedback.resource_usage.get("cpu_utilization", 0.0),
            "memory": feedback.resource_usage.get("memory_utilization", 0.0),
            "agents": sum(metrics.current_load for metrics in feedback.agent_performance.values()) / max(1, len(feedback.agent_performance))
        }

    async def _identify_bottlenecks(
        self, 
        feedback: 'WorkflowFeedback', 
        analysis_data: Dict[str, Any]
    ) -> List['PerformanceBottleneck']:
        """Identify performance bottlenecks from feedback data."""
        bottlenecks = []
        
        # Identify slow steps
        avg_time = sum(step.execution_time for step in feedback.step_results) / max(1, len(feedback.step_results))
        for step in feedback.step_results:
            if step.execution_time > avg_time * 2:
                bottlenecks.append(PerformanceBottleneck(
                    type="slow_execution",
                    severity="high" if step.execution_time > avg_time * 3 else "medium",
                    affected_steps=[step.step_id],
                    root_cause=f"Step execution time ({step.execution_time}s) significantly exceeds average ({avg_time:.1f}s)",
                    impact_score=step.execution_time / avg_time,
                    resolution_strategies=["resource_allocation", "agent_replacement", "step_optimization"]
                ))
        
        # Identify overloaded agents
        for agent_id, metrics in feedback.agent_performance.items():
            if metrics.current_load > 0.8:
                bottlenecks.append(PerformanceBottleneck(
                    type="agent_overload",
                    severity="high" if metrics.current_load > 0.9 else "medium",
                    affected_steps=[],  # Would need to track which steps this agent is handling
                    root_cause=f"Agent {agent_id} operating at {metrics.current_load*100:.1f}% capacity",
                    impact_score=metrics.current_load,
                    resolution_strategies=["load_balancing", "parallel_execution", "agent_addition"]
                ))
        
        return bottlenecks

    def _analyze_quality_distribution(self, feedback: 'WorkflowFeedback') -> Dict[str, int]:
        """Analyze distribution of quality scores."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for step in feedback.step_results:
            if step.quality_score >= 0.8:
                distribution["high"] += 1
            elif step.quality_score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution

    async def _parse_performance_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured performance analysis data."""
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract key information from text
            return {
                "inefficiencies": [],
                "critical_path": [],
                "parallel_opportunities": [],
                "specialization_mismatches": [],
                "error_patterns": []
            }

    async def _parse_improvement_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured improvement recommendations."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # Fallback: return empty list
            return []

    async def _parse_agent_assignments(self, response: str) -> Dict[str, str]:
        """Parse LLM response into agent assignment mappings."""
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: return empty dict
            return {}

    # =============================================
    # INTELLIGENT TOOL COORDINATION AND SHARING
    # =============================================

    async def coordinate_tool_sharing(self) -> ToolCoordinationResult:
        """
        Coordinate intelligent tool sharing between agents using LLM reasoning.
        
        This method analyzes tool usage patterns, resolves conflicts, optimizes
        allocation, and coordinates shared operations for maximum system efficiency.
        """
        try:
            logger.info("Starting intelligent tool coordination...")
            start_time = datetime.utcnow()
            
            # Analyze current tool usage patterns
            usage_patterns = await self._analyze_tool_usage_patterns()
            logger.debug(f"Analyzed {len(usage_patterns)} tool usage patterns")
            
            # Identify and resolve tool conflicts
            current_conflicts = list(self.active_tool_conflicts.values())
            resolutions = await self._resolve_tool_conflicts(current_conflicts) if current_conflicts else []
            logger.debug(f"Resolved {len(resolutions)} tool conflicts")
            
            # Optimize tool allocation based on patterns and priorities
            active_agents = [info for info in self.available_agents.values() if info.availability]
            allocation_plan = await self._optimize_tool_allocation(active_agents)
            logger.debug(f"Created allocation plan for {len(active_agents)} agents")
            
            # Manage tool queues for efficient access
            queue_status = await self._manage_tool_queues()
            logger.debug(f"Managed {len(self.tool_queues)} tool queues")
            
            # Coordinate any active shared operations
            shared_op_results = []
            for operation in self.shared_operations.values():
                if operation.status == "active":
                    coord_result = await self._coordinate_shared_operations(operation)
                    shared_op_results.append(coord_result)
            
            # Calculate performance metrics
            coordination_time = (datetime.utcnow() - start_time).total_seconds()
            efficiency_score = self._calculate_coordination_efficiency(allocation_plan, resolutions)
            
            # Generate insights and recommendations using LLM
            insights = await self._generate_coordination_insights(usage_patterns, resolutions, allocation_plan)
            
            # Create coordination result
            result = ToolCoordinationResult(
                allocations_made=allocation_plan.agent_assignments if allocation_plan else {},
                conflicts_resolved=resolutions,
                optimizations_made=self._extract_optimizations(allocation_plan),
                overall_efficiency=efficiency_score,
                resource_utilization=self._calculate_resource_utilization(),
                queue_improvements=self._calculate_queue_improvements(),
                active_shared_operations=len([op for op in self.shared_operations.values() if op.status == "active"]),
                coordination_events=len(resolutions) + len(shared_op_results),
                successful_coordinations=len([r for r in shared_op_results if r.coordination_success]),
                failed_coordinations=len([r for r in shared_op_results if not r.coordination_success]),
                usage_insights=insights.get("insights", []),
                optimization_recommendations=insights.get("recommendations", []),
                predicted_bottlenecks=insights.get("bottlenecks", []),
                next_coordination_schedule=datetime.utcnow() + timedelta(minutes=15),
                recommended_tool_additions=insights.get("tool_additions", []),
                capacity_warnings=insights.get("warnings", [])
            )
            
            # Store coordination result
            self.coordination_history.append(result)
            await self._store_coordination_result(result)
            
            logger.info(f"Tool coordination completed: {result.coordination_events} events, "
                       f"{efficiency_score:.2f} efficiency score, {coordination_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool coordination failed: {e}")
            return ToolCoordinationResult(
                allocations_made={},
                conflicts_resolved=[],
                optimizations_made=[],
                overall_efficiency=0.0
            )

    async def _analyze_tool_usage_patterns(self) -> Dict[str, UsagePattern]:
        """Analyze tool usage patterns across all active agents using LLM reasoning."""
        try:
            # Gather tool usage data from all agents
            usage_data = await self._collect_tool_usage_data()
            
            # Create LLM prompt for pattern analysis
            prompt = self._create_usage_pattern_analysis_prompt(usage_data)
            
            # Get LLM analysis
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
                agent_id=self.agent_id
            )
            
            # Parse patterns from LLM response
            patterns_data = await self._parse_usage_patterns(response)
            
            patterns = {}
            for pattern_data in patterns_data:
                pattern = UsagePattern(
                    agent_id=pattern_data.get("agent_id"),
                    tool_name=pattern_data.get("tool_name"),
                    usage_frequency=pattern_data.get("usage_frequency", 0.0),
                    average_duration=pattern_data.get("average_duration", 0.0),
                    peak_usage_times=pattern_data.get("peak_usage_times", []),
                    common_operations=pattern_data.get("common_operations", []),
                    success_rate=pattern_data.get("success_rate", 0.0),
                    resource_intensity=pattern_data.get("resource_intensity", "low"),
                    interdependencies=pattern_data.get("interdependencies", []),
                    pattern_confidence=pattern_data.get("confidence", 0.0)
                )
                
                pattern_key = f"{pattern.agent_id}_{pattern.tool_name}" if pattern.agent_id and pattern.tool_name else pattern.pattern_id
                patterns[pattern_key] = pattern
                self.tool_usage_patterns[pattern_key] = pattern
            
            return patterns
            
        except Exception as e:
            logger.error(f"Usage pattern analysis failed: {e}")
            return {}

    async def _resolve_tool_conflicts(self, conflicts: List[ToolConflict]) -> List[Resolution]:
        """Resolve tool conflicts using LLM reasoning and fair allocation strategies."""
        resolutions = []
        
        try:
            for conflict in conflicts:
                # Create LLM prompt for conflict resolution
                prompt = self._create_conflict_resolution_prompt(conflict)
                
                # Get LLM analysis and resolution strategy
                response = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=1500,
                    temperature=0.4,
                    agent_id=self.agent_id
                )
                
                # Parse resolution from LLM response
                resolution_data = await self._parse_conflict_resolution(response)
                
                resolution = Resolution(
                    conflict_id=conflict.conflict_id,
                    resolution_type=resolution_data.get("resolution_type", "priority_based"),
                    assigned_agent=resolution_data.get("assigned_agent", ""),
                    queued_agents=resolution_data.get("queued_agents", []),
                    estimated_wait_time=resolution_data.get("estimated_wait_time", 0.0),
                    alternative_tools=resolution_data.get("alternative_tools", []),
                    reasoning=resolution_data.get("reasoning", ""),
                    confidence_score=resolution_data.get("confidence_score", 0.0),
                    implementation_plan=resolution_data.get("implementation_plan", {})
                )
                
                # Implement the resolution
                await self._implement_conflict_resolution(conflict, resolution)
                resolutions.append(resolution)
                
                # Remove resolved conflict
                if conflict.conflict_id in self.active_tool_conflicts:
                    del self.active_tool_conflicts[conflict.conflict_id]
            
            return resolutions
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return []

    async def _optimize_tool_allocation(self, agents: List[Any]) -> AllocationPlan:
        """Optimize tool allocation across agents using LLM reasoning."""
        try:
            # Gather agent capabilities and current tool assignments
            agent_data = await self._collect_agent_tool_data(agents)
            
            # Create LLM prompt for allocation optimization
            prompt = self._create_allocation_optimization_prompt(agent_data)
            
            # Get LLM optimization recommendations
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.3,
                agent_id=self.agent_id
            )
            
            # Parse allocation plan from LLM response
            plan_data = await self._parse_allocation_plan(response)
            
            allocation_plan = AllocationPlan(
                agent_assignments=plan_data.get("agent_assignments", {}),
                tool_schedules=plan_data.get("tool_schedules", {}),
                estimated_completion=self._parse_completion_times(plan_data.get("estimated_completion", {})),
                resource_utilization=plan_data.get("resource_utilization", {}),
                optimization_score=plan_data.get("optimization_score", 0.0),
                conflicts_resolved=plan_data.get("conflicts_resolved", 0),
                efficiency_improvement=plan_data.get("efficiency_improvement", 0.0),
                implementation_order=plan_data.get("implementation_order", []),
                fallback_strategies=plan_data.get("fallback_strategies", {})
            )
            
            # Apply the allocation plan
            await self._apply_allocation_plan(allocation_plan)
            
            return allocation_plan
            
        except Exception as e:
            logger.error(f"Tool allocation optimization failed: {e}")
            return AllocationPlan()

    async def _manage_tool_queues(self) -> QueueStatus:
        """Manage tool operation queues for efficient access."""
        try:
            overall_status = QueueStatus(
                tool_name="all_tools",
                queue_length=0,
                average_processing_time=0.0,
                queue_efficiency=0.0
            )
            
            total_queue_length = 0
            total_processing_time = 0.0
            queue_count = 0
            
            for tool_name, queue in self.tool_queues.items():
                # Update queue metrics
                await self._update_queue_metrics(queue)
                
                # Optimize queue order based on priorities and dependencies
                await self._optimize_queue_order(queue)
                
                # Process waiting agents if tool becomes available
                await self._process_queue_if_available(queue)
                
                total_queue_length += queue.queue_length
                total_processing_time += queue.average_processing_time
                queue_count += 1
            
            if queue_count > 0:
                overall_status.queue_length = total_queue_length
                overall_status.average_processing_time = total_processing_time / queue_count
                overall_status.queue_efficiency = self._calculate_overall_queue_efficiency()
            
            return overall_status
            
        except Exception as e:
            logger.error(f"Queue management failed: {e}")
            return QueueStatus()

    async def _coordinate_shared_operations(self, operation: SharedOperation) -> CoordinationResult:
        """Coordinate shared operations between multiple agents."""
        try:
            logger.info(f"Coordinating shared operation: {operation.operation_id}")
            
            # Create coordination strategy based on operation type
            coordination_strategy = await self._create_coordination_strategy(operation)
            
            # Initialize coordination
            await self._initialize_shared_operation(operation, coordination_strategy)
            
            # Monitor and coordinate execution
            coordination_result = await self._execute_coordinated_operation(operation, coordination_strategy)
            
            # Collect results and finalize
            await self._finalize_shared_operation(operation, coordination_result)
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Shared operation coordination failed: {e}")
            return CoordinationResult(
                operation_id=operation.operation_id,
                coordination_success=False
            )

    # =============================================
    # SUPPORTING METHODS FOR TOOL COORDINATION
    # =============================================

    async def _collect_tool_usage_data(self) -> Dict[str, Any]:
        """Collect comprehensive tool usage data from all agents."""
        usage_data = {
            "agents": {},
            "tools": {},
            "current_time": datetime.utcnow().isoformat(),
            "system_load": await self._get_system_load()
        }
        
        # Collect data from each agent
        for agent_id, agent_info in self.available_agents.items():
            if hasattr(agent_info, 'get_tool_usage_stats'):
                usage_data["agents"][agent_id] = await agent_info.get_tool_usage_stats()
            else:
                # Simulate usage data collection
                usage_data["agents"][agent_id] = {
                    "active_tools": [],
                    "recent_operations": [],
                    "performance_metrics": {}
                }
        
        return usage_data

    async def _get_system_load(self) -> Dict[str, float]:
        """Get current system resource load."""
        return {
            "cpu_usage": 0.5,  # Mock values - replace with actual system monitoring
            "memory_usage": 0.6,
            "tool_contention": len(self.active_tool_conflicts) / max(1, len(self.tool_queues))
        }

    def _create_usage_pattern_analysis_prompt(self, usage_data: Dict[str, Any]) -> str:
        """Create LLM prompt for analyzing tool usage patterns."""
        return f"""
Analyze the tool usage patterns across all agents in the FlutterSwarm system:

CURRENT USAGE DATA:
{json.dumps(usage_data, indent=2)}

ACTIVE AGENTS: {len(usage_data.get('agents', {}))}
SYSTEM LOAD: {usage_data.get('system_load', {})}

Please analyze and identify:
1. Usage frequency patterns for each agent-tool combination
2. Peak usage times and resource contention periods
3. Common operation sequences and dependencies
4. Success rates and performance characteristics
5. Resource intensity classifications
6. Inter-agent coordination opportunities

For each significant pattern, provide:
- Agent ID and tool name
- Usage frequency (operations per hour)
- Average operation duration
- Peak usage times
- Common operations performed
- Success rate (0.0 to 1.0)
- Resource intensity (low/medium/high)
- Dependencies on other tools
- Confidence in pattern analysis (0.0 to 1.0)

Respond with a JSON array of usage patterns.
"""

    def _create_conflict_resolution_prompt(self, conflict: ToolConflict) -> str:
        """Create LLM prompt for resolving tool conflicts."""
        return f"""
Resolve the following tool access conflict using fair and efficient strategies:

CONFLICT DETAILS:
- Tool: {conflict.tool_name}
- Operation: {conflict.operation_type}
- Conflicting Agents: {conflict.conflicting_agents}
- Priority Scores: {conflict.priority_scores}
- Conflict Type: {conflict.conflict_type}
- Severity: {conflict.severity}
- Estimated Delay: {conflict.estimated_delay}s

CURRENT SYSTEM STATE:
- Active Conflicts: {len(self.active_tool_conflicts)}
- Tool Queues: {list(self.tool_queues.keys())}
- Available Alternatives: Analyze based on tool capabilities

Resolution strategies to consider:
1. Priority-based assignment (highest priority agent gets access)
2. Queue-based fairness (first-come, first-served with priority weights)
3. Parallel execution (if tool supports concurrent access)
4. Alternative tool suggestion (redirect to equivalent tools)
5. Time-slicing (shared access with time limits)

Provide resolution with:
- Resolution type
- Agent to be assigned immediate access
- Agents to be queued (with estimated wait times)
- Alternative tools for redirected agents
- Detailed reasoning for the decision
- Confidence score (0.0 to 1.0)
- Implementation steps

Respond with a JSON object containing the resolution plan.
"""

    def _create_allocation_optimization_prompt(self, agent_data: Dict[str, Any]) -> str:
        """Create LLM prompt for optimizing tool allocation."""
        return f"""
Optimize tool allocation across agents for maximum system efficiency:

AGENT DATA:
{json.dumps(agent_data, indent=2)}

OPTIMIZATION GOALS:
1. Maximize overall system throughput
2. Minimize resource contention and conflicts
3. Balance workload across agents
4. Respect agent specializations and capabilities
5. Reduce average task completion time

CONSTRAINTS:
- Tool dependencies and compatibility
- Agent capacity limits
- Priority requirements
- Resource availability

Please provide an optimized allocation plan including:
- Agent assignments (agent_id -> [tool_names])
- Tool schedules with time slots
- Estimated completion times
- Resource utilization projections
- Optimization score and expected efficiency improvement
- Implementation order for changes
- Fallback strategies for each assignment

Consider:
- Agent specialization matching
- Historical performance data
- Current workload distribution
- Future resource needs

Respond with a JSON object containing the complete allocation plan.
"""

    async def _parse_usage_patterns(self, response: str) -> List[Dict[str, Any]]:
        """Parse usage patterns from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse usage patterns response as JSON")
            return []

    async def _parse_conflict_resolution(self, response: str) -> Dict[str, Any]:
        """Parse conflict resolution from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse conflict resolution response as JSON")
            return {"resolution_type": "queue", "confidence_score": 0.0}

    async def _parse_allocation_plan(self, response: str) -> Dict[str, Any]:
        """Parse allocation plan from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse allocation plan response as JSON")
            return {"agent_assignments": {}, "optimization_score": 0.0}

    def _parse_completion_times(self, completion_data: Dict[str, str]) -> Dict[str, datetime]:
        """Parse completion time strings to datetime objects."""
        completion_times = {}
        for key, time_str in completion_data.items():
            try:
                if isinstance(time_str, str):
                    completion_times[key] = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                else:
                    completion_times[key] = datetime.utcnow() + timedelta(minutes=30)
            except ValueError:
                completion_times[key] = datetime.utcnow() + timedelta(minutes=30)
        return completion_times

    async def _implement_conflict_resolution(self, conflict: ToolConflict, resolution: Resolution) -> None:
        """Implement the resolved conflict strategy."""
        try:
            if resolution.resolution_type == "priority_based":
                # Assign tool to highest priority agent
                self.tool_allocations[conflict.tool_name] = resolution.assigned_agent
                
                # Queue other agents
                for agent_id in resolution.queued_agents:
                    await self._add_agent_to_queue(conflict.tool_name, agent_id)
                    
            elif resolution.resolution_type == "alternative_tool":
                # Redirect agents to alternative tools
                for i, agent_id in enumerate(conflict.conflicting_agents):
                    if i < len(resolution.alternative_tools):
                        alternative_tool = resolution.alternative_tools[i]
                        await self._redirect_agent_to_tool(agent_id, alternative_tool)
                        
            elif resolution.resolution_type == "parallel":
                # Enable parallel access if tool supports it
                await self._enable_parallel_tool_access(conflict.tool_name, conflict.conflicting_agents)
                
        except Exception as e:
            logger.error(f"Failed to implement conflict resolution: {e}")

    async def _apply_allocation_plan(self, plan: AllocationPlan) -> None:
        """Apply the optimized allocation plan."""
        try:
            # Update tool allocations
            for agent_id, tool_names in plan.agent_assignments.items():
                for tool_name in tool_names:
                    self.tool_allocations[tool_name] = agent_id
                    
            # Schedule tool operations
            for tool_name, schedule in plan.tool_schedules.items():
                await self._schedule_tool_operations(tool_name, schedule)
                
        except Exception as e:
            logger.error(f"Failed to apply allocation plan: {e}")

    async def _collect_agent_tool_data(self, agents: List[Any]) -> Dict[str, Any]:
        """Collect agent tool capability and usage data."""
        agent_data = {}
        
        for agent in agents:
            if hasattr(agent, 'agent_id'):
                agent_data[agent.agent_id] = {
                    "capabilities": getattr(agent, 'capabilities', []),
                    "current_tools": [],
                    "tool_preferences": {},
                    "performance_metrics": {},
                    "availability": True
                }
        
        return {"agents": agent_data, "tools_available": list(self.tool_queues.keys())}

    def _calculate_coordination_efficiency(self, allocation_plan: AllocationPlan, resolutions: List[Resolution]) -> float:
        """Calculate overall coordination efficiency score."""
        try:
            if not allocation_plan:
                return 0.0
                
            # Factor in allocation optimization score
            allocation_score = allocation_plan.optimization_score
            
            # Factor in conflict resolution effectiveness
            resolution_score = sum(r.confidence_score for r in resolutions) / max(1, len(resolutions))
            
            # Factor in resource utilization
            utilization_score = sum(allocation_plan.resource_utilization.values()) / max(1, len(allocation_plan.resource_utilization))
            
            # Weighted combination
            return (allocation_score * 0.4 + resolution_score * 0.3 + utilization_score * 0.3)
            
        except Exception:
            return 0.0

    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization across tools."""
        utilization = {}
        
        for tool_name, queue in self.tool_queues.items():
            if queue.current_user:
                utilization[tool_name] = 0.8  # Assume 80% when in use
            else:
                utilization[tool_name] = 0.0
                
        return utilization

    def _calculate_queue_improvements(self) -> Dict[str, float]:
        """Calculate queue efficiency improvements."""
        improvements = {}
        
        for tool_name, queue in self.tool_queues.items():
            # Mock improvement calculation
            improvements[tool_name] = queue.queue_efficiency
            
        return improvements

    def _extract_optimizations(self, allocation_plan: AllocationPlan) -> List[Dict[str, Any]]:
        """Extract optimization details from allocation plan."""
        if not allocation_plan:
            return []
            
        optimizations = []
        
        # Add allocation optimizations
        if allocation_plan.agent_assignments:
            optimizations.append({
                "type": "allocation_optimization",
                "description": f"Optimized allocation for {len(allocation_plan.agent_assignments)} agents",
                "improvement": allocation_plan.efficiency_improvement,
                "confidence": allocation_plan.optimization_score
            })
        
        return optimizations

    async def _generate_coordination_insights(
        self, 
        patterns: Dict[str, UsagePattern], 
        resolutions: List[Resolution], 
        plan: AllocationPlan
    ) -> Dict[str, List[str]]:
        """Generate insights and recommendations using LLM analysis."""
        try:
            # Create analysis prompt
            prompt = f"""
Analyze the tool coordination results and provide insights:

USAGE PATTERNS: {len(patterns)} patterns identified
CONFLICTS RESOLVED: {len(resolutions)} conflicts
OPTIMIZATION PLAN: {plan.optimization_score if plan else 0.0} score

Based on this coordination cycle, provide:

1. Key insights about system behavior and tool usage
2. Recommendations for improving coordination efficiency
3. Predicted bottlenecks and capacity issues
4. Suggested additional tools needed
5. Warning about potential problems

Respond with JSON containing insights, recommendations, bottlenecks, tool_additions, and warnings arrays.
"""
            
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.4,
                agent_id=self.agent_id
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to generate coordination insights: {e}")
            return {
                "insights": ["Coordination completed successfully"],
                "recommendations": ["Continue monitoring tool usage"],
                "bottlenecks": [],
                "tool_additions": [],
                "warnings": []
            }

    async def _store_coordination_result(self, result: ToolCoordinationResult) -> None:
        """Store coordination result in memory for learning."""
        try:
            await self.memory_manager.store_memory(
                content=f"Tool coordination completed: {result.coordination_events} events, "
                       f"{result.overall_efficiency:.2f} efficiency",
                metadata={
                    "coordination_id": result.coordination_id,
                    "conflicts_resolved": len(result.conflicts_resolved),
                    "optimizations_made": len(result.optimizations_made),
                    "efficiency_score": result.overall_efficiency
                },
                correlation_id=result.coordination_id,
                importance=0.8,
                long_term=True
            )
        except Exception as e:
            logger.error(f"Failed to store coordination result: {e}")

    # Mock helper methods for queue and operation management
    async def _update_queue_metrics(self, queue: QueueStatus) -> None:
        """Update queue performance metrics."""
        queue.last_updated = datetime.now()
        queue.queue_efficiency = min(1.0, 1.0 / max(1.0, queue.queue_length))

    async def _optimize_queue_order(self, queue: QueueStatus) -> None:
        """Optimize queue order based on priorities."""
        # Sort waiting agents by priority (mock implementation)
        queue.waiting_agents.sort(key=lambda x: x.get('priority', 0.5), reverse=True)

    async def _process_queue_if_available(self, queue: QueueStatus) -> None:
        """Process queue if tool becomes available."""
        if not queue.current_user and queue.waiting_agents:
            # Assign tool to next agent in queue
            next_agent = queue.waiting_agents.pop(0)
            queue.current_user = next_agent.get('agent_id')
            queue.queue_length = len(queue.waiting_agents)

    async def _add_agent_to_queue(self, tool_name: str, agent_id: str) -> None:
        """Add agent to tool queue."""
        if tool_name not in self.tool_queues:
            self.tool_queues[tool_name] = QueueStatus(tool_name=tool_name)
        
        queue = self.tool_queues[tool_name]
        queue.waiting_agents.append({'agent_id': agent_id, 'priority': 0.5, 'queued_at': datetime.now()})
        queue.queue_length = len(queue.waiting_agents)

    async def _redirect_agent_to_tool(self, agent_id: str, tool_name: str) -> None:
        """Redirect agent to alternative tool."""
        # Mock implementation - would integrate with actual agent communication
        logger.info(f"Redirecting agent {agent_id} to alternative tool {tool_name}")

    async def _enable_parallel_tool_access(self, tool_name: str, agents: List[str]) -> None:
        """Enable parallel access to tool for multiple agents."""
        # Mock implementation - would configure tool for parallel access
        logger.info(f"Enabling parallel access to {tool_name} for agents: {agents}")

    async def _schedule_tool_operations(self, tool_name: str, schedule: List[Dict[str, Any]]) -> None:
        """Schedule tool operations based on allocation plan."""
        # Mock implementation - would set up actual scheduling
        logger.info(f"Scheduling {len(schedule)} operations for tool {tool_name}")

    def _calculate_overall_queue_efficiency(self) -> float:
        """Calculate overall queue system efficiency."""
        if not self.tool_queues:
            return 1.0
        
        total_efficiency = sum(queue.queue_efficiency for queue in self.tool_queues.values())
        return total_efficiency / len(self.tool_queues)

    async def _create_coordination_strategy(self, operation: SharedOperation) -> Dict[str, Any]:
        """Create coordination strategy for shared operation."""
        return {
            "strategy_type": operation.coordination_strategy,
            "synchronization_points": operation.synchronization_points,
            "communication_protocol": operation.communication_protocol,
            "resource_allocation": {},
            "timeline": []
        }

    async def _initialize_shared_operation(self, operation: SharedOperation, strategy: Dict[str, Any]) -> None:
        """Initialize shared operation coordination."""
        operation.status = "active"
        operation.started_at = datetime.now()
        for agent_id in operation.participating_agents:
            operation.progress[agent_id] = 0.0

    async def _execute_coordinated_operation(self, operation: SharedOperation, strategy: Dict[str, Any]) -> CoordinationResult:
        """Execute coordinated operation between agents."""
        try:
            # Mock coordination execution
            result = CoordinationResult(
                operation_id=operation.operation_id,
                coordination_success=True,
                participants_coordinated=operation.participating_agents,
                synchronization_achieved=True,
                resource_conflicts_resolved=0,
                efficiency_score=0.85,
                total_coordination_time=30.0,
                individual_results={agent_id: {"status": "success"} for agent_id in operation.participating_agents},
                collective_outcome={"overall_status": "success", "quality_score": 0.9}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Coordinated operation execution failed: {e}")
            return CoordinationResult(
                operation_id=operation.operation_id,
                coordination_success=False
            )

    async def _finalize_shared_operation(self, operation: SharedOperation, result: CoordinationResult) -> None:
        """Finalize shared operation and update status."""
        operation.status = "completed" if result.coordination_success else "failed"
        operation.completed_at = datetime.now()
        operation.results = result.collective_outcome
