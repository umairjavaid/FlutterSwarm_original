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
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
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
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator agent."""
        return """
You are the Orchestrator Agent in the FlutterSwarm multi-agent system, responsible for coordinating all aspects of Flutter app development.

CORE RESPONSIBILITIES:
1. Task Decomposition: Break down complex development requests into manageable subtasks
2. Workflow Management: Design and execute efficient workflows across multiple agents
3. Agent Coordination: Assign tasks to the most suitable specialized agents
4. Progress Monitoring: Track task completion and handle dependencies
5. Quality Assurance: Ensure deliverables meet requirements and standards
6. Error Recovery: Handle failures and implement fallback strategies

SPECIALIZED AGENTS AVAILABLE:
- Architecture Agent: System design, patterns, project structure
- Implementation Agent: Code generation, feature development
- Testing Agent: Test creation, validation, quality assurance
- DevOps Agent: Deployment, CI/CD, infrastructure
- Security Agent: Security analysis, vulnerability assessment
- Documentation Agent: Documentation generation, API docs

DECISION-MAKING PRINCIPLES:
- Always consider the full project context and requirements
- Prioritize tasks based on dependencies and business value
- Choose the most appropriate execution strategy (sequential, parallel, hybrid)
- Ensure quality checkpoints throughout the workflow
- Maintain clear communication between agents
- Monitor resource utilization and optimize allocation

WORKFLOW EXECUTION STRATEGIES:
- Sequential: For tasks with strict dependencies
- Parallel: For independent tasks that can run concurrently
- Hybrid: Combination approach based on task characteristics

OUTPUT REQUIREMENTS:
- Provide clear, actionable task descriptions
- Include specific requirements and success criteria
- Define measurable deliverables
- Estimate realistic timeframes
- Consider error handling and rollback scenarios

You must use logical reasoning to make all decisions and never rely on hardcoded rules.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of orchestrator capabilities."""
        return [
            "task_decomposition",
            "workflow_management",
            "agent_coordination",
            "progress_monitoring",
            "resource_optimization",
            "error_recovery",
            "quality_assurance",
            "dependency_resolution"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute orchestrator-specific processing logic.
        
        For the orchestrator, this involves decomposing the task into a workflow
        and coordinating its execution across multiple agents.
        """
        try:
            # Create workflow from LLM analysis
            workflow = await self._create_workflow_from_analysis(task_context, llm_analysis)
            
            if not workflow:
                return {
                    "error": "Failed to create workflow from analysis",
                    "deliverables": {}
                }
            
            # Execute the workflow
            workflow_result = await self._execute_workflow(workflow)
            
            return {
                "workflow_id": workflow.workflow_id,
                "workflow_result": workflow_result,
                "deliverables": workflow_result.get("deliverables", {}),
                "execution_summary": {
                    "total_tasks": len(workflow.tasks),
                    "completed_tasks": workflow_result.get("completed_tasks", 0),
                    "execution_time": workflow_result.get("execution_time", 0),
                    "success": workflow_result.get("success", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator specialized processing: {e}")
            return {
                "error": str(e),
                "deliverables": {}
            }

    async def _create_workflow_from_analysis(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Optional[WorkflowDefinition]:
        """Create a workflow definition from LLM analysis."""
        try:
            # Use the existing decomposition method but with analysis context
            return await self._decompose_task_with_llm(task_context)
            
        except Exception as e:
            logger.error(f"Error creating workflow from analysis: {e}")
            return None
