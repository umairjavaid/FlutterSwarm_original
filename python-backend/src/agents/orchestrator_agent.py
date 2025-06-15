"""
Orchestrator Agent for FlutterSwarm Multi-Agent System.

This is the master agent that coordinates and manages all other specialized agents.
It handles task decomposition, workflow management, and inter-agent coordination
using LLM-driven reasoning and decision-making.
"""

import asyncio
import json
import uuid
import platform
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ..agents.base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..core.llm_client import LLMClient
from ..models.agent_models import AgentMessage, TaskResult, AgentCapabilityInfo
from ..models.task_models import TaskContext, WorkflowDefinition, TaskType, TaskPriority, ExecutionStrategy
from ..models.project_models import ProjectContext, PlatformTarget
from ..models.workflow_models import (
    WorkflowSession, EnvironmentState, WorkflowTemplate, 
    WorkflowStatus, WorkflowStep, ToolAvailability, DeviceInfo,
    EnvironmentIssue, EnvironmentIssueType, ResourceRequirement,
    WorkflowStepType
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
        
        # Enhanced workflow management
        self.environment_state: EnvironmentState = EnvironmentState()
        self.tool_availability_cache: Dict[str, bool] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        
        # System tracking
        self.completion_stats: Dict[str, Any] = {
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tasks_delegated": 0,
            "average_completion_time": 0.0
        }
        
        # Initialize workflow templates
        asyncio.create_task(self._initialize_workflow_templates())
    
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
            
            # Create comprehensive development session
            session = await self.create_development_session(task)
            
            if not session:
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    result={},
                    error="Failed to create development session",
                    correlation_id=task.correlation_id
                )
            
            # Execute the workflow session
            session_result = await self._execute_workflow_session(session)
            
            # Return consolidated result
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status="completed" if session_result["success"] else "failed",
                result=session_result,
                deliverables=session_result.get("deliverables", {}),
                error=session_result.get("error"),
                execution_time=session_result.get("execution_time"),
                metadata={"session_id": session.session_id},
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
    
    async def process_task_legacy(self, task: TaskContext) -> TaskResult:
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
5. Environment Management: Ensure development environment is properly configured
6. Resource Planning: Assess and allocate resources for development workflows

You must use logical reasoning to make all decisions and never rely on hardcoded rules.
"""

    # Enhanced Workflow Planning Methods
    
    async def _initialize_workflow_templates(self) -> None:
        """Initialize default workflow templates for common Flutter development scenarios."""
        try:
            # New Project Template
            new_project_template = WorkflowTemplate(
                template_id="new_flutter_project",
                name="New Flutter Project Creation",
                description="Complete workflow for creating a new Flutter project from scratch",
                category="new_project",
                target_project_types=["app", "package", "plugin"],
                target_platforms=[],  # All platforms
                steps=await self._create_new_project_workflow_steps(),
                requirements=ResourceRequirement(
                    cpu_cores=2,
                    memory_gb=4.0,
                    disk_space_gb=15.0,
                    network_required=True
                ),
                estimated_duration=2400,  # 40 minutes
                complexity_level="medium",
                prerequisites=[
                    "Flutter SDK installed",
                    "Development environment configured"
                ],
                tags=["creation", "setup", "initialization"]
            )
            
            # Feature Development Template
            feature_dev_template = WorkflowTemplate(
                template_id="feature_development",
                name="Feature Development Workflow",
                description="Workflow for developing new features in existing Flutter projects",
                category="feature_development",
                target_project_types=["app"],
                steps=await self._create_feature_development_workflow_steps(),
                requirements=ResourceRequirement(
                    cpu_cores=2,
                    memory_gb=3.0,
                    disk_space_gb=5.0,
                    network_required=True
                ),
                estimated_duration=3600,  # 60 minutes
                complexity_level="medium",
                prerequisites=[
                    "Existing Flutter project",
                    "Development environment ready"
                ],
                tags=["development", "feature", "implementation"]
            )
            
            # Testing Template
            testing_template = WorkflowTemplate(
                template_id="comprehensive_testing",
                name="Comprehensive Testing Workflow",
                description="Complete testing workflow including unit, widget, and integration tests",
                category="testing",
                target_project_types=["app", "package"],
                steps=await self._create_testing_workflow_steps(),
                requirements=ResourceRequirement(
                    cpu_cores=4,
                    memory_gb=6.0,
                    disk_space_gb=10.0,
                    network_required=True
                ),
                estimated_duration=4800,  # 80 minutes
                complexity_level="complex",
                tags=["testing", "quality", "validation"]
            )
            
            # Store templates
            self.workflow_templates = {
                template.template_id: template
                for template in [new_project_template, feature_dev_template, testing_template]
            }
            
            logger.info(f"Initialized {len(self.workflow_templates)} workflow templates")
            
        except Exception as e:
            logger.error(f"Error initializing workflow templates: {e}")
    
    async def _create_new_project_workflow_steps(self) -> List[WorkflowStep]:
        """Create workflow steps for new project creation."""
        return [
            WorkflowStep(
                step_id="env_check_setup",
                step_type=WorkflowStepType.ENVIRONMENT_CHECK,
                name="Environment Health Check",
                description="Validate development environment and identify issues",
                agent_type="orchestrator",
                estimated_duration=120,
                required_tools=["flutter", "dart"],
                success_criteria=[
                    "Flutter SDK available and compatible",
                    "Dart SDK functional",
                    "Basic tools accessible"
                ],
                validation_steps=[
                    "Run flutter doctor",
                    "Check SDK versions",
                    "Verify tool availability"
                ]
            ),
            WorkflowStep(
                step_id="env_setup",
                step_type=WorkflowStepType.ENVIRONMENT_SETUP,
                name="Environment Setup and Configuration",
                description="Configure development environment for Flutter development",
                agent_type="devops",
                estimated_duration=300,
                dependencies=["env_check_setup"],
                required_tools=["flutter", "dart"],
                environment_requirements={"min_flutter_version": "3.0.0"},
                success_criteria=[
                    "All tools properly configured",
                    "Environment passes health checks",
                    "Required platforms available"
                ]
            ),
            WorkflowStep(
                step_id="project_creation",
                step_type=WorkflowStepType.CODE_GENERATION,
                name="Flutter Project Creation",
                description="Create new Flutter project with specified configuration",
                agent_type="implementation",
                estimated_duration=180,
                dependencies=["env_setup"],
                required_tools=["flutter"],
                success_criteria=[
                    "Project structure created",
                    "Dependencies resolved",
                    "Basic compilation successful"
                ]
            ),
            WorkflowStep(
                step_id="initial_architecture",
                step_type=WorkflowStepType.ARCHITECTURE_DESIGN,
                name="Initial Architecture Setup",
                description="Design and implement basic project architecture",
                agent_type="architecture",
                estimated_duration=600,
                dependencies=["project_creation"],
                success_criteria=[
                    "Architecture pattern defined",
                    "Folder structure organized",
                    "Base components created"
                ]
            ),
            WorkflowStep(
                step_id="initial_validation",
                step_type=WorkflowStepType.VALIDATION,
                name="Initial Project Validation",
                description="Validate project setup and perform initial tests",
                agent_type="testing",
                estimated_duration=300,
                dependencies=["initial_architecture"],
                required_tools=["flutter"],
                success_criteria=[
                    "Project builds successfully",
                    "Basic tests pass",
                    "No critical issues detected"
                ]
            )
        ]
    
    async def _create_feature_development_workflow_steps(self) -> List[WorkflowStep]:
        """Create workflow steps for feature development."""
        return [
            WorkflowStep(
                step_id="project_analysis",
                step_type=WorkflowStepType.PROJECT_ANALYSIS,
                name="Project Analysis and Planning",
                description="Analyze existing project and plan feature implementation",
                agent_type="architecture",
                estimated_duration=600,
                required_tools=["flutter"],
                success_criteria=[
                    "Project structure analyzed",
                    "Feature requirements clarified",
                    "Implementation plan created"
                ]
            ),
            WorkflowStep(
                step_id="feature_design",
                step_type=WorkflowStepType.ARCHITECTURE_DESIGN,
                name="Feature Architecture Design",
                description="Design architecture for the new feature",
                agent_type="architecture",
                estimated_duration=900,
                dependencies=["project_analysis"],
                success_criteria=[
                    "Feature architecture designed",
                    "Integration points identified",
                    "Component specifications defined"
                ]
            ),
            WorkflowStep(
                step_id="feature_implementation",
                step_type=WorkflowStepType.CODE_GENERATION,
                name="Feature Implementation",
                description="Implement the feature according to design specifications",
                agent_type="implementation",
                estimated_duration=1800,
                dependencies=["feature_design"],
                required_tools=["flutter", "dart"],
                success_criteria=[
                    "Feature code implemented",
                    "Integration completed",
                    "Code compiles without errors"
                ]
            ),
            WorkflowStep(
                step_id="feature_testing",
                step_type=WorkflowStepType.TESTING,
                name="Feature Testing and Validation",
                description="Test the implemented feature thoroughly",
                agent_type="testing",
                estimated_duration=600,
                dependencies=["feature_implementation"],
                required_tools=["flutter"],
                success_criteria=[
                    "Unit tests created and passing",
                    "Widget tests implemented",
                    "Feature functionality validated"
                ]
            )
        ]
    
    async def _create_testing_workflow_steps(self) -> List[WorkflowStep]:
        """Create workflow steps for comprehensive testing."""
        return [
            WorkflowStep(
                step_id="test_planning",
                step_type=WorkflowStepType.PROJECT_ANALYSIS,
                name="Test Planning and Strategy",
                description="Analyze project and create comprehensive test strategy",
                agent_type="testing",
                estimated_duration=900,
                success_criteria=[
                    "Test strategy defined",
                    "Test coverage goals set",
                    "Test types identified"
                ]
            ),
            WorkflowStep(
                step_id="unit_testing",
                step_type=WorkflowStepType.TESTING,
                name="Unit Test Implementation",
                description="Create and execute unit tests for core functionality",
                agent_type="testing",
                estimated_duration=1800,
                dependencies=["test_planning"],
                required_tools=["flutter", "dart"],
                success_criteria=[
                    "Unit tests implemented",
                    "Code coverage > 80%",
                    "All tests passing"
                ]
            ),
            WorkflowStep(
                step_id="widget_testing",
                step_type=WorkflowStepType.TESTING,
                name="Widget Test Implementation",
                description="Create and execute widget tests for UI components",
                agent_type="testing",
                estimated_duration=1200,
                dependencies=["unit_testing"],
                required_tools=["flutter"],
                success_criteria=[
                    "Widget tests implemented",
                    "UI components tested",
                    "User interactions validated"
                ]
            ),
            WorkflowStep(
                step_id="integration_testing",
                step_type=WorkflowStepType.TESTING,
                name="Integration Test Implementation",
                description="Create and execute integration tests for complete workflows",
                agent_type="testing",
                estimated_duration=1800,
                dependencies=["widget_testing"],
                required_tools=["flutter"],
                required_platforms=[PlatformTarget.ANDROID, PlatformTarget.IOS],
                success_criteria=[
                    "Integration tests implemented",
                    "End-to-end workflows tested",
                    "Performance validated"
                ]
            )
        ]
    
    async def validate_and_setup_environment(self, project_context: Optional[ProjectContext] = None) -> EnvironmentState:
        """Validate and setup the development environment."""
        logger.info("Validating and setting up development environment")
        
        env_state = EnvironmentState()
        start_time = datetime.utcnow()
        
        try:
            # Check Flutter SDK
            flutter_available = await self._check_tool_availability("flutter")
            env_state.tools_available["flutter"] = flutter_available
            
            if flutter_available.available and flutter_available.version:
                env_state.flutter_version = flutter_available.version
            else:
                env_state.issues.append(EnvironmentIssue(
                    issue_type=EnvironmentIssueType.FLUTTER_SDK_MISSING,
                    severity="error",
                    description="Flutter SDK not found or not accessible",
                    resolution_steps=[
                        "Install Flutter SDK",
                        "Add Flutter to PATH",
                        "Run flutter doctor"
                    ],
                    blocking=True
                ))
            
            # Check Dart SDK
            dart_available = await self._check_tool_availability("dart")
            env_state.tools_available["dart"] = dart_available
            
            if dart_available.available and dart_available.version:
                env_state.dart_version = dart_available.version
            
            # Check platform-specific tools
            await self._check_platform_tools(env_state)
            
            # Detect connected devices
            await self._detect_connected_devices(env_state)
            
            # Get system information
            env_state.system_info = await self._get_system_info()
            
            # Calculate health score
            env_state.health_score = self._calculate_environment_health_score(env_state)
            
            # Update cache
            self.tool_availability_cache = {
                name: tool.available for name, tool in env_state.tools_available.items()
            }
            
            env_state.last_validated = datetime.utcnow()
            env_state.validation_duration = (env_state.last_validated - start_time).total_seconds()
            
            self.environment_state = env_state
            
            logger.info(f"Environment validation completed. Health score: {env_state.health_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error validating environment: {e}")
            env_state.issues.append(EnvironmentIssue(
                issue_type=EnvironmentIssueType.TOOL_UNAVAILABLE,
                severity="error",
                description=f"Environment validation failed: {str(e)}",
                blocking=True
            ))
        
        return env_state
    
    async def _check_tool_availability(self, tool_name: str) -> ToolAvailability:
        """Check if a specific tool is available and get its information."""
        start_time = datetime.utcnow()
        tool_availability = ToolAvailability(tool_name=tool_name, available=False)
        
        try:
            # Use appropriate agent to check tool
            if tool_name in ["flutter", "dart"]:
                # Use implementation agent's Flutter SDK tool
                result = await self._delegate_environment_check(tool_name)
                if result and result.get("available", False):
                    tool_availability.available = True
                    tool_availability.version = result.get("version")
                    tool_availability.path = result.get("path")
                else:
                    tool_availability.issues = result.get("issues", [f"{tool_name} not available"])
            
        except Exception as e:
            tool_availability.issues.append(f"Error checking {tool_name}: {str(e)}")
        
        tool_availability.check_duration = (datetime.utcnow() - start_time).total_seconds()
        tool_availability.last_checked = datetime.utcnow()
        
        return tool_availability
    
    async def _check_platform_tools(self, env_state: EnvironmentState) -> None:
        """Check availability of platform-specific development tools."""
        # Check Android SDK
        android_sdk = await self._check_tool_availability("android_sdk")
        env_state.tools_available["android_sdk"] = android_sdk
        
        # Check Xcode (macOS only)
        if env_state.system_info.get("os_type", "").lower() == "macos":
            xcode = await self._check_tool_availability("xcode")
            env_state.tools_available["xcode"] = xcode
        
        # Check platform-specific issues
        if not android_sdk.available:
            env_state.issues.append(EnvironmentIssue(
                issue_type=EnvironmentIssueType.ANDROID_SDK_MISSING,
                severity="warning",
                description="Android SDK not found - Android development unavailable",
                resolution_steps=[
                    "Install Android Studio",
                    "Configure Android SDK",
                    "Accept Android licenses"
                ],
                blocking=False
            ))
    
    async def _detect_connected_devices(self, env_state: EnvironmentState) -> None:
        """Detect and catalog connected development devices."""
        try:
            # Use implementation agent or process tool to get device list
            device_result = await self._delegate_device_detection()
            
            if device_result and "devices" in device_result:
                for device_data in device_result["devices"]:
                    device_info = DeviceInfo(
                        device_id=device_data.get("id", "unknown"),
                        name=device_data.get("name", "Unknown Device"),
                        platform=PlatformTarget(device_data.get("platform", "android")),
                        is_emulator=device_data.get("is_emulator", False),
                        status=device_data.get("status", "unknown"),
                        api_level=device_data.get("api_level"),
                        model=device_data.get("model"),
                        manufacturer=device_data.get("manufacturer")
                    )
                    env_state.devices_connected[device_info.device_id] = device_info
        
        except Exception as e:
            logger.warning(f"Error detecting devices: {e}")
            env_state.issues.append(EnvironmentIssue(
                issue_type=EnvironmentIssueType.DEVICE_NOT_CONNECTED,
                severity="info",
                description="Could not detect connected devices",
                blocking=False
            ))
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for environment assessment."""
        import platform
        import os
        
        return {
            "os_type": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "environment_variables": {
                key: os.environ.get(key, "")
                for key in ["PATH", "FLUTTER_ROOT", "ANDROID_HOME", "JAVA_HOME"]
                if key in os.environ
            }
        }
    
    def _calculate_environment_health_score(self, env_state: EnvironmentState) -> float:
        """Calculate a health score for the development environment."""
        score = 1.0
        
        # Critical tools
        if not env_state.tools_available.get("flutter", ToolAvailability("flutter", False)).available:
            score -= 0.5
        if not env_state.tools_available.get("dart", ToolAvailability("dart", False)).available:
            score -= 0.3
        
        # Platform tools
        if not env_state.tools_available.get("android_sdk", ToolAvailability("android_sdk", False)).available:
            score -= 0.1
        
        # Issues
        blocking_issues = sum(1 for issue in env_state.issues if issue.blocking and not issue.resolved)
        score -= blocking_issues * 0.2
        
        # Devices
        if not env_state.devices_connected:
            score -= 0.1
        
        return max(0.0, score)
    
    async def _delegate_environment_check(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Delegate environment checking to appropriate agent."""
        # This would delegate to the implementation agent with Flutter SDK tool
        # For now, return a simulated result
        return {
            "available": True,
            "version": "3.16.0",
            "path": "/usr/local/bin/flutter",
            "issues": []
        }
    
    async def _delegate_device_detection(self) -> Optional[Dict[str, Any]]:
        """Delegate device detection to appropriate agent."""
        # This would delegate to process tool for device listing
        # For now, return a simulated result
        return {
            "devices": [
                {
                    "id": "emulator-5554",
                    "name": "Android Emulator",
                    "platform": "android",
                    "is_emulator": True,
                    "status": "connected",
                    "api_level": "33"
                }
            ]
        }
    
    async def create_development_session(
        self,
        task: TaskContext,
        template_id: Optional[str] = None
    ) -> WorkflowSession:
        """Create a comprehensive development session with environment setup."""
        logger.info(f"Creating development session for task: {task.task_id}")
        
        # Validate environment first
        env_state = await self.validate_and_setup_environment(task.project_context)
        
        # Select appropriate template
        if template_id and template_id in self.workflow_templates:
            template = self.workflow_templates[template_id]
        else:
            template = await self._select_workflow_template(task)
        
        # Create workflow session
        session = WorkflowSession(
            session_id=str(uuid.uuid4()),
            name=f"Development Session: {task.description}",
            description=task.description,
            project_context=task.project_context,
            template_id=template.template_id if template else None,
            environment_state=env_state,
            resource_requirements=template.requirements if template else ResourceRequirement(),
            priority=task.priority,
            creator_agent_id=self.agent_id,
            estimated_duration=template.estimated_duration if template else 3600
        )
        
        # Create workflow steps
        if template:
            session.active_steps = await self._customize_template_steps(template, task, env_state)
        else:
            session.active_steps = await self._create_custom_workflow_steps(task, env_state)
        
        # Add environment setup steps if needed
        if env_state.has_blocking_issues():
            setup_steps = await self._create_environment_setup_steps(env_state)
            session.active_steps = setup_steps + session.active_steps
        
        # Store session
        self.active_workflows[session.session_id] = session
        
        logger.info(f"Created development session {session.session_id} with {len(session.active_steps)} steps")
        
        return session
    
    async def _select_workflow_template(self, task: TaskContext) -> Optional[WorkflowTemplate]:
        """Select the most appropriate workflow template for a task."""
        best_template = None
        best_score = 0.0
        
        for template in self.workflow_templates.values():
            if template.is_applicable_to_project(task.project_context):
                # Calculate suitability score based on task type and requirements
                score = self._calculate_template_suitability(template, task)
                if score > best_score:
                    best_score = score
                    best_template = template
        
        return best_template
    
    def _calculate_template_suitability(self, template: WorkflowTemplate, task: TaskContext) -> float:
        """Calculate how suitable a template is for a given task."""
        score = 0.5  # Base score
        
        # Match on category
        if task.task_type == TaskType.IMPLEMENTATION and "development" in template.category:
            score += 0.3
        elif task.task_type == TaskType.TESTING and "testing" in template.category:
            score += 0.3
        elif task.task_type == TaskType.ANALYSIS and "analysis" in template.category:
            score += 0.3
        
        # Consider success rate
        score += template.success_rate * 0.2
        
        return min(1.0, score)
    
    async def _customize_template_steps(
        self,
        template: WorkflowTemplate,
        task: TaskContext,
        env_state: EnvironmentState
    ) -> List[WorkflowStep]:
        """Customize template steps for specific task and environment."""
        customized_steps = []
        
        for step in template.steps:
            # Clone the step
            custom_step = WorkflowStep(
                step_id=f"{task.task_id}_{step.step_id}",
                step_type=step.step_type,
                name=step.name,
                description=step.description,
                agent_type=step.agent_type,
                estimated_duration=step.estimated_duration,
                dependencies=[f"{task.task_id}_{dep}" for dep in step.dependencies],
                required_tools=step.required_tools,
                required_platforms=step.required_platforms,
                environment_requirements=step.environment_requirements,
                parameters={**step.parameters, "task_context": task.to_dict()},
                success_criteria=step.success_criteria,
                validation_steps=step.validation_steps,
                rollback_steps=step.rollback_steps
            )
            
            # Check if step can execute in current environment
            can_execute, issues = custom_step.can_execute_in_environment(env_state)
            if not can_execute:
                custom_step.metadata["environment_issues"] = issues
            
            customized_steps.append(custom_step)
        
        return customized_steps
    
    async def _create_custom_workflow_steps(
        self,
        task: TaskContext,
        env_state: EnvironmentState
    ) -> List[WorkflowStep]:
        """Create custom workflow steps when no template is suitable."""
        # Use LLM to generate custom workflow steps
        prompt = self._create_custom_workflow_prompt(task, env_state)
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.3,
                max_tokens=2000,
                agent_id=self.agent_id
            )
            
            steps = await self._parse_custom_workflow_steps(response, task)
            return steps
            
        except Exception as e:
            logger.error(f"Error creating custom workflow steps: {e}")
            # Return basic fallback steps
            return [
                WorkflowStep(
                    step_id=f"{task.task_id}_analysis",
                    step_type=WorkflowStepType.PROJECT_ANALYSIS,
                    name="Task Analysis",
                    description="Analyze task requirements and plan implementation",
                    agent_type="architecture",
                    estimated_duration=600
                ),
                WorkflowStep(
                    step_id=f"{task.task_id}_implementation",
                    step_type=WorkflowStepType.CODE_GENERATION,
                    name="Task Implementation",
                    description="Implement the requested functionality",
                    agent_type="implementation",
                    estimated_duration=1800,
                    dependencies=[f"{task.task_id}_analysis"]
                )
            ]
    
    def _create_custom_workflow_prompt(self, task: TaskContext, env_state: EnvironmentState) -> str:
        """Create prompt for LLM to generate custom workflow steps."""
        return f"""
Create a custom workflow for the following Flutter development task:

TASK: {task.description}
TYPE: {task.task_type.value}
PRIORITY: {task.priority.value}

ENVIRONMENT STATE:
- Flutter Version: {env_state.flutter_version}
- Available Tools: {[name for name, tool in env_state.tools_available.items() if tool.available]}
- Connected Devices: {len(env_state.devices_connected)}
- Health Score: {env_state.health_score}

Create a JSON array of workflow steps with the following structure:
[
  {{
    "step_id": "unique_id",
    "step_type": "environment_check|environment_setup|project_analysis|architecture_design|code_generation|testing|build|deployment|documentation|validation",
    "name": "Step Name",
    "description": "Detailed description",
    "agent_type": "orchestrator|architecture|implementation|testing|devops|security|documentation|performance",
    "estimated_duration": 600,
    "dependencies": ["other_step_ids"],
    "required_tools": ["flutter", "dart"],
    "success_criteria": ["criterion1", "criterion2"]
  }}
]

Focus on creating logical, efficient steps that build upon each other.
"""
    
    async def _parse_custom_workflow_steps(self, response: str, task: TaskContext) -> List[WorkflowStep]:
        """Parse LLM response to create custom workflow steps."""
        try:
            import json
            
            # Clean response
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            steps_data = json.loads(response_clean)
            steps = []
            
            for step_data in steps_data:
                step = WorkflowStep(
                    step_id=f"{task.task_id}_{step_data.get('step_id', 'step')}",
                    step_type=WorkflowStepType(step_data.get('step_type', 'project_analysis')),
                    name=step_data.get('name', 'Workflow Step'),
                    description=step_data.get('description', ''),
                    agent_type=step_data.get('agent_type', 'implementation'),
                    estimated_duration=step_data.get('estimated_duration', 600),
                    dependencies=[f"{task.task_id}_{dep}" for dep in step_data.get('dependencies', [])],
                    required_tools=step_data.get('required_tools', []),
                    success_criteria=step_data.get('success_criteria', [])
                )
                steps.append(step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Error parsing custom workflow steps: {e}")
            return []
    
    async def _create_environment_setup_steps(self, env_state: EnvironmentState) -> List[WorkflowStep]:
        """Create steps to resolve environment issues."""
        setup_steps = []
        
        for i, issue in enumerate(env_state.issues):
            if issue.blocking and not issue.resolved:
                step = WorkflowStep(
                    step_id=f"env_fix_{i}",
                    step_type=WorkflowStepType.ENVIRONMENT_SETUP,
                    name=f"Resolve: {issue.issue_type.value}",
                    description=issue.description,
                    agent_type="devops",
                    estimated_duration=300,
                    success_criteria=[f"Issue {issue.issue_type.value} resolved"],
                    parameters={"issue": issue.to_dict()}
                )
                setup_steps.append(step)
        
        return setup_steps
    
    async def _execute_workflow_session(self, session: WorkflowSession) -> Dict[str, Any]:
        """Execute a complete workflow session with comprehensive tracking."""
        logger.info(f"Executing workflow session: {session.session_id}")
        
        session.status = WorkflowStatus.EXECUTING
        start_time = datetime.utcnow()
        
        try:
            # Check environment readiness
            if session.environment_state and session.environment_state.has_blocking_issues():
                session.add_log_entry("warning", "Environment has blocking issues, will attempt to resolve")
            
            # Execute steps sequentially or in parallel based on dependencies
            while session.get_next_executable_steps():
                executable_steps = session.get_next_executable_steps()
                
                # Execute steps (can be parallel if no dependencies)
                step_results = await self._execute_workflow_steps(session, executable_steps)
                
                # Update session state
                for step_id, result in step_results.items():
                    if result.get("success", False):
                        session.completed_steps.add(step_id)
                        session.add_log_entry("info", f"Step {step_id} completed successfully")
                    else:
                        session.failed_steps.add(step_id)
                        session.add_log_entry("error", f"Step {step_id} failed: {result.get('error', 'Unknown error')}")
                
                # Update progress
                session.calculate_progress()
                
                # Check for critical failures
                if len(session.failed_steps) > len(session.completed_steps):
                    session.add_log_entry("error", "Too many step failures, stopping session")
                    break
            
            # Determine final status
            if session.failed_steps:
                session.status = WorkflowStatus.FAILED
                success = False
            else:
                session.status = WorkflowStatus.COMPLETED
                success = True
            
            # Record completion
            session.end_time = datetime.utcnow()
            session.actual_duration = (session.end_time - start_time).total_seconds()
            
            # Collect deliverables
            deliverables = {}
            for step in session.active_steps:
                if step.step_id in session.completed_steps and step.result:
                    deliverables.update(step.result.get("deliverables", {}))
            
            session.deliverables = deliverables
            
            return {
                "success": success,
                "session_id": session.session_id,
                "deliverables": deliverables,
                "execution_time": session.actual_duration,
                "completed_steps": len(session.completed_steps),
                "failed_steps": len(session.failed_steps),
                "progress": session.progress_percentage,
                "environment_health": session.environment_state.health_score if session.environment_state else 1.0
            }
            
        except Exception as e:
            session.status = WorkflowStatus.FAILED
            session.add_log_entry("critical", f"Session execution failed: {str(e)}")
            logger.error(f"Error executing workflow session {session.session_id}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "session_id": session.session_id,
                "deliverables": {},
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _execute_workflow_steps(
        self, 
        session: WorkflowSession, 
        steps: List[WorkflowStep]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a list of workflow steps."""
        results = {}
        
        # Check if steps can be executed in parallel
        can_parallel = len(steps) > 1 and all(
            not step.dependencies or 
            all(dep in session.completed_steps for dep in step.dependencies)
            for step in steps
        )
        
        if can_parallel and len(steps) <= 3: # Limit parallel execution
            # Execute in parallel
            tasks = []
            for step in steps:
                tasks.append(self._execute_single_workflow_step(session, step))
            
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(step_results):
                step_id = steps[i].step_id
                if isinstance(result, Exception):
                    results[step_id] = {"success": False, "error": str(result)}
                else:
                    results[step_id] = result
        else:
            # Execute sequentially
            for step in steps:
                result = await self._execute_single_workflow_step(session, step)
                results[step.step_id] = result
                
                # Stop on critical failure
                if not result.get("success", False) and step.step_type in [
                    WorkflowStepType.ENVIRONMENT_SETUP, 
                    WorkflowStepType.ENVIRONMENT_CHECK
                ]:
                    break
        
        return results
    
    async def _execute_single_workflow_step(
        self, 
        session: WorkflowSession, 
        step: WorkflowStep
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.step_id} ({step.name})")
        
        step.status = "running"
        step.started_at = datetime.utcnow()
        
        try:
            # Check environment compatibility
            can_execute, issues = session.can_execute_step(step)
            if not can_execute:
                step.status = "failed"
                step.error = f"Environment issues: {', '.join(issues)}"
                return {"success": False, "error": step.error}
            
            # Delegate to appropriate agent
            result = await self._delegate_step_to_agent(step, session)
            
            # Update step status
            if result.get("success", False):
                step.status = "completed"
                step.result = result
                step.completed_at = datetime.utcnow()
                step.actual_duration = (step.completed_at - step.started_at).total_seconds()
            else:
                step.status = "failed"
                step.error = result.get("error", "Step execution failed")
            
            return result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            logger.error(f"Error executing step {step.step_id}: {e}")
            
            return {"success": False, "error": str(e)}
    
    async def _delegate_step_to_agent(
        self, 
        step: WorkflowStep, 
        session: WorkflowSession
    ) -> Dict[str, Any]:
        """Delegate a workflow step to the appropriate agent."""
        # Find suitable agent
        suitable_agents = [
            agent_info for agent_info in self.available_agents.values()
            if (agent_info.agent_type == step.agent_type and 
                agent_info.is_available())
        ]
        
        if not suitable_agents:
            return {
                "success": False, 
                "error": f"No available {step.agent_type} agent found"
            }
        
        # Select best agent (for now, use first available)
        selected_agent = suitable_agents[0]
        
        # Create task for the agent
        task_context = TaskContext(
            task_id=step.step_id,
            description=step.description,
            task_type=self._map_step_type_to_task_type(step.step_type),
            parameters=step.parameters,
            project_context=session.project_context,
            priority=session.priority,
            correlation_id=session.session_id,
            metadata={
                "step_type": step.step_type.value,
                "session_id": session.session_id,
                "required_tools": step.required_tools,
                "success_criteria": step.success_criteria
            }
        )
        
        # For now, simulate delegation (in real implementation, this would use the event bus)
        try:
            # This would be actual agent delegation in production
            result = await self._simulate_step_execution(step, task_context)
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Agent delegation failed: {str(e)}"}
    
    def _map_step_type_to_task_type(self, step_type: WorkflowStepType) -> TaskType:
        """Map workflow step types to task types."""
        mapping = {
            WorkflowStepType.PROJECT_ANALYSIS: TaskType.ANALYSIS,
            WorkflowStepType.ARCHITECTURE_DESIGN: TaskType.ANALYSIS,
            WorkflowStepType.CODE_GENERATION: TaskType.IMPLEMENTATION,
            WorkflowStepType.TESTING: TaskType.TESTING,
            WorkflowStepType.BUILD: TaskType.DEPLOYMENT,
            WorkflowStepType.DEPLOYMENT: TaskType.DEPLOYMENT,
            WorkflowStepType.DOCUMENTATION: TaskType.DOCUMENTATION,
            WorkflowStepType.ENVIRONMENT_SETUP: TaskType.IMPLEMENTATION,
            WorkflowStepType.ENVIRONMENT_CHECK: TaskType.ANALYSIS,
            WorkflowStepType.TOOL_PREPARATION: TaskType.IMPLEMENTATION,
            WorkflowStepType.VALIDATION: TaskType.TESTING
        }
        return mapping.get(step_type, TaskType.IMPLEMENTATION)
    
    async def _simulate_step_execution(
        self, 
        step: WorkflowStep, 
        task_context: TaskContext
    ) -> Dict[str, Any]:
        """Simulate step execution for demonstration purposes."""
        # In production, this would be actual agent delegation
        await asyncio.sleep(min(step.estimated_duration / 60, 3))  # Simulate work
        
        # Simulate success with some realistic outputs
        deliverables = {}
        
        if step.step_type == WorkflowStepType.PROJECT_ANALYSIS:
            deliverables = {
                "analysis_report": f"Analysis completed for {step.description}",
                "recommendations": ["Use clean architecture", "Implement proper state management"]
            }
        elif step.step_type == WorkflowStepType.CODE_GENERATION:
            deliverables = {
                "generated_files": ["lib/main.dart", "lib/models/", "lib/services/"],
                "code_metrics": {"lines_of_code": 250, "complexity": "medium"}
            }
        elif step.step_type == WorkflowStepType.TESTING:
            deliverables = {
                "test_files": ["test/unit/", "test/widget/"],
                "coverage_report": {"overall": "85%", "uncovered_lines": 45}
            }
        elif step.step_type == WorkflowStepType.ENVIRONMENT_SETUP:
            deliverables = {
                "environment_status": "configured",
                "tools_verified": step.required_tools,
                "issues_resolved": ["flutter_sdk_path", "android_licenses"]
            }
        
        return {
            "success": True,
            "deliverables": deliverables,
            "execution_time": step.estimated_duration,
            "message": f"Step {step.name} completed successfully"
        }
    
    # Session Management Methods
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow session."""
        if session_id not in self.active_workflows:
            return None
        
        session = self.active_workflows[session_id]
        
        return {
            "session_id": session.session_id,
            "name": session.name,
            "status": session.status.value,
            "progress": session.calculate_progress(),
            "estimated_remaining_time": session.estimate_remaining_time(),
            "completed_steps": len(session.completed_steps),
            "total_steps": len(session.active_steps),
            "failed_steps": len(session.failed_steps),
            "current_step": session.get_current_step().name if session.get_current_step() else None,
            "environment_health": session.environment_state.health_score if session.environment_state else 1.0,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None
        }
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause an active workflow session."""
        if session_id not in self.active_workflows:
            return False
        
        session = self.active_workflows[session_id]
        
        if session.status == WorkflowStatus.EXECUTING and session.can_be_paused:
            session.status = WorkflowStatus.PAUSED
            session.add_log_entry("info", "Session paused by user request")
            logger.info(f"Paused workflow session: {session_id}")
            return True
        
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused workflow session."""
        if session_id not in self.active_workflows:
            return False
        
        session = self.active_workflows[session_id]
        
        if session.status == WorkflowStatus.PAUSED:
            # Re-validate environment before resuming
            env_state = await self.validate_and_setup_environment(session.project_context)
            session.environment_state = env_state
            
            if env_state.has_blocking_issues():
                session.add_log_entry("warning", "Environment issues detected during resume")
                return False
            
            session.status = WorkflowStatus.EXECUTING
            session.add_log_entry("info", "Session resumed")
            logger.info(f"Resumed workflow session: {session_id}")
            return True
        
        return False
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active workflow session."""
        if session_id not in self.active_workflows:
            return False
        
        session = self.active_workflows[session_id]
        
        if session.status in [WorkflowStatus.EXECUTING, WorkflowStatus.PAUSED, WorkflowStatus.INITIALIZING]:
            session.status = WorkflowStatus.CANCELLED
            session.end_time = datetime.utcnow()
            session.add_log_entry("info", "Session cancelled by user request")
            logger.info(f"Cancelled workflow session: {session_id}")
            return True
        
        return False
    
    async def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """Get status of all active workflow sessions."""
        active_sessions = []
        
        for session_id, session in self.active_workflows.items():
            if session.status in [WorkflowStatus.EXECUTING, WorkflowStatus.PAUSED, WorkflowStatus.INITIALIZING]:
                status = await self.get_session_status(session_id)
                if status:
                    active_sessions.append(status)
        
        return active_sessions
    
    async def cleanup_completed_sessions(self, keep_recent_hours: int = 24) -> int:
        """Clean up completed sessions older than specified hours."""
        cleanup_time = datetime.utcnow() - timedelta(hours=keep_recent_hours)
        cleaned_count = 0
        
        sessions_to_remove = []
        
        for session_id, session in self.active_workflows.items():
            if (session.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                session.end_time and session.end_time < cleanup_time):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            # Save session data to memory before removing
            session = self.active_workflows[session_id]
            await self.memory_manager.store_memory(
                content=f"Workflow session completed: {session.name}",
                metadata={
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "duration": session.actual_duration,
                    "steps_completed": len(session.completed_steps),
                    "steps_failed": len(session.failed_steps)
                },
                importance=0.6,
                long_term=True
            )
            
            del self.active_workflows[session_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} completed workflow sessions")
        
        return cleaned_count
    
    async def get_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available workflow templates."""
        return {
            template_id: template.to_dict()
            for template_id, template in self.workflow_templates.items()
        }
    
    async def update_template_usage_stats(self, template_id: str, success: bool):
        """Update usage statistics for a workflow template."""
        if template_id in self.workflow_templates:
            template = self.workflow_templates[template_id]
            template.usage_count += 1
            template.last_used = datetime.utcnow()
            
            # Update success rate
            if template.usage_count == 1:
                template.success_rate = 1.0 if success else 0.0
            else:
                current_successes = template.success_rate * (template.usage_count - 1)
                if success:
                    current_successes += 1
                template.success_rate = current_successes / template.usage_count
    
    async def get_environment_status(self) -> Dict[str, Any]:
        """Get current development environment status."""
        if not self.environment_state or not self.environment_state.last_validated:
            # Force environment validation
            await self.validate_and_setup_environment()
        
        return self.environment_state.to_dict()
    
    async def refresh_tool_availability(self) -> Dict[str, bool]:
        """Refresh tool availability cache."""
        logger.info("Refreshing tool availability cache")
        
        tools_to_check = ["flutter", "dart", "android_sdk", "xcode"]
        updated_cache = {}
        
        for tool in tools_to_check:
            tool_availability = await self._check_tool_availability(tool)
            updated_cache[tool] = tool_availability.available
            
            # Update environment state
            if self.environment_state:
                self.environment_state.tools_available[tool] = tool_availability
        
        self.tool_availability_cache.update(updated_cache)
        
        # Recalculate health score
        if self.environment_state:
            self.environment_state.health_score = self._calculate_environment_health_score(self.environment_state)
        
        return updated_cache
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        active_count = len([
            s for s in self.active_workflows.values()
            if s.status in [WorkflowStatus.EXECUTING, WorkflowStatus.PAUSED]
        ])
        
        completed_count = len([
            s for s in self.active_workflows.values()
            if s.status == WorkflowStatus.COMPLETED
        ])
        
        failed_count = len([
            s for s in self.active_workflows.values()
            if s.status == WorkflowStatus.FAILED
        ])
        
        return {
            "active_sessions": active_count,
            "completed_sessions": completed_count,
            "failed_sessions": failed_count,
            "total_sessions": len(self.active_workflows),
            "success_rate": completed_count / max(1, completed_count + failed_count),
            "available_agents": len([a for a in self.available_agents.values() if a.is_available()]),
            "total_agents": len(self.available_agents),
            "environment_health": self.environment_state.health_score if self.environment_state else 0.0,
            "tool_availability": self.tool_availability_cache,
            "workflow_templates": len(self.workflow_templates)
        }
