"""
Orchestrator Agent - The primary coordinator for the multi-agent system.
Implements the Tier 1 agent that manages task delegation and workflow coordination.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum

from core.agent_types import (
    AgentType, AgentMessage, MessageType, TaskDefinition, TaskStatus,
    AgentResponse, Priority, ProjectContext, WorkflowState
)
from core.message_broker import MessageBroker, MessageHandler, create_agent_handler
from core.memory_manager import MemoryManager
from core.project_analyzer import ProjectAnalysisEngine
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TaskLedger:
    """Manages task tracking and coordination."""
    
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
    
    def add_task(self, task: TaskDefinition) -> bool:
        """Add a new task to the ledger."""
        try:
            self.tasks[task.task_id] = task
            self.task_dependencies[task.task_id] = task.depends_on.copy()
            logger.debug(f"Added task {task.task_id} to ledger")
            return True
        except Exception as e:
            logger.error(f"Failed to add task to ledger: {e}")
            return False
    
    def get_ready_tasks(self) -> List[TaskDefinition]:
        """Get tasks that are ready to execute (all dependencies satisfied)."""
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and 
                all(dep_id in self.completed_tasks for dep_id in self.task_dependencies[task_id])):
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks
    
    def mark_task_completed(self, task_id: str) -> bool:
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].completed_at = datetime.now()
            self.completed_tasks.add(task_id)
            logger.debug(f"Marked task {task_id} as completed")
            return True
        return False
    
    def mark_task_failed(self, task_id: str, error_message: str) -> bool:
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].error_message = error_message
            self.failed_tasks.add(task_id)
            logger.debug(f"Marked task {task_id} as failed: {error_message}")
            return True
        return False


class ProgressLedger:
    """Tracks overall workflow progress and metrics."""
    
    def __init__(self):
        self.session_id: str = ""
        self.start_time: datetime = datetime.now()
        self.milestones: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.quality_gates: Dict[str, bool] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
    
    def record_milestone(self, name: str, description: str, data: Dict[str, Any] = None):
        """Record a workflow milestone."""
        milestone = {
            "name": name,
            "description": description,
            "timestamp": datetime.now(),
            "data": data or {}
        }
        self.milestones.append(milestone)
        logger.info(f"Milestone reached: {name}")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update workflow metrics."""
        self.metrics.update(metrics)
        self.metrics["last_updated"] = datetime.now()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of workflow progress."""
        duration = datetime.now() - self.start_time
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration.total_seconds(),
            "milestones_count": len(self.milestones),
            "latest_milestone": self.milestones[-1] if self.milestones else None,
            "quality_gates": self.quality_gates,
            "metrics": self.metrics,
            "agent_performance": self.agent_performance
        }


class OrchestratorAgent(BaseAgent):
    """
    Primary orchestrator agent that coordinates the entire multi-agent workflow.
    Implements intelligent task decomposition, delegation, and quality gate management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.ORCHESTRATOR, config)
        
        # Core components
        self.task_ledger = TaskLedger()
        self.progress_ledger = ProgressLedger()
        self.project_analyzer = ProjectAnalysisEngine(config.get("analysis", {}))
        
        # Agent management
        self.available_agents: Dict[AgentType, bool] = {}
        self.agent_capabilities: Dict[AgentType, List[str]] = {}
        self.agent_workload: Dict[AgentType, int] = {}
        
        # Initialize specialized agents
        self._initialize_agent_capabilities()
        
        # Workflow state
        self.current_workflow: Optional[WorkflowState] = None
        self.current_project_context: Optional[ProjectContext] = None
        
        # Configuration
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 3)
        self.task_timeout = config.get("task_timeout", 300)  # 5 minutes
        self.quality_gate_enforcement = config.get("enforce_quality_gates", True)
        
        logger.info(f"OrchestratorAgent initialized with config: {config}")
    
    async def initialize(self):
        """Initialize the orchestrator agent."""
        try:
            await super().initialize()
            
            # Register message handlers
            handler = create_agent_handler(
                self.agent_id,
                [MessageType.TASK_REQUEST, MessageType.TASK_RESPONSE, MessageType.STATUS_UPDATE],
                self._handle_message
            )
            
            await self.message_broker.subscribe(handler)
            
            # Initialize project analyzer
            # Additional initialization if needed
            
            logger.info("OrchestratorAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OrchestratorAgent: {e}")
            raise
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process orchestration tasks."""
        try:
            self.current_workflow = state
            
            # Determine the next action based on workflow state
            if state.project_state == "analyzing":
                return await self._handle_project_analysis(state)
            elif state.project_state == "planning":
                return await self._handle_task_planning(state)
            elif state.project_state == "implementing":
                return await self._handle_task_execution(state)
            elif state.project_state == "testing":
                return await self._handle_quality_validation(state)
            else:
                return await self._handle_default_coordination(state)
            
        except Exception as e:
            logger.error(f"Failed to process orchestrator task: {e}")
            return self._create_error_response(str(e))
    
    async def _handle_project_analysis(self, state: WorkflowState) -> AgentResponse:
        """Handle project analysis phase."""
        try:
            logger.info(f"Starting project analysis for: {state.project_path}")
            
            # Analyze the project
            project_context = await self.project_analyzer.analyze_project(state.project_path)
            self.current_project_context = project_context
            
            # Store project context in memory
            session_memory = self.memory_manager.get_short_term_memory(state.session_id)
            await session_memory.store("project_context", project_context)
            
            # Record milestone
            self.progress_ledger.record_milestone(
                "project_analysis_complete",
                "Project analysis completed successfully",
                {
                    "project_state": project_context.project_state.value,
                    "file_count": project_context.file_count,
                    "architecture_pattern": project_context.architecture_pattern
                }
            )
            
            # Determine next phase
            if project_context.project_state.value in ["new", "initialized"]:
                next_state = "planning"
                next_tasks = await self._create_initial_setup_tasks(project_context)
            else:
                next_state = "planning"
                next_tasks = await self._create_development_tasks(project_context)
            
            # Add tasks to queue
            for task in next_tasks:
                self.task_ledger.add_task(task)
            
            # Create response message
            message = AgentMessage(
                sender_id=self.agent_id,
                message_type=MessageType.PROJECT_ANALYSIS,
                content=f"Project analysis completed. State: {project_context.project_state.value}",
                context={
                    "project_context": project_context.model_dump(),
                    "next_state": next_state,
                    "tasks_created": len(next_tasks)
                }
            )
            
            return AgentResponse(
                task_id="project_analysis",
                agent_id=self.agent_id,
                success=True,
                message=message,
                context={"next_state": next_state}
            )
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return self._create_error_response(f"Project analysis failed: {e}")
    
    async def _handle_task_planning(self, state: WorkflowState) -> AgentResponse:
        """Handle task planning and decomposition."""
        try:
            logger.info("Starting task planning phase")
            
            # Get ready tasks
            ready_tasks = self.task_ledger.get_ready_tasks()
            
            if not ready_tasks:
                # No tasks ready, transition to implementation
                message = AgentMessage(
                    sender_id=self.agent_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content="Task planning completed, moving to implementation",
                    context={"next_state": "implementing"}
                )
                
                return AgentResponse(
                    task_id="task_planning",
                    agent_id=self.agent_id,
                    success=True,
                    message=message,
                    context={"next_state": "implementing"}
                )
            
            # Assign tasks to agents
            task_assignments = []
            for task in ready_tasks[:self.max_concurrent_tasks]:
                agent_type = await self._select_best_agent(task)
                if agent_type:
                    task_assignments.append({
                        "task_id": task.task_id,
                        "agent_type": agent_type,
                        "task": task
                    })
                    
                    # Update task status
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    task.assigned_agent = agent_type
                    
                    # Send task to agent
                    await self._delegate_task_to_agent(task, agent_type)
            
            # Record milestone
            self.progress_ledger.record_milestone(
                "tasks_assigned",
                f"Assigned {len(task_assignments)} tasks to agents",
                {"assignments": task_assignments}
            )
            
            message = AgentMessage(
                sender_id=self.agent_id,
                message_type=MessageType.COORDINATION_REQUEST,
                content=f"Assigned {len(task_assignments)} tasks to agents",
                context={
                    "assignments": task_assignments,
                    "remaining_tasks": len(ready_tasks) - len(task_assignments)
                }
            )
            
            return AgentResponse(
                task_id="task_planning",
                agent_id=self.agent_id,
                success=True,
                message=message,
                context={"task_assignments": task_assignments}
            )
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return self._create_error_response(f"Task planning failed: {e}")
    
    async def _handle_task_execution(self, state: WorkflowState) -> AgentResponse:
        """Handle task execution coordination."""
        try:
            logger.info("Coordinating task execution")
            
            # Check status of running tasks
            running_tasks = [task for task in self.task_ledger.tasks.values() 
                           if task.status == TaskStatus.RUNNING]
            
            # Check for timeouts
            now = datetime.now()
            for task in running_tasks:
                if (task.started_at and 
                    (now - task.started_at).total_seconds() > self.task_timeout):
                    logger.warning(f"Task {task.task_id} timed out")
                    self.task_ledger.mark_task_failed(task.task_id, "Task timeout")
            
            # Get completion status
            total_tasks = len(self.task_ledger.tasks)
            completed_tasks = len(self.task_ledger.completed_tasks)
            failed_tasks = len(self.task_ledger.failed_tasks)
            
            # Update progress
            progress = completed_tasks / total_tasks if total_tasks > 0 else 0
            self.progress_ledger.update_metrics({
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "progress": progress
            })
            
            # Determine next action
            if completed_tasks + failed_tasks == total_tasks:
                # All tasks completed, move to testing
                next_state = "testing"
            else:
                # Continue execution
                next_state = "implementing"
            
            message = AgentMessage(
                sender_id=self.agent_id,
                message_type=MessageType.STATUS_UPDATE,
                content=f"Task execution progress: {completed_tasks}/{total_tasks} completed",
                context={
                    "progress": progress,
                    "next_state": next_state,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks
                }
            )
            
            return AgentResponse(
                task_id="task_execution",
                agent_id=self.agent_id,
                success=True,
                message=message,
                context={"next_state": next_state}
            )
            
        except Exception as e:
            logger.error(f"Task execution coordination failed: {e}")
            return self._create_error_response(f"Task execution failed: {e}")
    
    async def _handle_quality_validation(self, state: WorkflowState) -> AgentResponse:
        """Handle quality gate validation."""
        try:
            logger.info("Starting quality validation")
            
            # Check quality gates
            quality_results = {}
            
            # Code quality gate
            if self.current_project_context:
                code_quality_score = self.current_project_context.code_quality_score or 0
                quality_results["code_quality"] = code_quality_score >= 7.0
            
            # Test coverage gate
            if self.current_project_context:
                test_coverage = self.current_project_context.test_coverage or 0
                quality_results["test_coverage"] = test_coverage >= 0.8
            
            # Overall quality gate
            all_gates_passed = all(quality_results.values()) if quality_results else False
            
            # Update progress ledger
            self.progress_ledger.quality_gates = quality_results
            
            # Record milestone
            self.progress_ledger.record_milestone(
                "quality_validation_complete",
                f"Quality validation completed. Gates passed: {all_gates_passed}",
                {"quality_results": quality_results}
            )
            
            # Determine next action
            if all_gates_passed or not self.quality_gate_enforcement:
                next_state = "deploying"
            else:
                next_state = "implementing"  # Return to implementation for fixes
            
            message = AgentMessage(
                sender_id=self.agent_id,
                message_type=MessageType.QUALITY_GATE_RESULT,
                content=f"Quality validation completed. Gates passed: {all_gates_passed}",
                context={
                    "quality_results": quality_results,
                    "all_gates_passed": all_gates_passed,
                    "next_state": next_state
                }
            )
            
            return AgentResponse(
                task_id="quality_validation",
                agent_id=self.agent_id,
                success=True,
                message=message,
                context={
                    "next_state": next_state,
                    "quality_gates": quality_results
                }
            )
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return self._create_error_response(f"Quality validation failed: {e}")
    
    async def _handle_default_coordination(self, state: WorkflowState) -> AgentResponse:
        """Handle default coordination tasks."""
        try:
            # Default coordination logic
            message = AgentMessage(
                sender_id=self.agent_id,
                message_type=MessageType.STATUS_UPDATE,
                content="Orchestrator coordination in progress",
                context={"state": state.project_state}
            )
            
            return AgentResponse(
                task_id="coordination",
                agent_id=self.agent_id,
                success=True,
                message=message,
                context={}
            )
            
        except Exception as e:
            logger.error(f"Default coordination failed: {e}")
            return self._create_error_response(f"Default coordination failed: {e}")
    
    async def _create_initial_setup_tasks(self, project_context: ProjectContext) -> List[TaskDefinition]:
        """Create tasks for initial project setup."""
        tasks = []
        
        # Architecture setup task
        tasks.append(TaskDefinition(
            task_type="architecture_setup",
            description="Set up project architecture and folder structure",
            assigned_agent=AgentType.ARCHITECTURE,
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.HIGH
        ))
        
        # Initial implementation task
        tasks.append(TaskDefinition(
            task_type="initial_implementation",
            description="Create basic app structure and main components",
            assigned_agent=AgentType.IMPLEMENTATION,
            depends_on=[tasks[0].task_id] if tasks else [],
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.MEDIUM
        ))
        
        # Test setup task
        tasks.append(TaskDefinition(
            task_type="test_setup",
            description="Set up testing framework and initial tests",
            assigned_agent=AgentType.TESTING,
            depends_on=[tasks[1].task_id] if len(tasks) > 1 else [],
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.MEDIUM
        ))
        
        return tasks
    
    async def _create_development_tasks(self, project_context: ProjectContext) -> List[TaskDefinition]:
        """Create tasks for ongoing development."""
        tasks = []
        
        # Architecture review task
        tasks.append(TaskDefinition(
            task_type="architecture_review",
            description="Review and optimize project architecture",
            assigned_agent=AgentType.ARCHITECTURE,
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.MEDIUM
        ))
        
        # Implementation improvement task
        tasks.append(TaskDefinition(
            task_type="implementation_improvement",
            description="Improve existing code and add new features",
            assigned_agent=AgentType.IMPLEMENTATION,
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.MEDIUM
        ))
        
        # Test enhancement task
        tasks.append(TaskDefinition(
            task_type="test_enhancement",
            description="Enhance test coverage and add missing tests",
            assigned_agent=AgentType.TESTING,
            parameters={"project_context": project_context.model_dump()},
            priority=Priority.MEDIUM
        ))
        
        return tasks
    
    def _initialize_agent_capabilities(self):
        """Initialize the capabilities of all available agents."""
        self.agent_capabilities = {
            AgentType.ARCHITECTURE: [
                'analyze_project_structure',
                'detect_architecture_patterns',
                'recommend_structure_improvements',
                'generate_folder_structure',
                'architecture_review'
            ],
            AgentType.IMPLEMENTATION: [
                'generate_widgets',
                'implement_screens',
                'create_models',
                'setup_state_management',
                'code_refactoring'
            ],
            AgentType.TESTING: [
                'generate_unit_tests',
                'create_widget_tests',
                'setup_integration_tests',
                'run_test_coverage',
                'test_automation'
            ],
            AgentType.DEVOPS: [
                'setup_cicd',
                'setup_deployment',
                'deploy_app',
                'setup_infrastructure',
                'environment_config',
                'deployment_status',
                'rollback_deployment',
                'setup_monitoring',
                'docker_setup',
                'kubernetes_deploy',
                'ssl_setup',
                'backup_config',
                'security_scan'
            ]
        }
        
        # Initialize all agents as available
        for agent_type in self.agent_capabilities.keys():
            self.available_agents[agent_type] = True
            self.agent_workload[agent_type] = 0
        
        logger.info(f"Initialized capabilities for {len(self.agent_capabilities)} agent types")
    
    async def _select_best_agent(self, task: TaskDefinition) -> Optional[AgentType]:
        """Select the best agent for a given task."""
        try:
            # If task has assigned agent, use it
            if task.assigned_agent:
                return task.assigned_agent
            
            # Map task types to agent types
            task_agent_mapping = {
                "architecture_setup": AgentType.ARCHITECTURE,
                "architecture_review": AgentType.ARCHITECTURE,
                "initial_implementation": AgentType.IMPLEMENTATION,
                "implementation_improvement": AgentType.IMPLEMENTATION,
                "test_setup": AgentType.TESTING,
                "test_enhancement": AgentType.TESTING,
                "deployment": AgentType.DEVOPS,
                "ci_cd_setup": AgentType.DEVOPS
            }
            
            return task_agent_mapping.get(task.task_type)
            
        except Exception as e:
            logger.error(f"Failed to select agent for task: {e}")
            return None
    
    async def _delegate_task_to_agent(self, task: TaskDefinition, agent_type: AgentType):
        """Delegate a task to a specific agent."""
        try:
            # Create task request message
            message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=str(agent_type),
                message_type=MessageType.TASK_REQUEST,
                content=f"Task assignment: {task.description}",
                context={
                    "task": task.model_dump(),
                    "project_context": self.current_project_context.model_dump() if self.current_project_context else {}
                },
                requires_response=True,
                priority=task.priority
            )
            
            # Send message to agent
            await self.message_broker.send_direct_message(message, str(agent_type))
            
            # Track assignment
            self.task_ledger.task_assignments[task.task_id] = str(agent_type)
            
            logger.info(f"Delegated task {task.task_id} to {agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to delegate task: {e}")
            raise
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages."""
        try:
            if message.message_type == MessageType.TASK_RESPONSE:
                await self._handle_task_response(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.TASK_REQUEST:
                await self._handle_task_request(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
    
    async def _handle_task_response(self, message: AgentMessage):
        """Handle task completion responses."""
        try:
            task_data = message.context.get("task", {})
            task_id = task_data.get("task_id")
            success = message.context.get("success", False)
            
            if not task_id:
                logger.warning("Task response missing task_id")
                return
            
            if success:
                self.task_ledger.mark_task_completed(task_id)
                logger.info(f"Task {task_id} completed successfully")
            else:
                error_message = message.context.get("error", "Unknown error")
                self.task_ledger.mark_task_failed(task_id, error_message)
                logger.warning(f"Task {task_id} failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to handle task response: {e}")
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle agent status updates."""
        try:
            agent_id = message.sender_id
            status = message.context.get("status", "unknown")
            
            # Update agent performance tracking
            if agent_id not in self.progress_ledger.agent_performance:
                self.progress_ledger.agent_performance[agent_id] = {
                    "last_update": datetime.now(),
                    "status": status,
                    "task_count": 0,
                    "success_count": 0
                }
            else:
                self.progress_ledger.agent_performance[agent_id].update({
                    "last_update": datetime.now(),
                    "status": status
                })
            
            logger.debug(f"Status update from {agent_id}: {status}")
            
        except Exception as e:
            logger.error(f"Failed to handle status update: {e}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle incoming task requests."""
        try:
            # This could be used for agent-to-orchestrator requests
            logger.info(f"Received task request from {message.sender_id}: {message.content}")
            
            # Process the request and potentially create new tasks
            # Implementation depends on specific requirements
            
        except Exception as e:
            logger.error(f"Failed to handle task request: {e}")
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Create an error response."""
        message = AgentMessage(
            sender_id=self.agent_id,
            message_type=MessageType.ERROR_REPORT,
            content=f"Orchestrator error: {error_message}",
            context={"error": error_message}
        )
        
        return AgentResponse(
            task_id="error",
            agent_id=self.agent_id,
            success=False,
            message=message,
            error_message=error_message
        )
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        try:
            return {
                "orchestrator_status": self.state.status.value,
                "current_workflow": self.current_workflow.model_dump() if self.current_workflow else None,
                "task_ledger": {
                    "total_tasks": len(self.task_ledger.tasks),
                    "completed_tasks": len(self.task_ledger.completed_tasks),
                    "failed_tasks": len(self.task_ledger.failed_tasks),
                    "ready_tasks": len(self.task_ledger.get_ready_tasks())
                },
                "progress_summary": self.progress_ledger.get_progress_summary(),
                "project_context": self.current_project_context.model_dump() if self.current_project_context else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {"error": str(e)}
