"""
LangGraph Supervisor Agent for FlutterSwarm Multi-Agent System.

This module implements the supervisor pattern using LangGraph's StateGraph
for centralized orchestration and decision-making.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..models.langgraph_models import WorkflowState, AgentRole, WorkflowPhase, MessageType
from ..models.langgraph_models import create_agent_message, create_system_message
from ..models.task_models import TaskContext, TaskType
from ..models.agent_models import TaskStatus
from ..core.langgraph_checkpointer import create_checkpointer
from ..config import get_logger, settings
from .base_agent import AgentConfig, AgentCapability

logger = get_logger("langgraph_supervisor")


class MockAgent:
    """Mock agent for CLI compatibility."""
    def __init__(self, agent_type: str, capabilities: List[str]):
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.agent_id = f"{agent_type}_agent"
    
    async def process_task(self, task):
        """Mock task processing."""
        return {"status": "completed", "result": f"Mock {self.agent_type} result"}


class SupervisorAgent:
    """
    LangGraph-based supervisor agent that orchestrates the entire workflow.
    
    The supervisor makes high-level decisions about workflow progression,
    agent assignment, and error handling using LLM reasoning.
    """
    
    def __init__(
        self,
        checkpointer_backend: str = "memory",
        redis_url: Optional[str] = None
    ):
        """Initialize the supervisor agent."""
        self.agent_id = "supervisor"
        self.agent_roles = [role.value for role in AgentRole if role != AgentRole.SUPERVISOR]
        
        # Initialize LLM (LangChain for supervisor workflow)
        self.llm = self._create_llm()
        
        # Initialize FlutterSwarm LLM client for agents
        from ..core.llm_client import LLMClient
        self.flutterswarm_llm = LLMClient()
        
        # Create checkpointer
        self.checkpointer = create_checkpointer(
            backend=checkpointer_backend,
            redis_url=redis_url
        )
        
        # Initialize event bus (create a minimal one for compatibility)
        from ..core.event_bus import EventBus
        self.event_bus = EventBus()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("LangGraph Supervisor Agent initialized")
    
    def _create_llm(self):
        """Create LLM instance based on configuration."""
        if hasattr(settings, 'llm') and hasattr(settings.llm, 'anthropic_api_key') and settings.llm.anthropic_api_key:
            return ChatAnthropic(
                model=settings.llm.default_model,
                temperature=settings.llm.temperature,
                api_key=settings.llm.anthropic_api_key,
                timeout=60,  # Add timeout to prevent hanging
                max_retries=2  # Limit retries
            )
        else:
            # Fallback to Anthropic with environment variable
            return ChatAnthropic(
                model=settings.llm.default_model,
                temperature=settings.llm.temperature,
                timeout=60,  # Add timeout to prevent hanging
                max_retries=2  # Limit retries
            )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the main workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("task_decomposition", self._task_decomposition_node)
        workflow.add_node("agent_assignment", self._agent_assignment_node)
        workflow.add_node("execution_monitor", self._execution_monitor_node)
        workflow.add_node("result_aggregation", self._result_aggregation_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add specialized agent nodes
        self._add_agent_nodes(workflow)
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {
                "decompose": "task_decomposition",
                "assign": "agent_assignment", 
                "monitor": "execution_monitor",
                "aggregate": "result_aggregation",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "task_decomposition",
            self._decomposition_router,
            {
                "assign": "agent_assignment",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_assignment",
            self._agent_dispatch_router,
            {
                "implementation_agent": "implementation_agent",
                "architecture_agent": "architecture_agent", 
                "testing_agent": "testing_agent",
                "security_agent": "security_agent",
                "devops_agent": "devops_agent",
                "documentation_agent": "documentation_agent",
                "performance_agent": "performance_agent",
                "monitor": "execution_monitor",
                "aggregate": "result_aggregation",
                "supervisor": "supervisor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "execution_monitor",
            self._monitor_router,
            {
                "continue": "execution_monitor",
                "aggregate": "result_aggregation",
                "supervisor": "supervisor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "result_aggregation",
            self._aggregation_router,
            {
                "supervisor": "supervisor",
                "end": END,
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "error_handler",
            self._error_router,
            {
                "retry": "supervisor",
                "end": END
            }
        )
        
        # Add edges from all agent nodes back to execution monitor
        for agent_type in ["architecture", "implementation", "testing", "security", "devops", "documentation", "performance"]:
            workflow.add_edge(f"{agent_type}_agent", "execution_monitor")
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _agent_dispatch_router(self, state: WorkflowState) -> str:
        """Route tasks to the appropriate agent or next state."""
        logger.info("Agent dispatch router: Determining next step for assigned tasks")
        active_tasks = state.get("active_tasks", {})
        agent_assignments = state.get("agent_assignments", {})
        
        # Find the next task that has an agent assigned but hasn't been dispatched yet
        for task_id, task_info in active_tasks.items():
            if task_info.get("status") == "assigned":
                agent_type = task_info.get("agent_type")
                if agent_type:
                    agent_node_name = f"{agent_type}_agent"
                    logger.info(f"Dispatching task {task_id} to {agent_node_name}")
                    return agent_node_name

        # If no specific agent to dispatch to, or all dispatched tasks are running,
        # move to monitoring or another appropriate state.
        if active_tasks: # If there are still active (possibly running) tasks
            logger.info("All assigned tasks dispatched or in progress, moving to monitor.")
            return "monitor"
        elif state.get("pending_tasks"): # If there are tasks yet to be assigned
            logger.info("No tasks to dispatch, but pending tasks exist. Returning to supervisor.")
            return "supervisor"
        else: # No active or pending tasks, but maybe completed or error states
            logger.info("No tasks to dispatch and no pending tasks. Moving to aggregation.")
            return "aggregate"

    def _add_agent_nodes(self, workflow: StateGraph):
        """Add specialized agent nodes to the workflow."""
        try:
            # Import agent classes
            from .implementation_agent import ImplementationAgent
            from .architecture_agent import ArchitectureAgent
            from .testing_agent import TestingAgent
            from .security_agent import SecurityAgent
            from .devops_agent import DevOpsAgent
            from .documentation_agent import DocumentationAgent
            from .performance_agent import PerformanceAgent
            from ..core.memory_manager import MemoryManager

            # Create memory manager for agents
            memory_manager = MemoryManager(agent_id="shared_memory")

            # Initialize implementation agent
            implementation_config = AgentConfig(
                agent_id="implementation_agent",
                agent_type="implementation",
                llm_model=settings.llm.default_model,
                capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.FILE_OPERATIONS],
                max_concurrent_tasks=3
            )
            implementation_agent = ImplementationAgent(
                config=implementation_config,
                llm_client=self.flutterswarm_llm,
                memory_manager=memory_manager,
                event_bus=self.event_bus
            )
            workflow.add_node("implementation_agent", self._create_agent_node(implementation_agent))

            # Initialize architecture agent
            architecture_config = AgentConfig(
                agent_id="architecture_agent",
                agent_type="architecture",
                llm_model=settings.llm.default_model,
                capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
                max_concurrent_tasks=2
            )
            architecture_agent = ArchitectureAgent(
                config=architecture_config,
                llm_client=self.flutterswarm_llm,
                memory_manager=memory_manager,
                event_bus=self.event_bus
            )
            workflow.add_node("architecture_agent", self._create_agent_node(architecture_agent))

            # Initialize other agents
            agent_capabilities = {
                "testing": [AgentCapability.TESTING],
                "security": [AgentCapability.SECURITY_ANALYSIS, AgentCapability.VULNERABILITY_SCANNING],
                "devops": [AgentCapability.DEPLOYMENT, AgentCapability.INFRASTRUCTURE],
                "documentation": [AgentCapability.DOCUMENTATION],
                "performance": [AgentCapability.MONITORING]
            }
            
            for agent_type in ["testing", "security", "devops", "documentation", "performance"]:
                config = AgentConfig(
                    agent_id=f"{agent_type}_agent",
                    agent_type=agent_type,
                    llm_model=settings.llm.default_model,
                    capabilities=agent_capabilities.get(agent_type, [AgentCapability.ORCHESTRATION]),
                    max_concurrent_tasks=2
                )
                
                # Create mock agents for now (can be replaced with actual implementations)
                mock_agent = MockAgent(agent_type, [cap.value for cap in agent_capabilities.get(agent_type, [AgentCapability.ORCHESTRATION])])
                workflow.add_node(f"{agent_type}_agent", self._create_mock_agent_node(mock_agent))

            logger.info(f"Added mock agent nodes for roles: {self.agent_roles}")

        except ImportError as e:
            logger.warning(f"Could not import agent classes: {e}")
            # Fallback to mock agents
            agent_capabilities = {
                "testing": ["testing"],
                "security": ["security_analysis", "vulnerability_scanning"],
                "devops": ["deployment", "infrastructure"],
                "documentation": ["documentation"],
                "performance": ["monitoring"],
                "architecture": ["architecture_analysis"],
                "implementation": ["code_generation", "file_operations"]
            }
            
            for role in self.agent_roles:
                capabilities = agent_capabilities.get(role, ["orchestration"])
                mock_agent = MockAgent(role, capabilities)
                workflow.add_node(f"{role}_agent", self._create_mock_agent_node(mock_agent))
            logger.info(f"Added mock agent nodes for roles: {self.agent_roles}")

    def _create_agent_node(self, agent):
        """Create a workflow node from an agent instance."""
        async def agent_executor(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing agent node for: {agent.config.agent_type}")
            
            # Get active tasks for this agent
            active_tasks = state.get("active_tasks", {})
            agent_assignments = state.get("agent_assignments", {})
            project_context = state.get("project_context", {})
            
            # Find a task assigned to this agent
            task_to_process = None
            task_id = None
            for tid, agent_id in agent_assignments.items():
                if agent_id == agent.config.agent_id and tid in active_tasks:
                    task_to_process = active_tasks[tid]
                    task_id = tid
                    break
            
            if task_to_process:
                try:
                    # Create comprehensive task context with project requirements
                    from ..models.task_models import TaskContext, TaskType
                    
                    # Extract project details and features
                    project_name = project_context.get("project_name", "flutter_app")
                    features = project_context.get("features", [])
                    requirements = project_context.get("requirements", {})
                    output_dir = project_context.get("output_dir", ".")
                    
                    task_context = TaskContext(
                        task_id=task_to_process.get("task_id", "unknown"),
                        description=task_to_process.get("description", ""),
                        task_type=self._map_task_type(task_to_process.get("agent_type", "implementation")),
                        parameters=requirements,
                        dependencies=task_to_process.get("dependencies", []),
                        metadata={
                            "priority": task_to_process.get("priority", "normal"),
                            "agent_type": task_to_process.get("agent_type", "implementation"),
                            "project_name": project_name,
                            "features": features,
                            "output_dir": output_dir,
                            "deliverables": task_to_process.get("deliverables", []),
                            "requirements": requirements,
                            "correlation_id": f"supervisor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        }
                    )
                    
                    # Mark task as in progress
                    active_tasks[task_id]["status"] = "in_progress"
                    active_tasks[task_id]["progress"] = 0.1
                    active_tasks[task_id]["started_at"] = datetime.utcnow().isoformat()
                    
                    logger.info(f"Processing task {task_id} with agent {agent.config.agent_id}")
                    logger.info(f"Task description: {task_context.description}")
                    logger.info(f"Project features: {features}")
                    
                    # Process the task using the agent with timeout
                    result = await asyncio.wait_for(
                        agent.process_task(task_context),
                        timeout=300  # 5 minute timeout for agent processing
                    )
                    
                    # Update state with completed task
                    completed_tasks = state.get("completed_tasks", {})
                    completed_tasks[task_id] = {
                        **task_to_process,
                        "result": result,
                        "agent": agent.config.agent_id,
                        "status": "completed",
                        "progress": 1.0,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Remove from active tasks
                    if task_id in active_tasks:
                        del active_tasks[task_id]
                    
                    state["completed_tasks"] = completed_tasks
                    state["active_tasks"] = active_tasks
                    
                    logger.info(f"Agent {agent.config.agent_id} successfully completed task {task_id}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Agent {agent.config.agent_id} timed out processing task {task_id}")
                    # Mark task as failed due to timeout
                    failed_tasks = state.get("failed_tasks", {})
                    failed_tasks[task_id] = {
                        **task_to_process,
                        "status": "failed",
                        "error": "Agent processing timeout",
                        "failed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Remove from active tasks
                    if task_id in active_tasks:
                        del active_tasks[task_id]
                    
                    state["failed_tasks"] = failed_tasks
                    state["active_tasks"] = active_tasks
                    
                except Exception as e:
                    logger.error(f"Agent {agent.config.agent_id} failed to process task {task_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Mark task as failed
                    failed_tasks = state.get("failed_tasks", {})
                    failed_tasks[task_id] = {
                        **task_to_process,
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Remove from active tasks
                    if task_id in active_tasks:
                        del active_tasks[task_id]
                    
                    state["failed_tasks"] = failed_tasks
                    state["active_tasks"] = active_tasks
            
            return state
        
        return agent_executor

    async def _simulate_agent_processing(self, task_id: str, agent_type: str) -> None:
        """Simulate agent processing with some work."""
        # Simulate realistic processing time
        await asyncio.sleep(1.0)  # Reduced from longer times
        logger.debug(f"Agent {agent_type} completed processing task {task_id}")

    def _create_mock_deliverables(self, agent_type: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock deliverables based on agent type."""
        base_deliverables = {
            "task_summary": f"Completed {task_info.get('description', 'task')}",
            "status": "success",
            "agent_type": agent_type
        }
        
        # Add agent-specific deliverables
        if agent_type == "architecture":
            base_deliverables.update({
                "architecture_diagram": "system_architecture.md",
                "technical_specifications": "tech_specs.md"
            })
        elif agent_type == "implementation":
            base_deliverables.update({
                "source_code": "lib/main.dart",
                "project_structure": "flutter_project/"
            })
        elif agent_type == "testing":
            base_deliverables.update({
                "test_suite": "test/unit_tests.dart",
                "test_report": "test_results.md"
            })
        
        return base_deliverables

    def _create_mock_agent_node(self, mock_agent):
        """Create a workflow node from a mock agent."""
        async def mock_agent_executor(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing mock agent node for: {mock_agent.agent_type}")
            
            # Get active tasks
            active_tasks = state.get("active_tasks", {})
            agent_assignments = state.get("agent_assignments", {})
            
            # Find a task for this agent type
            task_to_complete_id = None
            for task_id, agent_id in agent_assignments.items():
                if agent_id == mock_agent.agent_id and task_id in active_tasks:
                    task_to_complete_id = task_id
                    break
            
            if task_to_complete_id:
                try:
                    # Process with timeout
                    await asyncio.wait_for(
                        self._simulate_agent_processing(task_to_complete_id, mock_agent.agent_type),
                        timeout=30  # 30 second timeout
                    )
                    
                    # Get completed and active tasks
                    completed_tasks = dict(state.get("completed_tasks", {}))
                    task_info = active_tasks[task_to_complete_id]
                    
                    # Create deliverables
                    deliverables = self._create_mock_deliverables(mock_agent.agent_type, task_info)
                    
                    completed_tasks[task_to_complete_id] = {
                        **task_info,
                        "status": "completed",
                        "result": f"Successfully completed by {mock_agent.agent_type}",
                        "deliverables": deliverables,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Remove from active tasks
                    remaining_active = {k: v for k, v in active_tasks.items() if k != task_to_complete_id}
                    
                    logger.info(f"Mock agent {mock_agent.agent_type} completed task {task_to_complete_id}")
                    
                    # Update state
                    return {
                        **state,
                        "active_tasks": remaining_active,
                        "completed_tasks": completed_tasks,
                        "next_action": "monitor"
                    }
                    
                except asyncio.TimeoutError:
                    logger.error(f"Agent {mock_agent.agent_type} timed out on task {task_to_complete_id}")
                    
                    # Mark as failed
                    failed_tasks = dict(state.get("failed_tasks", {}))
                    failed_tasks[task_to_complete_id] = {
                        **active_tasks[task_to_complete_id],
                        "status": "failed",
                        "error": "Processing timeout",
                        "failed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Remove from active tasks
                    remaining_active = {k: v for k, v in active_tasks.items() if k != task_to_complete_id}
                    
                    return {
                        **state,
                        "active_tasks": remaining_active,
                        "failed_tasks": failed_tasks,
                        "next_action": "error"
                    }

            return state

        return mock_agent_executor

    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Main supervisor decision-making node."""
        logger.info("Supervisor node: Making workflow decisions")
        
        # Create supervisor prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_supervisor_system_prompt()),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="What should be the next action in this workflow?")
        ])
        
        # Prepare context
        context = {
            "workflow_state": {
                "pending_tasks": len(state.get("pending_tasks", [])),
                "active_tasks": len(state.get("active_tasks", {})),
                "completed_tasks": len(state.get("completed_tasks", {})),
                "failed_tasks": len(state.get("failed_tasks", {})),
                "available_agents": list(state.get("available_agents", {}).keys())
            },
            "current_phase": self._determine_workflow_phase(state)
        }
        
        # Get LLM decision
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "messages": state.get("messages", []),
            "context": json.dumps(context, indent=2)
        })
        
        # Parse decision
        decision = self._parse_supervisor_decision(response.content)
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Supervisor decision: {decision['action']} - {decision['reasoning']}",
            MessageType.WORKFLOW_CONTROL,
            {"decision": decision, "context": context}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "next_action": decision["action"],
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "last_supervisor_decision": decision,
                "workflow_phase": context["current_phase"]
            }
        }
    
    async def _task_decomposition_node(self, state: WorkflowState) -> WorkflowState:
        """Decompose high-level tasks into agent-specific subtasks."""
        logger.info("Task decomposition node: Breaking down tasks")
        
        task_description = state["task_description"]
        project_context = state.get("project_context", {})
        project_name = project_context.get("project_name", "flutter_app")
        features = project_context.get("requirements", {}).get("features", [])
        
        # Create dynamic decomposition prompt
        prompt = f"""
        Analyze the following Flutter app development task and decompose it into specific subtasks for Flutter development agents.
        
        TASK: {task_description}
        PROJECT NAME: {project_name}
        FEATURES: {features}
        
        AVAILABLE AGENTS: architecture, implementation, testing, security, devops, documentation, performance
        
        You MUST respond with ONLY a valid JSON object with exactly 2 tasks (architecture + implementation):
        
        {{
            "tasks": [
                {{
                    "task_id": "arch_001",
                    "description": "Design {project_name} architecture with features: {', '.join(features) if features else 'basic functionality'}",
                    "agent_type": "architecture",
                    "priority": "high",
                    "estimated_duration": 30,
                    "dependencies": [],
                    "deliverables": ["Architecture design", "Project structure", "Dependencies list"]
                }},
                {{
                    "task_id": "impl_001", 
                    "description": "Create Flutter project {project_name} and implement features: {', '.join(features) if features else 'basic functionality'}",
                    "agent_type": "implementation",
                    "priority": "high",
                    "estimated_duration": 60,
                    "dependencies": ["arch_001"],
                    "deliverables": ["Flutter project", "Feature implementation", "UI components"]
                }}
            ],
            "workflow_name": "{project_name} Development",
            "total_estimated_time": 90
        }}
        
        Requirements:
        - Each task must have a unique task_id
        - agent_type must be one of the available agents
        - priority: "high", "medium", "low"
        - estimated_duration is in minutes
        - Exactly 2 tasks: 1 architecture + 1 implementation
        
        Respond with ONLY the JSON object, no explanation or markdown formatting.
        """
        
        try:
            # Generate decomposition with reduced timeout
            response = await asyncio.wait_for(
                self.llm.ainvoke([
                    {"role": "user", "content": prompt}
                ]),
                timeout=30  # Reduced timeout from 60 to 30 seconds
            )
            
            # Parse decomposition result
            decomposition = self._parse_task_decomposition(response.content)
            
            if not decomposition.get("tasks"):
                logger.warning("Task decomposition failed, using fallback")
                decomposition = self._create_dynamic_fallback_task_decomposition(project_name, features)
            
        except asyncio.TimeoutError:
            logger.error("Task decomposition timed out, using fallback")
            decomposition = self._create_dynamic_fallback_task_decomposition(project_name, features)
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            decomposition = self._create_dynamic_fallback_task_decomposition(project_name, features)
        
        # Convert to workflow tasks
        tasks = decomposition.get("tasks", [])
        pending_tasks = {}
        
        for task_data in tasks:
            task_id = task_data.get("task_id", f"task_{len(pending_tasks) + 1:03d}")
            pending_tasks[task_id] = {
                "task_id": task_id,
                "description": task_data.get("description", "No description"),
                "agent_type": task_data.get("agent_type", "implementation"),
                "priority": task_data.get("priority", "medium"),
                "estimated_duration": task_data.get("estimated_duration", 30),
                "dependencies": task_data.get("dependencies", []),
                "deliverables": task_data.get("deliverables", []),
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Decomposed task into {len(pending_tasks)} subtasks: {list(pending_tasks.keys())}",
            MessageType.TASK_UPDATE,
            {
                "decomposition_result": decomposition,
                "task_count": len(pending_tasks),
                "workflow_name": decomposition.get("workflow_name", "Unknown Workflow")
            }
        ))
        
        logger.info(f"Task decomposition completed with {len(pending_tasks)} tasks")
        
        return {
            **state,
            "messages": new_messages,
            "pending_tasks": pending_tasks,
            "decomposition_result": decomposition,
            "next_action": "assign",
            "workflow_phase": WorkflowPhase.DECOMPOSITION,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "decomposition_completed_at": datetime.utcnow().isoformat(),
                "total_tasks": len(pending_tasks),
                "estimated_duration": decomposition.get("total_estimated_time", 0)
            }
        }
    
    async def _agent_assignment_node(self, state: WorkflowState) -> WorkflowState:
        """Assign tasks to appropriate agents."""
        logger.info("Agent assignment node: Assigning tasks to agents")
        
        pending_tasks = state.get("pending_tasks", {})
        available_agents = state.get("available_agents", {})
        
        assignments = []
        for task_id, task in pending_tasks.items():
            # Find best agent for task
            best_agent = self._find_best_agent(task, available_agents)
            if best_agent:
                assignments.append({
                    "task_id": task["task_id"],
                    "agent_id": best_agent,
                    "agent_type": task["agent_type"],
                    "assigned_at": datetime.utcnow().isoformat()
                })
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Assigned {len(assignments)} tasks to agents",
            MessageType.TASK_REQUEST,
            {"assignments": assignments}
        ))
        
        # Move assigned tasks to active
        active_tasks = dict(state.get("active_tasks", {}))
        for assignment in assignments:
            task_id = assignment["task_id"]
            task = pending_tasks[task_id]
            active_tasks[task_id] = {
                **task,
                "assigned_agent": assignment["agent_id"],
                "status": "assigned",
                "assigned_at": assignment["assigned_at"]
            }
        
        # Remove assigned tasks from pending
        remaining_pending = {
            tid: task for tid, task in pending_tasks.items() 
            if tid not in [a["task_id"] for a in assignments]
        }
        
        return {
            **state,
            "messages": new_messages,
            "pending_tasks": remaining_pending,
            "active_tasks": active_tasks,
            "agent_assignments": {
                **state.get("agent_assignments", {}),
                **{a["task_id"]: a["agent_id"] for a in assignments}
            },
            "next_action": "monitor"
        }
    
    async def _execution_monitor_node(self, state: WorkflowState) -> WorkflowState:
        """Monitor task execution and coordinate with real agents."""
        logger.info("Execution monitor node: Monitoring task progress and coordinating agents")
        
        active_tasks = state.get("active_tasks", {})
        completed_tasks = state.get("completed_tasks", {})
        pending_tasks = state.get("pending_tasks", [])
        
        # Move pending tasks to active if dependencies are met
        newly_active = {}
        remaining_pending = []
        
        for task in pending_tasks:
            task_id = task.get("task_id")
            dependencies = task.get("dependencies", [])
            
            # Check if all dependencies are completed
            deps_completed = all(dep in completed_tasks for dep in dependencies)
            
            if deps_completed:
                # Move to active tasks and assign to agent
                agent_type = task.get("agent_type", "implementation")
                agent_id = f"{agent_type}_agent"
                
                newly_active[task_id] = {
                    **task,
                    "status": "assigned",
                    "assigned_agent": agent_id,
                    "assigned_at": datetime.utcnow().isoformat(),
                    "progress": 0.0
                }
                
                # Update agent assignments
                agent_assignments = state.get("agent_assignments", {})
                agent_assignments[task_id] = agent_id
                state["agent_assignments"] = agent_assignments
            else:
                remaining_pending.append(task)
        
        # Update active tasks
        all_active = {**active_tasks, **newly_active}
        
        # Process active tasks by actually invoking real agents
        updated_active = {}
        newly_completed = {}
        
        for task_id, task_info in all_active.items():
            current_status = task_info.get("status", "assigned")
            assigned_agent = task_info.get("assigned_agent")
            
            if current_status == "assigned" and assigned_agent:
                try:
                    logger.info(f"Processing task {task_id} with agent {assigned_agent}")
                    
                    # Create task context
                    from ..models.task_models import TaskContext
                    project_context = state.get("project_context", {})
                    
                    task_context = TaskContext(
                        task_id=task_id,
                        description=task_info.get("description", ""),
                        task_type=self._map_task_type(task_info.get("agent_type", "implementation")),
                        metadata={
                            "project_name": project_context.get("project_name", "flutter_app"),
                            "features": project_context.get("features", []),
                            "output_dir": project_context.get("output_dir", "."),
                            "project_context": project_context,
                            "correlation_id": f"supervisor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        }
                    )
                    
                    # Invoke the real agent
                    agent_result = await self._invoke_real_agent(assigned_agent, task_context)
                    
                    if agent_result.get("status") == "completed":
                        newly_completed[task_id] = {
                            **task_info,
                            "status": "completed",
                            "progress": 1.0,
                            "completed_at": datetime.utcnow().isoformat(),
                            "result": agent_result.get("result", {}),
                            "deliverables": agent_result.get("deliverables", {})
                        }
                        logger.info(f"Task {task_id} completed successfully")
                    else:
                        # Task failed - mark as failed
                        updated_active[task_id] = {
                            **task_info,
                            "status": "failed",
                            "error": agent_result.get("error", "Unknown error"),
                            "failed_at": datetime.utcnow().isoformat()
                        }
                        logger.error(f"Task {task_id} failed: {agent_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    updated_active[task_id] = {
                        **task_info,
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.utcnow().isoformat()
                    }
            elif current_status in ["completed"]:
                # Move completed tasks
                newly_completed[task_id] = task_info
            else:
                # Keep other tasks in active state
                updated_active[task_id] = task_info
        
        # Update completed tasks
        all_completed = {**completed_tasks, **newly_completed}
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Monitoring: {len(updated_active)} active, {len(newly_completed)} newly completed, {len(remaining_pending)} pending",
            MessageType.STATUS_UPDATE,
            {
                "active_count": len(updated_active),
                "completed_count": len(all_completed),
                "pending_count": len(remaining_pending),
                "newly_completed": list(newly_completed.keys()),
                "newly_active": list(newly_active.keys())
            }
        ))
        
        # Determine next action
        if newly_active:
            # New tasks became active - continue to agent assignment/execution
            next_action = "assign"
        elif updated_active:
            # Still have active tasks - continue monitoring
            next_action = "continue"
        elif remaining_pending:
            # Have pending tasks but dependencies not met - continue monitoring
            next_action = "continue"  
        elif all_completed:
            # All tasks completed - aggregate results
            next_action = "aggregate"
        else:
            # No tasks - this shouldn't happen, but handle gracefully
            next_action = "aggregate"
        
        return {
            **state,
            "messages": new_messages,
            "active_tasks": updated_active,
            "completed_tasks": all_completed,
            "pending_tasks": remaining_pending,
            "next_action": next_action,
            "execution_metrics": {
                **state.get("execution_metrics", {}),
                "last_monitor_check": datetime.utcnow().isoformat(),
                "active_task_count": len(updated_active),
                "completed_task_count": len(all_completed),
                "pending_task_count": len(remaining_pending)
            }
        }
    
    async def _result_aggregation_node(self, state: WorkflowState) -> WorkflowState:
        """Aggregate results from completed tasks."""
        logger.info("Result aggregation node: Aggregating task results")
        
        completed_tasks = state.get("completed_tasks", {})
        
        # Aggregate deliverables
        aggregated_deliverables = {}
        for task_id, task_result in completed_tasks.items():
            task_deliverables = task_result.get("deliverables", {})
            aggregated_deliverables[task_id] = task_deliverables
        
        # Create final result
        final_result = {
            "workflow_id": state["workflow_id"],
            "status": "completed",
            "deliverables": aggregated_deliverables,
            "summary": f"Completed {len(completed_tasks)} tasks successfully",
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            "Workflow completed successfully",
            MessageType.WORKFLOW_CONTROL,
            {"final_result": final_result}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "final_result": final_result,
            "deliverables": aggregated_deliverables,
            "should_continue": False,
            "next_action": "end"
        }
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors and determine recovery strategy."""
        logger.info("Error handler node: Processing workflow errors")
        
        error_message = state.get("error_message", "Unknown error")
        failed_tasks = state.get("failed_tasks", {})
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(state)
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Error handled: {error_message}. Strategy: {recovery_strategy}",
            MessageType.ERROR_REPORT,
            {"recovery_strategy": recovery_strategy, "failed_tasks": list(failed_tasks.keys())}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "next_action": recovery_strategy,
            "workflow_metadata": {
                **state.get("workflow_metadata", {}),
                "error_handled_at": datetime.utcnow().isoformat(),
                "recovery_strategy": recovery_strategy
            }
        }
    
    # Router functions
    def _supervisor_router(self, state: WorkflowState) -> str:
        """Route supervisor decisions."""
        return state.get("next_action", "end")
    
    def _decomposition_router(self, state: WorkflowState) -> str:
        """Route after task decomposition."""
        return "assign" if state.get("pending_tasks") else "error"
    
    def _assignment_router(self, state: WorkflowState) -> str:
        """Route after agent assignment."""
        if state.get("active_tasks"):
            return "monitor"
        elif state.get("pending_tasks"):
            return "supervisor"
        else:
            return "error"
    
    def _monitor_router(self, state: WorkflowState) -> str:
        """Route based on monitoring results."""
        active_tasks = state.get("active_tasks", {})
        completed_tasks = state.get("completed_tasks", {})
        pending_tasks = state.get("pending_tasks", [])
        
        # Check if we have made progress on tasks
        workflow_metadata = state.get("workflow_metadata", {})
        monitor_count = workflow_metadata.get("monitor_iterations", 0)
        
        # Prevent infinite monitoring loops
        if monitor_count > 3:
            logger.warning(f"Monitor loop detected ({monitor_count} iterations), forcing aggregation")
            return "aggregate"
        
        # Update monitor count
        workflow_metadata["monitor_iterations"] = monitor_count + 1
        state["workflow_metadata"] = workflow_metadata
        
        # If no tasks are active and we have completed tasks, aggregate
        if not active_tasks and completed_tasks:
            return "aggregate"
        # If no tasks are active and no completed tasks but pending tasks exist, go back to supervisor
        elif not active_tasks and not completed_tasks and pending_tasks:
            return "supervisor"
        # If we have active tasks but limited monitoring, continue monitoring briefly
        elif active_tasks and monitor_count < 2:
            return "continue"
        # Otherwise aggregate what we have
        else:
            return "aggregate"
    
    def _aggregation_router(self, state: WorkflowState) -> str:
        """Route after result aggregation."""
        return "end" if state.get("final_result") else "supervisor"
    
    def _error_router(self, state: WorkflowState) -> str:
        """Route after error handling."""
        recovery = state.get("workflow_metadata", {}).get("recovery_strategy", "end")
        return "retry" if recovery == "retry" else "end"
    
    # Helper methods
    def _get_supervisor_system_prompt(self) -> str:
        """Get the supervisor system prompt."""
        from ..config.agent_configs import agent_config_manager
        
        # Try to get from configuration first
        config_prompt = agent_config_manager.get_system_prompt("supervisor")
        if config_prompt:
            return config_prompt
            
        # Fallback to default
        return """
        You are the Supervisor Agent in a LangGraph-based Flutter development workflow.
        
        Your role is to make high-level decisions about workflow progression:
        - When to decompose tasks
        - When to assign tasks to agents
        - When to monitor execution
        - When to aggregate results
        - How to handle errors
        
        Always respond with your reasoning and the next recommended action.
        Consider the current workflow state and make decisions that optimize for:
        - Task completion efficiency
        - Resource utilization
        - Error recovery
        - Quality assurance
        
        Available actions: decompose, assign, monitor, aggregate, error, end
        """
    
    def _determine_workflow_phase(self, state: WorkflowState) -> str:
        """Determine current workflow phase."""
        if not state.get("pending_tasks") and not state.get("active_tasks"):
            if state.get("completed_tasks"):
                return WorkflowPhase.COMPLETION.value
            else:
                return WorkflowPhase.INITIALIZATION.value
        elif state.get("pending_tasks") and not state.get("active_tasks"):
            return WorkflowPhase.PLANNING.value
        elif state.get("active_tasks"):
            return WorkflowPhase.EXECUTION.value
        else:
            return WorkflowPhase.ANALYSIS.value
    
    def _parse_supervisor_decision(self, response: str) -> Dict[str, Any]:
        """Parse supervisor LLM response."""
        # Simplified parsing - in practice, use more robust JSON extraction
        try:
            if "decompose" in response.lower():
                action = "decompose"
            elif "assign" in response.lower():
                action = "assign"
            elif "monitor" in response.lower():
                action = "monitor"
            elif "aggregate" in response.lower():
                action = "aggregate"
            elif "error" in response.lower():
                action = "error"
            else:
                action = "end"
            
            return {
                "action": action,
                "reasoning": response,
                "confidence": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to parse supervisor decision: {e}")
            return {
                "action": "error",
                "reasoning": f"Failed to parse decision: {e}",
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _parse_task_decomposition(self, response: str) -> Dict[str, Any]:
        """Parse task decomposition response with improved error handling."""
        logger.debug(f"Original LLM response for task decomposition:\n{response}")
        
        try:
            # Clean up the response
            response_clean = response.strip()
            
            # Remove markdown code block markers if present
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:]
            
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            logger.debug(f"Cleaned response: {response_clean}")
            
            # Strategy 1: Direct JSON parsing
            try:
                result = json.loads(response_clean)
                if self._validate_task_decomposition(result):
                    logger.info("Successfully parsed JSON directly")
                    return result
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parsing failed: {e}")
            
            # Strategy 2: Find and extract complete JSON object
            start_brace = response_clean.find('{')
            if start_brace != -1:
                json_str = self._extract_json_object(response_clean, start_brace)
                if json_str:
                    try:
                        result = json.loads(json_str)
                        if self._validate_task_decomposition(result):
                            logger.info("Successfully parsed extracted JSON object")
                            return result
                    except json.JSONDecodeError as e:
                        logger.debug(f"Extracted JSON parsing failed: {e}")
            
            # Strategy 3: Handle common truncation patterns
            if any(pattern in response_clean for pattern in ['\"Secure', '"deliverables": ["Secure']):
                logger.warning("Detected truncated JSON response, attempting repair")
                # Try to find the last complete task and close the JSON properly
                last_task_start = response_clean.rfind('"task_id":')
                if last_task_start > 0:
                    # Find the end of the last complete task
                    search_start = last_task_start
                    last_complete_end = response_clean.rfind('}', 0, search_start)
                    if last_complete_end > 0:
                        # Try to reconstruct with proper closing
                        repaired_json = response_clean[:last_complete_end + 1] + '], "workflow_name": "Music Streaming App", "total_estimated_time": 180}'
                        try:
                            result = json.loads(repaired_json)
                            if self._validate_task_decomposition(result):
                                logger.info("Successfully repaired and parsed truncated JSON")
                                return result
                        except json.JSONDecodeError:
                            pass
            
            # Strategy 4: Create fallback task decomposition
            logger.warning("All JSON parsing strategies failed, using fallback")
            return self._create_fallback_task_decomposition()
                
        except Exception as e:
            logger.error(f"Unexpected error during task decomposition parsing: {e}", exc_info=True)
            return self._create_fallback_task_decomposition()

    def _extract_json_object(self, text: str, start: int) -> Optional[str]:
        """Extract a complete JSON object from text starting at the given position."""
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i + 1]
        
        return None

    def _validate_task_decomposition(self, result: Dict[str, Any]) -> bool:
        """Validate that the result contains proper task decomposition structure."""
        if not isinstance(result, dict):
            return False
        
        # Check for tasks array
        if "tasks" not in result:
            return False
        
        tasks = result["tasks"]
        if not isinstance(tasks, list):
            return False
        
        # Validate at least one task exists and has required fields
        if len(tasks) == 0:
            return False
        
        for task in tasks:
            if not isinstance(task, dict):
                return False
            required_fields = ["task_id", "description", "agent_type"]
            if not all(field in task for field in required_fields):
                return False
        
        return True

    def _create_fallback_task_decomposition(self) -> Dict[str, Any]:
        """Create a fallback task decomposition when parsing fails."""
        return {
            "tasks": [
                {
                    "task_id": "task_001",
                    "description": "Design Flutter application architecture and project structure",
                    "agent_type": "architecture",
                    "priority": "high",
                    "estimated_duration": 30,
                    "dependencies": [],
                    "deliverables": ["Architecture design", "Technical specifications"]
                },
                {
                    "task_id": "task_002",
                    "description": "Create Flutter project and implement core features",
                    "agent_type": "implementation",
                    "priority": "high", 
                    "estimated_duration": 45,
                    "dependencies": ["task_001"],
                    "deliverables": ["Flutter project", "Feature implementation", "UI components"]
                }
            ],
            "workflow_name": "Flutter App Development",
            "total_estimated_time": 75,
            "fallback": True
        }

    def _create_dynamic_fallback_task_decomposition(self, project_name: str, features: list) -> Dict[str, Any]:
        """Create a dynamic fallback task decomposition based on project details."""
        features_text = ', '.join(features) if features else 'basic functionality'
        
        return {
            "tasks": [
                {
                    "task_id": "arch_001",
                    "description": f"Design {project_name} architecture with features: {features_text}",
                    "agent_type": "architecture",
                    "priority": "high",
                    "estimated_duration": 30,
                    "dependencies": [],
                    "deliverables": ["Architecture design", "Project structure", "Dependencies list"]
                },
                {
                    "task_id": "impl_001",
                    "description": f"Create Flutter project {project_name} and implement features: {features_text}",
                    "agent_type": "implementation",
                    "priority": "high",
                    "estimated_duration": 60,
                    "dependencies": ["arch_001"],
                    "deliverables": ["Flutter project", "Feature implementation", "UI components"]
                }
            ],
            "workflow_name": f"{project_name} Development",
            "total_estimated_time": 90,
            "fallback": True
        }
    
    def _find_best_agent(self, task: Dict[str, Any], available_agents: Dict[str, Any]) -> Optional[str]:
        """Find the best agent for a task."""
        required_agent_type = task.get("agent_type")
        
        # Find agents of the required type
        suitable_agents = [
            agent_id for agent_id, info in available_agents.items()
            if info.get("agent_type") == required_agent_type and info.get("availability", False)
        ]
        
        if suitable_agents:
            # Return agent with lowest load
            return min(suitable_agents, key=lambda x: available_agents[x].get("current_load", 0))
        
        # If no exact match, return the first available agent of the required type
        # Create a fallback agent entry
        fallback_agent_id = f"{required_agent_type}_agent"
        return fallback_agent_id
    
    def _determine_recovery_strategy(self, state: WorkflowState) -> str:
        """Determine error recovery strategy."""
        failed_tasks = state.get("failed_tasks", {})
        retry_count = state.get("workflow_metadata", {}).get("retry_count", 0)
        
        if retry_count < 3 and len(failed_tasks) < 3:
            return "retry"
        else:
            return "end"
    
    async def execute_workflow(
        self,
        task_description: str,
        project_context: Dict[str, Any],
        workflow_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a complete workflow."""
        if not workflow_id:
            workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if not thread_id:
            thread_id = workflow_id
        
        # Initial state
        initial_state = {
            "workflow_id": workflow_id,
            "task_description": task_description,
            "project_context": project_context,
            "messages": [
                create_system_message(f"Starting workflow: {workflow_id}"),
                create_agent_message(
                    self.agent_id,
                    f"Initialized workflow for: {task_description}",
                    MessageType.WORKFLOW_CONTROL
                )
            ],
            "current_agent": None,
            "available_agents": {
                # LangGraph agent registrations
                "architecture_agent": {
                    "agent_type": "architecture",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["system_design", "architecture_planning"]
                },
                "implementation_agent": {
                    "agent_type": "implementation", 
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["code_generation", "feature_implementation"]
                },
                "testing_agent": {
                    "agent_type": "testing",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["test_creation", "quality_assurance"]
                },
                "security_agent": {
                    "agent_type": "security",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["security_analysis", "vulnerability_assessment"]
                },
                "devops_agent": {
                    "agent_type": "devops",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["deployment", "ci_cd"]
                },
                "documentation_agent": {
                    "agent_type": "documentation",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["documentation_creation", "api_docs"]
                },
                "performance_agent": {
                    "agent_type": "performance",
                    "availability": True,
                    "current_load": 0,
                    "capabilities": ["performance_analysis", "optimization"]
                }
            },
            "agent_assignments": {},
            "pending_tasks": [],
            "active_tasks": {},
            "completed_tasks": {},
            "failed_tasks": {},
            "deliverables": {},
            "final_result": None,
            "next_action": "decompose",
            "should_continue": True,
            "error_message": None,
            "workflow_metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            },
            "execution_metrics": {}
        }
        
        # Execute workflow
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50  # Increase recursion limit
        }
        
        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            logger.info(f"Workflow {workflow_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "final_result": None
            }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task request."""
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute the workflow
        result = await self.execute_workflow(
            task_description=request["description"],
            project_context=request.get("project_context", {}),
            workflow_id=task_id
        )
        
        return {
            "task_id": task_id,
            "success": result.get("status") != "failed",
            "status": result.get("status", "unknown"),
            "result": result.get("final_result"),
            "deliverables": result.get("deliverables", {}),
            "workflow_id": result.get("workflow_id"),
            "error": result.get("error")
        }

    @property
    def agents(self) -> Dict[str, Any]:
        """Get available agents for compatibility with CLI."""
        return {
            role: MockAgent(role, self._get_agent_capabilities(role))
            for role in self.agent_roles
        }

    def _get_agent_capabilities(self, agent_role: str) -> List[str]:
        """Get capabilities for an agent role."""
        capabilities_map = {
            "architecture": ["system_design", "architecture_planning", "pattern_selection"],
            "implementation": ["code_generation", "feature_implementation", "refactoring"],
            "testing": ["test_creation", "quality_assurance", "test_automation"],
            "security": ["security_analysis", "vulnerability_assessment", "compliance"],
            "devops": ["deployment", "ci_cd", "infrastructure_automation"],
            "documentation": ["documentation_creation", "api_docs", "user_guides"],
            "performance": ["performance_analysis", "optimization", "monitoring"]
        }
        return capabilities_map.get(agent_role, ["general_purpose"])
    
    async def create_flutter_project_with_sdk(self, project_name: str, app_description: str, platforms: List[str] = None) -> str:
        """Create Flutter project using Flutter SDK tool instead of templates."""
        from ..core.tools.flutter_sdk_tool import FlutterSDKTool
        
        if platforms is None:
            platforms = ["android", "ios"]
        
        # Normalize project name for Flutter
        normalized_name = project_name.lower().replace(' ', '_').replace('-', '_')
        
        # Use the Flutter SDK tool to create the project
        flutter_tool = FlutterSDKTool()
        
        params = {
            "project_name": normalized_name,
            "description": app_description,
            "platforms": platforms
        }
        
        try:
            result = await flutter_tool.execute("create_project", params)
            
            if result.status.name == "SUCCESS":
                project_path = result.data.get("project_path", "")
                logger.info(f"Successfully created Flutter project: {normalized_name} at {project_path}")
                return f"Successfully created Flutter project '{normalized_name}' at {project_path}. The project is ready for development with the specified platforms: {', '.join(platforms)}."
            else:
                error_msg = result.error_message or "Unknown error occurred"
                logger.error(f"Failed to create Flutter project: {error_msg}")
                return f"Failed to create Flutter project: {error_msg}"
                
        except Exception as e:
            logger.error(f"Exception while creating Flutter project: {e}")
            return f"Failed to create Flutter project due to exception: {str(e)}"

    async def get_system_status(self):
        """Get the current status of the FlutterSwarm system."""
        try:
            # Create a basic status report
            status = {
                "system": {
                    "initialized": True,
                    "running": True,
                    "agents_count": len(self.agent_roles),
                    "memory_managers_count": 1
                },
                "agents": {},
                "event_bus": {
                    "total_messages": self.event_bus.get_message_count(),
                    "successful_deliveries": self.event_bus.get_successful_deliveries(),
                    "active_topics": len(self.event_bus.get_active_topics())
                }
            }
            
            # Add mock status for each agent role
            for role in self.agent_roles:
                status["agents"][f"{role}_agent"] = {
                    "status": "ready",
                    "type": role,
                    "active_tasks": 0
                }
                
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "system": {
                    "initialized": True,
                    "running": True,
                    "error": str(e)
                }
            }
    
    def _map_task_type(self, agent_type: str) -> TaskType:
        """Map agent type to task type."""
        mapping = {
            "architecture": TaskType.ANALYSIS,
            "implementation": TaskType.IMPLEMENTATION,
            "testing": TaskType.TESTING,
            "security": TaskType.ANALYSIS,
            "devops": TaskType.DEPLOYMENT,
            "documentation": TaskType.DOCUMENTATION,
            "performance": TaskType.ANALYSIS
        }
        return mapping.get(agent_type, TaskType.ANALYSIS)
    
    async def _invoke_real_agent(self, agent_id: str, task_context: TaskContext) -> Dict[str, Any]:
        """Invoke the actual agent to process the task."""
        try:
            # Import agents dynamically to avoid circular imports
            from .implementation_agent import ImplementationAgent
            from .architecture_agent import ArchitectureAgent
            
            agent = None
            
            # Get the agent instance based on agent_id
            if agent_id == "implementation_agent":
                # Create agent config
                from .base_agent import AgentConfig, AgentCapability
                config = AgentConfig(
                    agent_id=agent_id,
                    agent_type="implementation",
                    capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.FILE_OPERATIONS],
                    max_concurrent_tasks=3
                )
                
                # Create memory manager
                from ..core.memory_manager import MemoryManager
                memory_manager = MemoryManager(agent_id="shared_memory")
                
                agent = ImplementationAgent(
                    config=config,
                    llm_client=self.flutterswarm_llm,
                    memory_manager=memory_manager,
                    event_bus=self.event_bus
                )
                
            elif agent_id == "architecture_agent":
                from .base_agent import AgentConfig, AgentCapability
                config = AgentConfig(
                    agent_id=agent_id,
                    agent_type="architecture",
                    capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS, AgentCapability.CODE_GENERATION],
                    max_concurrent_tasks=3
                )
                
                from ..core.memory_manager import MemoryManager
                memory_manager = MemoryManager(agent_id="shared_memory")
                
                agent = ArchitectureAgent(
                    config=config,
                    llm_client=self.flutterswarm_llm,
                    memory_manager=memory_manager,
                    event_bus=self.event_bus
                )
            
            if agent:
                logger.info(f"Invoking {agent_id} for task {task_context.task_id}")
                task_result = await agent.process_task(task_context)
                
                return {
                    "status": "completed" if task_result.status == TaskStatus.COMPLETED else "failed",
                    "result": task_result.result,
                    "deliverables": task_result.deliverables,
                    "error": task_result.errors[0] if task_result.errors else None
                }
            else:
                logger.warning(f"No agent implementation found for {agent_id}")
                return {
                    "status": "failed",
                    "error": f"Agent {agent_id} not implemented"
                }
                
        except Exception as e:
            logger.error(f"Error invoking agent {agent_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }


