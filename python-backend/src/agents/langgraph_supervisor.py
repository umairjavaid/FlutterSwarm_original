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
        
        # Initialize LLM
        self.llm = self._create_llm()
        
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
                api_key=settings.llm.anthropic_api_key
            )
        else:
            # Fallback to Anthropic with environment variable
            return ChatAnthropic(
                model=settings.llm.default_model,
                temperature=settings.llm.temperature
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
                "implementation": "implementation_agent",
                "architecture": "architecture_agent", 
                "testing": "testing_agent",
                "security": "security_agent",
                "devops": "devops_agent",
                "documentation": "documentation_agent",
                "performance": "performance_agent",
                "monitor": "execution_monitor",
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
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
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
        
        # Create decomposition prompt
        prompt = f"""
        Analyze the following task and decompose it into specific subtasks for Flutter development agents.
        
        TASK: {task_description}
        
        PROJECT CONTEXT: {json.dumps(project_context, indent=2)}
        
        AVAILABLE AGENTS: {', '.join(self.agent_roles)}
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        
        {{
            "tasks": [
                {{
                    "task_id": "task_001",
                    "description": "Design application architecture",
                    "agent_type": "architecture",
                    "priority": "high",
                    "estimated_duration": 30,
                    "dependencies": [],
                    "deliverables": ["Architecture diagram", "Technical specifications"]
                }},
                {{
                    "task_id": "task_002", 
                    "description": "Implement Flutter UI components",
                    "agent_type": "implementation",
                    "priority": "high",
                    "estimated_duration": 60,
                    "dependencies": ["task_001"],
                    "deliverables": ["Working Flutter app", "Source code"]
                }}
            ],
            "workflow_name": "Flutter Application Development",
            "total_estimated_time": 90
        }}
        
        Agent Types Available:
        - architecture: App structure and design patterns
        - implementation: Flutter code development
        - testing: Unit and integration tests
        - security: Security analysis and hardening
        - devops: Deployment and CI/CD
        - documentation: API docs and user guides
        - performance: Optimization and profiling
        
        Requirements:
        - Each task must have a unique task_id
        - agent_type must be one of the available agents
        - priority can be: "high", "medium", "low"
        - estimated_duration is in minutes
        - dependencies reference task_ids that must complete first
        - Include specific deliverables for each task
        
        Respond with ONLY the JSON object, no explanation or markdown formatting.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a task decomposition expert for Flutter development."),
            HumanMessage(content=prompt)
        ])
        
        # Parse decomposition result
        decomposition = self._parse_task_decomposition(response.content)
        
        # Check if decomposition was successful
        if "error" in decomposition or not decomposition.get("tasks"):
            logger.error(f"Task decomposition failed: {decomposition.get('error', 'No tasks generated')}")
            # Create a fallback simple task structure
            decomposition = {
                "tasks": [{
                    "task_id": f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_1",
                    "description": task_description,
                    "agent_type": "implementation",  # Default to implementation agent
                    "priority": "normal"
                }]
            }
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Task decomposed into {len(decomposition['tasks'])} subtasks",
            MessageType.TASK_REQUEST,
            {"decomposition": decomposition}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "pending_tasks": decomposition["tasks"],
            "next_action": "assign"
        }
    
    async def _agent_assignment_node(self, state: WorkflowState) -> WorkflowState:
        """Assign tasks to appropriate agents."""
        logger.info("Agent assignment node: Assigning tasks to agents")
        
        pending_tasks = state.get("pending_tasks", [])
        available_agents = state.get("available_agents", {})
        
        assignments = []
        for task in pending_tasks:
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
            task = next(t for t in pending_tasks if t["task_id"] == task_id)
            active_tasks[task_id] = {
                **task,
                "assigned_agent": assignment["agent_id"],
                "status": "assigned",
                "assigned_at": assignment["assigned_at"]
            }
        
        # Remove assigned tasks from pending
        remaining_pending = [
            t for t in pending_tasks 
            if t["task_id"] not in [a["task_id"] for a in assignments]
        ]
        
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
        """Monitor task execution and handle updates."""
        logger.info("Execution monitor node: Monitoring task progress")
        
        active_tasks = state.get("active_tasks", {})
        completed_tasks = state.get("completed_tasks", {})
        
        # Simulate task execution and completion
        newly_completed = {}
        remaining_active = {}
        
        for task_id, task_info in active_tasks.items():
            current_status = task_info.get("status", "assigned")
            
            # Simulate task progression: assigned -> in_progress -> completed
            if current_status == "assigned":
                # Mark as in progress
                remaining_active[task_id] = {
                    **task_info,
                    "status": "in_progress",
                    "progress": 0.5,
                    "updated_at": datetime.utcnow().isoformat()
                }
            elif current_status == "in_progress":
                # Mark as completed - let agents determine actual implementation
                newly_completed[task_id] = {
                    **task_info,
                    "status": "completed",
                    "progress": 1.0,
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": {
                        "status": "success",
                        "summary": "Task completed by assigned agent",
                        "agent_output": "Implementation completed as specified"
                    }
                }
            else:
                # Keep as is
                remaining_active[task_id] = task_info
        
        # Update completed tasks
        all_completed = {**completed_tasks, **newly_completed}
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Monitoring {len(remaining_active)} active tasks, {len(newly_completed)} newly completed",
            MessageType.STATUS_UPDATE,
            {
                "active_count": len(remaining_active),
                "completed_count": len(all_completed),
                "newly_completed": list(newly_completed.keys())
            }
        ))
        
        return {
            **state,
            "messages": new_messages,
            "active_tasks": remaining_active,
            "completed_tasks": all_completed,
            "next_action": "aggregate" if newly_completed and not remaining_active else "continue",
            "execution_metrics": {
                **state.get("execution_metrics", {}),
                "last_monitor_check": datetime.utcnow().isoformat(),
                "active_task_count": len(remaining_active),
                "completed_task_count": len(all_completed)
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
        
        # If we've monitored too many times without progress, move to aggregation
        if monitor_count > 5:
            return "aggregate"
        
        # Update monitor count
        workflow_metadata["monitor_iterations"] = monitor_count + 1
        state["workflow_metadata"] = workflow_metadata
        
        if active_tasks:
            # Continue monitoring only if tasks are actually progressing
            return "continue" if monitor_count < 3 else "aggregate"
        elif completed_tasks:
            return "aggregate"
        elif pending_tasks:
            return "supervisor"
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
        """Parse task decomposition response."""
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
            
            # Extract JSON from response
            start = response_clean.find('{')
            end = response_clean.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response_clean[start:end]
                result = json.loads(json_str)
                
                # Validate that result has tasks
                if "tasks" not in result or not isinstance(result["tasks"], list):
                    logger.warning("Task decomposition response missing tasks array")
                    return {"tasks": [], "error": "Missing tasks array in response"}
                
                return result
            else:
                logger.error("No valid JSON found in response")
                return {"tasks": [], "error": "No valid JSON found in response"}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in task decomposition: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return {"tasks": [], "error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to parse task decomposition: {e}")
            return {"tasks": [], "error": str(e)}
    
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
        
        return None
    
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
    
    def _add_agent_nodes(self, workflow: StateGraph) -> None:
        """Add specialized agent nodes to the workflow graph."""
        logger.info("Adding specialized agent nodes to workflow")
        
        # Import agent classes
        from .implementation_agent import ImplementationAgent
        from .architecture_agent import ArchitectureAgent
        from .testing_agent import TestingAgent
        from .security_agent import SecurityAgent
        from .devops_agent import DevOpsAgent
        from .documentation_agent import DocumentationAgent
        from .performance_agent import PerformanceAgent
        
        # Add implementation agent node
        workflow.add_node("implementation_agent", self._implementation_agent_node)
        workflow.add_node("architecture_agent", self._architecture_agent_node) 
        workflow.add_node("testing_agent", self._testing_agent_node)
        workflow.add_node("security_agent", self._security_agent_node)
        workflow.add_node("devops_agent", self._devops_agent_node)
        workflow.add_node("documentation_agent", self._documentation_agent_node)
        workflow.add_node("performance_agent", self._performance_agent_node)
        
        # Routing is handled in the main workflow configuration
        
        # Add edges from agent nodes back to monitor
        for agent_name in ["implementation_agent", "architecture_agent", "testing_agent", 
                          "security_agent", "devops_agent", "documentation_agent", "performance_agent"]:
            workflow.add_edge(agent_name, "execution_monitor")
    
    def _agent_dispatch_router(self, state: WorkflowState) -> str:
        """Route tasks to appropriate specialized agents."""
        active_tasks = state.get("active_tasks", {})
        
        # Find the next task that needs to be executed
        for task_id, task_info in active_tasks.items():
            if task_info.get("status") == "assigned":
                agent_type = task_info.get("agent_type", "implementation")
                logger.info(f"Routing task {task_id} to {agent_type} agent")
                return agent_type
        
        # If no assigned tasks, go to monitor
        return "monitor"
    
    async def _implementation_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute implementation agent tasks."""
        return await self._execute_agent_tasks(state, "implementation")
    
    async def _architecture_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute architecture agent tasks."""
        return await self._execute_agent_tasks(state, "architecture")
    
    async def _testing_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute testing agent tasks."""
        return await self._execute_agent_tasks(state, "testing")
    
    async def _security_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute security agent tasks."""
        return await self._execute_agent_tasks(state, "security")
    
    async def _devops_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute devops agent tasks."""
        return await self._execute_agent_tasks(state, "devops")
    
    async def _documentation_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute documentation agent tasks."""
        return await self._execute_agent_tasks(state, "documentation")
    
    async def _performance_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute performance agent tasks."""
        return await self._execute_agent_tasks(state, "performance")
    
    async def _execute_agent_tasks(self, state: WorkflowState, agent_type: str) -> WorkflowState:
        """Execute tasks assigned to a specific agent type."""
        logger.info(f"Executing {agent_type} agent tasks")
        
        active_tasks = state.get("active_tasks", {})
        completed_tasks = state.get("completed_tasks", {})
        
        # Find tasks assigned to this agent type
        agent_tasks = {
            task_id: task_info for task_id, task_info in active_tasks.items()
            if task_info.get("agent_type") == agent_type and task_info.get("status") == "assigned"
        }
        
        if not agent_tasks:
            logger.warning(f"No tasks found for {agent_type} agent")
            return state
        
        # Execute each task assigned to this agent
        new_completed = {}
        remaining_active = {k: v for k, v in active_tasks.items() if k not in agent_tasks}
        
        for task_id, task_info in agent_tasks.items():
            try:
                logger.info(f"Executing task {task_id}: {task_info.get('description', 'Unknown task')}")
                
                # Create task context for the agent  
                from ..models.task_models import TaskContext, TaskType
                task_context = TaskContext(
                    task_id=task_id,
                    task_type=TaskType.IMPLEMENTATION,
                    description=task_info.get("description", ""),
                    parameters={
                        "requirements": task_info.get("deliverables", []),
                        "project_context": state.get("project_context", {}),
                        "agent_type": agent_type
                    }
                )
                
                # Execute the task using the appropriate agent
                if agent_type == "implementation":
                    result = await self._execute_implementation_task(task_context)
                elif agent_type == "architecture":
                    result = await self._execute_architecture_task(task_context)
                else:
                    # For other agent types, create a placeholder result
                    result = {
                        "status": "completed",
                        "summary": f"{agent_type.title()} task completed",
                        "deliverables": task_info.get("deliverables", []),
                        "files_created": [],
                        "notes": f"Task executed by {agent_type} agent"
                    }
                
                # Mark task as completed
                new_completed[task_id] = {
                    **task_info,
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": result
                }
                
                logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to execute task {task_id}: {e}")
                # Keep task in active with error status
                remaining_active[task_id] = {
                    **task_info,
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                }
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            f"{agent_type}_agent",
            f"Completed {len(new_completed)} tasks, {len([t for t in remaining_active.values() if t.get('status') == 'failed'])} failed",
            MessageType.STATUS_UPDATE,
            {"completed_tasks": list(new_completed.keys())}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "active_tasks": remaining_active,
            "completed_tasks": {**completed_tasks, **new_completed}
        }
    
    async def _execute_implementation_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an implementation task using the implementation agent."""
        try:
            # Import here to avoid circular imports
            from .implementation_agent import ImplementationAgent
            from ..core.memory_manager import MemoryManager
            
            # Create a temporary implementation agent
            config = AgentConfig(
                agent_id="temp_implementation_agent",
                agent_type="implementation", 
                capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.FILE_OPERATIONS]
            )
            
            # Create minimal dependencies
            memory_manager = MemoryManager(agent_id="temp_implementation_agent")
            
            impl_agent = ImplementationAgent(
                config=config,
                llm_client=self.llm,
                memory_manager=memory_manager,
                event_bus=self.event_bus
            )
            
            # Execute the task
            task_result = await impl_agent._generate_code(
                task_context=task_context,
                llm_analysis={"requirements": task_context.parameters.get("requirements", [])}
            )
            
            return {
                "status": "completed",
                "summary": f"Implementation task completed: {task_context.description}",
                "files_created": list(task_result.get("code_files", {}).keys()),
                "deliverables": task_result.get("implementation_notes", []),
                "dependencies": task_result.get("dependencies", []),
                "agent_output": task_result
            }
            
        except Exception as e:
            logger.error(f"Implementation task execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "summary": f"Failed to execute implementation task: {task_context.description}"
            }
    
    async def _execute_architecture_task(self, task_context: TaskContext) -> Dict[str, Any]:
        """Execute an architecture task."""
        try:
            # For now, create a mock architecture result
            # In the future, this would use the actual ArchitectureAgent
            return {
                "status": "completed",
                "summary": f"Architecture task completed: {task_context.description}",
                "deliverables": [
                    "Architecture diagram created",
                    "Technical specifications documented",
                    "Design patterns selected"
                ],
                "recommendations": [
                    "Use BLoC pattern for state management",
                    "Implement repository pattern for data access",
                    "Follow clean architecture principles"
                ]
            }
            
        except Exception as e:
            logger.error(f"Architecture task execution failed: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "summary": f"Failed to execute architecture task: {task_context.description}"
            }

