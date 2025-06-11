"""
LangGraph Supervisor Agent for FlutterSwarm Multi-Agent System.

This module implements the supervisor pattern using LangGraph's StateGraph
for centralized orchestration and decision-making.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..models.langgraph_models import WorkflowState, AgentRole, WorkflowPhase, MessageType
from ..models.langgraph_models import create_agent_message, create_system_message
from ..core.langgraph_checkpointer import create_checkpointer
from ..config import get_logger, settings

logger = get_logger("langgraph_supervisor")


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
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("LangGraph Supervisor Agent initialized")
    
    def _create_llm(self):
        """Create LLM instance based on configuration."""
        if hasattr(settings, 'llm') and hasattr(settings.llm, 'openai_api_key') and settings.llm.openai_api_key:
            return ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                api_key=settings.llm.openai_api_key
            )
        elif hasattr(settings, 'llm') and hasattr(settings.llm, 'anthropic_api_key') and settings.llm.anthropic_api_key:
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.3,
                api_key=settings.llm.anthropic_api_key
            )
        else:
            # Fallback to OpenAI with environment variable
            return ChatOpenAI(
                model="gpt-4",
                temperature=0.3
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
            self._assignment_router,
            {
                "monitor": "execution_monitor",
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
        Analyze the following task and decompose it into specific subtasks for Flutter development agents:
        
        TASK: {task_description}
        
        PROJECT CONTEXT: {json.dumps(project_context, indent=2)}
        
        AVAILABLE AGENTS: {', '.join(self.agent_roles)}
        
        Provide a JSON response with subtasks, each assigned to the most appropriate agent type.
        Include dependencies, priorities, and expected deliverables.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a task decomposition expert for Flutter development."),
            HumanMessage(content=prompt)
        ])
        
        # Parse decomposition result
        decomposition = self._parse_task_decomposition(response.content)
        
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
        
        # Simulate task progress checking
        # In real implementation, this would check actual agent status
        progress_updates = []
        for task_id, task_info in active_tasks.items():
            # Check if task should be completed (simplified logic)
            if task_info.get("status") == "assigned":
                progress_updates.append({
                    "task_id": task_id,
                    "status": "in_progress",
                    "progress": 0.5,
                    "updated_at": datetime.utcnow().isoformat()
                })
        
        # Update state
        new_messages = list(state.get("messages", []))
        new_messages.append(create_agent_message(
            self.agent_id,
            f"Monitoring {len(active_tasks)} active tasks",
            MessageType.STATUS_UPDATE,
            {"progress_updates": progress_updates}
        ))
        
        return {
            **state,
            "messages": new_messages,
            "execution_metrics": {
                **state.get("execution_metrics", {}),
                "last_monitor_check": datetime.utcnow().isoformat(),
                "active_task_count": len(active_tasks)
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
        
        if active_tasks:
            return "continue"
        elif completed_tasks:
            return "aggregate"
        else:
            return "supervisor"
    
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
        # Simplified parsing - implement robust JSON extraction
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end] if start != -1 and end > start else '{}'
            
            result = json.loads(json_str)
            return result
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
            }
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
