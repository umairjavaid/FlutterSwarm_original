"""
LangGraph Agent Nodes for FlutterSwarm Multi-Agent System.

This module provides LangGraph-compatible node implementations that wrap
the existing specialized agents for seamless integration.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..models.langgraph_models import (
    WorkflowState, AgentRole, MessageType, WorkflowPhase,
    create_agent_message, create_task_context_from_state
)
from ..core.langsmith_integration import LangSmithWorkflowTracer, trace_agent_task
from ..config import get_logger

# Import all specialized agents
from .architecture_agent import ArchitectureAgent
from .implementation_agent import ImplementationAgent
from .testing_agent import TestingAgent
from .security_agent import SecurityAgent
from .devops_agent import DevOpsAgent
from .documentation_agent import DocumentationAgent
from .performance_agent import PerformanceAgent

logger = get_logger("langgraph_agent_nodes")


class LangGraphAgentNode:
    """
    Base class for LangGraph-compatible agent nodes.
    
    This class wraps existing specialized agents to work within
    the LangGraph workflow framework.
    """
    
    def __init__(
        self,
        agent_class: Type,
        agent_config: Any,
        llm_client: Any,
        memory_manager: 'MemoryManager',
        event_bus: Any,
        tracer: Optional[LangSmithWorkflowTracer] = None
    ):
        """Initialize the LangGraph agent node."""
        self.agent = agent_class(agent_config, llm_client, memory_manager, event_bus)
        self.tracer = tracer or LangSmithWorkflowTracer()
        self.agent_type = agent_config.agent_type
        
        logger.info(f"LangGraph node created for {self.agent_type} agent")
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the agent node within the LangGraph workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        return await self._execute_agent_node(state)
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return f"{self.agent_type}_node"
    
    async def _execute_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the specialized agent and update workflow state."""
        try:
            # Get tasks assigned to this agent
            assigned_tasks = self._get_assigned_tasks(state)
            
            if not assigned_tasks:
                logger.info(f"No tasks assigned to {self.agent_type} agent")
                return state
            
            # Process each assigned task
            results = {}
            new_messages = list(state.get("messages", []))
            
            for task_id, task_data in assigned_tasks.items():
                try:
                    # Create task context from workflow state
                    task_context = create_task_context_from_state(state, task_data)
                    
                    # Execute agent task with tracing
                    result = await trace_agent_task(
                        task_context,
                        agent_id=self.agent.agent_id,
                        task_id=task_id,
                        task_type=task_data.get("task_type", "unknown"),
                        func=self.agent.process_task,
                    )
                    
                    # Store result
                    results[task_id] = result
                    
                    # Add completion message
                    new_messages.append(create_agent_message(
                        self.agent.agent_id,
                        f"Task {task_id} completed: {result.status}",
                        MessageType.TASK_RESPONSE,
                        {
                            "task_id": task_id,
                            "result": result.to_dict(),
                            "deliverables": result.deliverables
                        }
                    ))
                    
                    # Move task from active to completed
                    if result.is_successful():
                        self._move_task_to_completed(state, task_id, result)
                    else:
                        self._move_task_to_failed(state, task_id, result)
                    
                except Exception as e:
                    logger.error(f"Task {task_id} failed in {self.agent_type} agent: {e}")
                    
                    # Add error message
                    new_messages.append(create_agent_message(
                        self.agent.agent_id,
                        f"Task {task_id} failed: {str(e)}",
                        MessageType.ERROR_REPORT,
                        {"task_id": task_id, "error": str(e)}
                    ))
                    
                    # Move to failed tasks
                    self._move_task_to_failed(state, task_id, None, str(e))
            
            # Update workflow state
            updated_state = {
                **state,
                "messages": new_messages,
                "current_agent": self.agent.agent_id,
                "execution_metrics": {
                    **state.get("execution_metrics", {}),
                    f"{self.agent_type}_tasks_processed": len(results),
                    f"{self.agent_type}_last_execution": datetime.utcnow().isoformat()
                }
            }
            
            # Update agent availability
            self._update_agent_availability(updated_state)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_type} agent node: {e}")
            
            # Add error to messages
            error_messages = list(state.get("messages", []))
            error_messages.append(create_agent_message(
                self.agent.agent_id,
                f"Agent node error: {str(e)}",
                MessageType.ERROR_REPORT,
                {"error": str(e), "agent_type": self.agent_type}
            ))
            
            return {
                **state,
                "messages": error_messages,
                "error_message": f"{self.agent_type} agent error: {str(e)}"
            }
    
    def _get_assigned_tasks(self, state: WorkflowState) -> Dict[str, Dict[str, Any]]:
        """Get tasks assigned to this agent."""
        active_tasks = state.get("active_tasks", {})
        agent_assignments = state.get("agent_assignments", {})
        
        assigned_tasks = {}
        for task_id, agent_id in agent_assignments.items():
            if (agent_id == self.agent.agent_id and 
                task_id in active_tasks and 
                active_tasks[task_id].get("status") in ["assigned", "in_progress"]):
                assigned_tasks[task_id] = active_tasks[task_id]
        
        return assigned_tasks
    
    def _move_task_to_completed(
        self, 
        state: WorkflowState, 
        task_id: str, 
        result: Any
    ) -> None:
        """Move task from active to completed."""
        active_tasks = dict(state.get("active_tasks", {}))
        completed_tasks = dict(state.get("completed_tasks", {}))
        
        if task_id in active_tasks:
            task_data = active_tasks[task_id]
            task_data.update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": result.to_dict() if result else None,
                "deliverables": result.deliverables if result else {}
            })
            
            completed_tasks[task_id] = task_data
            del active_tasks[task_id]
            
            state["active_tasks"] = active_tasks
            state["completed_tasks"] = completed_tasks
    
    def _move_task_to_failed(
        self, 
        state: WorkflowState, 
        task_id: str, 
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """Move task from active to failed."""
        active_tasks = dict(state.get("active_tasks", {}))
        failed_tasks = dict(state.get("failed_tasks", {}))
        
        if task_id in active_tasks:
            task_data = active_tasks[task_id]
            task_data.update({
                "status": "failed",
                "failed_at": datetime.utcnow().isoformat(),
                "error": error or (result.error if result else "Unknown error"),
                "result": result.to_dict() if result else None
            })
            
            failed_tasks[task_id] = task_data
            del active_tasks[task_id]
            
            state["active_tasks"] = active_tasks
            state["failed_tasks"] = failed_tasks
    
    def _update_agent_availability(self, state: WorkflowState) -> None:
        """Update agent availability in workflow state."""
        available_agents = dict(state.get("available_agents", {}))
        
        if self.agent.agent_id in available_agents:
            agent_info = available_agents[self.agent.agent_id]
            agent_info["current_load"] = len(self._get_assigned_tasks(state))
            agent_info["last_activity"] = datetime.utcnow().isoformat()
            
            available_agents[self.agent.agent_id] = agent_info
            state["available_agents"] = available_agents


# Specialized agent node implementations
class ArchitectureAgentNode(LangGraphAgentNode):
    """LangGraph node for Architecture Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            ArchitectureAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class ImplementationAgentNode(LangGraphAgentNode):
    """LangGraph node for Implementation Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            ImplementationAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class TestingAgentNode(LangGraphAgentNode):
    """LangGraph node for Testing Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            TestingAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class SecurityAgentNode(LangGraphAgentNode):
    """LangGraph node for Security Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            SecurityAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class DevOpsAgentNode(LangGraphAgentNode):
    """LangGraph node for DevOps Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            DevOpsAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class DocumentationAgentNode(LangGraphAgentNode):
    """LangGraph node for Documentation Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            DocumentationAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class PerformanceAgentNode(LangGraphAgentNode):
    """LangGraph node for Performance Agent."""
    
    def __init__(self, agent_config, llm_client, memory_manager, event_bus, tracer=None):
        super().__init__(
            PerformanceAgent, agent_config, llm_client, memory_manager, event_bus, tracer
        )


class AgentNodeFactory:
    """Factory for creating agent nodes."""
    
    NODE_CLASSES = {
        AgentRole.ARCHITECTURE: ArchitectureAgentNode,
        AgentRole.IMPLEMENTATION: ImplementationAgentNode,
        AgentRole.TESTING: TestingAgentNode,
        AgentRole.SECURITY: SecurityAgentNode,
        AgentRole.DEVOPS: DevOpsAgentNode,
        AgentRole.DOCUMENTATION: DocumentationAgentNode,
    }
    
    @classmethod
    def create_node(
        cls,
        agent_role: AgentRole,
        agent_config: Any,
        llm_client: Any,
        memory_manager: Any,
        event_bus: Any,
        tracer: Optional[LangSmithWorkflowTracer] = None
    ) -> LangGraphAgentNode:
        """Create an agent node for the specified role."""
        if agent_role not in cls.NODE_CLASSES:
            raise ValueError(f"Unsupported agent role: {agent_role}")
        
        node_class = cls.NODE_CLASSES[agent_role]
        return node_class(agent_config, llm_client, memory_manager, event_bus, tracer)
    
    @classmethod
    def create_all_nodes(
        cls,
        agent_configs: Dict[AgentRole, Any],
        llm_client: Any,
        memory_manager: Any,
        event_bus: Any,
        tracer: Optional[LangSmithWorkflowTracer] = None
    ) -> Dict[str, LangGraphAgentNode]:
        """Create all agent nodes."""
        nodes = {}
        
        for role, config in agent_configs.items():
            if role in cls.NODE_CLASSES:
                node = cls.create_node(role, config, llm_client, memory_manager, event_bus, tracer)
                nodes[f"{role.value}_node"] = node
        
        return nodes


def create_conditional_agent_router(available_nodes: Dict[str, LangGraphAgentNode]):
    """Create a conditional router for agent nodes."""
    
    def router(state: WorkflowState) -> str:
        """Route to appropriate agent based on workflow state."""
        try:
            # Get next agent assignment from state
            next_agent = state.get("next_action")
            
            if next_agent and f"{next_agent}_node" in available_nodes:
                return f"{next_agent}_node"
            
            # Determine next agent based on pending tasks
            pending_tasks = state.get("pending_tasks", [])
            active_tasks = state.get("active_tasks", {})
            
            if pending_tasks:
                # Route to supervisor for task assignment
                return "supervisor"
            elif active_tasks:
                # Find agent with assigned tasks
                agent_assignments = state.get("agent_assignments", {})
                for task_id, agent_id in agent_assignments.items():
                    if task_id in active_tasks:
                        agent_type = active_tasks[task_id].get("agent_type")
                        if agent_type and f"{agent_type}_node" in available_nodes:
                            return f"{agent_type}_node"
            
            # Default to supervisor
            return "supervisor"
            
        except Exception as e:
            logger.error(f"Error in agent router: {e}")
            return "supervisor"
    
    return router
