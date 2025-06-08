"""
LangGraph Multi-Agent Orchestration Framework
Implements the hierarchical agent architecture using LangGraph StateGraph.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from .agent_types import (
    AgentType, 
    AgentState, 
    AgentMessage, 
    TaskStatus,
    AgentResponse,
    WorkflowState
)
from .message_broker import MessageBroker

logger = logging.getLogger(__name__)


class LangGraphOrchestrator:
    """
    Main orchestrator that manages the multi-agent workflow using LangGraph.
    Implements the three-tier hierarchical architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.message_broker = MessageBroker()
        
        # Initialize checkpointer for state persistence
        self.checkpointer = SqliteSaver.from_conn_string(
            config.get("checkpoint_db", ":memory:")
        )
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the state graph
        self.workflow = self._build_workflow()
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
    def _initialize_agents(self):
        """Initialize all agent instances."""
        # Dynamic imports to avoid circular dependencies
        from agents.orchestrator_agent import OrchestratorAgent
        from agents.architecture_agent import ArchitectureAgent
        from agents.implementation_agent import ImplementationAgent
        from agents.testing_agent import TestingAgent
        from agents.devops_agent import DevOpsAgent
        
        self.agents = {
            AgentType.ORCHESTRATOR: OrchestratorAgent(self.config),
            AgentType.ARCHITECTURE: ArchitectureAgent(self.config),
            AgentType.IMPLEMENTATION: ImplementationAgent(self.config),
            AgentType.TESTING: TestingAgent(self.config),
            AgentType.DEVOPS: DevOpsAgent(self.config),
        }
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state graph for agent coordination."""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("architecture", self._architecture_node)
        workflow.add_node("implementation", self._implementation_node)
        workflow.add_node("testing", self._testing_node)
        workflow.add_node("devops", self._devops_node)
        workflow.add_node("quality_gate", self._quality_gate_node)
        
        # Define the workflow edges
        workflow.set_entry_point("orchestrator")
        
        # Orchestrator can delegate to any specialist agent
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "architecture": "architecture",
                "implementation": "implementation", 
                "testing": "testing",
                "devops": "devops",
                "quality_gate": "quality_gate",
                "end": END
            }
        )
        
        # All agents return to orchestrator for coordination
        for agent in ["architecture", "implementation", "testing", "devops"]:
            workflow.add_edge(agent, "orchestrator")
        
        # Quality gate can end or continue
        workflow.add_conditional_edges(
            "quality_gate",
            self._route_from_quality_gate,
            {
                "orchestrator": "orchestrator",
                "end": END
            }
        )
        
        return workflow
    
    async def _orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Process tasks through the orchestrator agent."""
        try:
            orchestrator = self.agents[AgentType.ORCHESTRATOR]
            
            # Update agent state
            agent_state = AgentState(
                agent_id=str(AgentType.ORCHESTRATOR),
                status=TaskStatus.RUNNING,
                current_task="Coordinating workflow"
            )
            state.active_agents[str(AgentType.ORCHESTRATOR)] = agent_state
            
            # Process current task
            response = await orchestrator.process_task(state)
            
            # Update state with response
            state.messages.append(response.message)
            state.shared_memory.update(response.context)
            state.updated_at = datetime.now()
            
            # Update agent status
            agent_state.status = TaskStatus.COMPLETED
            agent_state.last_activity = datetime.now()
            
            self.logger.info(f"Orchestrator completed task: {response.message.content}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in orchestrator node: {e}")
            state.active_agents[str(AgentType.ORCHESTRATOR)].status = TaskStatus.FAILED
            raise
    
    async def _architecture_node(self, state: WorkflowState) -> WorkflowState:
        """Process tasks through the architecture agent."""
        try:
            architecture_agent = self.agents[AgentType.ARCHITECTURE]
            
            # Update agent state
            agent_state = AgentState(
                agent_id=str(AgentType.ARCHITECTURE),
                status=TaskStatus.RUNNING,
                current_task="Analyzing architecture"
            )
            state.active_agents[str(AgentType.ARCHITECTURE)] = agent_state
            
            # Process architecture analysis
            response = await architecture_agent.process_task(state)
            
            # Update state
            state.messages.append(response.message)
            state.project_context.update(response.context)
            state.updated_at = datetime.now()
            
            # Update agent status
            agent_state.status = TaskStatus.COMPLETED
            agent_state.last_activity = datetime.now()
            
            self.logger.info(f"Architecture agent completed: {response.message.content}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in architecture node: {e}")
            state.active_agents[str(AgentType.ARCHITECTURE)].status = TaskStatus.FAILED
            raise
    
    async def _implementation_node(self, state: WorkflowState) -> WorkflowState:
        """Process tasks through the implementation agent."""
        try:
            impl_agent = self.agents[AgentType.IMPLEMENTATION]
            
            # Update agent state
            agent_state = AgentState(
                agent_id=str(AgentType.IMPLEMENTATION),
                status=TaskStatus.RUNNING,
                current_task="Implementing features"
            )
            state.active_agents[str(AgentType.IMPLEMENTATION)] = agent_state
            
            # Process implementation
            response = await impl_agent.process_task(state)
            
            # Update state
            state.messages.append(response.message)
            state.shared_memory.update(response.context)
            state.updated_at = datetime.now()
            
            # Update agent status
            agent_state.status = TaskStatus.COMPLETED
            agent_state.last_activity = datetime.now()
            
            self.logger.info(f"Implementation agent completed: {response.message.content}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in implementation node: {e}")
            state.active_agents[str(AgentType.IMPLEMENTATION)].status = TaskStatus.FAILED
            raise
    
    async def _testing_node(self, state: WorkflowState) -> WorkflowState:
        """Process tasks through the testing agent."""
        try:
            testing_agent = self.agents[AgentType.TESTING]
            
            # Update agent state
            agent_state = AgentState(
                agent_id=str(AgentType.TESTING),
                status=TaskStatus.RUNNING,
                current_task="Running tests"
            )
            state.active_agents[str(AgentType.TESTING)] = agent_state
            
            # Process testing
            response = await testing_agent.process_task(state)
            
            # Update state
            state.messages.append(response.message)
            state.quality_gates.update(response.context.get("quality_gates", {}))
            state.updated_at = datetime.now()
            
            # Update agent status
            agent_state.status = TaskStatus.COMPLETED
            agent_state.last_activity = datetime.now()
            
            self.logger.info(f"Testing agent completed: {response.message.content}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in testing node: {e}")
            state.active_agents[str(AgentType.TESTING)].status = TaskStatus.FAILED
            raise
    
    async def _devops_node(self, state: WorkflowState) -> WorkflowState:
        """Process tasks through the devops agent."""
        try:
            devops_agent = self.agents[AgentType.DEVOPS]
            
            # Update agent state
            agent_state = AgentState(
                agent_id=str(AgentType.DEVOPS),
                status=TaskStatus.RUNNING,
                current_task="Deployment operations"
            )
            state.active_agents[str(AgentType.DEVOPS)] = agent_state
            
            # Process devops tasks
            response = await devops_agent.process_task(state)
            
            # Update state
            state.messages.append(response.message)
            state.shared_memory.update(response.context)
            state.updated_at = datetime.now()
            
            # Update agent status
            agent_state.status = TaskStatus.COMPLETED
            agent_state.last_activity = datetime.now()
            
            self.logger.info(f"DevOps agent completed: {response.message.content}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in devops node: {e}")
            state.active_agents[str(AgentType.DEVOPS)].status = TaskStatus.FAILED
            raise
    
    async def _quality_gate_node(self, state: WorkflowState) -> WorkflowState:
        """Process quality gate validation."""
        try:
            # Evaluate quality gates
            gate_results = {}
            
            # Check code quality
            if "code_quality" in state.quality_gates:
                gate_results["code_quality"] = state.quality_gates["code_quality"]
            
            # Check test coverage
            if "test_coverage" in state.quality_gates:
                gate_results["test_coverage"] = state.quality_gates["test_coverage"]
            
            # Check security scan
            if "security_scan" in state.quality_gates:
                gate_results["security_scan"] = state.quality_gates["security_scan"]
            
            # Overall gate status
            all_passed = all(gate_results.values())
            
            # Create quality gate message
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id="quality_gate",
                message_type="quality_gate_result",
                content=f"Quality gates: {'PASSED' if all_passed else 'FAILED'}",
                context={"gate_results": gate_results, "overall_passed": all_passed}
            )
            
            state.messages.append(message)
            state.quality_gates["overall"] = all_passed
            state.updated_at = datetime.now()
            
            self.logger.info(f"Quality gate evaluation: {'PASSED' if all_passed else 'FAILED'}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in quality gate node: {e}")
            raise
    
    def _route_from_orchestrator(self, state: WorkflowState) -> str:
        """Route from orchestrator to appropriate agent."""
        
        # Check if there are pending tasks
        if state.task_queue:
            next_task = state.task_queue[0]
            task_type = next_task.get("type")
            
            if task_type == "architecture":
                return "architecture"
            elif task_type == "implementation":
                return "implementation"
            elif task_type == "testing":
                return "testing"
            elif task_type == "devops":
                return "devops"
        
        # Check if quality gates need to be evaluated
        if state.quality_gates and not state.quality_gates.get("overall"):
            return "quality_gate"
        
        # Default to end if no more tasks
        return "end"
    
    def _route_from_quality_gate(self, state: WorkflowState) -> str:
        """Route from quality gate based on results."""
        
        # If quality gates failed, return to orchestrator for remediation
        if not state.quality_gates.get("overall", False):
            return "orchestrator"
        
        # If all quality gates passed, end the workflow
        return "end"
    
    async def execute_workflow(
        self, 
        project_path: str, 
        initial_task: str,
        config: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute the complete multi-agent workflow."""
        
        try:
            # Create initial state
            initial_state = WorkflowState(
                project_path=project_path,
                task_queue=[{
                    "id": str(uuid.uuid4()),
                    "type": "orchestrator",
                    "description": initial_task,
                    "created_at": datetime.now()
                }]
            )
            
            # Execute the workflow
            result = await self.app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state.session_id}}
            )
            
            self.logger.info(f"Workflow completed for project: {project_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow session."""
        
        try:
            # Get current state from checkpointer
            config = {"configurable": {"thread_id": session_id}}
            state = await self.app.aget_state(config)
            
            if not state:
                return {"error": "Session not found"}
            
            return {
                "session_id": session_id,
                "project_state": state.values.get("project_state"),
                "active_agents": state.values.get("active_agents", {}),
                "task_queue": state.values.get("task_queue", []),
                "completed_tasks": state.values.get("completed_tasks", []),
                "quality_gates": state.values.get("quality_gates", {}),
                "updated_at": state.values.get("updated_at")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            return {"error": str(e)}
    
    async def pause_workflow(self, session_id: str) -> bool:
        """Pause a running workflow."""
        try:
            # Implementation for pausing workflow
            # This would involve stopping the current execution
            self.logger.info(f"Pausing workflow session: {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pause workflow: {e}")
            return False
    
    async def resume_workflow(self, session_id: str) -> bool:
        """Resume a paused workflow."""
        try:
            # Implementation for resuming workflow
            # This would involve restarting from the last checkpoint
            self.logger.info(f"Resuming workflow session: {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to resume workflow: {e}")
            return False
