"""
Base Agent - Abstract base class for all agents in the multi-agent system.
Provides common functionality for message handling, task processing, and state management.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from core.agent_types import (
    AgentType, AgentMessage, MessageType, TaskDefinition, TaskStatus, 
    AgentResponse, Priority, ProjectContext, AgentState, WorkflowState
)
from core.message_broker import MessageBroker, MessageHandler, create_agent_handler
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    Provides common functionality for message handling, task processing, and state management.
    """
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            agent_type: The type of agent
            config: Configuration dictionary
        """
        self.agent_type = agent_type
        self.agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        self.config = config
        
        # Core components
        self.message_broker: Optional[MessageBroker] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # Agent state
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=agent_type,
            status=TaskStatus.PENDING,
            config=config
        )
        
        # Task management
        self.current_task: Optional[TaskDefinition] = None
        self.task_history: List[TaskDefinition] = []
        self.active_sessions: Dict[str, Any] = {}
        
        # Performance metrics
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
            "last_activity": datetime.now()
        }
        
        # Capabilities
        self.capabilities: List[str] = self._define_capabilities()
        
        logger.info(f"Initialized {agent_type.value} agent with ID: {self.agent_id}")
    
    async def initialize(self):
        """Initialize the agent with external dependencies."""
        try:
            # Initialize message broker
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.message_broker = MessageBroker(redis_url)
            await self.message_broker.initialize()
            
            # Initialize memory manager
            memory_config = self.config.get("memory", {})
            self.memory_manager = MemoryManager(memory_config)
            await self.memory_manager.initialize()
            
            # Update agent capabilities
            self.state.capabilities = self.capabilities
            self.state.status = TaskStatus.PENDING
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.state.status = TaskStatus.FAILED
            self.state.last_error = str(e)
            raise
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        try:
            if self.message_broker:
                await self.message_broker.shutdown()
            
            if self.memory_manager:
                await self.memory_manager.shutdown()
            
            self.state.status = TaskStatus.CANCELLED
            logger.info(f"Agent {self.agent_id} shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down agent {self.agent_id}: {e}")
    
    @abstractmethod
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """
        Process a task based on the current workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResponse: Response containing results and updated state
        """
        pass
    
    @abstractmethod
    def _define_capabilities(self) -> List[str]:
        """
        Define the capabilities of this agent.
        
        Returns:
            List of capability strings
        """
        pass
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """
        Handle incoming messages from other agents.
        
        Args:
            message: The incoming message
            
        Returns:
            Optional response message
        """
        try:
            self.state.last_activity = datetime.now()
            
            # Handle different message types
            if message.message_type == MessageType.TASK_REQUEST:
                return await self._handle_task_request(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.STATE_SYNC:
                await self._handle_state_sync(message)
            elif message.message_type == MessageType.ERROR_REPORT:
                await self._handle_error_report(message)
            else:
                logger.warning(f"Unhandled message type: {message.message_type}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self._create_error_response(f"Message handling failed: {e}")
    
    async def _handle_task_request(self, message: AgentMessage) -> AgentResponse:
        """Handle task request messages."""
        try:
            # Extract task from message context
            task_data = message.context.get("task", {})
            if not task_data:
                return self._create_error_response("No task data in message")
            
            # Create task definition
            task = TaskDefinition(**task_data)
            self.current_task = task
            self.state.current_task = task.description
            self.state.current_task_id = task.task_id
            self.state.status = TaskStatus.RUNNING
            
            # Process the task
            start_time = datetime.now()
            
            # Create workflow state from message context
            workflow_state = self._create_workflow_state_from_message(message)
            
            # Process task
            response = await self.process_task(workflow_state)
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, response.success)
            
            # Update state
            if response.success:
                self.state.status = TaskStatus.COMPLETED
                self.state.success_count += 1
            else:
                self.state.status = TaskStatus.FAILED
                self.state.failure_count += 1
                self.state.last_error = response.error_message
            
            self.task_history.append(task)
            self.current_task = None
            
            return response
            
        except Exception as e:
            logger.error(f"Task request handling failed: {e}")
            self.state.status = TaskStatus.FAILED
            self.state.last_error = str(e)
            return self._create_error_response(f"Task processing failed: {e}")
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update messages."""
        try:
            status_info = message.context.get("status", {})
            logger.debug(f"Status update from {message.sender_id}: {status_info}")
            
        except Exception as e:
            logger.error(f"Status update handling failed: {e}")
    
    async def _handle_state_sync(self, message: AgentMessage):
        """Handle state synchronization messages."""
        try:
            sync_data = message.context.get("state", {})
            logger.debug(f"State sync from {message.sender_id}: {sync_data}")
            
        except Exception as e:
            logger.error(f"State sync handling failed: {e}")
    
    async def _handle_error_report(self, message: AgentMessage):
        """Handle error report messages."""
        try:
            error_info = message.context.get("error", {})
            logger.warning(f"Error report from {message.sender_id}: {error_info}")
            
        except Exception as e:
            logger.error(f"Error report handling failed: {e}")
    
    def _create_workflow_state_from_message(self, message: AgentMessage) -> WorkflowState:
        """Create a workflow state from message context."""
        # This is a basic implementation - should be enhanced based on actual needs
        project_context = message.context.get("project_context", {})
        
        return WorkflowState(
            project_path=project_context.get("project_path", ""),
            project_state="implementing",  # Default state
            messages=[message],
            project_context=project_context
        )
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance metrics."""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1
        
        total_tasks = self.performance_metrics["tasks_completed"] + self.performance_metrics["tasks_failed"]
        self.performance_metrics["total_execution_time"] += execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / total_tasks if total_tasks > 0 else 0.0
        )
        self.performance_metrics["last_activity"] = datetime.now()
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Create an error response."""
        error_message_obj = AgentMessage(
            sender_id=self.agent_id,
            message_type=MessageType.ERROR_REPORT,
            content=f"Agent error: {error_message}",
            context={"error": error_message}
        )
        
        return AgentResponse(
            task_id=self.current_task.task_id if self.current_task else "unknown",
            agent_id=self.agent_id,
            success=False,
            message=error_message_obj,
            error_message=error_message
        )
    
    def _create_success_response(
        self, 
        content: str, 
        task_id: str = None,
        context: Dict[str, Any] = None,
        artifacts: List[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Create a success response."""
        response_message = AgentMessage(
            sender_id=self.agent_id,
            message_type=MessageType.TASK_RESPONSE,
            content=content,
            context=context or {}
        )
        
        return AgentResponse(
            task_id=task_id or (self.current_task.task_id if self.current_task else "unknown"),
            agent_id=self.agent_id,
            success=True,
            message=response_message,
            context=context or {},
            artifacts=artifacts or []
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.state.status.value,
            "current_task": self.state.current_task,
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics,
            "last_activity": self.state.last_activity.isoformat(),
            "error_count": self.state.error_count,
            "last_error": self.state.last_error
        }
    
    async def send_status_update(self, status_message: str, context: Dict[str, Any] = None):
        """Send a status update to other agents."""
        if not self.message_broker:
            return
        
        message = AgentMessage(
            sender_id=self.agent_id,
            message_type=MessageType.STATUS_UPDATE,
            content=status_message,
            context={
                "status": self.state.status.value,
                "agent_type": self.agent_type.value,
                **(context or {})
            }
        )
        
        await self.message_broker.broadcast_message(message)
    
    async def request_assistance(self, request_message: str, target_agent: Optional[str] = None):
        """Request assistance from other agents."""
        if not self.message_broker:
            return
        
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=target_agent,
            message_type=MessageType.COORDINATION_REQUEST,
            content=request_message,
            context={
                "requesting_agent": self.agent_type.value,
                "current_task": self.state.current_task
            },
            requires_response=True
        )
        
        if target_agent:
            await self.message_broker.send_direct_message(message, target_agent)
        else:
            await self.message_broker.broadcast_message(message)
