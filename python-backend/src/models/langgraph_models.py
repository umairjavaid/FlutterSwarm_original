"""
LangGraph Models for FlutterSwarm Multi-Agent System.

This module defines the data structures and models used specifically
for LangGraph-based workflows and state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class AgentRole(Enum):
    """Enumeration of agent roles in the LangGraph workflow."""
    SUPERVISOR = "supervisor"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    SECURITY = "security"
    DEVOPS = "devops"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"


class WorkflowPhase(Enum):
    """Enumeration of workflow phases."""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"


class MessageType(Enum):
    """Types of messages in the workflow."""
    WORKFLOW_CONTROL = "workflow_control"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    AGENT_MESSAGE = "agent_message"


class WorkflowState(TypedDict, total=False):
    """State management for LangGraph workflows."""
    # Core workflow information
    workflow_id: str
    task_description: str
    project_context: Dict[str, Any]
    
    # Communication and messaging
    messages: List[BaseMessage]
    current_agent: Optional[str]
    
    # Agent management
    available_agents: Dict[str, Dict[str, Any]]
    agent_assignments: Dict[str, str]  # task_id -> agent_id
    
    # Task management
    pending_tasks: List[Dict[str, Any]]
    active_tasks: Dict[str, Dict[str, Any]]
    completed_tasks: Dict[str, Dict[str, Any]]
    failed_tasks: Dict[str, Dict[str, Any]]
    
    # Results and deliverables
    deliverables: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]
    
    # Workflow control
    next_action: Optional[str]
    should_continue: bool
    error_message: Optional[str]
    
    # Metadata and metrics
    workflow_metadata: Dict[str, Any]
    execution_metrics: Dict[str, Any]


@dataclass
class LangGraphTaskContext:
    """Task context specifically for LangGraph workflows."""
    task_id: str
    description: str
    task_type: str
    agent_type: str
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    expected_deliverables: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LangGraphAgentConfig:
    """Configuration for LangGraph agents."""
    agent_id: str
    agent_type: str
    agent_role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    availability: bool = True
    current_load: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_agent_message(
    agent_id: str,
    content: str,
    message_type: MessageType,
    metadata: Optional[Dict[str, Any]] = None
) -> AIMessage:
    """Create an agent message for the workflow."""
    message_metadata = {
        "agent_id": agent_id,
        "message_type": message_type.value,
        "timestamp": datetime.utcnow().isoformat(),
        **(metadata or {})
    }
    
    return AIMessage(
        content=content,
        additional_kwargs=message_metadata
    )


def create_system_message(content: str) -> SystemMessage:
    """Create a system message for the workflow."""
    return SystemMessage(
        content=content,
        additional_kwargs={
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": MessageType.WORKFLOW_CONTROL.value
        }
    )


def create_human_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> HumanMessage:
    """Create a human message for the workflow."""
    message_metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "message_type": MessageType.TASK_REQUEST.value,
        **(metadata or {})
    }
    
    return HumanMessage(
        content=content,
        additional_kwargs=message_metadata
    )


def create_task_context_from_state(
    state: WorkflowState,
    task_data: Dict[str, Any]
) -> LangGraphTaskContext:
    """Create a task context from workflow state and task data."""
    return LangGraphTaskContext(
        task_id=task_data.get("task_id", ""),
        description=task_data.get("description", ""),
        task_type=task_data.get("task_type", "analysis"),
        agent_type=task_data.get("agent_type", "general"),
        requirements=task_data.get("requirements", []),
        expected_deliverables=task_data.get("expected_deliverables", []),
        dependencies=task_data.get("dependencies", []),
        priority=task_data.get("priority", "normal"),
        metadata={
            **task_data.get("metadata", {}),
            "workflow_id": state.get("workflow_id"),
            "project_context": state.get("project_context", {})
        }
    )


@dataclass
class WorkflowResult:
    """Result of a LangGraph workflow execution."""
    workflow_id: str
    status: str  # "completed", "failed", "partial"
    deliverables: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_successful(self) -> bool:
        """Check if the workflow completed successfully."""
        return self.status == "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "deliverables": self.deliverables,
            "task_results": self.task_results,
            "execution_metrics": self.execution_metrics,
            "error_message": self.error_message,
            "completed_at": self.completed_at.isoformat(),
            "successful": self.is_successful()
        }


class LangGraphNodeType(Enum):
    """Types of nodes in the LangGraph workflow."""
    SUPERVISOR = "supervisor"
    AGENT = "agent"
    TOOL = "tool"
    CONDITIONAL = "conditional"
    AGGREGATOR = "aggregator"
    ERROR_HANDLER = "error_handler"


@dataclass
class LangGraphNodeConfig:
    """Configuration for a LangGraph node."""
    node_id: str
    node_type: LangGraphNodeType
    agent_role: Optional[AgentRole] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_workflow_state(state: WorkflowState) -> bool:
    """Validate that the workflow state is consistent."""
    try:
        # Check required fields
        required_fields = ["workflow_id", "task_description"]
        for field in required_fields:
            if field not in state:
                return False
        
        # Validate agent assignments
        agent_assignments = state.get("agent_assignments", {})
        active_tasks = state.get("active_tasks", {})
        
        for task_id in agent_assignments:
            if task_id not in active_tasks and task_id not in state.get("completed_tasks", {}):
                return False
        
        return True
        
    except Exception:
        return False


def create_initial_workflow_state(
    workflow_id: str,
    task_description: str,
    project_context: Dict[str, Any]
) -> WorkflowState:
    """Create an initial workflow state."""
    return WorkflowState(
        workflow_id=workflow_id,
        task_description=task_description,
        project_context=project_context,
        messages=[],
        current_agent=None,
        available_agents={},
        agent_assignments={},
        pending_tasks=[],
        active_tasks={},
        completed_tasks={},
        failed_tasks={},
        deliverables={},
        final_result=None,
        next_action="initialize",
        should_continue=True,
        error_message=None,
        workflow_metadata={
            "created_at": datetime.utcnow().isoformat(),
            "phase": WorkflowPhase.INITIALIZATION.value
        },
        execution_metrics={}
    )
