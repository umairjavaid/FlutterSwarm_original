"""
Agent Models for FlutterSwarm LangGraph-based Multi-Agent System.

This module defines the data structures and models used for LangGraph agent
communication and state management.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LangGraphAgentStatus(Enum):
    """Status of agents in LangGraph workflows."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class LangGraphAgentCapability(Enum):
    """LangGraph agent capabilities."""
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    SECURITY_ANALYSIS = "security_analysis"
    DOCUMENTATION = "documentation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    MONITORING = "monitoring"
    FILE_OPERATIONS = "file_operations"


@dataclass
class LangGraphAgentInfo:
    """Information about an agent in the LangGraph system."""
    agent_id: str
    agent_type: str
    status: LangGraphAgentStatus
    capabilities: List[LangGraphAgentCapability] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    current_load: int = 0
    max_concurrent_tasks: int = 5
    availability: bool = True
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.status == LangGraphAgentStatus.AVAILABLE and
            self.availability and
            self.current_load < self.max_concurrent_tasks
        )
    
    def can_handle_capability(self, capability: LangGraphAgentCapability) -> bool:
        """Check if agent can handle a specific capability."""
        return capability in self.capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "specializations": self.specializations,
            "current_load": self.current_load,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "availability": self.availability,
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class LangGraphTaskRequest:
    """Task request in LangGraph workflow."""
    task_id: str
    description: str
    task_type: str
    required_capability: LangGraphAgentCapability
    priority: str = "normal"
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type,
            "required_capability": self.required_capability.value,
            "priority": self.priority,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class LangGraphTaskResult:
    """Task result in LangGraph workflow."""
    task_id: str
    agent_id: str
    status: str  # "completed", "failed", "partial"
    result: Any = None
    deliverables: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "result": self.result,
            "deliverables": self.deliverables,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
            "successful": self.is_successful()
        }


# Error class for LangGraph agents
class LangGraphAgentError(Exception):
    """Exception raised by LangGraph agents."""
    
    def __init__(self, message: str, agent_id: str = None, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message": str(self),
            "agent_id": self.agent_id,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat()
        }
