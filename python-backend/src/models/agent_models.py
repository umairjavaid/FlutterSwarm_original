"""
Agent Models for FlutterSwarm Multi-Agent System.

This module defines the data structures and models used for agent communication,
task management, and state tracking throughout the system.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class AgentStatus(Enum):
    """Agent operational status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    SHUTTING_DOWN = "shutting_down"


class MessageType(Enum):
    """Message type enumeration for agent communication."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STATUS_UPDATE = "agent_status_update"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class AgentMessage:
    """
    Structured message for inter-agent communication.
    
    This class defines the standard format for all messages exchanged
    between agents through the event bus system.
    
    Attributes:
        type: Type of message being sent
        source: ID of the sending agent
        target: ID of the target agent or "broadcast" for all agents
        payload: Message data and context
        timestamp: When the message was created
        correlation_id: Unique ID for tracking related messages
        priority: Message priority (higher numbers = higher priority)
        metadata: Additional message metadata
    """
    type: str
    source: str
    target: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            type=data["type"],
            source=data["source"],
            target=data["target"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"],
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class TaskResult:
    """
    Result of task execution by an agent.
    
    This class encapsulates the outcome of task processing, including
    success/failure status, deliverables, and execution metadata.
    
    Attributes:
        task_id: Unique identifier of the completed task
        agent_id: ID of the agent that processed the task
        status: Execution status (completed, failed, partial)
        result: Main result data from task processing
        deliverables: Specific deliverables produced by the task
        error: Error message if task failed
        execution_time: Time taken to complete the task in seconds
        metadata: Additional execution metadata and metrics
        correlation_id: Correlation ID for tracking related tasks
        created_at: When the result was created
    """
    task_id: str
    agent_id: str
    status: str
    result: Dict[str, Any]
    deliverables: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "result": self.result,
            "deliverables": self.deliverables,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create result from dictionary."""
        return cls(
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            result=data["result"],
            deliverables=data.get("deliverables", {}),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            metadata=data.get("metadata", {}),
            correlation_id=data["correlation_id"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == "completed"
    
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == "failed"
    
    def is_partial(self) -> bool:
        """Check if task completed partially."""
        return self.status == "partial"


@dataclass
class AgentCapabilityInfo:
    """
    Information about an agent's capabilities.
    
    Used for capability discovery and task assignment optimization.
    """
    agent_id: str
    agent_type: str
    capabilities: List[str]
    specializations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    availability: bool = True
    max_concurrent_tasks: int = 5
    current_load: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "specializations": self.specializations,
            "performance_metrics": self.performance_metrics,
            "availability": self.availability,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_load": self.current_load,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapabilityInfo':
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            capabilities=data["capabilities"],
            specializations=data.get("specializations", []),
            performance_metrics=data.get("performance_metrics", {}),
            availability=data.get("availability", True),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 5),
            current_load=data.get("current_load", 0),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
    
    def can_handle_task(self, required_capability: str) -> bool:
        """Check if agent can handle a task requiring specific capability."""
        return (
            self.availability and
            self.current_load < self.max_concurrent_tasks and
            required_capability in self.capabilities
        )
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage of capacity."""
        return (self.current_load / self.max_concurrent_tasks) * 100 if self.max_concurrent_tasks > 0 else 0


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class LLMError(AgentError):
    """Exception for LLM-related errors."""
    pass


class CommunicationError(AgentError):
    """Exception for agent communication errors."""
    pass


class TaskExecutionError(AgentError):
    """Exception for task execution errors."""
    pass


@dataclass
class LLMInteraction:
    """
    Record of an LLM interaction for memory and audit purposes.
    
    Used to track all LLM calls made by agents for learning,
    debugging, and optimization purposes.
    """
    agent_id: str
    prompt: str
    response: str
    context: Dict[str, Any]
    model: str
    temperature: float
    tokens_used: int
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "agent_id": self.agent_id,
            "prompt": self.prompt,
            "response": self.response,
            "context": self.context,
            "model": self.model,
            "temperature": self.temperature,
            "tokens_used": self.tokens_used,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "success": self.success,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMInteraction':
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            prompt=data["prompt"],
            response=data["response"],
            context=data["context"],
            model=data["model"],
            temperature=data["temperature"],
            tokens_used=data["tokens_used"],
            response_time=data["response_time"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"],
            success=data.get("success", True),
            error=data.get("error")
        )


@dataclass
class MemoryEntry:
    """
    Represents a memory entry in the agent's memory system.
    
    Stores content with metadata, embeddings for semantic search,
    and tracking information for relevance and cleanup.
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    importance: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    embedding: Optional[Any] = None  # numpy array for embeddings
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "agent_id": self.agent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        import numpy as np
        
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
            importance=data.get("importance", 1.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            agent_id=data.get("agent_id")
        )
    
    def update_access(self) -> None:
        """Update access tracking information."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
