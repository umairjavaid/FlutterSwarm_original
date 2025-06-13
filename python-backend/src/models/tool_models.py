"""
Tool Models for FlutterSwarm Multi-Agent System.

This module defines data structures for tool system including usage tracking,
learning models, and operational patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class ToolStatus(Enum):
    """Status of tool operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolPermission(Enum):
    """Required permissions for tool operations."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    NETWORK_ACCESS = "network_access"
    PROCESS_SPAWN = "process_spawn"
    PROCESS_KILL = "process_kill"
    SYSTEM_INFO = "system_info"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"


@dataclass
class ToolOperation:
    """Represents a tool operation request."""
    operation_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ToolResult:
    """Result of a tool operation."""
    operation_id: str
    status: ToolStatus
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ToolCapabilities:
    """Describes tool capabilities for agent understanding."""
    available_operations: List[Dict[str, Any]] = field(default_factory=list)
    input_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_codes: Dict[str, str] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    supported_contexts: List[str] = field(default_factory=list)
    performance_characteristics: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


@dataclass
class ToolUsageEntry:
    """Records tool usage for learning and optimization."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    tool_name: str = ""
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    success: bool = False
    error_details: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    outcome_quality: float = 0.0
    agent_satisfaction: float = 0.0


@dataclass
class ToolMetrics:
    """Performance and usage metrics for tools."""
    total_uses: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    error_frequency: Dict[str, int] = field(default_factory=dict)
    resource_efficiency: float = 0.0
    agent_preference_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ToolUnderstanding:
    """Agent's understanding of a tool's capabilities."""
    tool_name: str
    agent_id: str
    capability_summary: str
    usage_scenarios: List[str] = field(default_factory=list)
    parameter_patterns: Dict[str, List[Any]] = field(default_factory=dict)
    success_indicators: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ToolUsagePlan:
    """Plan for using tools to accomplish a task."""
    task_description: str
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    tool_sequence: List[ToolOperation] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    fallback_strategies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)


@dataclass
class ToolDiscovery:
    """Information about newly discovered tool patterns or insights."""
    agent_id: str
    discovery_type: str  # "pattern", "optimization", "limitation", "combination"
    description: str
    discovery_id: str = field(default_factory=lambda: str(uuid4()))
    tool_names: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    applicability: List[str] = field(default_factory=list)  # contexts where this applies
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskOutcome:
    """Outcome of a task that used tools."""
    task_id: str
    success: bool
    quality_score: float
    efficiency_score: float
    tools_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    errors_encountered: List[str] = field(default_factory=list)
    user_feedback: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolLearningModel:
    """Machine learning model for tool selection and usage optimization."""
    agent_type: str
    model_id: str = field(default_factory=lambda: str(uuid4()))
    tool_preferences: Dict[str, float] = field(default_factory=dict)
    parameter_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    success_predictors: Dict[str, float] = field(default_factory=dict)
    context_mappings: Dict[str, List[str]] = field(default_factory=dict)
    last_training: datetime = field(default_factory=datetime.now)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)


