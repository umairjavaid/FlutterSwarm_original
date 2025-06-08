"""
Agent types and data models for the multi-agent system.
Defines the structure for agent communication and state management.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class AgentType(Enum):
    """Enumeration of all agent types in the system."""
    
    # Tier 1: Orchestrator
    ORCHESTRATOR = "orchestrator"
    
    # Tier 2: Specialized Agents
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEVOPS = "devops"
    
    # Additional Specialized Agents
    SEARCH_FETCH = "search_fetch"
    NAVIGATION_ROUTING = "navigation_routing"
    ANIMATION_EFFECTS = "animation_effects"
    LOCALIZATION = "localization"
    ACCESSIBILITY = "accessibility"
    DATA_MODEL = "data_model"
    LOCAL_STORAGE = "local_storage"
    API_INTEGRATION = "api_integration"
    REPOSITORY = "repository"
    BUSINESS_LOGIC = "business_logic"
    
    # Platform-Specific Deployment Agents
    ANDROID_DEPLOYMENT = "android_deployment"
    IOS_DEPLOYMENT = "ios_deployment"
    WEB_DEPLOYMENT = "web_deployment"
    DESKTOP_DEPLOYMENT = "desktop_deployment"
    
    # Quality and Documentation Agents
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_MONITORING = "performance_monitoring"
    
    # Advanced Agents
    MEMORY = "memory"
    STATE_MANAGEMENT = "state_management"
    UI_UX_IMPLEMENTATION = "ui_ux_implementation"


class TaskStatus(Enum):
    """Status of agent tasks."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    STATE_SYNC = "state_sync"
    ERROR_REPORT = "error_report"
    QUALITY_GATE_RESULT = "quality_gate_result"
    PROJECT_ANALYSIS = "project_analysis"
    CODE_GENERATION = "code_generation"
    TEST_RESULTS = "test_results"
    DEPLOYMENT_STATUS = "deployment_status"
    COORDINATION_REQUEST = "coordination_request"


class Priority(Enum):
    """Task priority levels."""
    
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentMessage(BaseModel):
    """Message structure for inter-agent communication."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    sender_id: str
    recipient_id: Optional[str] = None
    broadcast: bool = False
    
    message_type: MessageType
    content: str
    context: Dict[str, Any] = Field(default_factory=dict)
    
    priority: Priority = Priority.MEDIUM
    requires_response: bool = False
    response_timeout: Optional[int] = None  # seconds
    
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Tracing and debugging
    trace_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """Current state of an agent."""
    
    agent_id: str
    agent_type: AgentType
    status: TaskStatus = TaskStatus.PENDING
    
    current_task: Optional[str] = None
    current_task_id: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Activity tracking
    last_activity: datetime = Field(default_factory=datetime.now)
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    
    # Error information
    last_error: Optional[str] = None
    error_count: int = 0
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class TaskDefinition(BaseModel):
    """Definition of a task to be executed by an agent."""
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    description: str
    
    # Agent assignment
    assigned_agent: Optional[AgentType] = None
    required_capabilities: List[str] = Field(default_factory=list)
    
    # Task parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Task IDs
    blocks: List[str] = Field(default_factory=list)  # Task IDs
    
    # Execution constraints
    priority: Priority = Priority.MEDIUM
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    
    # Quality gates
    quality_requirements: Dict[str, Any] = Field(default_factory=dict)
    blocking_quality_gates: List[str] = Field(default_factory=list)
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class AgentResponse(BaseModel):
    """Response from an agent after processing a task."""
    
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    agent_id: str
    
    success: bool
    message: AgentMessage
    
    # Results
    result: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Context and state changes
    context: Dict[str, Any] = Field(default_factory=dict)
    state_changes: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality information
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    quality_gates_passed: Dict[str, bool] = Field(default_factory=dict)
    
    # Performance metrics
    execution_time: Optional[float] = None
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.now)


class ProjectState(Enum):
    """Project maturity and development state."""
    
    NEW = "new"
    INITIALIZED = "initialized"
    MID_DEVELOPMENT = "mid_development"
    MATURE = "mature"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


class QualityGate(BaseModel):
    """Quality gate definition and status."""
    
    gate_id: str
    name: str
    description: str
    
    gate_type: str  # "blocking" or "non_blocking"
    category: str  # "code_quality", "testing", "security", etc.
    
    # Criteria
    criteria: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Status
    status: str = "pending"  # "pending", "running", "passed", "failed"
    passed: bool = False
    score: Optional[float] = None
    
    # Results
    results: Dict[str, Any] = Field(default_factory=dict)
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Execution info
    executed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ProjectContext(BaseModel):
    """Context information about the Flutter project."""
    
    project_path: str
    project_name: str
    project_state: ProjectState = ProjectState.NEW
    
    # Project metadata
    flutter_version: Optional[str] = None
    dart_version: Optional[str] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)
    dev_dependencies: Dict[str, str] = Field(default_factory=dict)
    
    # Architecture information
    architecture_pattern: Optional[str] = None  # "clean", "mvvm", "bloc", etc.
    state_management: Optional[str] = None  # "bloc", "provider", "riverpod", etc.
    
    # Structure analysis
    file_count: int = 0
    line_count: int = 0
    test_coverage: float = 0.0
    
    # Quality metrics
    code_quality_score: Optional[float] = None
    complexity_score: Optional[float] = None
    maintainability_index: Optional[float] = None
    
    # Capabilities and features
    platforms: List[str] = Field(default_factory=list)  # "android", "ios", "web", "windows", etc.
    features: List[str] = Field(default_factory=list)
    
    # Development status
    has_tests: bool = False
    has_documentation: bool = False
    has_ci_cd: bool = False
    
    # Analysis timestamps
    last_analyzed: Optional[datetime] = None
    analysis_version: str = "1.0"
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowConfiguration(BaseModel):
    """Configuration for workflow execution."""
    
    # Agent configuration
    enabled_agents: List[AgentType] = Field(default_factory=list)
    agent_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Quality gates
    quality_gates: List[QualityGate] = Field(default_factory=list)
    enforce_quality_gates: bool = True
    
    # Execution settings
    parallel_execution: bool = True
    max_concurrent_agents: int = 3
    task_timeout: int = 300  # seconds
    
    # Integration settings
    flutter_cli_path: Optional[str] = None
    dart_cli_path: Optional[str] = None
    
    # External services
    enable_ai_assistance: bool = True
    openai_api_key: Optional[str] = None
    
    # Monitoring and logging
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowState(BaseModel):
    """State model for the LangGraph workflow."""
    
    # Core state
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_path: str
    project_state: str = "analyzing"  # analyzing, planning, implementing, testing, deploying
    
    # Messages and communication
    messages: List[AgentMessage] = Field(default_factory=list)
    active_agents: Dict[str, AgentState] = Field(default_factory=dict)
    
    # Task management
    task_queue: List[Dict[str, Any]] = Field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    failed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Context and memory
    project_context: Dict[str, Any] = Field(default_factory=dict)
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality gates
    quality_gates: Dict[str, bool] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
