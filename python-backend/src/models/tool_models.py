"""
Tool Models for FlutterSwarm Multi-Agent System.

This module defines data structures for tool system including usage tracking,
learning models, and operational patterns.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Coroutine, Callable, Set
from uuid import uuid4


class ToolStatus(Enum):
    """Status of tool operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskStatus(Enum):
    """Status of async tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolPermission(Enum):
    """Permissions required for tool operations."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"
    PROCESS_SPAWN = "process_spawn"
    PROCESS_KILL = "process_kill"
    NETWORK_ACCESS = "network_access"
    SYSTEM_INFO = "system_info"


@dataclass
class AsyncTask:
    """
    Manages long-running operations with async execution.
    """
    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    operation: str = ""
    tool_name: str = ""
    agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal asyncio task reference
    _asyncio_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)
    _completion_callback: Optional[Callable] = field(default=None, init=False, repr=False)
    
    def start_task(self, coroutine: Coroutine, callback: Optional[Callable] = None) -> None:
        """Start the async task execution."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self._completion_callback = callback
        self._asyncio_task = asyncio.create_task(self._execute_with_tracking(coroutine))
    
    async def _execute_with_tracking(self, coroutine: Coroutine) -> Any:
        """Execute coroutine with status tracking."""
        try:
            self.result = await coroutine
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()
            self.progress = 1.0
            
            if self._completion_callback:
                await self._completion_callback(self)
                
            return self.result
            
        except asyncio.CancelledError:
            self.status = TaskStatus.CANCELLED
            self.completed_at = datetime.now()
            raise
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.completed_at = datetime.now()
            self.error = str(e)
            
            if self._completion_callback:
                await self._completion_callback(self)
                
            raise
    
    async def cancel(self) -> bool:
        """Cancel the running task."""
        if self._asyncio_task and not self._asyncio_task.done():
            self._asyncio_task.cancel()
            try:
                await self._asyncio_task
            except asyncio.CancelledError:
                pass
            return True
        return False
    
    async def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for task completion with optional timeout."""
        if not self._asyncio_task:
            raise RuntimeError("Task not started")
        
        if timeout:
            try:
                return await asyncio.wait_for(self._asyncio_task, timeout=timeout)
            except asyncio.TimeoutError:
                self.status = TaskStatus.TIMEOUT
                self.completed_at = datetime.now()
                await self.cancel()
                raise
        else:
            return await self._asyncio_task
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING and self._asyncio_task and not self._asyncio_task.done()
    
    def is_completed(self) -> bool:
        """Check if task has completed (success or failure)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]
    
    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return (end_time - self.started_at).total_seconds()
        return None
    
    def update_progress(self, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update task progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        if metadata:
            self.metadata.update(metadata)


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
    result: Any = None
    reasoning: str = ""
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
    avg_duration: float = 0.0
    error_count: int = 0
    last_used: Optional[datetime] = None
    error_frequency: Dict[str, int] = field(default_factory=dict)
    resource_efficiency: float = 0.0
    agent_preference_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Backward compatibility aliases
    @property
    def average_execution_time(self) -> float:
        """Backward compatibility alias for avg_duration."""
        return self.avg_duration
    
    @average_execution_time.setter
    def average_execution_time(self, value: float) -> None:
        """Backward compatibility setter for avg_duration."""
        self.avg_duration = value


@dataclass
class ToolUnderstanding:
    """Agent's understanding of a tool's capabilities."""
    tool_name: str
    agent_id: str
    capabilities_summary: str
    usage_scenarios: List[str] = field(default_factory=list)
    parameter_patterns: Dict[str, List[Any]] = field(default_factory=dict)
    success_indicators: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Backward compatibility alias
    @property
    def capability_summary(self) -> str:
        """Backward compatibility alias for capabilities_summary."""
        return self.capabilities_summary


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


# Workflow Adaptation Models

@dataclass
class WorkflowStepResult:
    """Result of a single workflow step execution."""
    step_id: str
    agent_id: str
    status: str  # "completed", "failed", "partial", "skipped"
    execution_time: float
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    output_size: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent in a workflow."""
    agent_id: str
    agent_type: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    resource_efficiency: float = 0.0
    quality_average: float = 0.0
    error_rate: float = 0.0
    current_load: int = 0
    availability_score: float = 1.0
    specialization_match: float = 0.0


@dataclass
class WorkflowFeedback:
    """Comprehensive feedback about workflow execution and performance."""
    workflow_id: str
    feedback_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Step-level feedback
    step_results: List[WorkflowStepResult] = field(default_factory=list)
    
    # Agent performance feedback
    agent_performance: Dict[str, AgentPerformanceMetrics] = field(default_factory=dict)
    
    # Resource utilization
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # User and system feedback
    user_feedback: Optional[str] = None
    user_satisfaction: float = 0.0
    system_alerts: List[str] = field(default_factory=list)
    
    # Performance indicators
    overall_completion_time: float = 0.0
    target_completion_time: float = 0.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Identified issues
    bottlenecks: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck in workflow execution."""
    type: str  # "agent_overload", "resource_contention", "dependency_chain", "tool_inefficiency"
    severity: str  # "low", "medium", "high", "critical"
    root_cause: str = ""
    bottleneck_id: str = field(default_factory=lambda: str(uuid4()))
    affected_steps: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    resolution_strategies: List[str] = field(default_factory=list)
    estimated_improvement: float = 0.0


@dataclass
class PerformanceAnalysis:
    """Comprehensive analysis of workflow performance."""
    workflow_id: str
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Overall metrics
    efficiency_score: float = 0.0  # 0.0 to 1.0
    completion_rate: float = 0.0   # percentage of tasks completed successfully
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Identified issues
    bottlenecks: List[PerformanceBottleneck] = field(default_factory=list)
    inefficiencies: List[str] = field(default_factory=list)
    
    # Timing analysis
    critical_path: List[str] = field(default_factory=list)
    parallel_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Agent analysis
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    agent_specialization_mismatches: List[Dict[str, str]] = field(default_factory=list)
    
    # Quality metrics
    output_quality_distribution: Dict[str, int] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)


@dataclass
class WorkflowImprovement:
    """Specific improvement recommendation for workflow optimization."""
    type: str  # "reorder", "parallelize", "replace_agent", "add_step", "remove_step", "resource_reallocation"
    priority: str  # "low", "medium", "high", "critical"
    description: str = ""
    improvement_id: str = field(default_factory=lambda: str(uuid4()))
    target_steps: List[str] = field(default_factory=list)
    
    # Change specifications
    proposed_changes: Dict[str, Any] = field(default_factory=dict)
    expected_benefit: float = 0.0  # Expected improvement percentage
    implementation_cost: float = 0.0  # Cost/complexity of implementation
    risk_level: str = "low"  # "low", "medium", "high"
    
    # Validation
    confidence_score: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    
    
@dataclass
class WorkflowSession:
    """Active workflow session with real-time state and modification capabilities."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_id: str = ""
    original_definition: Optional[Any] = None  # WorkflowDefinition
    
    # Current state
    current_steps: List[Dict[str, Any]] = field(default_factory=list)
    step_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Agent assignments
    agent_assignments: Dict[str, str] = field(default_factory=dict)  # step_id -> agent_id
    agent_availability: Dict[str, bool] = field(default_factory=dict)
    
    # Resource management
    resource_allocations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    completed_steps: Set[str] = field(default_factory=set)
    active_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    
    # Timing and estimates
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    step_timing_estimates: Dict[str, float] = field(default_factory=dict)
    
    # Adaptation history
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_count: int = 0
    last_adaptation: Optional[datetime] = None
    
    # Performance tracking
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_checkpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AdaptationResult:
    """Result of workflow adaptation process."""
    adaptation_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Changes made
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    steps_modified: List[str] = field(default_factory=list)
    agents_reassigned: Dict[str, str] = field(default_factory=dict)  # step_id -> new_agent_id
    
    # Expected improvements
    expected_improvements: Dict[str, float] = field(default_factory=dict)
    updated_timeline: Dict[str, datetime] = field(default_factory=dict)
    resource_reallocation: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Validation and confidence
    confidence_score: float = 0.0
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    rollback_plan: Optional[Dict[str, Any]] = None
    
    # Impact prediction
    estimated_time_savings: float = 0.0
    estimated_quality_improvement: float = 0.0
    estimated_resource_efficiency: float = 0.0
    
    # Monitoring requirements
    success_criteria: List[str] = field(default_factory=list)
    monitoring_checkpoints: List[Dict[str, Any]] = field(default_factory=list)


# Tool Coordination Models

@dataclass
class ToolConflict:
    """Represents a tool access conflict between agents."""
    conflict_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    operation_type: str = ""
    conflicting_agents: List[str] = field(default_factory=list)
    priority_scores: Dict[str, float] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=datetime.now)
    conflict_type: str = "resource_contention"  # "resource_contention", "exclusive_access", "version_mismatch"
    severity: str = "medium"  # "low", "medium", "high", "critical"
    estimated_delay: float = 0.0  # estimated delay in seconds
    resolution_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Resolution:
    """Represents a resolution to a tool conflict."""
    resolution_id: str = field(default_factory=lambda: str(uuid4()))
    conflict_id: str = ""
    resolution_type: str = ""  # "priority_based", "queue", "parallel", "fallback", "alternative_tool"
    assigned_agent: str = ""
    queued_agents: List[str] = field(default_factory=list)
    estimated_wait_time: float = 0.0
    alternative_tools: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence_score: float = 0.0
    implementation_plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """Tool usage pattern analysis for an agent or tool."""
    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    usage_frequency: float = 0.0  # operations per hour
    average_duration: float = 0.0  # average operation duration
    peak_usage_times: List[str] = field(default_factory=list)
    common_operations: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    resource_intensity: str = "low"  # "low", "medium", "high"
    typical_parameters: Dict[str, Any] = field(default_factory=dict)
    interdependencies: List[str] = field(default_factory=list)
    pattern_confidence: float = 0.0


@dataclass
class AllocationPlan:
    """Tool allocation plan for optimizing resource distribution."""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    agent_assignments: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> tool_names
    tool_schedules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # tool_name -> schedule
    estimated_completion: Dict[str, datetime] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    optimization_score: float = 0.0
    conflicts_resolved: int = 0
    efficiency_improvement: float = 0.0
    implementation_order: List[str] = field(default_factory=list)
    fallback_strategies: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class QueueStatus:
    """Status of tool operation queues."""
    queue_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    current_user: Optional[str] = None
    queue_length: int = 0
    waiting_agents: List[Dict[str, Any]] = field(default_factory=list)
    estimated_wait_times: Dict[str, float] = field(default_factory=dict)
    priority_queue: List[str] = field(default_factory=list)
    average_processing_time: float = 0.0
    queue_efficiency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SharedOperation:
    """Coordinated operation involving multiple agents."""
    operation_id: str = field(default_factory=lambda: str(uuid4()))
    operation_type: str = ""
    participating_agents: List[str] = field(default_factory=list)
    coordination_strategy: str = "sequential"  # "sequential", "parallel", "leader_follower", "collaborative"
    primary_agent: Optional[str] = None
    shared_resources: List[str] = field(default_factory=list)
    synchronization_points: List[Dict[str, Any]] = field(default_factory=list)
    communication_protocol: str = "event_driven"
    status: str = "planned"  # "planned", "active", "completed", "failed", "cancelled"
    progress: Dict[str, float] = field(default_factory=dict)  # agent_id -> progress
    results: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CoordinationResult:
    """Result of coordinating a shared operation."""
    result_id: str = field(default_factory=lambda: str(uuid4()))
    operation_id: str = ""
    coordination_success: bool = False
    participants_coordinated: List[str] = field(default_factory=list)
    synchronization_achieved: bool = False
    resource_conflicts_resolved: int = 0
    efficiency_score: float = 0.0
    total_coordination_time: float = 0.0
    individual_results: Dict[str, Any] = field(default_factory=dict)
    collective_outcome: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class ToolCoordinationResult:
    """Result of overall tool coordination process."""
    coordination_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Allocation results
    allocations_made: Dict[str, str] = field(default_factory=dict)  # agent_id -> tool_name
    conflicts_resolved: List[Resolution] = field(default_factory=list)
    optimizations_made: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    overall_efficiency: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    queue_improvements: Dict[str, float] = field(default_factory=dict)
    
    # Coordination statistics
    active_shared_operations: int = 0
    coordination_events: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    
    # Insights and recommendations
    usage_insights: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)
    predicted_bottlenecks: List[str] = field(default_factory=list)
    
    # Future planning
    next_coordination_schedule: Optional[datetime] = None
    recommended_tool_additions: List[str] = field(default_factory=list)
    capacity_warnings: List[str] = field(default_factory=list)


# =============================================
# SESSION MANAGEMENT MODELS
# =============================================

class SessionState(Enum):
    """State of development sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    TERMINATED = "terminated"
    RECOVERING = "recovering"
    INITIALIZING = "initializing"
    COMPLETED = "completed"


class InterruptionType(Enum):
    """Types of session interruptions."""
    USER_REQUEST = "user_request"
    SYSTEM_ERROR = "system_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    AGENT_FAILURE = "agent_failure"
    TOOL_FAILURE = "tool_failure"
    EXTERNAL_DEPENDENCY = "external_dependency"
    TIMEOUT = "timeout"
    CONFLICT = "conflict"


class RecoveryStrategy(Enum):
    """Recovery strategies for interrupted sessions."""
    RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"
    RESTART_CURRENT_STEP = "restart_current_step"
    SKIP_FAILED_STEP = "skip_failed_step"
    ROLLBACK_TO_SAFE_STATE = "rollback_to_safe_state"
    TERMINATE_SESSION = "terminate_session"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class SessionResource:
    """Represents a resource allocated to a development session."""
    resource_id: str = field(default_factory=lambda: str(uuid4()))
    resource_type: str = ""  # "tool", "agent", "process", "connection", "file"
    resource_name: str = ""
    allocation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource lifecycle tracking
    initialization_time: float = 0.0  # seconds
    cleanup_time: Optional[float] = None
    total_usage_time: float = 0.0
    access_count: int = 0
    
    # Resource health and status
    health_status: str = "healthy"  # "healthy", "degraded", "failed"
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class Interruption:
    """Represents an interruption in a development session."""
    interruption_id: str = field(default_factory=lambda: str(uuid4()))
    interruption_type: InterruptionType = InterruptionType.USER_REQUEST
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"
    
    # Context of interruption
    affected_agents: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    affected_steps: List[str] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    is_recoverable: bool = True
    estimated_recovery_time: float = 0.0  # seconds
    recovery_complexity: str = "simple"  # "simple", "moderate", "complex"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStep:
    """Individual step in a session recovery plan."""
    step_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    step_type: str = ""  # "validation", "cleanup", "restoration", "verification"
    estimated_duration: float = 0.0  # seconds
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    
    # Execution tracking
    status: str = "pending"  # "pending", "running", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Comprehensive plan for recovering from session interruptions."""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    interruption_id: str = ""
    strategy: RecoveryStrategy = RecoveryStrategy.RESUME_FROM_CHECKPOINT
    created_at: datetime = field(default_factory=datetime.now)
    
    # Recovery details
    recovery_steps: List[RecoveryStep] = field(default_factory=list)
    estimated_time: float = 0.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.8  # 0.0 to 1.0
    
    # Dependencies and constraints
    required_agents: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Risk assessment
    potential_risks: List[str] = field(default_factory=list)
    mitigation_strategies: Dict[str, str] = field(default_factory=dict)
    fallback_plan: Optional[str] = None
    
    # Execution tracking
    status: str = "planned"  # "planned", "executing", "completed", "failed", "aborted"
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionCheckpoint:
    """Checkpoint for session state persistence."""
    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # State snapshot
    session_state: SessionState = SessionState.ACTIVE
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resource_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Progress tracking
    completed_steps: List[str] = field(default_factory=list)
    active_step: Optional[str] = None
    pending_steps: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    # Context preservation
    environment_context: Dict[str, Any] = field(default_factory=dict)
    project_context: Dict[str, Any] = field(default_factory=dict)
    task_context: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    recovery_instructions: List[str] = field(default_factory=list)
    rollback_checkpoints: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DevelopmentSession:
    """Comprehensive development session for Flutter projects."""
    # Core identification
    session_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    
    # Project and task context
    project_context: Dict[str, Any] = field(default_factory=dict)
    task_context: Dict[str, Any] = field(default_factory=dict)
    workflow_definition: Dict[str, Any] = field(default_factory=dict)
    
    # Session state management
    state: SessionState = SessionState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Agent collaboration
    active_agents: Set[str] = field(default_factory=set)
    agent_assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    agent_coordination: Dict[str, Any] = field(default_factory=dict)
    
    # Resource lifecycle management
    resources: Dict[str, SessionResource] = field(default_factory=dict)
    resource_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    resource_cleanup_queue: List[str] = field(default_factory=list)
    
    # Workflow and progress tracking
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: str = "initialization"
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    
    # Interruption and recovery
    interruptions: List[Interruption] = field(default_factory=list)
    current_interruption: Optional[str] = None
    recovery_plans: Dict[str, RecoveryPlan] = field(default_factory=dict)
    
    # State persistence
    checkpoints: Dict[str, SessionCheckpoint] = field(default_factory=dict)
    last_checkpoint: Optional[str] = None
    auto_checkpoint_interval: int = 300  # seconds
    
    # Session configuration
    max_duration: Optional[int] = None  # seconds
    auto_pause_on_idle: bool = True
    idle_timeout: int = 1800  # seconds
    
    # Metrics and monitoring
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_timeline_entry(self, event_type: str, description: str, metadata: Dict[str, Any] = None) -> None:
        """Add an entry to the session timeline."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "metadata": metadata or {}
        }
        self.timeline.append(entry)
    
    def get_active_resources(self) -> List[SessionResource]:
        """Get all currently active resources."""
        return [resource for resource in self.resources.values() if resource.is_active]
    
    def get_session_duration(self) -> Optional[float]:
        """Get total session duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def is_healthy(self) -> bool:
        """Check if session is in a healthy state."""
        return (
            self.state in [SessionState.ACTIVE, SessionState.PAUSED] and
            len([r for r in self.resources.values() if r.health_status == "failed"]) == 0 and
            self.current_interruption is None
        )


@dataclass 
class PauseResult:
    """Result of pausing a development session."""
    session_id: str = ""
    success: bool = False
    pause_timestamp: datetime = field(default_factory=datetime.now)
    checkpoint_id: Optional[str] = None
    
    # State preservation
    preserved_state: Dict[str, Any] = field(default_factory=dict)
    resource_cleanup_actions: List[str] = field(default_factory=list)
    agent_pause_confirmations: Dict[str, bool] = field(default_factory=dict)
    
    # Resume preparation
    resume_instructions: List[str] = field(default_factory=list)
    estimated_resume_time: float = 0.0  # seconds
    
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeResult:
    """Result of resuming a development session."""
    session_id: str = ""
    success: bool = False
    resume_timestamp: datetime = field(default_factory=datetime.now)
    checkpoint_used: Optional[str] = None
    
    # State restoration
    restored_state: Dict[str, Any] = field(default_factory=dict)
    resource_restoration_actions: List[str] = field(default_factory=list)
    agent_resume_confirmations: Dict[str, bool] = field(default_factory=dict)
    
    # Validation results
    state_validation_passed: bool = False
    resource_validation_passed: bool = False
    environment_validation_passed: bool = False
    
    # Continuation information
    next_steps: List[str] = field(default_factory=list)
    estimated_completion_time: Optional[datetime] = None
    
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationResult:
    """Result of terminating a development session."""
    session_id: str = ""
    success: bool = False
    termination_timestamp: datetime = field(default_factory=datetime.now)
    termination_reason: str = "user_request"
    
    # Cleanup results
    resource_cleanup_results: Dict[str, bool] = field(default_factory=dict)
    agent_cleanup_confirmations: Dict[str, bool] = field(default_factory=dict)
    data_preservation_results: Dict[str, bool] = field(default_factory=dict)
    
    # Session summary
    total_duration: Optional[float] = None  # seconds
    completion_percentage: float = 0.0
    deliverables_saved: List[str] = field(default_factory=list)
    artifacts_preserved: List[str] = field(default_factory=list)
    
    # Post-termination actions
    cleanup_actions_pending: List[str] = field(default_factory=list)
    manual_cleanup_required: List[str] = field(default_factory=list)
    
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)