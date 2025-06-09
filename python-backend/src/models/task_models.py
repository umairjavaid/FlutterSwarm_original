"""
Task Management Models for FlutterSwarm Multi-Agent System.

This module defines the data structures and models used for task definition,
execution tracking, and workflow management throughout the system.
"""

import json
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TaskType(Enum):
    """Task type enumeration for different kinds of work."""
    ANALYSIS = "analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    FEATURE_IMPLEMENTATION = "feature_implementation"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionStrategy(Enum):
    """Task execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


@dataclass
class TaskRequirement:
    """
    Specific requirement for task execution.
    
    Used to define constraints, inputs, and expected outputs for tasks.
    """
    name: str
    description: str
    required: bool = True
    data_type: str = "string"
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    default_value: Optional[Any] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this requirement."""
        if self.required and value is None:
            return False
        
        # Add type validation and rule checking as needed
        return True


@dataclass
class TaskDeliverable:
    """
    Expected deliverable from task execution.
    
    Defines what output is expected from a task and how to validate it.
    """
    name: str
    description: str
    type: str  # "file", "data", "report", "analysis", etc.
    format: Optional[str] = None  # File format or data structure
    validation_schema: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    
    def validate(self, deliverable_data: Any) -> bool:
        """Validate a deliverable against this specification."""
        # Implement validation logic based on type and schema
        return True


@dataclass
class TaskContext:
    """
    Complete context information for task execution.
    
    This class encapsulates all information needed for an agent to
    understand and execute a task effectively.
    
    Attributes:
        task_id: Unique identifier for the task
        description: Human-readable description of the task
        task_type: Type/category of the task
        requirements: List of specific requirements for execution
        expected_deliverables: List of expected outputs
        project_context: Current project state and information
        constraints: Execution constraints and limitations
        priority: Task priority level
        correlation_id: ID for tracking related tasks in a workflow
        metadata: Additional task-specific information
        created_at: When the task was created
        deadline: Optional deadline for task completion
        estimated_duration: Estimated time to complete the task
    """
    task_id: str
    description: str
    task_type: TaskType
    requirements: List[TaskRequirement] = field(default_factory=list)
    expected_deliverables: List[TaskDeliverable] = field(default_factory=list)
    project_context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task context to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "requirements": [req.__dict__ for req in self.requirements],
            "expected_deliverables": [deliv.__dict__ for deliv in self.expected_deliverables],
            "project_context": self.project_context,
            "constraints": self.constraints,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create task context from dictionary."""
        requirements = [TaskRequirement(**req) for req in data.get("requirements", [])]
        deliverables = [TaskDeliverable(**deliv) for deliv in data.get("expected_deliverables", [])]
        
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            requirements=requirements,
            expected_deliverables=deliverables,
            project_context=data.get("project_context", {}),
            constraints=data.get("constraints", {}),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
            correlation_id=data["correlation_id"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            estimated_duration=timedelta(seconds=data["estimated_duration"]) if data.get("estimated_duration") else None
        )
    
    def is_expired(self) -> bool:
        """Check if task has passed its deadline."""
        return self.deadline is not None and datetime.utcnow() > self.deadline
    
    def get_urgency_score(self) -> float:
        """Calculate urgency score based on priority and deadline."""
        base_score = self.priority.value
        
        if self.deadline:
            time_remaining = (self.deadline - datetime.utcnow()).total_seconds()
            if time_remaining <= 0:
                return 10.0  # Overdue
            elif time_remaining <= 3600:  # Less than 1 hour
                return base_score * 2
            elif time_remaining <= 86400:  # Less than 1 day
                return base_score * 1.5
        
        return base_score


@dataclass
class WorkflowDefinition:
    """
    Definition of a workflow containing multiple related tasks.
    
    Used by the orchestrator to manage complex multi-step processes
    that involve coordination between multiple agents.
    
    Attributes:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name for the workflow
        description: Detailed description of the workflow purpose
        tasks: List of tasks in the workflow
        dependencies: Task dependency graph
        execution_strategy: How tasks should be executed
        success_criteria: Conditions for considering workflow successful
        failure_handling: How to handle task failures
        metadata: Additional workflow information
        created_at: When the workflow was created
        estimated_duration: Expected total duration
    """
    workflow_id: str
    name: str
    description: str
    tasks: List[TaskContext] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> [dependency_task_ids]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    estimated_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "dependencies": self.dependencies,
            "execution_strategy": self.execution_strategy.value,
            "success_criteria": self.success_criteria,
            "failure_handling": self.failure_handling,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowDefinition':
        """Create workflow from dictionary."""
        tasks = [TaskContext.from_dict(task_data) for task_data in data.get("tasks", [])]
        
        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data["description"],
            tasks=tasks,
            dependencies=data.get("dependencies", {}),
            execution_strategy=ExecutionStrategy(data.get("execution_strategy", ExecutionStrategy.SEQUENTIAL.value)),
            success_criteria=data.get("success_criteria", {}),
            failure_handling=data.get("failure_handling", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            estimated_duration=timedelta(seconds=data["estimated_duration"]) if data.get("estimated_duration") else None
        )
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[TaskContext]:
        """Get tasks that are ready to execute based on completed dependencies."""
        ready_tasks = []
        
        for task in self.tasks:
            task_dependencies = self.dependencies.get(task.task_id, [])
            
            # Check if all dependencies are completed
            if all(dep_id in completed_tasks for dep_id in task_dependencies):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def validate_dependencies(self) -> bool:
        """Validate that the dependency graph is valid (no cycles, all tasks exist)."""
        task_ids = {task.task_id for task in self.tasks}
        
        # Check that all dependency references exist
        for task_id, deps in self.dependencies.items():
            if task_id not in task_ids:
                return False
            for dep_id in deps:
                if dep_id not in task_ids:
                    return False
        
        # Check for cycles using DFS
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for task_id in task_ids:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    return False
        
        return True


@dataclass
class TaskExecution:
    """
    Runtime information about task execution.
    
    Tracks the actual execution of a task including timing,
    resource usage, and intermediate states.
    """
    task_id: str
    agent_id: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def get_duration(self) -> Optional[timedelta]:
        """Get execution duration if task is completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == "in_progress" and self.started_at is not None
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task dependencies are satisfied."""
        return self.dependencies.issubset(completed_tasks)
    
    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, output_data: Dict[str, Any]) -> None:
        """Mark task as completed with output data."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.output_data = output_data
    
    def mark_error(self, error_message: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.ERROR
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
    
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return (self.status == TaskStatus.ERROR and 
                self.retry_count < self.max_retries)
    
    def increment_retry(self) -> None:
        """Increment retry counter and reset status."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.error_message = None


@dataclass
class Workflow:
    """Workflow definition containing multiple tasks."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, 'TaskContext'] = field(default_factory=dict)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.HYBRID
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: 'TaskContext') -> None:
        """Add a task to the workflow."""
        self.tasks[task.id] = task
    
    def get_ready_tasks(self) -> List['TaskContext']:
        """Get tasks that are ready to execute."""
        completed_task_ids = {
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }
        
        return [
            task for task in self.tasks.values()
            if (task.status == TaskStatus.PENDING and 
                task.can_execute(completed_task_ids))
        ]
    
    def get_parallel_tasks(self) -> List['TaskContext']:
        """Get tasks that can be executed in parallel."""
        ready_tasks = self.get_ready_tasks()
        
        if self.execution_strategy == ExecutionStrategy.PARALLEL:
            return ready_tasks
        elif self.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            return ready_tasks[:1] if ready_tasks else []
        else:  # HYBRID or CONDITIONAL
            # Group by agent type to allow parallel execution within agents
            return ready_tasks
    
    def is_completed(self) -> bool:
        """Check if all tasks in workflow are completed."""
        return all(
            task.status == TaskStatus.COMPLETED
            for task in self.tasks.values()
        )
    
    def has_errors(self) -> bool:
        """Check if any tasks have errors."""
        return any(
            task.status == TaskStatus.ERROR
            for task in self.tasks.values()
        )
    
    def get_progress(self) -> float:
        """Get workflow completion progress (0-1)."""
        if not self.tasks:
            return 0.0
        
        completed = sum(
            1 for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED
        )
        return completed / len(self.tasks)


@dataclass
class TaskDecomposition:
    """Result of task decomposition analysis."""
    original_request: str
    tasks: List['TaskContext']
    workflow: Workflow
    execution_strategy: ExecutionStrategy
    success_criteria: List[str]
    estimated_duration: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
