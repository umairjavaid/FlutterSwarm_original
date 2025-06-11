"""
Task Models for FlutterSwarm LangGraph-based System.

This module defines task-related data structures specifically for 
LangGraph workflows.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class LangGraphTaskType(Enum):
    """Types of tasks in LangGraph workflows."""
    ANALYSIS = "analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class LangGraphTaskPriority(Enum):
    """Priority levels for LangGraph tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LangGraphTaskDefinition:
    """Definition of a task in LangGraph workflow."""
    task_id: str
    description: str
    task_type: LangGraphTaskType
    agent_type: str
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    expected_deliverables: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: LangGraphTaskPriority = LangGraphTaskPriority.NORMAL
    estimated_duration: int = 300  # seconds
    timeout: int = 600  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "agent_type": self.agent_type,
            "requirements": self.requirements,
            "expected_deliverables": self.expected_deliverables,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "estimated_duration": self.estimated_duration,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass 
class LangGraphWorkflowDefinition:
    """Definition of a complete LangGraph workflow."""
    workflow_id: str
    name: str
    description: str
    tasks: List[LangGraphTaskDefinition] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
        
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
    
    def get_ready_tasks(self, completed_task_ids: Set[str]) -> List[LangGraphTaskDefinition]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for task in self.tasks:
            if task.task_id in completed_task_ids:
                continue
                
            # Check if all dependencies are completed
            task_dependencies = self.dependencies.get(task.task_id, [])
            if all(dep_id in completed_task_ids for dep_id in task_dependencies):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "dependencies": self.dependencies,
            "success_criteria": self.success_criteria,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "valid_dependencies": self.validate_dependencies()
        }
