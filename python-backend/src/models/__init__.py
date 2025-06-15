"""
Models package initialization.
"""
from .agent_models import (
    AgentMessage,
    TaskResult,
    MemoryEntry,
    TaskStatus,
    MessageType
)
from .task_models import (
    TaskContext,
    Workflow,
    TaskDecomposition,
    ExecutionStrategy,
    TaskType
)
from .project_models import (
    ProjectContext,
    ProjectAnalysis,
    ProjectDependency,
    FileStructure,
    CodeMetrics,
    ProjectType,
    ArchitecturePattern,
    PlatformTarget
)
from .code_models import (
    CodeGeneration,
    CodeUnderstanding,
    ProjectContext as CodeProjectContext,
    GenerationEntry,
    CodePattern,
    ProjectStructure,
    CodeType,
    ArchitectureStyle,
    CodeConvention,
    CodeAnalysisResult
)

__all__ = [
    # Agent models
    "AgentMessage",
    "TaskResult",
    "MemoryEntry",
    "TaskStatus",
    "MessageType",
    
    # Task models
    "TaskContext",
    "Workflow",
    "TaskDecomposition",
    "ExecutionStrategy",
    "TaskType",
    
    # Project models
    "ProjectContext",
    "ProjectAnalysis",
    "ProjectDependency",
    "FileStructure",
    "CodeMetrics",
    "ProjectType",
    "ArchitecturePattern",
    "PlatformTarget",
    
    # Code models
    "CodeGeneration",
    "CodeUnderstanding", 
    "CodeProjectContext",
    "GenerationEntry",
    "CodePattern",
    "ProjectStructure",
    "CodeType",
    "ArchitectureStyle",
    "CodeConvention",
    "CodeAnalysisResult"
]
