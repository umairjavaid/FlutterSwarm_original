"""
Project representation models for Flutter projects.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path


class ProjectType(Enum):
    """Types of Flutter projects."""
    APP = "app"
    PACKAGE = "package"
    PLUGIN = "plugin"
    MODULE = "module"


class ArchitecturePattern(Enum):
    """Common Flutter architecture patterns."""
    CLEAN_ARCHITECTURE = "clean_architecture"
    BLOC = "bloc"
    PROVIDER = "provider"
    RIVERPOD = "riverpod"
    GETX = "getx"
    MVC = "mvc"
    MVVM = "mvvm"
    CUSTOM = "custom"


class PlatformTarget(Enum):
    """Target platforms for Flutter projects."""
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


@dataclass
class ProjectDependency:
    """Flutter project dependency information."""
    name: str
    version: str
    is_dev_dependency: bool = False
    source: str = "pub"  # pub, git, path, etc.
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileStructure:
    """Project file structure representation."""
    path: str
    is_directory: bool
    size: Optional[int] = None
    children: List["FileStructure"] = field(default_factory=list)
    last_modified: Optional[datetime] = None
    
    def get_all_files(self) -> List[str]:
        """Get all file paths in the structure."""
        files = []
        if not self.is_directory:
            files.append(self.path)
        for child in self.children:
            files.extend(child.get_all_files())
        return files
    
    def get_files_by_extension(self, extension: str) -> List[str]:
        """Get files with specific extension."""
        return [
            file for file in self.get_all_files()
            if file.endswith(f".{extension}")
        ]


@dataclass
class CodeMetrics:
    """Code quality and complexity metrics."""
    lines_of_code: int = 0
    lines_of_comments: int = 0
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
    duplicate_code_percentage: float = 0.0
    technical_debt_ratio: float = 0.0
    code_smells: List[str] = field(default_factory=list)
    security_vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Complete project context information."""
    id: str
    name: str
    path: str
    project_type: ProjectType
    architecture_pattern: Optional[ArchitecturePattern] = None
    target_platforms: Set[PlatformTarget] = field(default_factory=set)
    flutter_version: Optional[str] = None
    dart_version: Optional[str] = None
    dependencies: Dict[str, ProjectDependency] = field(default_factory=dict)
    file_structure: Optional[FileStructure] = None
    code_metrics: Optional[CodeMetrics] = None
    team_info: Dict[str, Any] = field(default_factory=dict)
    build_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_analyzed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_dependency_names(self) -> List[str]:
        """Get list of dependency names."""
        return list(self.dependencies.keys())
    
    def get_dev_dependencies(self) -> List[str]:
        """Get list of dev dependency names."""
        return [
            name for name, dep in self.dependencies.items()
            if dep.is_dev_dependency
        ]
    
    def get_production_dependencies(self) -> List[str]:
        """Get list of production dependency names."""
        return [
            name for name, dep in self.dependencies.items()
            if not dep.is_dev_dependency
        ]
    
    def has_dependency(self, name: str) -> bool:
        """Check if project has specific dependency."""
        return name in self.dependencies
    
    def get_dart_files(self) -> List[str]:
        """Get all Dart files in the project."""
        if not self.file_structure:
            return []
        return self.file_structure.get_files_by_extension("dart")
    
    def get_test_files(self) -> List[str]:
        """Get all test files in the project."""
        dart_files = self.get_dart_files()
        return [
            file for file in dart_files
            if "test" in file.lower() or file.endswith("_test.dart")
        ]
    
    def estimate_complexity(self) -> str:
        """Estimate project complexity based on metrics."""
        if not self.code_metrics:
            return "unknown"
        
        loc = self.code_metrics.lines_of_code
        complexity = self.code_metrics.cyclomatic_complexity
        dep_count = len(self.dependencies)
        
        if loc < 1000 and complexity < 10 and dep_count < 10:
            return "simple"
        elif loc < 10000 and complexity < 50 and dep_count < 50:
            return "moderate"
        else:
            return "complex"


@dataclass
class ProjectAnalysis:
    """Results of project analysis."""
    project_context: ProjectContext
    architecture_assessment: Dict[str, Any] = field(default_factory=dict)
    security_assessment: Dict[str, Any] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
