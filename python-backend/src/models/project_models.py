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


class HealthStatus(Enum):
    """Overall health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"


class IssueCategory(Enum):
    """Categories of project issues."""
    BUILD = "build"
    DEPENDENCIES = "dependencies"
    CODE_QUALITY = "code_quality"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"


class IssueSeverity(Enum):
    """Severity levels for project issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyHealthStatus(Enum):
    """Status of project dependencies."""
    UP_TO_DATE = "up_to_date"
    OUTDATED = "outdated"
    VULNERABLE = "vulnerable"
    DEPRECATED = "deprecated"
    MISSING = "missing"


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
class ProjectIssue:
    """Represents a specific project issue with actionable information."""
    severity: IssueSeverity
    category: IssueCategory
    description: str
    affected_files: List[str] = field(default_factory=list)
    suggested_fix: str = ""
    auto_fixable: bool = False
    impact_score: float = 0.0  # 0.0 to 1.0
    first_detected: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    resolution_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_blocking(self) -> bool:
        """Check if this issue blocks development."""
        return self.severity in [IssueSeverity.HIGH, IssueSeverity.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "affected_files": self.affected_files,
            "suggested_fix": self.suggested_fix,
            "auto_fixable": self.auto_fixable,
            "impact_score": self.impact_score,
            "first_detected": self.first_detected.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "resolution_url": self.resolution_url,
            "metadata": self.metadata
        }


@dataclass
class BuildHealthStatus:
    """Health status of project build system."""
    can_build: bool
    last_build_time: Optional[datetime] = None
    build_duration: Optional[float] = None  # seconds
    build_errors: List[str] = field(default_factory=list)
    build_warnings: List[str] = field(default_factory=list)
    platforms_tested: List[PlatformTarget] = field(default_factory=list)
    artifact_sizes: Dict[str, int] = field(default_factory=dict)  # platform -> size in bytes
    optimization_score: float = 0.0  # 0.0 to 1.0
    last_successful_build: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def is_healthy(self) -> bool:
        """Check if build system is healthy."""
        return self.can_build and len(self.build_errors) == 0
    
    def get_status(self) -> HealthStatus:
        """Get overall build health status."""
        if not self.can_build or self.consecutive_failures > 3:
            return HealthStatus.FAILING
        elif self.build_errors:
            return HealthStatus.CRITICAL
        elif self.build_warnings:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class DependencyStatus:
    """Status of project dependencies and package management."""
    total_dependencies: int = 0
    up_to_date: int = 0
    outdated: int = 0
    vulnerable: int = 0
    deprecated: int = 0
    missing: int = 0
    dependency_health: Dict[str, DependencyHealthStatus] = field(default_factory=dict)
    security_advisories: List[Dict[str, Any]] = field(default_factory=list)
    update_recommendations: List[str] = field(default_factory=list)
    last_dependency_check: Optional[datetime] = None
    pubspec_lock_exists: bool = True
    
    def get_health_score(self) -> float:
        """Calculate dependency health score."""
        if self.total_dependencies == 0:
            return 1.0
        
        score = 1.0
        score -= (self.vulnerable * 0.3) / self.total_dependencies
        score -= (self.deprecated * 0.2) / self.total_dependencies
        score -= (self.missing * 0.4) / self.total_dependencies
        score -= (self.outdated * 0.1) / self.total_dependencies
        
        return max(0.0, score)
    
    def get_status(self) -> HealthStatus:
        """Get overall dependency health status."""
        health_score = self.get_health_score()
        
        if health_score >= 0.9:
            return HealthStatus.HEALTHY
        elif health_score >= 0.7:
            return HealthStatus.WARNING
        elif health_score >= 0.5:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILING


@dataclass
class CodeQualityMetrics:
    """Code quality and static analysis metrics."""
    maintainability_index: float = 0.0  # 0.0 to 100.0
    cyclomatic_complexity: float = 0.0
    lines_of_code: int = 0
    lines_of_comments: int = 0
    comment_ratio: float = 0.0
    duplicate_code_percentage: float = 0.0
    technical_debt_minutes: int = 0
    code_smells: List[str] = field(default_factory=list)
    bugs: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    dart_analysis_issues: int = 0
    linting_violations: List[str] = field(default_factory=list)
    last_analysis: Optional[datetime] = None
    
    def get_quality_grade(self) -> str:
        """Get quality grade (A-F) based on metrics."""
        if self.maintainability_index >= 80:
            return "A"
        elif self.maintainability_index >= 70:
            return "B"
        elif self.maintainability_index >= 60:
            return "C"
        elif self.maintainability_index >= 50:
            return "D"
        else:
            return "F"
    
    def get_status(self) -> HealthStatus:
        """Get overall code quality health status."""
        if len(self.vulnerabilities) > 0 or len(self.bugs) > 10:
            return HealthStatus.CRITICAL
        elif self.maintainability_index < 50 or len(self.code_smells) > 20:
            return HealthStatus.WARNING
        elif self.maintainability_index >= 70:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.WARNING


@dataclass
class TestCoverageReport:
    """Test coverage analysis results."""
    overall_coverage: float = 0.0  # 0.0 to 100.0
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    uncovered_lines: int = 0
    total_lines: int = 0
    test_files_count: int = 0
    unit_tests: int = 0
    widget_tests: int = 0
    integration_tests: int = 0
    test_failures: int = 0
    test_skipped: int = 0
    test_execution_time: float = 0.0  # seconds
    coverage_by_file: Dict[str, float] = field(default_factory=dict)
    low_coverage_files: List[str] = field(default_factory=list)
    last_test_run: Optional[datetime] = None
    
    def get_status(self) -> HealthStatus:
        """Get test coverage health status."""
        if self.overall_coverage >= 80:
            return HealthStatus.HEALTHY
        elif self.overall_coverage >= 60:
            return HealthStatus.WARNING
        elif self.overall_coverage >= 40:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILING


@dataclass
class ResourceUsageMetrics:
    """Resource usage and performance metrics."""
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    build_cache_size_mb: float = 0.0
    dependencies_size_mb: float = 0.0
    generated_files_size_mb: float = 0.0
    startup_time_ms: float = 0.0
    hot_reload_time_ms: float = 0.0
    frame_render_time_ms: float = 0.0
    last_measured: Optional[datetime] = None
    
    def get_efficiency_score(self) -> float:
        """Calculate resource efficiency score."""
        score = 1.0
        
        # Penalize high resource usage
        if self.memory_usage_mb > 500:
            score -= 0.2
        if self.cpu_usage_percent > 50:
            score -= 0.2
        if self.startup_time_ms > 3000:
            score -= 0.3
        if self.hot_reload_time_ms > 1000:
            score -= 0.2
        if self.frame_render_time_ms > 16.67:  # 60fps threshold
            score -= 0.1
            
        return max(0.0, score)


@dataclass
class ProjectHealthReport:
    """Comprehensive project health assessment report."""
    project_id: str
    project_path: str
    overall_status: HealthStatus
    health_score: float = 0.0  # 0.0 to 1.0
    
    # Component health statuses
    build_health: BuildHealthStatus = field(default_factory=BuildHealthStatus)
    dependencies: DependencyStatus = field(default_factory=DependencyStatus)
    code_quality: CodeQualityMetrics = field(default_factory=CodeQualityMetrics)
    test_coverage: TestCoverageReport = field(default_factory=TestCoverageReport)
    resource_usage: ResourceUsageMetrics = field(default_factory=ResourceUsageMetrics)
    
    # Issues and recommendations
    issues: List[ProjectIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    auto_fix_suggestions: List[str] = field(default_factory=list)
    
    # Monitoring metadata
    last_assessment: datetime = field(default_factory=datetime.utcnow)
    assessment_duration: float = 0.0  # seconds
    next_scheduled_check: Optional[datetime] = None
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate overall health score after initialization."""
        self.health_score = self._calculate_overall_health_score()
        self.overall_status = self._determine_overall_status()
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate weighted overall health score."""
        weights = {
            "build": 0.3,
            "dependencies": 0.2,
            "code_quality": 0.25,
            "test_coverage": 0.15,
            "resource_usage": 0.1
        }
        
        scores = {
            "build": 1.0 if self.build_health.is_healthy() else 0.5,
            "dependencies": self.dependencies.get_health_score(),
            "code_quality": self.code_quality.maintainability_index / 100.0,
            "test_coverage": self.test_coverage.overall_coverage / 100.0,
            "resource_usage": self.resource_usage.get_efficiency_score()
        }
        
        return sum(weights[component] * scores[component] for component in weights)
    
    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall status based on health score and critical issues."""
        critical_issues = [issue for issue in self.issues if issue.severity == IssueSeverity.CRITICAL]
        
        if critical_issues or self.health_score < 0.3:
            return HealthStatus.CRITICAL
        elif self.health_score < 0.6:
            return HealthStatus.WARNING
        elif not self.build_health.can_build:
            return HealthStatus.FAILING
        else:
            return HealthStatus.HEALTHY
    
    def get_critical_issues(self) -> List[ProjectIssue]:
        """Get list of critical issues that need immediate attention."""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.CRITICAL]
    
    def get_blocking_issues(self) -> List[ProjectIssue]:
        """Get list of issues that block development."""
        return [issue for issue in self.issues if issue.is_blocking()]
    
    def get_auto_fixable_issues(self) -> List[ProjectIssue]:
        """Get list of issues that can be automatically fixed."""
        return [issue for issue in self.issues if issue.auto_fixable]
    
    def get_trend_direction(self) -> str:
        """Get health trend direction (improving, stable, degrading)."""
        # This would require historical data comparison
        # For now, return based on current state
        if self.health_score >= 0.8:
            return "stable"
        elif len(self.get_critical_issues()) > 0:
            return "degrading"
        else:
            return "improving"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "project_id": self.project_id,
            "project_path": self.project_path,
            "overall_status": self.overall_status.value,
            "health_score": self.health_score,
            "build_health": {
                "can_build": self.build_health.can_build,
                "status": self.build_health.get_status().value,
                "errors": len(self.build_health.build_errors),
                "warnings": len(self.build_health.build_warnings),
                "consecutive_failures": self.build_health.consecutive_failures
            },
            "dependencies": {
                "total": self.dependencies.total_dependencies,
                "status": self.dependencies.get_status().value,
                "health_score": self.dependencies.get_health_score(),
                "vulnerable": self.dependencies.vulnerable,
                "outdated": self.dependencies.outdated
            },
            "code_quality": {
                "maintainability_index": self.code_quality.maintainability_index,
                "status": self.code_quality.get_status().value,
                "grade": self.code_quality.get_quality_grade(),
                "technical_debt_minutes": self.code_quality.technical_debt_minutes,
                "coverage_percentage": self.code_quality.coverage_percentage
            },
            "test_coverage": {
                "overall_coverage": self.test_coverage.overall_coverage,
                "status": self.test_coverage.get_status().value,
                "test_failures": self.test_coverage.test_failures,
                "total_tests": self.test_coverage.unit_tests + self.test_coverage.widget_tests + self.test_coverage.integration_tests
            },
            "issues_summary": {
                "total": len(self.issues),
                "critical": len([i for i in self.issues if i.severity == IssueSeverity.CRITICAL]),
                "high": len([i for i in self.issues if i.severity == IssueSeverity.HIGH]),
                "blocking": len(self.get_blocking_issues()),
                "auto_fixable": len(self.get_auto_fixable_issues())
            },
            "recommendations_count": len(self.recommendations),
            "last_assessment": self.last_assessment.isoformat(),
            "trend_direction": self.get_trend_direction()
        }


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
