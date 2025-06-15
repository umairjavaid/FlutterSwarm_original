"""
Environment management models for FlutterSwarm.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class RequirementType(Enum):
    """Types of environment requirements."""
    SDK = "sdk"
    TOOL = "tool"
    DEPENDENCY = "dependency"
    PLATFORM = "platform"
    DEVICE = "device"
    CONFIGURATION = "configuration"


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"


class SetupStatus(Enum):
    """Status of environment setup."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Requirement:
    """Represents an environment requirement."""
    name: str
    type: RequirementType
    description: str
    version_constraint: Optional[str] = None
    optional: bool = False
    platform_specific: Optional[str] = None
    justification: str = ""
    priority: int = 1  # 1=critical, 2=important, 3=nice-to-have
    
    def __post_init__(self):
        if not self.justification:
            self.justification = f"{self.type.value} required for Flutter development"


@dataclass
class ValidationResult:
    """Result of validating a tool or requirement."""
    tool_name: str
    status: ValidationStatus
    available: bool
    version: Optional[str] = None
    expected_version: Optional[str] = None
    path: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    check_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_compatible(self) -> bool:
        """Check if the tool version is compatible."""
        if not self.available:
            return False
        if not self.expected_version or not self.version:
            return True
        
        # Simple version comparison (can be enhanced)
        try:
            current_parts = [int(x) for x in self.version.split('.')]
            expected_parts = [int(x) for x in self.expected_version.split('.')]
            
            # Major version compatibility
            return current_parts[0] >= expected_parts[0]
        except:
            return True


@dataclass
class SetupResult:
    """Result of environment setup operation."""
    operation: str
    status: SetupStatus
    message: str
    artifacts: List[str] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None
    rollback_possible: bool = False
    
    def is_successful(self) -> bool:
        return self.status in [SetupStatus.SUCCESS, SetupStatus.PARTIAL_SUCCESS]


@dataclass
class Recommendation:
    """Recommendation for fixing environment issues."""
    title: str
    description: str
    priority: int  # 1=critical, 2=important, 3=nice-to-have
    category: str  # installation, configuration, update, etc.
    automated: bool = False  # Can be fixed automatically
    steps: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 15
    risk_level: str = "low"  # low, medium, high
    prerequisites: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.steps:
            self.steps = [f"Please resolve: {self.description}"]


@dataclass
class EnvironmentSetupResult:
    """Comprehensive result of environment setup and validation."""
    overall_status: SetupStatus
    setup_time: float
    environment_health_score: float  # 0.0 to 1.0
    
    # Validation results
    tool_validations: Dict[str, ValidationResult] = field(default_factory=dict)
    requirement_checks: Dict[str, ValidationResult] = field(default_factory=dict)
    
    # Setup results
    setup_operations: List[SetupResult] = field(default_factory=list)
    
    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    
    # Environment info
    flutter_version: Optional[str] = None
    dart_version: Optional[str] = None
    available_platforms: List[str] = field(default_factory=list)
    connected_devices: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    project_type: Optional[str] = None
    target_platforms: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_ready_for_development(self) -> bool:
        """Check if environment is ready for Flutter development."""
        return (
            self.overall_status in [SetupStatus.SUCCESS, SetupStatus.PARTIAL_SUCCESS] and
            len(self.critical_issues) == 0 and
            self.environment_health_score >= 0.7
        )
    
    def get_blocking_issues(self) -> List[str]:
        """Get issues that block development."""
        blocking = []
        
        # Check critical validations
        for validation in self.tool_validations.values():
            if not validation.available and validation.tool_name in ["flutter", "dart"]:
                blocking.append(f"{validation.tool_name} is not available")
        
        blocking.extend(self.critical_issues)
        return blocking
    
    def get_actionable_recommendations(self) -> List[Recommendation]:
        """Get recommendations sorted by priority and automation capability."""
        return sorted(
            [rec for rec in self.recommendations if rec.automated or len(rec.steps) > 0],
            key=lambda x: (x.priority, not x.automated)
        )
