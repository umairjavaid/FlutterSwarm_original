"""
Workflow Models for FlutterSwarm Development Sessions.

This module defines comprehensive workflow session models that support
complete Flutter development lifecycles with environment management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from pathlib import Path

from .project_models import ProjectContext, PlatformTarget
from .task_models import LangGraphTaskDefinition, TaskPriority


class WorkflowStatus(Enum):
    """Status of workflow sessions."""
    INITIALIZING = "initializing"
    ENVIRONMENT_SETUP = "environment_setup"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EnvironmentIssueType(Enum):
    """Types of development environment issues."""
    FLUTTER_SDK_MISSING = "flutter_sdk_missing"
    FLUTTER_SDK_OUTDATED = "flutter_sdk_outdated"
    DART_SDK_MISSING = "dart_sdk_missing"
    ANDROID_SDK_MISSING = "android_sdk_missing"
    XCODE_MISSING = "xcode_missing"
    DEVICE_NOT_CONNECTED = "device_not_connected"
    EMULATOR_NOT_AVAILABLE = "emulator_not_available"
    DEPENDENCIES_MISSING = "dependencies_missing"
    NETWORK_CONNECTIVITY = "network_connectivity"
    PERMISSIONS_INSUFFICIENT = "permissions_insufficient"
    TOOL_UNAVAILABLE = "tool_unavailable"
    VERSION_CONFLICT = "version_conflict"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    ENVIRONMENT_CHECK = "environment_check"
    ENVIRONMENT_SETUP = "environment_setup"
    TOOL_PREPARATION = "tool_preparation"
    PROJECT_ANALYSIS = "project_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    BUILD = "build"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    VALIDATION = "validation"


class RequirementType(Enum):
    """Types of environment requirements."""
    SDK = "sdk"
    TOOL = "tool"
    PLUGIN = "plugin"
    DEPENDENCY = "dependency"
    DEVICE = "device"
    PLATFORM = "platform"
    CONFIGURATION = "configuration"


class SetupStatus(Enum):
    """Status of environment setup operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    REQUIRES_MANUAL_INTERVENTION = "requires_manual_intervention"
    SKIPPED = "skipped"


@dataclass
class Requirement:
    """Environment requirement specification."""
    name: str
    type: RequirementType
    version_constraint: Optional[str] = None
    optional: bool = False
    description: str = ""
    platform_specific: List[PlatformTarget] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_satisfied(self, available_version: Optional[str] = None) -> bool:
        """Check if requirement is satisfied by available version."""
        if not self.version_constraint:
            return True
        
        if not available_version:
            return False
        
        # Simple version comparison logic
        try:
            # Extract major.minor.patch versions for comparison
            constraint_parts = self.version_constraint.replace(">=", "").replace(">", "").replace("=", "").strip().split(".")
            available_parts = available_version.split(".")
            
            for i, (constraint_part, available_part) in enumerate(zip(constraint_parts, available_parts)):
                constraint_num = int(constraint_part)
                available_num = int(available_part)
                
                if available_num > constraint_num:
                    return True
                elif available_num < constraint_num:
                    return False
                # If equal, continue to next part
            
            return True  # All parts are equal or better
        except (ValueError, IndexError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "version_constraint": self.version_constraint,
            "optional": self.optional,
            "description": self.description,
            "platform_specific": [p.value for p in self.platform_specific],
            "alternatives": self.alternatives,
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """Result of tool validation."""
    tool_name: str
    available: bool
    version: Optional[str] = None
    path: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_time: datetime = field(default_factory=datetime.utcnow)
    validation_duration: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible(self, requirement: Requirement) -> bool:
        """Check if this validation result satisfies a requirement."""
        if not self.available:
            return requirement.optional
        
        return requirement.is_satisfied(self.version)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "available": self.available,
            "version": self.version,
            "path": self.path,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "validation_time": self.validation_time.isoformat(),
            "validation_duration": self.validation_duration,
            "metadata": self.metadata
        }


@dataclass
class EnvironmentSetupResult:
    """Result of environment setup operation."""
    status: SetupStatus
    setup_time: datetime = field(default_factory=datetime.utcnow)
    setup_duration: float = 0.0  # seconds
    issues: List[EnvironmentIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    completed_requirements: List[Requirement] = field(default_factory=list)
    failed_requirements: List[Requirement] = field(default_factory=list)
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    environment_state: Optional[EnvironmentState] = None
    next_steps: List[str] = field(default_factory=list)
    manual_intervention_required: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready_for_development(self) -> bool:
        """Check if environment is ready for development."""
        if self.status == SetupStatus.FAILED:
            return False
        
        # Check for blocking issues
        blocking_issues = [issue for issue in self.issues if issue.blocking and not issue.resolved]
        if blocking_issues:
            return False
        
        # Check if critical requirements are met
        critical_failed = [req for req in self.failed_requirements if not req.optional]
        return len(critical_failed) == 0
    
    def get_setup_summary(self) -> Dict[str, Any]:
        """Get a summary of the setup results."""
        return {
            "status": self.status.value,
            "ready_for_development": self.is_ready_for_development(),
            "total_requirements": len(self.completed_requirements) + len(self.failed_requirements),
            "completed_requirements": len(self.completed_requirements),
            "failed_requirements": len(self.failed_requirements),
            "blocking_issues": len([issue for issue in self.issues if issue.blocking and not issue.resolved]),
            "setup_duration": self.setup_duration,
            "manual_intervention_count": len(self.manual_intervention_required)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "setup_time": self.setup_time.isoformat(),
            "setup_duration": self.setup_duration,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "completed_requirements": [req.to_dict() for req in self.completed_requirements],
            "failed_requirements": [req.to_dict() for req in self.failed_requirements],
            "validation_results": {name: result.to_dict() for name, result in self.validation_results.items()},
            "environment_state": self.environment_state.to_dict() if self.environment_state else None,
            "next_steps": self.next_steps,
            "manual_intervention_required": self.manual_intervention_required,
            "summary": self.get_setup_summary(),
            "metadata": self.metadata
        }


@dataclass
class EnvironmentIssue:
    """Represents an issue in the development environment."""
    issue_type: EnvironmentIssueType
    severity: str  # "error", "warning", "info"
    description: str
    resolution_steps: List[str] = field(default_factory=list)
    blocking: bool = True
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity,
            "description": self.description,
            "resolution_steps": self.resolution_steps,
            "blocking": self.blocking,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "metadata": self.metadata
        }


@dataclass
class DeviceInfo:
    """Information about connected development devices."""
    device_id: str
    name: str
    platform: PlatformTarget
    is_emulator: bool
    status: str  # "connected", "disconnected", "unauthorized"
    api_level: Optional[str] = None
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    screen_resolution: Optional[str] = None
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "platform": self.platform.value,
            "is_emulator": self.is_emulator,
            "status": self.status,
            "api_level": self.api_level,
            "model": self.model,
            "manufacturer": self.manufacturer,
            "screen_resolution": self.screen_resolution,
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ToolAvailability:
    """Information about tool availability in the environment."""
    tool_name: str
    available: bool
    version: Optional[str] = None
    path: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.utcnow)
    check_duration: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "available": self.available,
            "version": self.version,
            "path": self.path,
            "issues": self.issues,
            "last_checked": self.last_checked.isoformat(),
            "check_duration": self.check_duration,
            "metadata": self.metadata
        }


@dataclass
class EnvironmentState:
    """Current development environment status."""
    flutter_version: Optional[str] = None
    dart_version: Optional[str] = None
    tools_available: Dict[str, ToolAvailability] = field(default_factory=dict)
    devices_connected: Dict[str, DeviceInfo] = field(default_factory=dict)
    issues: List[EnvironmentIssue] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    last_validated: Optional[datetime] = None
    validation_duration: float = 0.0  # seconds
    health_score: float = 1.0  # 0.0 to 1.0
    
    def has_blocking_issues(self) -> bool:
        """Check if environment has blocking issues."""
        return any(issue.blocking and not issue.resolved for issue in self.issues)
    
    def get_available_platforms(self) -> Set[PlatformTarget]:
        """Get platforms that can be targeted based on available tools."""
        platforms = set()
        
        # Always available if Flutter is available
        if self.tools_available.get("flutter", ToolAvailability("flutter", False)).available:
            platforms.add(PlatformTarget.WEB)
        
        # Platform-specific checks
        if self.tools_available.get("android_sdk", ToolAvailability("android_sdk", False)).available:
            platforms.add(PlatformTarget.ANDROID)
        
        if self.tools_available.get("xcode", ToolAvailability("xcode", False)).available:
            platforms.add(PlatformTarget.IOS)
            platforms.add(PlatformTarget.MACOS)
        
        # Desktop platforms based on OS
        os_type = self.system_info.get("os_type", "").lower()
        if os_type == "windows":
            platforms.add(PlatformTarget.WINDOWS)
        elif os_type == "linux":
            platforms.add(PlatformTarget.LINUX)
        
        return platforms
    
    def get_connected_devices_by_platform(self, platform: PlatformTarget) -> List[DeviceInfo]:
        """Get connected devices for a specific platform."""
        return [
            device for device in self.devices_connected.values()
            if device.platform == platform and device.status == "connected"
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "flutter_version": self.flutter_version,
            "dart_version": self.dart_version,
            "tools_available": {name: tool.to_dict() for name, tool in self.tools_available.items()},
            "devices_connected": {id: device.to_dict() for id, device in self.devices_connected.items()},
            "issues": [issue.to_dict() for issue in self.issues],
            "environment_variables": self.environment_variables,
            "system_info": self.system_info,
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "validation_duration": self.validation_duration,
            "health_score": self.health_score
        }


@dataclass
class WorkflowStep:
    """Individual step in a workflow session."""
    step_id: str
    step_type: WorkflowStepType
    name: str
    description: str
    agent_type: str
    estimated_duration: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_platforms: List[PlatformTarget] = field(default_factory=list)
    environment_requirements: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    
    # Execution tracking
    status: str = "pending"  # pending, running, completed, failed, skipped
    assigned_agent_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[float] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready_to_execute(self, completed_steps: Set[str]) -> bool:
        """Check if step is ready to execute based on dependencies."""
        return all(dep_id in completed_steps for dep_id in self.dependencies)
    
    def can_execute_in_environment(self, env_state: EnvironmentState) -> Tuple[bool, List[str]]:
        """Check if step can execute in current environment."""
        issues = []
        
        # Check required tools
        for tool in self.required_tools:
            if tool not in env_state.tools_available or not env_state.tools_available[tool].available:
                issues.append(f"Required tool '{tool}' is not available")
        
        # Check required platforms
        available_platforms = env_state.get_available_platforms()
        for platform in self.required_platforms:
            if platform not in available_platforms:
                issues.append(f"Required platform '{platform.value}' is not available")
        
        # Check environment requirements
        for req_name, req_value in self.environment_requirements.items():
            if req_name == "min_flutter_version":
                if not env_state.flutter_version or env_state.flutter_version < req_value:
                    issues.append(f"Flutter version {req_value} or higher required")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "required_tools": self.required_tools,
            "required_platforms": [p.value for p in self.required_platforms],
            "environment_requirements": self.environment_requirements,
            "parameters": self.parameters,
            "success_criteria": self.success_criteria,
            "validation_steps": self.validation_steps,
            "rollback_steps": self.rollback_steps,
            "status": self.status,
            "assigned_agent_id": self.assigned_agent_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "actual_duration": self.actual_duration,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class ResourceRequirement:
    """Resource requirements for workflow execution."""
    cpu_cores: int = 2
    memory_gb: float = 4.0
    disk_space_gb: float = 10.0
    network_required: bool = True
    gpu_required: bool = False
    concurrent_processes: int = 1
    estimated_peak_memory: float = 2.0
    estimated_disk_usage: float = 5.0


@dataclass
class WorkflowTemplate:
    """Reusable workflow pattern template."""
    template_id: str
    name: str
    description: str
    category: str  # "new_project", "feature_development", "testing", "deployment", etc.
    target_project_types: List[str] = field(default_factory=list)
    target_platforms: List[PlatformTarget] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    estimated_duration: int = 1800  # seconds
    complexity_level: str = "medium"  # "simple", "medium", "complex"
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    common_issues: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_applicable_to_project(self, project_context: ProjectContext) -> bool:
        """Check if template is applicable to a project."""
        # Check project type
        if self.target_project_types and project_context.project_type.value not in self.target_project_types:
            return False
        
        # Check platforms
        if self.target_platforms:
            has_common_platform = bool(
                set(self.target_platforms) & project_context.target_platforms
            )
            if not has_common_platform:
                return False
        
        return True
    
    def estimate_duration_for_project(self, project_context: ProjectContext) -> int:
        """Estimate duration based on project complexity."""
        base_duration = self.estimated_duration
        
        # Adjust based on project complexity
        complexity = project_context.estimate_complexity()
        if complexity == "simple":
            return int(base_duration * 0.7)
        elif complexity == "complex":
            return int(base_duration * 1.5)
        
        return base_duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "target_project_types": self.target_project_types,
            "target_platforms": [p.value for p in self.target_platforms],
            "steps": [step.to_dict() for step in self.steps],
            "requirements": {
                "cpu_cores": self.requirements.cpu_cores,
                "memory_gb": self.requirements.memory_gb,
                "disk_space_gb": self.requirements.disk_space_gb,
                "network_required": self.requirements.network_required,
                "gpu_required": self.requirements.gpu_required,
                "concurrent_processes": self.requirements.concurrent_processes,
                "estimated_peak_memory": self.requirements.estimated_peak_memory,
                "estimated_disk_usage": self.requirements.estimated_disk_usage
            },
            "estimated_duration": self.estimated_duration,
            "complexity_level": self.complexity_level,
            "prerequisites": self.prerequisites,
            "tags": self.tags,
            "success_criteria": self.success_criteria,
            "common_issues": self.common_issues,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata
        }


@dataclass
class WorkflowSession:
    """Complete workflow session for Flutter development."""
    session_id: str
    name: str
    description: str
    project_context: ProjectContext
    template_id: Optional[str] = None
    active_steps: List[WorkflowStep] = field(default_factory=list)
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)
    
    # Execution tracking
    status: WorkflowStatus = WorkflowStatus.INITIALIZING
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    estimated_duration: int = 1800  # seconds
    actual_duration: Optional[float] = None
    progress_percentage: float = 0.0
    
    # Environment and resources
    environment_state: Optional[EnvironmentState] = None
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    allocated_agents: Set[str] = field(default_factory=set)
    
    # Results and artifacts
    deliverables: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # file paths
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Workflow management
    priority: TaskPriority = TaskPriority.NORMAL
    can_be_paused: bool = True
    auto_retry_failed_steps: bool = True
    max_retries_per_step: int = 3
    timeout_seconds: int = 7200  # 2 hours
    
    # Metadata
    creator_agent_id: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_next_executable_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to execute."""
        return [
            step for step in self.active_steps
            if (step.status == "pending" and 
                step.is_ready_to_execute(self.completed_steps))
        ]
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get currently executing step."""
        running_steps = [step for step in self.active_steps if step.status == "running"]
        return running_steps[0] if running_steps else None
    
    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage."""
        if not self.active_steps:
            return 0.0
        
        completed_count = len(self.completed_steps)
        total_count = len(self.active_steps)
        
        # Add partial progress for running step
        current_step = self.get_current_step()
        if current_step and current_step.started_at:
            elapsed = (datetime.utcnow() - current_step.started_at).total_seconds()
            step_progress = min(elapsed / current_step.estimated_duration, 0.9)
            completed_count += step_progress
        
        self.progress_percentage = (completed_count / total_count) * 100
        return self.progress_percentage
    
    def estimate_remaining_time(self) -> Optional[int]:
        """Estimate remaining time in seconds."""
        if not self.active_steps:
            return None
        
        remaining_steps = [
            step for step in self.active_steps
            if step.step_id not in self.completed_steps
        ]
        
        if not remaining_steps:
            return 0
        
        # Sum estimated durations for remaining steps
        total_estimated = sum(step.estimated_duration for step in remaining_steps)
        
        # Adjust for current step progress
        current_step = self.get_current_step()
        if current_step and current_step.started_at:
            elapsed = (datetime.utcnow() - current_step.started_at).total_seconds()
            remaining_for_current = max(0, current_step.estimated_duration - elapsed)
            total_estimated = total_estimated - current_step.estimated_duration + remaining_for_current
        
        return int(total_estimated)
    
    def can_execute_step(self, step: WorkflowStep) -> Tuple[bool, List[str]]:
        """Check if a step can be executed in current environment."""
        if not self.environment_state:
            return False, ["Environment state not available"]
        
        # Check environment compatibility
        can_execute, env_issues = step.can_execute_in_environment(self.environment_state)
        
        # Check for blocking environment issues
        if self.environment_state.has_blocking_issues():
            env_issues.append("Environment has blocking issues")
            can_execute = False
        
        return can_execute, env_issues
    
    def add_log_entry(self, level: str, message: str, metadata: Dict[str, Any] = None):
        """Add a log entry to the workflow session."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "metadata": metadata or {}
        }
        
        if level in ["error", "critical"]:
            self.error_log.append(entry)
        else:
            self.logs.append(entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "project_context": self.project_context.to_dict() if hasattr(self.project_context, 'to_dict') else str(self.project_context),
            "template_id": self.template_id,
            "active_steps": [step.to_dict() for step in self.active_steps],
            "completed_steps": list(self.completed_steps),
            "failed_steps": list(self.failed_steps),
            "skipped_steps": list(self.skipped_steps),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "progress_percentage": self.progress_percentage,
            "environment_state": self.environment_state.to_dict() if self.environment_state else None,
            "resource_requirements": {
                "cpu_cores": self.resource_requirements.cpu_cores,
                "memory_gb": self.resource_requirements.memory_gb,
                "disk_space_gb": self.resource_requirements.disk_space_gb,
                "network_required": self.resource_requirements.network_required,
                "gpu_required": self.resource_requirements.gpu_required,
                "concurrent_processes": self.resource_requirements.concurrent_processes,
                "estimated_peak_memory": self.resource_requirements.estimated_peak_memory,
                "estimated_disk_usage": self.resource_requirements.estimated_disk_usage
            },
            "allocated_agents": list(self.allocated_agents),
            "deliverables": self.deliverables,
            "artifacts": self.artifacts,
            "priority": self.priority.value,
            "creator_agent_id": self.creator_agent_id,
            "tags": self.tags,
            "metadata": self.metadata
        }
