"""
Code Models for FlutterSwarm Multi-Agent System.

This module defines data structures for code generation, understanding,
and project-aware development patterns.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class CodeType(Enum):
    """Types of Flutter code."""
    WIDGET = "widget"
    BLOC = "bloc"
    CUBIT = "cubit" 
    PROVIDER = "provider"
    REPOSITORY = "repository"
    MODEL = "model"
    SERVICE = "service"
    CONTROLLER = "controller"
    SCREEN = "screen"
    PAGE = "page"
    COMPONENT = "component"
    UTILITY = "utility"
    CONFIGURATION = "configuration"
    TEST = "test"


class ArchitectureStyle(Enum):
    """Flutter architecture styles."""
    CLEAN_ARCHITECTURE = "clean_architecture"
    BLOC_PATTERN = "bloc_pattern"
    PROVIDER_PATTERN = "provider_pattern"
    RIVERPOD_PATTERN = "riverpod_pattern"
    GETX_PATTERN = "getx_pattern"
    MVC_PATTERN = "mvc_pattern"
    MVVM_PATTERN = "mvvm_pattern"
    CUSTOM = "custom"


class CodeConvention(Enum):
    """Code conventions and patterns."""
    NAMING_CONVENTION = "naming_convention"
    FILE_ORGANIZATION = "file_organization"
    IMPORT_STYLE = "import_style"
    COMMENT_STYLE = "comment_style"
    ERROR_HANDLING = "error_handling"
    STATE_MANAGEMENT = "state_management"
    WIDGET_STRUCTURE = "widget_structure"
    FOLDER_STRUCTURE = "folder_structure"


class RefactoringType(Enum):
    """Types of refactoring operations."""
    RENAME_FILE = "rename_file"
    MOVE_FILE = "move_file"
    EXTRACT_CLASS = "extract_class"
    MERGE_FILES = "merge_files"
    SPLIT_FILE = "split_file"
    REORGANIZE_STRUCTURE = "reorganize_structure"
    RENAME_CLASS = "rename_class"
    RENAME_METHOD = "rename_method"
    MOVE_CLASS = "move_class"
    UPDATE_IMPORTS = "update_imports"
    RESTRUCTURE_DIRECTORY = "restructure_directory"
    REFACTOR_ARCHITECTURE = "refactor_architecture"


class RiskLevel(Enum):
    """Risk levels for refactoring operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueType(Enum):
    """Types of validation issues."""
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    NULL_SAFETY = "null_safety"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    PERFORMANCE_ISSUE = "performance_issue"
    STYLE_VIOLATION = "style_violation"
    LINT_WARNING = "lint_warning"
    UNUSED_IMPORT = "unused_import"
    DEAD_CODE = "dead_code"
    COMPLEXITY_WARNING = "complexity_warning"
    SECURITY_ISSUE = "security_issue"


class ChangeType(Enum):
    """Types of code changes."""
    FILE_ADDED = "file_added"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_RENAMED = "file_renamed"
    CONTENT_CHANGED = "content_changed"


@dataclass
class CodePattern:
    """Represents a discovered code pattern in the project."""
    pattern_id: str
    pattern_type: str
    description: str
    examples: List[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0
    file_paths: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "examples": self.examples,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "file_paths": self.file_paths,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat()
        }


@dataclass
class ProjectStructure:
    """Represents the understood project structure."""
    root_path: str
    structure_map: Dict[str, Any] = field(default_factory=dict)
    key_directories: Dict[str, str] = field(default_factory=dict)
    architecture_layers: Dict[str, List[str]] = field(default_factory=dict)
    module_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    configuration_files: List[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_layer_for_path(self, file_path: str) -> Optional[str]:
        """Determine which architecture layer a file belongs to."""
        relative_path = str(Path(file_path).relative_to(self.root_path))
        
        for layer, paths in self.architecture_layers.items():
            for path_pattern in paths:
                if path_pattern in relative_path:
                    return layer
        return None
    
    def suggest_file_location(self, code_type: CodeType, feature_name: str) -> str:
        """Suggest appropriate file location based on project structure."""
        layer = self._map_code_type_to_layer(code_type)
        base_path = self.key_directories.get(layer, f"lib/{layer}")
        
        return f"{base_path}/{feature_name}"
    
    def _map_code_type_to_layer(self, code_type: CodeType) -> str:
        """Map code type to architecture layer."""
        mapping = {
            CodeType.WIDGET: "presentation",
            CodeType.SCREEN: "presentation", 
            CodeType.PAGE: "presentation",
            CodeType.BLOC: "presentation",
            CodeType.CUBIT: "presentation",
            CodeType.PROVIDER: "presentation",
            CodeType.REPOSITORY: "data",
            CodeType.SERVICE: "domain",
            CodeType.MODEL: "data",
            CodeType.CONTROLLER: "domain"
        }
        return mapping.get(code_type, "lib")


@dataclass
class CodeGeneration:
    """Result of code generation operation."""
    generation_id: str
    request_description: str
    generated_code: Dict[str, str]  # file_path -> content
    target_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)
    imports_required: List[str] = field(default_factory=list)
    configuration_changes: List[Dict[str, Any]] = field(default_factory=list)
    test_requirements: List[str] = field(default_factory=list)
    documentation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_main_file(self) -> Optional[str]:
        """Get the main file from generated code."""
        if not self.generated_code:
            return None
        return list(self.generated_code.keys())[0]
    
    def get_total_lines(self) -> int:
        """Get total lines of generated code."""
        return sum(len(content.splitlines()) for content in self.generated_code.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generation_id": self.generation_id,
            "request_description": self.request_description,
            "generated_code": self.generated_code,
            "target_files": self.target_files,
            "dependencies": self.dependencies,
            "integration_points": self.integration_points,
            "imports_required": self.imports_required,
            "configuration_changes": self.configuration_changes,
            "test_requirements": self.test_requirements,
            "documentation": self.documentation,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
            "total_lines": self.get_total_lines()
        }


@dataclass
class CodeUnderstanding:
    """Result of analyzing existing code."""
    file_path: str
    code_type: Optional[CodeType] = None
    structure: Dict[str, Any] = field(default_factory=dict)
    patterns: List[CodePattern] = field(default_factory=list)
    conventions: Dict[CodeConvention, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_indicators: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_primary_pattern(self) -> Optional[CodePattern]:
        """Get the most confident pattern."""
        if not self.patterns:
            return None
        return max(self.patterns, key=lambda p: p.confidence)
    
    def has_pattern_type(self, pattern_type: str) -> bool:
        """Check if code has specific pattern type."""
        return any(p.pattern_type == pattern_type for p in self.patterns)
    
    def get_convention_value(self, convention: CodeConvention) -> Optional[str]:
        """Get convention value with fallback."""
        return self.conventions.get(convention)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "code_type": self.code_type.value if self.code_type else None,
            "structure": self.structure,
            "patterns": [p.to_dict() for p in self.patterns],
            "conventions": {k.value: v for k, v in self.conventions.items()},
            "dependencies": self.dependencies,
            "relationships": self.relationships,
            "complexity_metrics": self.complexity_metrics,
            "quality_indicators": self.quality_indicators,
            "suggestions": self.suggestions,
            "analyzed_at": self.analyzed_at.isoformat()
        }


@dataclass 
class ProjectContext:
    """Enhanced project context for code generation."""
    root_path: str
    architecture_style: Optional[ArchitectureStyle] = None
    conventions: Dict[CodeConvention, str] = field(default_factory=dict)
    existing_patterns: Dict[str, CodePattern] = field(default_factory=dict)
    structure: Optional[ProjectStructure] = None
    dependencies: List[str] = field(default_factory=list)
    flutter_version: Optional[str] = None
    dart_version: Optional[str] = None
    state_management: Optional[str] = None
    navigation_pattern: Optional[str] = None
    testing_framework: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_pattern_by_type(self, pattern_type: str) -> List[CodePattern]:
        """Get patterns by type."""
        return [p for p in self.existing_patterns.values() if p.pattern_type == pattern_type]
    
    def has_convention(self, convention: CodeConvention) -> bool:
        """Check if project has specific convention."""
        return convention in self.conventions
    
    def suggest_naming_convention(self, code_type: CodeType) -> str:
        """Suggest naming convention based on project patterns."""
        # Look for existing naming patterns
        naming_convention = self.conventions.get(CodeConvention.NAMING_CONVENTION, "snake_case")
        
        # Customize based on code type
        if code_type in [CodeType.WIDGET, CodeType.SCREEN, CodeType.PAGE]:
            return f"{naming_convention}_widget"
        elif code_type in [CodeType.BLOC, CodeType.CUBIT]:
            return f"{naming_convention}_bloc"
        
        return naming_convention
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "root_path": self.root_path,
            "architecture_style": self.architecture_style.value if self.architecture_style else None,
            "conventions": {k.value: v for k, v in self.conventions.items()},
            "existing_patterns": {k: v.to_dict() for k, v in self.existing_patterns.items()},
            "structure": self.structure.__dict__ if self.structure else None,
            "dependencies": self.dependencies,
            "flutter_version": self.flutter_version,
            "dart_version": self.dart_version,
            "state_management": self.state_management,
            "navigation_pattern": self.navigation_pattern,
            "testing_framework": self.testing_framework,
            "metadata": self.metadata,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class GenerationEntry:
    """Track code generation history."""
    entry_id: str
    timestamp: datetime
    request: str
    generated_code: CodeGeneration
    files_affected: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    agent_id: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_generation_size(self) -> int:
        """Get size of generated code."""
        return self.generated_code.get_total_lines()
    
    def was_successful(self) -> bool:
        """Check if generation was successful."""
        return self.success and not self.error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "request": self.request,
            "generated_code": self.generated_code.to_dict(),
            "files_affected": self.files_affected,
            "success": self.success,
            "error_message": self.error_message,
            "agent_id": self.agent_id,
            "project_context": self.project_context,
            "feedback": self.feedback,
            "performance_metrics": self.performance_metrics,
            "generation_size": self.get_generation_size()
        }


@dataclass
class CodeAnalysisResult:
    """Result of comprehensive code analysis."""
    project_context: ProjectContext
    file_understandings: Dict[str, CodeUnderstanding] = field(default_factory=dict)
    overall_patterns: List[CodePattern] = field(default_factory=list)
    architecture_assessment: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_files_by_type(self, code_type: CodeType) -> List[str]:
        """Get files by code type."""
        return [
            path for path, understanding in self.file_understandings.items()
            if understanding.code_type == code_type
        ]
    
    def get_dominant_patterns(self, min_frequency: int = 2) -> List[CodePattern]:
        """Get patterns that appear frequently across the project."""
        return [p for p in self.overall_patterns if p.frequency >= min_frequency]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_context": self.project_context.to_dict(),
            "file_understandings": {k: v.to_dict() for k, v in self.file_understandings.items()},
            "overall_patterns": [p.to_dict() for p in self.overall_patterns],
            "architecture_assessment": self.architecture_assessment,
            "quality_metrics": self.quality_metrics,
            "recommendations": self.recommendations,
            "analysis_metadata": self.analysis_metadata,
            "analyzed_at": self.analyzed_at.isoformat()
        }


@dataclass
class CodeExample:
    """Represents a code example found in the project for reference."""
    file_path: str
    code_snippet: str
    code_type: CodeType
    description: str
    patterns_used: List[str] = field(default_factory=list)
    conventions_followed: List[CodeConvention] = field(default_factory=list)
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "code_snippet": self.code_snippet,
            "code_type": self.code_type.value,
            "description": self.description,
            "patterns_used": self.patterns_used,
            "conventions_followed": [c.value for c in self.conventions_followed],
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at.isoformat()
        }


@dataclass
class IntegrationPlan:
    """Plan for integrating new code into existing project structure."""
    plan_id: str
    feature_description: str
    affected_files: List[str] = field(default_factory=list)
    new_files: List[Dict[str, str]] = field(default_factory=list)  # path, purpose
    dependencies_to_add: List[str] = field(default_factory=list)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)
    required_modifications: List[Dict[str, Any]] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    configuration_changes: List[Dict[str, Any]] = field(default_factory=list)
    architectural_impact: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: str = "medium"  # low, medium, high
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    implementation_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_file_count(self) -> Dict[str, int]:
        """Get count of affected vs new files."""
        return {
            "affected": len(self.affected_files),
            "new": len(self.new_files),
            "total": len(self.affected_files) + len(self.new_files)
        }
    
    def has_breaking_changes(self) -> bool:
        """Check if plan includes breaking changes."""
        return any(
            mod.get("breaking", False) 
            for mod in self.required_modifications
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "feature_description": self.feature_description,
            "affected_files": self.affected_files,
            "new_files": self.new_files,
            "dependencies_to_add": self.dependencies_to_add,
            "integration_points": self.integration_points,
            "required_modifications": self.required_modifications,
            "testing_requirements": self.testing_requirements,
            "configuration_changes": self.configuration_changes,
            "architectural_impact": self.architectural_impact,
            "estimated_complexity": self.estimated_complexity,
            "risk_assessment": self.risk_assessment,
            "implementation_order": self.implementation_order,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "file_count": self.get_file_count(),
            "has_breaking_changes": self.has_breaking_changes()
        }


@dataclass
class GeneratedCode:
    """Represents generated code with metadata for intelligent placement."""
    content: str
    file_type: CodeType
    dependencies: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    target_path: Optional[str] = None
    purpose: str = ""
    architecture_layer: Optional[str] = None
    requires_barrel_export: bool = False
    part_of_library: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_main_class_name(self) -> Optional[str]:
        """Extract main class name from generated code."""
        import re
        match = re.search(r'class\s+(\w+)', self.content)
        return match.group(1) if match else None
    
    def get_widget_name(self) -> Optional[str]:
        """Extract widget name if this is a widget."""
        if self.file_type in [CodeType.WIDGET, CodeType.SCREEN, CodeType.PAGE]:
            return self.get_main_class_name()
        return None
    
    def requires_state_management(self) -> bool:
        """Check if code requires state management integration."""
        return any(dep in self.content.lower() for dep in ['bloc', 'cubit', 'provider', 'riverpod'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "file_type": self.file_type.value,
            "dependencies": self.dependencies,
            "exports": self.exports,
            "imports": self.imports,
            "target_path": self.target_path,
            "purpose": self.purpose,
            "architecture_layer": self.architecture_layer,
            "requires_barrel_export": self.requires_barrel_export,
            "part_of_library": self.part_of_library,
            "metadata": self.metadata,
            "main_class_name": self.get_main_class_name(),
            "widget_name": self.get_widget_name(),
            "requires_state_management": self.requires_state_management()
        }


@dataclass
class PlacementResult:
    """Result of intelligent code placement operation."""
    placement_id: str
    placed_files: List[str] = field(default_factory=list)
    created_directories: List[str] = field(default_factory=list)
    updated_files: List[str] = field(default_factory=list)
    backup_paths: List[str] = field(default_factory=list)
    barrel_exports_updated: List[str] = field(default_factory=list)
    imports_added: Dict[str, List[str]] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    placement_time: datetime = field(default_factory=datetime.utcnow)
    
    def get_total_files_affected(self) -> int:
        """Get total number of files affected by placement."""
        return len(set(self.placed_files + self.updated_files))
    
    def has_errors(self) -> bool:
        """Check if placement had errors."""
        return not self.success or self.error_message is not None
    
    def get_rollback_info(self) -> Dict[str, Any]:
        """Get information needed for rollback."""
        return {
            "placement_id": self.placement_id,
            "backup_paths": self.backup_paths,
            "created_directories": self.created_directories,
            "placed_files": self.placed_files,
            "updated_files": self.updated_files
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "placement_id": self.placement_id,
            "placed_files": self.placed_files,
            "created_directories": self.created_directories,
            "updated_files": self.updated_files,
            "backup_paths": self.backup_paths,
            "barrel_exports_updated": self.barrel_exports_updated,
            "imports_added": self.imports_added,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "placement_time": self.placement_time.isoformat(),
            "total_files_affected": self.get_total_files_affected(),
            "has_errors": self.has_errors(),
            "rollback_info": self.get_rollback_info()
        }


@dataclass
class UpdateResult:
    """Result of updating related files with imports and exports."""
    update_id: str
    files_updated: List[str] = field(default_factory=list)
    imports_added: Dict[str, List[str]] = field(default_factory=dict)
    exports_modified: Dict[str, List[str]] = field(default_factory=dict)
    barrel_files_created: List[str] = field(default_factory=list)
    part_files_updated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def has_errors(self) -> bool:
        """Check if update had errors."""
        return not self.success or len(self.errors) > 0
    
    def get_total_modifications(self) -> int:
        """Get total number of modifications made."""
        return (len(self.files_updated) + 
                len(self.barrel_files_created) + 
                len(self.part_files_updated))
    
    def get_affected_files(self) -> List[str]:
        """Get all files affected by the update."""
        return list(set(
            self.files_updated + 
            self.barrel_files_created + 
            self.part_files_updated
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "update_id": self.update_id,
            "files_updated": self.files_updated,
            "imports_added": self.imports_added,
            "exports_modified": self.exports_modified,
            "barrel_files_created": self.barrel_files_created,
            "part_files_updated": self.part_files_updated,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
            "rollback_data": self.rollback_data,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
            "has_errors": self.has_errors(),
            "total_modifications": self.get_total_modifications(),
            "affected_files": self.get_affected_files()
        }


@dataclass
class RefactoringRequest:
    """Request for code refactoring operation."""
    refactoring_id: str
    refactoring_type: RefactoringType
    target_files: List[str] = field(default_factory=list)
    new_structure: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    preserve_git_history: bool = True
    backup_before_refactoring: bool = True
    verify_after_refactoring: bool = True
    dry_run: bool = False
    requested_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_primary_target(self) -> Optional[str]:
        """Get the primary target file for refactoring."""
        return self.target_files[0] if self.target_files else None
    
    def is_structural_change(self) -> bool:
        """Check if this refactoring involves structural changes."""
        structural_types = {
            RefactoringType.MOVE_FILE,
            RefactoringType.REORGANIZE_STRUCTURE,
            RefactoringType.RESTRUCTURE_DIRECTORY,
            RefactoringType.REFACTOR_ARCHITECTURE
        }
        return self.refactoring_type in structural_types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "refactoring_id": self.refactoring_id,
            "refactoring_type": self.refactoring_type.value,
            "target_files": self.target_files,
            "new_structure": self.new_structure,
            "options": self.options,
            "description": self.description,
            "preserve_git_history": self.preserve_git_history,
            "backup_before_refactoring": self.backup_before_refactoring,
            "verify_after_refactoring": self.verify_after_refactoring,
            "dry_run": self.dry_run,
            "requested_by": self.requested_by,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "primary_target": self.get_primary_target(),
            "is_structural_change": self.is_structural_change()
        }


@dataclass
class ImpactAnalysis:
    """Analysis of refactoring impact on the project."""
    analysis_id: str
    affected_files: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    dependencies_to_update: List[str] = field(default_factory=list)
    import_changes: Dict[str, List[str]] = field(default_factory=dict)
    test_files_affected: List[str] = field(default_factory=list)
    configuration_changes: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    rollback_complexity: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_total_impact_score(self) -> float:
        """Calculate total impact score based on various factors."""
        score = 0.0
        
        # Base score from number of affected files
        score += len(self.affected_files) * 0.1
        
        # Risk level multiplier
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 2.0,
            RiskLevel.HIGH: 4.0,
            RiskLevel.CRITICAL: 8.0
        }
        score *= risk_multipliers.get(self.risk_level, 2.0)
        
        # Breaking changes penalty
        score += len(self.breaking_changes) * 2.0
        
        # Test file impact
        score += len(self.test_files_affected) * 0.5
        
        return min(score, 100.0)  # Cap at 100
    
    def is_safe_to_proceed(self) -> bool:
        """Determine if it's safe to proceed with refactoring."""
        if self.risk_level == RiskLevel.CRITICAL:
            return False
        if len(self.breaking_changes) > 10:
            return False
        if self.get_total_impact_score() > 50:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "analysis_id": self.analysis_id,
            "affected_files": self.affected_files,
            "breaking_changes": self.breaking_changes,
            "risk_level": self.risk_level.value,
            "dependencies_to_update": self.dependencies_to_update,
            "import_changes": self.import_changes,
            "test_files_affected": self.test_files_affected,
            "configuration_changes": self.configuration_changes,
            "estimated_effort": self.estimated_effort,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "rollback_complexity": self.rollback_complexity,
            "metadata": self.metadata,
            "analyzed_at": self.analyzed_at.isoformat(),
            "total_impact_score": self.get_total_impact_score(),
            "safe_to_proceed": self.is_safe_to_proceed()
        }


@dataclass
class MovementPlan:
    """Detailed plan for file movements and structural changes."""
    plan_id: str
    file_moves: List[Dict[str, str]] = field(default_factory=list)  # old_path -> new_path
    directory_changes: List[Dict[str, str]] = field(default_factory=list)
    dependency_updates: List[Dict[str, Any]] = field(default_factory=list)
    import_updates: Dict[str, List[str]] = field(default_factory=dict)
    export_updates: Dict[str, List[str]] = field(default_factory=dict)
    barrel_file_updates: List[str] = field(default_factory=list)
    git_operations: List[Dict[str, str]] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    rollback_steps: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_move_count(self) -> int:
        """Get total number of file moves."""
        return len(self.file_moves)
    
    def get_affected_directories(self) -> Set[str]:
        """Get all directories affected by the movement plan."""
        directories = set()
        
        for move in self.file_moves:
            old_dir = str(Path(move.get("old_path", "")).parent)
            new_dir = str(Path(move.get("new_path", "")).parent)
            directories.add(old_dir)
            directories.add(new_dir)
        
        for change in self.directory_changes:
            directories.add(change.get("old_path", ""))
            directories.add(change.get("new_path", ""))
        
        return directories
    
    def estimate_duration_minutes(self) -> int:
        """Estimate duration in minutes for executing the plan."""
        base_time = 5  # Base 5 minutes
        move_time = len(self.file_moves) * 2  # 2 minutes per file move
        update_time = len(self.dependency_updates) * 1  # 1 minute per dependency update
        git_time = len(self.git_operations) * 3  # 3 minutes per git operation
        
        return base_time + move_time + update_time + git_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "file_moves": self.file_moves,
            "directory_changes": self.directory_changes,
            "dependency_updates": self.dependency_updates,
            "import_updates": self.import_updates,
            "export_updates": self.export_updates,
            "barrel_file_updates": self.barrel_file_updates,
            "git_operations": self.git_operations,
            "execution_order": self.execution_order,
            "rollback_steps": self.rollback_steps,
            "prerequisites": self.prerequisites,
            "validation_steps": self.validation_steps,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "move_count": self.get_move_count(),
            "affected_directories": list(self.get_affected_directories()),
            "estimated_duration_minutes": self.estimate_duration_minutes()
        }


@dataclass
class RefactoringResult:
    """Result of code refactoring operation."""
    refactoring_id: str
    files_moved: List[Dict[str, str]] = field(default_factory=list)  # old_path -> new_path
    references_updated: List[str] = field(default_factory=list)
    imports_updated: Dict[str, List[str]] = field(default_factory=dict)
    exports_updated: Dict[str, List[str]] = field(default_factory=dict)
    directories_created: List[str] = field(default_factory=list)
    directories_removed: List[str] = field(default_factory=list)
    barrel_files_updated: List[str] = field(default_factory=list)
    git_operations_performed: List[str] = field(default_factory=list)
    backup_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    partial_success: bool = False
    rollback_info: Dict[str, Any] = field(default_factory=dict)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_total_files_affected(self) -> int:
        """Get total number of files affected by refactoring."""
        return len(set(
            [move.get("old_path", "") for move in self.files_moved] +
            [move.get("new_path", "") for move in self.files_moved] +
            self.references_updated +
            list(self.imports_updated.keys()) +
            list(self.exports_updated.keys())
        ))
    
    def has_errors(self) -> bool:
        """Check if refactoring had errors."""
        return not self.success or len(self.errors) > 0
    
    def was_successful(self) -> bool:
        """Check if refactoring was completely successful."""
        return self.success and not self.errors
    
    def needs_rollback(self) -> bool:
        """Determine if rollback is needed."""
        return self.has_errors() and not self.partial_success
    
    def get_rollback_info(self) -> Dict[str, Any]:
        """Get detailed rollback information."""
        return {
            "refactoring_id": self.refactoring_id,
            "backup_paths": self.backup_paths,
            "files_to_restore": [move.get("old_path", "") for move in self.files_moved],
            "git_operations_to_undo": self.git_operations_performed,
            "directories_to_cleanup": self.directories_created,
            "rollback_complexity": len(self.files_moved) + len(self.git_operations_performed)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "refactoring_id": self.refactoring_id,
            "files_moved": self.files_moved,
            "references_updated": self.references_updated,
            "imports_updated": self.imports_updated,
            "exports_updated": self.exports_updated,
            "directories_created": self.directories_created,
            "directories_removed": self.directories_removed,
            "barrel_files_updated": self.barrel_files_updated,
            "git_operations_performed": self.git_operations_performed,
            "backup_paths": self.backup_paths,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
            "partial_success": self.partial_success,
            "rollback_info": self.rollback_info,
            "verification_results": self.verification_results,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
            "total_files_affected": self.get_total_files_affected(),
            "has_errors": self.has_errors(),
            "was_successful": self.was_successful(),
            "needs_rollback": self.needs_rollback(),
            "rollback_details": self.get_rollback_info()
        }


@dataclass
class CodeChanges:
    """Represents code changes to be validated."""
    change_id: str
    changed_files: List[str] = field(default_factory=list)
    change_types: Dict[str, ChangeType] = field(default_factory=dict)
    file_contents: Dict[str, str] = field(default_factory=dict)  # file_path -> content
    context: Optional[ProjectContext] = None
    validate_full_project: bool = False
    include_tests: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_affected_files(self) -> List[str]:
        """Get all files affected by the changes."""
        return list(set(self.changed_files + list(self.file_contents.keys())))
    
    def has_file_type(self, file_extension: str) -> bool:
        """Check if changes include files of specific type."""
        return any(f.endswith(file_extension) for f in self.get_affected_files())
    
    def get_dart_files(self) -> List[str]:
        """Get only Dart files from the changes."""
        return [f for f in self.get_affected_files() if f.endswith('.dart')]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "change_id": self.change_id,
            "changed_files": self.changed_files,
            "change_types": {k: v.value for k, v in self.change_types.items()},
            "file_contents": self.file_contents,
            "context": self.context.to_dict() if self.context else None,
            "validate_full_project": self.validate_full_project,
            "include_tests": self.include_tests,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "affected_files": self.get_affected_files(),
            "dart_files_count": len(self.get_dart_files())
        }


@dataclass
class ValidationIssue:
    """Represents a validation issue found in code."""
    issue_id: str
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggested_fix: Optional[str] = None
    rule_name: Optional[str] = None
    category: Optional[str] = None
    auto_fixable: bool = False
    fix_confidence: float = 0.0
    related_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_blocking(self) -> bool:
        """Check if this issue blocks compilation or execution."""
        blocking_types = {
            IssueType.SYNTAX_ERROR,
            IssueType.TYPE_ERROR,
            IssueType.NULL_SAFETY
        }
        return self.issue_type in blocking_types or self.severity == IssueSeverity.CRITICAL
    
    def can_auto_fix(self) -> bool:
        """Check if this issue can be automatically fixed."""
        return self.auto_fixable and self.fix_confidence > 0.7
    
    def get_priority_score(self) -> float:
        """Calculate priority score for fixing order."""
        severity_scores = {
            IssueSeverity.CRITICAL: 10.0,
            IssueSeverity.ERROR: 8.0,
            IssueSeverity.WARNING: 5.0,
            IssueSeverity.INFO: 2.0
        }
        
        base_score = severity_scores.get(self.severity, 1.0)
        
        # Boost score for blocking issues
        if self.is_blocking():
            base_score *= 2.0
        
        # Boost score for auto-fixable issues
        if self.can_auto_fix():
            base_score *= 1.5
        
        return base_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "code_snippet": self.code_snippet,
            "suggested_fix": self.suggested_fix,
            "rule_name": self.rule_name,
            "category": self.category,
            "auto_fixable": self.auto_fixable,
            "fix_confidence": self.fix_confidence,
            "related_issues": self.related_issues,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
            "is_blocking": self.is_blocking(),
            "can_auto_fix": self.can_auto_fix(),
            "priority_score": self.get_priority_score()
        }


@dataclass
class SyntaxIssue:
    """Specific syntax issue representation."""
    error_type: str
    message: str
    line: int
    column: int
    context: str = ""
    expected: Optional[str] = None
    actual: Optional[str] = None
    
    def to_validation_issue(self, file_path: str, issue_id: str) -> ValidationIssue:
        """Convert to ValidationIssue."""
        return ValidationIssue(
            issue_id=issue_id,
            issue_type=IssueType.SYNTAX_ERROR,
            severity=IssueSeverity.ERROR,
            description=f"{self.error_type}: {self.message}",
            file_path=file_path,
            line_number=self.line,
            column_number=self.column,
            code_snippet=self.context,
            metadata={
                "expected": self.expected,
                "actual": self.actual,
                "error_type": self.error_type
            }
        )


@dataclass
class ArchitectureIssue:
    """Architecture compliance issue."""
    violation_type: str
    pattern_expected: str
    pattern_found: str
    recommendation: str
    impact_level: str = "medium"
    
    def to_validation_issue(self, file_path: str, issue_id: str, line_number: Optional[int] = None) -> ValidationIssue:
        """Convert to ValidationIssue."""
        severity_map = {
            "low": IssueSeverity.INFO,
            "medium": IssueSeverity.WARNING,
            "high": IssueSeverity.ERROR,
            "critical": IssueSeverity.CRITICAL
        }
        
        return ValidationIssue(
            issue_id=issue_id,
            issue_type=IssueType.ARCHITECTURE_VIOLATION,
            severity=severity_map.get(self.impact_level, IssueSeverity.WARNING),
            description=f"Architecture violation: {self.violation_type}",
            file_path=file_path,
            line_number=line_number,
            suggested_fix=self.recommendation,
            metadata={
                "violation_type": self.violation_type,
                "pattern_expected": self.pattern_expected,
                "pattern_found": self.pattern_found,
                "impact_level": self.impact_level
            }
        )


@dataclass
class PerformanceIssue:
    """Performance issue representation."""
    issue_category: str
    description: str
    impact: str
    suggestion: str
    complexity_increase: Optional[str] = None
    
    def to_validation_issue(self, file_path: str, issue_id: str, line_number: Optional[int] = None) -> ValidationIssue:
        """Convert to ValidationIssue."""
        impact_severity = {
            "low": IssueSeverity.INFO,
            "medium": IssueSeverity.WARNING,
            "high": IssueSeverity.ERROR
        }
        
        return ValidationIssue(
            issue_id=issue_id,
            issue_type=IssueType.PERFORMANCE_ISSUE,
            severity=impact_severity.get(self.impact, IssueSeverity.WARNING),
            description=f"Performance: {self.description}",
            file_path=file_path,
            line_number=line_number,
            suggested_fix=self.suggestion,
            category="performance",
            metadata={
                "issue_category": self.issue_category,
                "impact": self.impact,
                "complexity_increase": self.complexity_increase
            }
        )


@dataclass
class StyleFixResult:
    """Result of applying style fixes."""
    files_processed: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    formatting_changes: Dict[str, int] = field(default_factory=dict)  # file -> line count
    lint_fixes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = True
    
    def get_total_changes(self) -> int:
        """Get total number of changes made."""
        return sum(self.formatting_changes.values()) + len(self.lint_fixes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "files_processed": self.files_processed,
            "fixes_applied": self.fixes_applied,
            "formatting_changes": self.formatting_changes,
            "lint_fixes": self.lint_fixes,
            "errors": self.errors,
            "success": self.success,
            "total_changes": self.get_total_changes()
        }


@dataclass
class ValidationResult:
    """Result of comprehensive code validation."""
    validation_id: str
    syntax_issues: List[ValidationIssue] = field(default_factory=list)
    architecture_issues: List[ValidationIssue] = field(default_factory=list)
    performance_issues: List[ValidationIssue] = field(default_factory=list)
    style_issues: List[ValidationIssue] = field(default_factory=list)
    all_issues: List[ValidationIssue] = field(default_factory=list)
    auto_fixed: List[str] = field(default_factory=list)
    files_validated: List[str] = field(default_factory=list)
    validation_passed: bool = True
    blocking_issues_count: int = 0
    total_issues_count: int = 0
    fix_suggestions: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.all_issues = (
            self.syntax_issues + 
            self.architecture_issues + 
            self.performance_issues + 
            self.style_issues
        )
        self.total_issues_count = len(self.all_issues)
        self.blocking_issues_count = len([issue for issue in self.all_issues if issue.is_blocking()])
        self.validation_passed = self.blocking_issues_count == 0
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.all_issues if issue.severity == severity]
    
    def get_issues_by_type(self, issue_type: IssueType) -> List[ValidationIssue]:
        """Get issues filtered by type."""
        return [issue for issue in self.all_issues if issue.issue_type == issue_type]
    
    def get_auto_fixable_issues(self) -> List[ValidationIssue]:
        """Get issues that can be automatically fixed."""
        return [issue for issue in self.all_issues if issue.can_auto_fix()]
    
    def get_issues_by_file(self, file_path: str) -> List[ValidationIssue]:
        """Get issues for a specific file."""
        return [issue for issue in self.all_issues if issue.file_path == file_path]
    
    def get_priority_ordered_issues(self) -> List[ValidationIssue]:
        """Get issues ordered by priority score."""
        return sorted(self.all_issues, key=lambda x: x.get_priority_score(), reverse=True)
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == IssueSeverity.CRITICAL for issue in self.all_issues)
    
    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics."""
        stats = {
            "total_issues": self.total_issues_count,
            "blocking_issues": self.blocking_issues_count,
            "auto_fixable": len(self.get_auto_fixable_issues()),
            "files_with_issues": len(set(issue.file_path for issue in self.all_issues))
        }
        
        for severity in IssueSeverity:
            stats[f"{severity.value}_count"] = len(self.get_issues_by_severity(severity))
        
        for issue_type in IssueType:
            stats[f"{issue_type.value}_count"] = len(self.get_issues_by_type(issue_type))
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validation_id": self.validation_id,
            "syntax_issues": [issue.to_dict() for issue in self.syntax_issues],
            "architecture_issues": [issue.to_dict() for issue in self.architecture_issues],
            "performance_issues": [issue.to_dict() for issue in self.performance_issues],
            "style_issues": [issue.to_dict() for issue in self.style_issues],
            "auto_fixed": self.auto_fixed,
            "files_validated": self.files_validated,
            "validation_passed": self.validation_passed,
            "blocking_issues_count": self.blocking_issues_count,
            "total_issues_count": self.total_issues_count,
            "fix_suggestions": self.fix_suggestions,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
            "validated_at": self.validated_at.isoformat(),
            "has_critical_issues": self.has_critical_issues(),
            "summary_stats": self.get_summary_stats()
        }


@dataclass
class FixResult:
    """Result of fixing validation issues."""
    fix_id: str
    issues_fixed: List[str] = field(default_factory=list)  # issue IDs
    changes_made: Dict[str, List[str]] = field(default_factory=dict)  # file -> changes
    remaining_issues: List[ValidationIssue] = field(default_factory=list)
    new_issues_introduced: List[ValidationIssue] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    backup_paths: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    overall_success: bool = False
    fix_strategies_used: List[str] = field(default_factory=list)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fixed_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.issues_fixed and len(self.issues_fixed) > 0:
            total_attempted = len(self.issues_fixed) + len(self.remaining_issues)
            if total_attempted > 0:
                self.success_rate = len(self.issues_fixed) / total_attempted
        
        self.overall_success = (
            self.success_rate > 0.8 and 
            len(self.new_issues_introduced) == 0 and
            len(self.remaining_issues) == 0
        )
    
    def get_fix_summary(self) -> Dict[str, int]:
        """Get summary of fix results."""
        return {
            "issues_fixed": len(self.issues_fixed),
            "issues_remaining": len(self.remaining_issues),
            "new_issues": len(self.new_issues_introduced),
            "files_modified": len(self.files_modified),
            "changes_made": sum(len(changes) for changes in self.changes_made.values())
        }
    
    def was_successful(self) -> bool:
        """Check if fix operation was successful."""
        return self.overall_success and len(self.new_issues_introduced) == 0
    
    def needs_manual_review(self) -> bool:
        """Check if manual review is needed."""
        return (
            self.success_rate < 0.5 or 
            len(self.new_issues_introduced) > 0 or
            len(self.remaining_issues) > len(self.issues_fixed)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fix_id": self.fix_id,
            "issues_fixed": self.issues_fixed,
            "changes_made": self.changes_made,
            "remaining_issues": [issue.to_dict() for issue in self.remaining_issues],
            "new_issues_introduced": [issue.to_dict() for issue in self.new_issues_introduced],
            "files_modified": self.files_modified,
            "backup_paths": self.backup_paths,
            "success_rate": self.success_rate,
            "overall_success": self.overall_success,
            "fix_strategies_used": self.fix_strategies_used,
            "performance_impact": self.performance_impact,
            "verification_results": self.verification_results,
            "metadata": self.metadata,
            "fixed_at": self.fixed_at.isoformat(),
            "fix_summary": self.get_fix_summary(),
            "was_successful": self.was_successful(),
            "needs_manual_review": self.needs_manual_review()
        }


@dataclass
class PackageInfo:
    """Information about a Flutter/Dart package."""
    name: str
    version: str
    description: str = ""
    platform_support: Dict[str, bool] = field(default_factory=dict)  # ios, android, web, desktop
    alternatives: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dev_dependency: bool = False
    required_permissions: List[str] = field(default_factory=list)
    configuration_required: bool = False
    bundle_size_impact: Optional[str] = None  # 'small', 'medium', 'large'
    popularity_score: Optional[float] = None
    maintenance_score: Optional[float] = None
    feature_coverage: List[str] = field(default_factory=list)
    initialization_code: Optional[str] = None
    import_statement: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def supports_platform(self, platform: str) -> bool:
        """Check if package supports a specific platform."""
        return self.platform_support.get(platform, True)  # Default to True if unknown
    
    def get_version_constraint(self) -> str:
        """Get recommended version constraint for pubspec.yaml."""
        # Use caret constraint for semantic versioning
        if self.version and self.version != "latest":
            return f"^{self.version}"
        return "^1.0.0"  # Default constraint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "platform_support": self.platform_support,
            "alternatives": self.alternatives,
            "dependencies": self.dependencies,
            "dev_dependency": self.dev_dependency,
            "required_permissions": self.required_permissions,
            "configuration_required": self.configuration_required,
            "bundle_size_impact": self.bundle_size_impact,
            "popularity_score": self.popularity_score,
            "maintenance_score": self.maintenance_score,
            "feature_coverage": self.feature_coverage,
            "initialization_code": self.initialization_code,
            "import_statement": self.import_statement,
            "metadata": self.metadata
        }


@dataclass
class CompatibilityReport:
    """Report on package version compatibility."""
    compatible_packages: List[str] = field(default_factory=list)
    incompatible_packages: List[str] = field(default_factory=list)
    conflict_details: Dict[str, str] = field(default_factory=dict)
    suggested_resolutions: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    overall_compatibility: bool = True
    flutter_version_required: Optional[str] = None
    dart_version_required: Optional[str] = None
    
    def has_conflicts(self) -> bool:
        """Check if there are any compatibility conflicts."""
        return len(self.incompatible_packages) > 0 or not self.overall_compatibility
    
    def get_resolution_plan(self) -> Dict[str, Any]:
        """Get a plan to resolve compatibility issues."""
        return {
            "conflicts_found": len(self.incompatible_packages),
            "resolutions_available": len(self.suggested_resolutions),
            "action_required": self.has_conflicts(),
            "suggested_actions": list(self.suggested_resolutions.values()),
            "warnings": self.warnings
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "compatible_packages": self.compatible_packages,
            "incompatible_packages": self.incompatible_packages,
            "conflict_details": self.conflict_details,
            "suggested_resolutions": self.suggested_resolutions,
            "warnings": self.warnings,
            "overall_compatibility": self.overall_compatibility,
            "flutter_version_required": self.flutter_version_required,
            "dart_version_required": self.dart_version_required,
            "has_conflicts": self.has_conflicts(),
            "resolution_plan": self.get_resolution_plan()
        }


@dataclass
class ConfigurationResult:
    """Result of package configuration setup."""
    package_name: str
    configuration_successful: bool
    files_modified: List[str] = field(default_factory=list)
    configuration_steps: List[str] = field(default_factory=list)
    initialization_code_added: bool = False
    permissions_added: List[str] = field(default_factory=list)
    platform_specific_setup: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    def was_successful(self) -> bool:
        """Check if configuration was successful."""
        return self.configuration_successful and len(self.errors) == 0
    
    def needs_manual_setup(self) -> bool:
        """Check if manual setup is required."""
        return len(self.next_steps) > 0 or len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "package_name": self.package_name,
            "configuration_successful": self.configuration_successful,
            "files_modified": self.files_modified,
            "configuration_steps": self.configuration_steps,
            "initialization_code_added": self.initialization_code_added,
            "permissions_added": self.permissions_added,
            "platform_specific_setup": self.platform_specific_setup,
            "errors": self.errors,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
            "was_successful": self.was_successful(),
            "needs_manual_setup": self.needs_manual_setup()
        }


@dataclass
class DependencyUpdate:
    """Result of dependency management operation."""
    update_id: str
    added_packages: Dict[str, PackageInfo] = field(default_factory=dict)
    updated_versions: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # package -> (old_version, new_version)
    removed_packages: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, ConfigurationResult] = field(default_factory=dict)
    verification_results: Dict[str, bool] = field(default_factory=dict)
    pubspec_backup_path: Optional[str] = None
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    bundle_size_impact: Optional[str] = None
    build_time_impact: Optional[str] = None
    compatibility_report: Optional[CompatibilityReport] = None
    next_steps: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_package_count_changes(self) -> Dict[str, int]:
        """Get summary of package count changes."""
        return {
            "added": len(self.added_packages),
            "updated": len(self.updated_versions),
            "removed": len(self.removed_packages),
            "configured": len([r for r in self.configuration_changes.values() if r.was_successful()])
        }
    
    def has_issues(self) -> bool:
        """Check if there are any issues with the update."""
        return len(self.errors) > 0 or not self.success
    
    def needs_manual_intervention(self) -> bool:
        """Check if manual intervention is needed."""
        return (
            len(self.next_steps) > 0 or
            any(not result for result in self.verification_results.values()) or
            any(config.needs_manual_setup() for config in self.configuration_changes.values())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "update_id": self.update_id,
            "added_packages": {name: pkg.to_dict() for name, pkg in self.added_packages.items()},
            "updated_versions": self.updated_versions,
            "removed_packages": self.removed_packages,
            "configuration_changes": {name: config.to_dict() for name, config in self.configuration_changes.items()},
            "verification_results": self.verification_results,
            "pubspec_backup_path": self.pubspec_backup_path,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "bundle_size_impact": self.bundle_size_impact,
            "build_time_impact": self.build_time_impact,
            "compatibility_report": self.compatibility_report.to_dict() if self.compatibility_report else None,
            "next_steps": self.next_steps,
            "updated_at": self.updated_at.isoformat(),
            "package_count_changes": self.get_package_count_changes(),
            "has_issues": self.has_issues(),
            "needs_manual_intervention": self.needs_manual_intervention()
        }


@dataclass
class DependencyOptimization:
    """Result of dependency optimization operation."""
    optimization_id: str
    removed_packages: List[str] = field(default_factory=list)
    updated_packages: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # package -> (old_version, new_version)
    suggested_alternatives: Dict[str, str] = field(default_factory=dict)  # current_package -> suggested_alternative
    size_savings: Dict[str, Any] = field(default_factory=dict)  # bundle_size_mb, assets_reduced, etc.
    build_improvements: Dict[str, Any] = field(default_factory=dict)  # build_time_saved, dependency_count, etc.
    unused_dependencies: List[str] = field(default_factory=list)
    outdated_dependencies: Dict[str, str] = field(default_factory=dict)  # package -> latest_version
    security_improvements: List[str] = field(default_factory=list)
    performance_gains: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    applied_optimizations: List[str] = field(default_factory=list)
    skipped_optimizations: Dict[str, str] = field(default_factory=dict)  # optimization -> reason
    verification_passed: bool = True
    backup_created: bool = False
    backup_path: Optional[str] = None
    optimized_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            "packages_removed": len(self.removed_packages),
            "packages_updated": len(self.updated_packages),
            "alternatives_suggested": len(self.suggested_alternatives),
            "optimizations_applied": len(self.applied_optimizations),
            "optimizations_skipped": len(self.skipped_optimizations),
            "estimated_size_savings_mb": self.size_savings.get("bundle_size_mb", 0),
            "estimated_build_time_saved_seconds": self.build_improvements.get("build_time_saved_seconds", 0)
        }
    
    def has_significant_improvements(self) -> bool:
        """Check if optimization provides significant improvements."""
        size_savings = self.size_savings.get("bundle_size_mb", 0)
        build_time_saved = self.build_improvements.get("build_time_saved_seconds", 0)
        packages_optimized = len(self.removed_packages) + len(self.updated_packages)
        
        return (
            size_savings > 1.0 or  # More than 1MB saved
            build_time_saved > 10 or  # More than 10 seconds saved
            packages_optimized > 0  # Any packages optimized
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "optimization_id": self.optimization_id,
            "removed_packages": self.removed_packages,
            "updated_packages": self.updated_packages,
            "suggested_alternatives": self.suggested_alternatives,
            "size_savings": self.size_savings,
            "build_improvements": self.build_improvements,
            "unused_dependencies": self.unused_dependencies,
            "outdated_dependencies": self.outdated_dependencies,
            "security_improvements": self.security_improvements,
            "performance_gains": self.performance_gains,
            "recommendations": self.recommendations,
            "applied_optimizations": self.applied_optimizations,
            "skipped_optimizations": self.skipped_optimizations,
            "verification_passed": self.verification_passed,
            "backup_created": self.backup_created,
            "backup_path": self.backup_path,
            "optimized_at": self.optimized_at.isoformat(),
            "optimization_summary": self.get_optimization_summary(),
            "has_significant_improvements": self.has_significant_improvements()
        }


class ChangeType(Enum):
    """Types of code changes for hot reload analysis."""
    WIDGET_UPDATE = "widget_update"
    STATE_CHANGE = "state_change"
    FUNCTION_UPDATE = "function_update"
    CLASS_MODIFICATION = "class_modification"
    IMPORT_CHANGE = "import_change"
    CONSTANT_UPDATE = "constant_update"
    CONSTRUCTOR_CHANGE = "constructor_change"
    BUILD_METHOD_CHANGE = "build_method_change"
    LIFECYCLE_METHOD_CHANGE = "lifecycle_method_change"
    GLOBAL_VARIABLE_CHANGE = "global_variable_change"
    NEW_FILE = "new_file"
    FILE_DELETION = "file_deletion"
    ANNOTATION_CHANGE = "annotation_change"


class StateImpact(Enum):
    """Impact on widget state during hot reload."""
    NONE = "none"  # No state impact
    PRESERVES = "preserves"  # State is preserved
    RESETS = "resets"  # State will be reset
    UNKNOWN = "unknown"  # Impact cannot be determined


class ReloadOutcome(Enum):
    """Possible outcomes of hot reload operation."""
    SUCCESS = "success"
    FAILED_RESTART_REQUIRED = "failed_restart_required"
    FAILED_COMPILATION_ERROR = "failed_compilation_error"
    FAILED_RUNTIME_ERROR = "failed_runtime_error"
    SKIPPED_INCOMPATIBLE = "skipped_incompatible"


@dataclass
class CodeChange:
    """Information about a code change for hot reload analysis."""
    change_id: str
    file_path: str
    change_type: ChangeType
    affected_widgets: List[str] = field(default_factory=list)
    state_impact: StateImpact = StateImpact.UNKNOWN
    line_start: int = 0
    line_end: int = 0
    content_before: Optional[str] = None
    content_after: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_hot_reload_friendly(self) -> bool:
        """Check if this change is typically hot reload friendly."""
        friendly_changes = [
            ChangeType.WIDGET_UPDATE,
            ChangeType.FUNCTION_UPDATE,
            ChangeType.CONSTANT_UPDATE,
            ChangeType.BUILD_METHOD_CHANGE
        ]
        return self.change_type in friendly_changes
    
    def requires_restart(self) -> bool:
        """Check if this change requires a full restart."""
        restart_required = [
            ChangeType.CONSTRUCTOR_CHANGE,
            ChangeType.GLOBAL_VARIABLE_CHANGE,
            ChangeType.NEW_FILE,
            ChangeType.FILE_DELETION,
            ChangeType.IMPORT_CHANGE
        ]
        return self.change_type in restart_required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "change_id": self.change_id,
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "affected_widgets": self.affected_widgets,
            "state_impact": self.state_impact.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_before": self.content_before,
            "content_after": self.content_after,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_hot_reload_friendly": self.is_hot_reload_friendly(),
            "requires_restart": self.requires_restart()
        }


@dataclass
class ReloadCompatibility:
    """Prediction about hot reload compatibility for a set of changes."""
    can_hot_reload: bool
    requires_restart: bool
    problematic_changes: List[str] = field(default_factory=list)
    compatibility_score: float = 1.0  # 0.0 to 1.0
    estimated_success_rate: float = 1.0
    state_preservation_expected: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    batch_optimization_possible: bool = True
    
    def get_reload_strategy(self) -> str:
        """Get recommended reload strategy."""
        if self.requires_restart:
            return "full_restart"
        elif self.can_hot_reload and self.compatibility_score > 0.8:
            return "hot_reload"
        elif self.can_hot_reload and self.batch_optimization_possible:
            return "batched_hot_reload"
        else:
            return "manual_verification_needed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "can_hot_reload": self.can_hot_reload,
            "requires_restart": self.requires_restart,
            "problematic_changes": self.problematic_changes,
            "compatibility_score": self.compatibility_score,
            "estimated_success_rate": self.estimated_success_rate,
            "state_preservation_expected": self.state_preservation_expected,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "batch_optimization_possible": self.batch_optimization_possible,
            "reload_strategy": self.get_reload_strategy()
        }


@dataclass
class ReloadFailure:
    """Information about a hot reload failure."""
    failure_id: str
    failure_type: str
    error_message: str
    affected_files: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    can_auto_recover: bool = False
    recovery_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "failure_id": self.failure_id,
            "failure_type": self.failure_type,
            "error_message": self.error_message,
            "affected_files": self.affected_files,
            "stack_trace": self.stack_trace,
            "suggested_fixes": self.suggested_fixes,
            "can_auto_recover": self.can_auto_recover,
            "recovery_confidence": self.recovery_confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RecoveryPlan:
    """Plan for recovering from hot reload failures."""
    plan_id: str
    recovery_steps: List[str] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    rollback_changes: List[str] = field(default_factory=list)
    estimated_success_rate: float = 0.0
    requires_manual_intervention: bool = True
    automated_fixes_available: bool = False
    
    def can_auto_execute(self) -> bool:
        """Check if recovery plan can be executed automatically."""
        return (
            self.automated_fixes_available and 
            not self.requires_manual_intervention and 
            self.estimated_success_rate > 0.7
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "recovery_steps": self.recovery_steps,
            "files_to_modify": self.files_to_modify,
            "rollback_changes": self.rollback_changes,
            "estimated_success_rate": self.estimated_success_rate,
            "requires_manual_intervention": self.requires_manual_intervention,
            "automated_fixes_available": self.automated_fixes_available,
            "can_auto_execute": self.can_auto_execute()
        }


@dataclass
class DevelopmentSession:
    """Information about an active development session."""
    session_id: str
    active_files: List[str] = field(default_factory=list)
    running_processes: List[str] = field(default_factory=list)
    reload_history: List[Dict[str, Any]] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    hot_reload_enabled: bool = True
    target_device: Optional[str] = None
    flutter_mode: str = "debug"  # debug, profile, release
    watched_directories: List[str] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if session is currently active."""
        if not self.running_processes:
            return False
        
        # Consider session inactive if no activity for more than 30 minutes
        inactive_threshold = datetime.utcnow() - timedelta(minutes=30)
        return self.last_activity > inactive_threshold
    
    def get_session_duration(self) -> timedelta:
        """Get total session duration."""
        end_time = self.last_activity if not self.is_active() else datetime.utcnow()
        return end_time - self.session_start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "active_files": self.active_files,
            "running_processes": self.running_processes,
            "reload_history": self.reload_history,
            "session_start": self.session_start.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "hot_reload_enabled": self.hot_reload_enabled,
            "target_device": self.target_device,
            "flutter_mode": self.flutter_mode,
            "watched_directories": self.watched_directories,
            "excluded_patterns": self.excluded_patterns,
            "performance_metrics": self.performance_metrics,
            "is_active": self.is_active(),
            "session_duration_minutes": self.get_session_duration().total_seconds() / 60
        }


@dataclass
class HotReloadExperience:
    """Results and metrics from a hot reload development session."""
    experience_id: str
    session_id: str
    successful_reloads: int = 0
    failed_reloads: int = 0
    restart_count: int = 0
    optimizations_applied: List[str] = field(default_factory=list)
    productivity_metrics: Dict[str, float] = field(default_factory=dict)
    reload_times: List[float] = field(default_factory=list)  # in seconds
    failure_patterns: List[str] = field(default_factory=list)
    success_patterns: List[str] = field(default_factory=list)
    code_changes_processed: int = 0
    state_preservation_rate: float = 0.0
    developer_satisfaction_score: Optional[float] = None
    recommendations_for_improvement: List[str] = field(default_factory=list)
    experience_start: datetime = field(default_factory=datetime.utcnow)
    experience_end: Optional[datetime] = None
    
    def get_success_rate(self) -> float:
        """Calculate hot reload success rate."""
        total_attempts = self.successful_reloads + self.failed_reloads
        if total_attempts == 0:
            return 1.0
        return self.successful_reloads / total_attempts
    
    def get_average_reload_time(self) -> float:
        """Get average hot reload time in seconds."""
        if not self.reload_times:
            return 0.0
        return sum(self.reload_times) / len(self.reload_times)
    
    def get_productivity_score(self) -> float:
        """Calculate overall productivity score (0-100)."""
        success_rate = self.get_success_rate()
        avg_reload_time = self.get_average_reload_time()
        
        # Base score from success rate
        score = success_rate * 70
        
        # Penalty for slow reloads (ideal is under 1 second)
        if avg_reload_time > 1.0:
            time_penalty = min(20, (avg_reload_time - 1.0) * 10)
            score -= time_penalty
        
        # Bonus for optimizations
        optimization_bonus = min(10, len(self.optimizations_applied) * 2)
        score += optimization_bonus
        
        # Penalty for too many restarts
        restart_penalty = min(10, self.restart_count * 2)
        score -= restart_penalty
        
        return max(0, min(100, score))
    
    def needs_improvement(self) -> bool:
        """Check if the development experience needs improvement."""
        return (
            self.get_success_rate() < 0.8 or
            self.get_average_reload_time() > 2.0 or
            self.restart_count > 5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experience_id": self.experience_id,
            "session_id": self.session_id,
            "successful_reloads": self.successful_reloads,
            "failed_reloads": self.failed_reloads,
            "restart_count": self.restart_count,
            "optimizations_applied": self.optimizations_applied,
            "productivity_metrics": self.productivity_metrics,
            "reload_times": self.reload_times,
            "failure_patterns": self.failure_patterns,
            "success_patterns": self.success_patterns,
            "code_changes_processed": self.code_changes_processed,
            "state_preservation_rate": self.state_preservation_rate,
            "developer_satisfaction_score": self.developer_satisfaction_score,
            "recommendations_for_improvement": self.recommendations_for_improvement,
            "experience_start": self.experience_start.isoformat(),
            "experience_end": self.experience_end.isoformat() if self.experience_end else None,
            "success_rate": self.get_success_rate(),
            "average_reload_time_seconds": self.get_average_reload_time(),
            "productivity_score": self.get_productivity_score(),
            "needs_improvement": self.needs_improvement()
        }


# Feature-Complete Generation Models

class FeatureType(Enum):
    """Types of features that can be generated."""
    AUTHENTICATION = "authentication"
    CRUD_OPERATIONS = "crud_operations"
    USER_INTERFACE = "user_interface"
    DATA_VISUALIZATION = "data_visualization"
    PAYMENT_INTEGRATION = "payment_integration"
    SOCIAL_FEATURES = "social_features"
    NOTIFICATION_SYSTEM = "notification_system"
    FILE_MANAGEMENT = "file_management"
    SEARCH_FUNCTIONALITY = "search_functionality"
    MESSAGING_SYSTEM = "messaging_system"
    ANALYTICS_INTEGRATION = "analytics_integration"
    OFFLINE_SUPPORT = "offline_support"
    CUSTOM_BUSINESS_LOGIC = "custom_business_logic"


class ComponentType(Enum):
    """Types of components in a feature implementation."""
    MODEL = "model"
    REPOSITORY = "repository"
    SERVICE = "service"
    BLOC_CUBIT = "bloc_cubit"
    WIDGET = "widget"
    SCREEN = "screen"
    ROUTE = "route"
    TEST = "test"
    CONFIGURATION = "configuration"
    DEPENDENCY_INJECTION = "dependency_injection"


class ArchitectureLayer(Enum):
    """Architecture layers for feature organization."""
    PRESENTATION = "presentation"
    DOMAIN = "domain"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    SHARED = "shared"


class StylePattern(Enum):
    """Code style patterns that can be adapted."""
    NAMING_CONVENTION = "naming_convention"
    FILE_STRUCTURE = "file_structure"
    IMPORT_ORGANIZATION = "import_organization"
    WIDGET_COMPOSITION = "widget_composition"
    STATE_MANAGEMENT = "state_management"
    ERROR_HANDLING = "error_handling"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


class StyleComplexity(Enum):
    """Complexity levels for style patterns."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class UIRequirement:
    """Represents UI requirements for a feature."""
    screen_name: str
    widget_types: List[str]
    layout_pattern: str
    navigation_flow: List[str]
    state_management_approach: str
    styling_requirements: Dict[str, Any]
    responsive_behavior: Dict[str, str]
    accessibility_requirements: List[str]
    animations: List[str] = field(default_factory=list)
    custom_components: List[str] = field(default_factory=list)


@dataclass
class DataRequirement:
    """Represents data requirements for a feature."""
    models: List[str]
    relationships: Dict[str, List[str]]
    persistence_strategy: str
    caching_requirements: List[str]
    validation_rules: Dict[str, List[str]]
    transformation_needs: Dict[str, str]
    migration_requirements: List[str] = field(default_factory=list)
    indexing_strategy: Dict[str, str] = field(default_factory=dict)


@dataclass
class BusinessLogicRequirement:
    """Represents business logic requirements for a feature."""
    use_cases: List[str]
    business_rules: Dict[str, List[str]]
    workflows: List[Dict[str, Any]]
    integration_points: List[str]
    security_requirements: List[str]
    performance_requirements: Dict[str, str]
    error_scenarios: List[str]
    notification_triggers: List[str] = field(default_factory=list)


@dataclass
class APIRequirement:
    """Represents API requirements for a feature."""
    endpoints: List[Dict[str, Any]]
    authentication_method: str
    data_formats: List[str]
    error_handling_strategy: str
    caching_strategy: str
    offline_behavior: str
    rate_limiting: Dict[str, Any]
    security_headers: List[str] = field(default_factory=list)


@dataclass
class TestingRequirement:
    """Represents testing requirements for a feature."""
    unit_test_coverage: float
    widget_tests: List[str]
    integration_tests: List[str]
    performance_tests: List[str]
    accessibility_tests: List[str]
    mock_strategies: Dict[str, str]
    test_data_requirements: List[str]
    automation_level: str = "high"


@dataclass
class FeatureSpecification:
    """Complete specification for feature generation."""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    description: str
    ui_requirements: UIRequirement
    data_requirements: DataRequirement
    business_logic_requirements: BusinessLogicRequirement
    api_requirements: Optional[APIRequirement]
    testing_requirements: TestingRequirement
    architecture_constraints: Dict[str, Any]
    dependencies: List[str]
    priority: str
    timeline: Dict[str, str]
    acceptance_criteria: List[str]
    performance_targets: Dict[str, float] = field(default_factory=dict)
    security_requirements: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneratedComponent:
    """Represents a generated code component."""
    component_id: str
    component_type: ComponentType
    file_path: str
    content: str
    dependencies: List[str]
    exports: List[str]
    architecture_layer: ArchitectureLayer
    test_file_path: Optional[str] = None
    documentation: str = ""
    complexity_score: float = 0.0
    performance_considerations: List[str] = field(default_factory=list)


@dataclass
class ImplementationPlan:
    """Plan for implementing a complete feature."""
    plan_id: str
    feature_id: str
    components: List[Dict[str, Any]]
    dependency_graph: Dict[str, List[str]]
    implementation_order: List[str]
    integration_points: List[Dict[str, Any]]
    risk_assessment: Dict[str, str]
    estimated_effort: Dict[str, float]
    architecture_decisions: List[str]
    validation_checkpoints: List[str] = field(default_factory=list)


@dataclass
class FeatureImplementation:
    """Complete implementation result for a feature."""
    implementation_id: str
    feature_specification: FeatureSpecification
    implementation_plan: ImplementationPlan
    generated_components: List[GeneratedComponent]
    wiring_configuration: Dict[str, Any]
    routing_setup: Dict[str, Any]
    dependency_injection_setup: Dict[str, Any]
    testing_suite: Dict[str, Any]
    documentation: str
    validation_results: List[ValidationResult]
    performance_metrics: Dict[str, float]
    implementation_time: float
    success_indicators: Dict[str, bool]
    follow_up_tasks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_implementation_quality_score(self) -> float:
        """Calculate overall implementation quality score."""
        score = 0.0
        total_weight = 0.0
        
        # Component quality (30%)
        if self.generated_components:
            component_scores = [comp.complexity_score for comp in self.generated_components if comp.complexity_score > 0]
            if component_scores:
                avg_component_score = sum(component_scores) / len(component_scores)
                score += avg_component_score * 0.3
                total_weight += 0.3
        
        # Validation results (25%)
        if self.validation_results:
            passed_validations = sum(1 for result in self.validation_results if result.is_valid)
            validation_score = (passed_validations / len(self.validation_results)) * 100
            score += validation_score * 0.25
            total_weight += 0.25
        
        # Success indicators (25%)
        if self.success_indicators:
            success_rate = sum(self.success_indicators.values()) / len(self.success_indicators)
            score += success_rate * 100 * 0.25
            total_weight += 0.25
        
        # Performance metrics (20%)
        if self.performance_metrics:
            # Normalize performance metrics (assuming higher values are better)
            performance_score = min(100, sum(self.performance_metrics.values()) / len(self.performance_metrics))
            score += performance_score * 0.2
            total_weight += 0.2
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_completion_status(self) -> str:
        """Get the completion status of the implementation."""
        if not self.success_indicators:
            return "incomplete"
        
        success_rate = sum(self.success_indicators.values()) / len(self.success_indicators)
        
        if success_rate >= 0.95:
            return "complete"
        elif success_rate >= 0.8:
            return "mostly_complete"
        elif success_rate >= 0.5:
            return "partially_complete"
        else:
            return "incomplete"


# Code Style Adaptation Models

@dataclass
class StyleRule:
    """Represents a code style rule."""
    rule_id: str
    pattern: StylePattern
    description: str
    example_correct: str
    example_incorrect: str
    enforcement_level: str  # "error", "warning", "suggestion"
    context_conditions: List[str]
    auto_fixable: bool = True
    complexity: StyleComplexity = StyleComplexity.SIMPLE


@dataclass
class StyleAnalysis:
    """Analysis of existing code style patterns."""
    project_path: str
    analyzed_files: List[str]
    discovered_patterns: Dict[StylePattern, List[str]]
    consistency_scores: Dict[StylePattern, float]
    common_violations: List[Dict[str, Any]]
    recommended_rules: List[StyleRule]
    confidence_scores: Dict[str, float]
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CodeStyle:
    """Comprehensive code style definition."""
    style_id: str
    style_name: str
    description: str
    rules: List[StyleRule]
    naming_conventions: Dict[str, str]
    file_organization: Dict[str, List[str]]
    architecture_preferences: Dict[str, str]
    formatting_preferences: Dict[str, Any]
    linting_configuration: Dict[str, Any]
    documentation_standards: Dict[str, str]
    testing_standards: Dict[str, str]
    performance_guidelines: List[str] = field(default_factory=list)
    security_guidelines: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_style_complexity(self) -> StyleComplexity:
        """Determine the overall complexity of this style."""
        complex_rules = sum(1 for rule in self.rules if rule.complexity in [StyleComplexity.COMPLEX, StyleComplexity.ENTERPRISE])
        
        if complex_rules > len(self.rules) * 0.5:
            return StyleComplexity.ENTERPRISE
        elif complex_rules > len(self.rules) * 0.3:
            return StyleComplexity.COMPLEX
        elif complex_rules > len(self.rules) * 0.1:
            return StyleComplexity.MODERATE
        else:
            return StyleComplexity.SIMPLE


@dataclass
class StyleApplication:
    """Result of applying a style rule to code."""
    rule_id: str
    file_path: str
    original_code: str
    modified_code: str
    changes_applied: List[str]
    confidence_score: float
    manual_review_needed: bool
    application_time: float
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class StyleAdaptation:
    """Result of adapting code to a target style."""
    adaptation_id: str
    source_files: List[str]
    target_style: CodeStyle
    style_analysis: StyleAnalysis
    applications: List[StyleApplication]
    adaptation_summary: Dict[str, Any]
    quality_improvements: Dict[str, float]
    consistency_improvements: Dict[str, float]
    adaptation_time: float
    success_rate: float
    manual_review_items: List[str] = field(default_factory=list)
    follow_up_recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_adaptation_quality_score(self) -> float:
        """Calculate the quality of the style adaptation."""
        if not self.applications:
            return 0.0
        
        # Base score from success rate
        base_score = self.success_rate * 60
        
        # Bonus for confidence
        avg_confidence = sum(app.confidence_score for app in self.applications) / len(self.applications)
        confidence_bonus = avg_confidence * 25
        
        # Bonus for consistency improvements
        consistency_bonus = sum(self.consistency_improvements.values()) / len(self.consistency_improvements) * 15 if self.consistency_improvements else 0
        
        # Penalty for manual review items
        manual_review_penalty = min(10, len(self.manual_review_items) * 2)
        
        return max(0, min(100, base_score + confidence_bonus + consistency_bonus - manual_review_penalty))


# Performance Benchmarking Models

@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    baseline_value: Optional[float] = None
    
    def get_improvement_percentage(self) -> Optional[float]:
        """Calculate improvement percentage from baseline."""
        if self.baseline_value is None or self.baseline_value == 0:
            return None
        return ((self.value - self.baseline_value) / self.baseline_value) * 100


@dataclass
class BenchmarkResult:
    """Complete benchmark results for agent performance."""
    benchmark_id: str
    agent_version: str
    test_suite: str
    metrics: List[PerformanceMetric]
    execution_time: float
    resource_usage: Dict[str, float]
    success_indicators: Dict[str, bool]
    comparison_baseline: Optional[str] = None
    environment_info: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_overall_performance_score(self) -> float:
        """Calculate overall performance score."""
        if not self.metrics:
            return 0.0
        
        # Weight different metrics appropriately
        weighted_scores = []
        
        for metric in self.metrics:
            improvement = metric.get_improvement_percentage()
            if improvement is not None:
                # Convert improvement to a 0-100 score
                score = min(100, max(0, 50 + improvement))
                weighted_scores.append(score)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
