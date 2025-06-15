"""
Code Models for FlutterSwarm Multi-Agent System.

This module defines data structures for code generation, understanding,
and project-aware development patterns.
"""
from dataclasses import dataclass, field
from datetime import datetime
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
