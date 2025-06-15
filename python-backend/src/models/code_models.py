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
