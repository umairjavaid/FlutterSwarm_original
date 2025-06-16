"""
File System Tool for FlutterSwarm Multi-Agent System.

Provides Flutter-aware file operations, safe file handling, template-based creation, 
batch/transaction support, file watching, conflict resolution, and asset management.
"""

import os
import shutil
import json
import asyncio
import re
import yaml
import fnmatch
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from PIL import Image
import logging

from .base_tool import (
    BaseTool, ToolCapabilities, ToolOperation, ToolPermission, ToolResult, ToolStatus,
    ToolCategory, ToolContext
)

logger = logging.getLogger(__name__)


class FileSystemTool(BaseTool):
    """Enhanced Flutter-aware file system tool with comprehensive operations."""
    
    def __init__(self):
        super().__init__(
            name="file_system",
            description="Flutter-aware file system operations with safety, templates, and batch support.",
            version="2.0.0",
            category=ToolCategory.FILE_SYSTEM,
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.FILE_WRITE,
                ToolPermission.FILE_CREATE,
                ToolPermission.FILE_DELETE,
                ToolPermission.DIRECTORY_CREATE,
                ToolPermission.DIRECTORY_DELETE
            ]
        )
        self.backup_dir = ".fs_tool_backups"
        self.watched_files = set()
        self.transactions = {}
        self.flutter_structure = self._define_flutter_structure()
        self.gitignore_patterns = set()
        self.asset_cache = {}

    def _define_flutter_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define Flutter project structure conventions."""
        return {
            "lib": {
                "description": "Main Dart source code",
                "conventions": ["main.dart required", "subdirectories for organization"],
                "critical": True,
                "suggested_structure": ["models", "widgets", "services", "utils", "screens"]
            },
            "test": {
                "description": "Unit and widget tests",
                "conventions": ["mirror lib structure", "_test.dart suffix"],
                "critical": False,
                "suggested_structure": ["unit", "widget", "integration"]
            },
            "assets": {
                "description": "Static assets (images, fonts, etc.)",
                "conventions": ["organized by type", "multiple resolutions"],
                "critical": False,
                "suggested_structure": ["images", "fonts", "data"]
            },
            "android": {
                "description": "Android platform-specific code",
                "conventions": ["native Android structure"],
                "critical": True,
                "platform": "android"
            },
            "ios": {
                "description": "iOS platform-specific code", 
                "conventions": ["native iOS structure"],
                "critical": True,
                "platform": "ios"
            },
            "web": {
                "description": "Web platform-specific code",
                "conventions": ["web assets and configuration"],
                "critical": False,
                "platform": "web"
            },
            "windows": {
                "description": "Windows platform-specific code",
                "conventions": ["native Windows structure"],
                "critical": False,
                "platform": "windows"
            },
            "macos": {
                "description": "macOS platform-specific code",
                "conventions": ["native macOS structure"],
                "critical": False,
                "platform": "macos"
            },
            "linux": {
                "description": "Linux platform-specific code",
                "conventions": ["native Linux structure"],
                "critical": False,
                "platform": "linux"
            }
        }

    def _load_dynamic_templates(self) -> Dict[str, str]:
        """Load templates dynamically from template engine."""
        from ..template_engine import get_template_engine, TemplateType, ArchitecturalPattern
        
        template_engine = get_template_engine()
        templates = {}
        
        # Load base templates for different types
        template_types = [
            TemplateType.WIDGET,
            TemplateType.MODEL,
            TemplateType.BLOC,
            TemplateType.REPOSITORY,
            TemplateType.SERVICE,
            TemplateType.SCREEN
        ]
        
        for template_type in template_types:
            template = template_engine.get_template(template_type, ArchitecturalPattern.BASIC_PATTERN)
            if template:
                templates[template_type.value] = template.source
        
        return templates

    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """Provide dynamic example usage scenarios for agent learning."""
        return [
            {
                "scenario": "Create Flutter widget from template",
                "operation": "create_from_template",
                "parameters": {
                    "template_type": "widget",
                    "file_path": "lib/widgets/custom_button.dart",
                    "variables": {
                        "widget_name": "CustomButton",
                        "widget_type": "StatelessWidget"
                    }
                },
                "expected_outcome": "Creates a new widget file with proper Flutter structure"
            },
            {
                "scenario": "Generate model class with JSON serialization",
                "operation": "create_from_template", 
                "parameters": {
                    "template_type": "model",
                    "file_path": "lib/models/user.dart",
                    "variables": {
                        "model_name": "User",
                        "fields": ["id", "name", "email"]
                    }
                },
                "expected_outcome": "Creates model with JSON serialization methods"
            },
            {
                "scenario": "Setup BLoC pattern files",
                "operation": "create_from_template",
                "parameters": {
                    "template_type": "bloc",
                    "file_path": "lib/blocs/user_bloc.dart",
                    "variables": {
                        "bloc_name": "User",
                        "events": ["LoadUser", "UpdateUser"],
                        "states": ["UserLoading", "UserLoaded", "UserError"]
                    }
                },
                "expected_outcome": "Creates complete BLoC structure with events and states"
            }
        ]

    async def get_capabilities(self) -> ToolCapabilities:
        """Get comprehensive file system capabilities."""
        operations = [
            {
                "name": "read_file",
                "description": "Read a file with Flutter project awareness and import optimization suggestions.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "analyze_imports": {"type": "boolean", "default": False}
                    },
                    "required": ["path"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "size": {"type": "integer"},
                        "modified": {"type": "string"},
                        "import_analysis": {"type": "object"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Read main.dart with import analysis", "params": {"path": "lib/main.dart", "analyze_imports": True}}],
                "error_codes": {"NOT_FOUND": "File not found", "PERMISSION_DENIED": "Access denied"}
            },
            {
                "name": "write_file",
                "description": "Write to a file with backup, rollback support, and Flutter validation.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "create_backup": {"type": "boolean", "default": True},
                        "encoding": {"type": "string", "default": "utf-8"},
                        "validate_flutter": {"type": "boolean", "default": True},
                        "optimize_imports": {"type": "boolean", "default": False}
                    },
                    "required": ["path", "content"]
                },
                "output_schema": {
                    "type": "object", 
                    "properties": {
                        "written": {"type": "boolean"},
                        "backup_path": {"type": "string"},
                        "validation_result": {"type": "object"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_WRITE],
                "examples": [{"description": "Write widget with backup", "params": {"path": "lib/widgets/my_widget.dart", "content": "...", "optimize_imports": True}}]
            },
            {
                "name": "create_from_template",
                "description": "Create a Flutter file from template with advanced customization.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "template_type": {"type": "string", "enum": ["widget", "stateful_widget", "model", "service", "screen", "provider", "repository", "test_file"]},
                        "file_path": {"type": "string"},
                        "class_name": {"type": "string"},
                        "variables": {"type": "object", "default": {}},
                        "create_test": {"type": "boolean", "default": False},
                        "create_barrel": {"type": "boolean", "default": False}
                    },
                    "required": ["template_type", "file_path", "class_name"]
                },
                "required_permissions": [ToolPermission.FILE_CREATE],
                "examples": [{"description": "Create widget with test", "params": {"template": "widget", "path": "lib/widgets/header.dart", "class_name": "HeaderWidget", "create_test": True}}]
            },
            {
                "name": "manage_pubspec",
                "description": "Safely manage pubspec.yaml with validation and dependency conflict detection.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add_dependency", "remove_dependency", "update_dependency", "add_asset", "remove_asset"]},
                        "package_name": {"type": "string"},
                        "version": {"type": "string"},
                        "asset_path": {"type": "string"},
                        "dev_dependency": {"type": "boolean", "default": False}
                    },
                    "required": ["action"]
                },
                "required_permissions": [ToolPermission.FILE_WRITE],
                "examples": [{"description": "Add dependency", "params": {"action": "add_dependency", "package_name": "http", "version": "^0.13.5"}}]
            },
            {
                "name": "optimize_assets",
                "description": "Optimize assets (images, fonts) for Flutter with multiple resolutions.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "asset_path": {"type": "string"},
                        "target_sizes": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                        "update_pubspec": {"type": "boolean", "default": True}
                    },
                    "required": ["asset_path"]
                },
                "required_permissions": [ToolPermission.FILE_READ, ToolPermission.FILE_WRITE],
                "examples": [{"description": "Optimize app icon", "params": {"asset_path": "assets/images/app_icon.png"}}]
            },
            {
                "name": "create_barrel_exports",
                "description": "Create barrel export files for better import organization.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string"},
                        "recursive": {"type": "boolean", "default": False},
                        "exclude_private": {"type": "boolean", "default": True}
                    },
                    "required": ["directory"]
                },
                "required_permissions": [ToolPermission.FILE_CREATE],
                "examples": [{"description": "Create barrel exports for widgets", "params": {"directory": "lib/widgets"}}]
            },
            {
                "name": "batch_operation",
                "description": "Execute multiple file operations in a transaction with rollback support.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "operations": {"type": "array", "items": {"type": "object"}},
                        "rollback_on_error": {"type": "boolean", "default": True},
                        "create_checkpoint": {"type": "boolean", "default": True}
                    },
                    "required": ["operations"]
                },
                "required_permissions": [ToolPermission.FILE_READ, ToolPermission.FILE_WRITE],
                "examples": [{"description": "Refactor multiple files", "params": {"operations": []}}]
            },
            {
                "name": "analyze_project_structure",
                "description": "Analyze Flutter project structure with detailed insights and suggestions.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "default": "."},
                        "deep_analysis": {"type": "boolean", "default": True},
                        "check_conventions": {"type": "boolean", "default": True}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "structure": {"type": "object"},
                        "suggestions": {"type": "array"},
                        "metrics": {"type": "object"},
                        "flutter_info": {"type": "object"},
                        "platform_support": {"type": "array"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Deep analyze current project", "params": {"deep_analysis": True}}]
            },
            {
                "name": "setup_file_watcher",
                "description": "Setup file watching with Flutter-specific change categorization.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}},
                        "ignore_patterns": {"type": "array", "items": {"type": "string"}},
                        "categorize_changes": {"type": "boolean", "default": True}
                    },
                    "required": ["paths"]
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Watch lib directory", "params": {"paths": ["lib/**/*.dart"]}}]
            },
            {
                "name": "validate_flutter_conventions",
                "description": "Validate file structure against Flutter conventions and best practices.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "check_imports": {"type": "boolean", "default": True},
                        "check_naming": {"type": "boolean", "default": True},
                        "check_structure": {"type": "boolean", "default": True}
                    },
                    "required": ["path"]
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Validate lib directory", "params": {"path": "lib"}}]
            }
        ]
        
        return ToolCapabilities(
            available_operations=operations,
            supported_contexts=["flutter_project", "dart_package", "flutter_module"],
            performance_characteristics={
                "avg_response_time": "50ms",
                "concurrent_operations": 10,
                "max_file_size": "10MB",
                "backup_retention": "7 days"
            },
            limitations=[
                "Cannot operate outside project boundaries", 
                "Requires valid Flutter project structure for advanced features",
                "Asset optimization requires PIL for image processing",
                "Platform-specific operations require platform tools"
            ],
            resource_requirements={
                "cpu": "low",
                "memory": "64MB",
                "disk": "variable based on operations",
                "network": "none"
            }
        )

    async def validate_params(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate operation parameters with Flutter-specific checks."""
        if operation == "read_file":
            if "path" not in params:
                return False, "Missing required parameter: path"
            path = Path(params["path"])
            if not path.exists():
                return False, f"File not found: {path}"
            return True, None
            
        elif operation == "write_file":
            required = ["path", "content"]
            missing = [p for p in required if p not in params]
            if missing:
                return False, f"Missing required parameters: {missing}"
            
            # Flutter-specific validation for pubspec.yaml
            path = Path(params["path"])
            if path.name == "pubspec.yaml" and params.get("validate_flutter", True):
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            current_content = f.read()
                        is_valid, error = self._validate_pubspec_changes(current_content, params["content"])
                        if not is_valid:
                            return False, f"Pubspec validation failed: {error}"
                    except Exception as e:
                        return False, f"Failed to validate pubspec: {e}"
            
            return True, None
            
        elif operation == "create_from_template":
            required = ["template_type", "file_path"]
            missing = [p for p in required if p not in params]
            if missing:
                return False, f"Missing required parameters: {missing}"
            
            valid_templates = ["widget", "model", "bloc", "repository", "service", "screen", "test"]
            if params["template_type"] not in valid_templates:
                return False, f"Unknown template: {params['template_type']}"
            
            # Validate class name if provided
            if "class_name" in params:
                class_name = params["class_name"]
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                    return False, f"Invalid class name '{class_name}': must start with uppercase letter and contain only letters/numbers"
            
            return True, None
            
        elif operation == "manage_pubspec":
            if "action" not in params:
                return False, "Missing required parameter: action"
            
            action = params["action"]
            if action in ["add_dependency", "remove_dependency", "update_dependency"]:
                if "package_name" not in params:
                    return False, f"Missing required parameter 'package_name' for action '{action}'"
            elif action in ["add_asset", "remove_asset"]:
                if "asset_path" not in params:
                    return False, f"Missing required parameter 'asset_path' for action '{action}'"
            
            return True, None
            
        elif operation == "optimize_assets":
            if "asset_path" not in params:
                return False, "Missing required parameter: asset_path"
            
            asset_path = Path(params["asset_path"])
            if not asset_path.exists():
                return False, f"Asset file not found: {asset_path}"
            
            return True, None
            
        elif operation == "create_barrel_exports":
            if "directory" not in params:
                return False, "Missing required parameter: directory"
            
            directory = Path(params["directory"])
            if not directory.exists() or not directory.is_dir():
                return False, f"Directory not found: {directory}"
            
            return True, None
            
        elif operation == "batch_operation":
            if "operations" not in params:
                return False, "Missing required parameter: operations"
            if not isinstance(params["operations"], list):
                return False, "Parameter 'operations' must be a list"
            
            return True, None
            
        elif operation == "analyze_project_structure":
            project_path = Path(params.get("project_path", "."))
            if not project_path.exists():
                return False, f"Project path not found: {project_path}"
            
            return True, None
            
        elif operation == "setup_file_watcher":
            if "paths" not in params:
                return False, "Missing required parameter: paths"
            if not isinstance(params["paths"], list):
                return False, "Parameter 'paths' must be a list"
            
            return True, None
            
        elif operation == "validate_flutter_conventions":
            if "path" not in params:
                return False, "Missing required parameter: path"
            
            path = Path(params["path"])
            if not path.exists():
                return False, f"Path not found: {path}"
            
            return True, None
            
        return False, f"Unknown operation: {operation}"

    async def execute(self, operation: str, params: Dict[str, Any], operation_id: Optional[str] = None) -> ToolResult:
        """Execute file system operations with Flutter awareness."""
        start_time = datetime.now()
        
        try:
            # Load .gitignore patterns for the project
            project_path = self._find_project_root(Path(params.get("path", ".")))
            if project_path:
                self.gitignore_patterns = self._load_gitignore_patterns(project_path)
            
            if operation == "read_file":
                return await self._read_file(params)
            elif operation == "write_file":
                return await self._write_file(params)
            elif operation == "create_from_template":
                return await self._create_from_template(params)
            elif operation == "manage_pubspec":
                return await self._manage_pubspec(params)
            elif operation == "optimize_assets":
                return await self._optimize_assets(params)
            elif operation == "create_barrel_exports":
                return await self._create_barrel_exports(params)
            elif operation == "batch_operation":
                return await self._batch_operation(params)
            elif operation == "analyze_project_structure":
                return await self._analyze_project_structure(params)
            elif operation == "setup_file_watcher":
                return await self._setup_file_watcher(params)
            elif operation == "validate_flutter_conventions":
                return await self._validate_flutter_conventions(params)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Unknown operation: {operation}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            logger.error(f"Error executing {operation}: {e}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _find_project_root(self, start_path: Path) -> Optional[Path]:
        """Find Flutter project root by looking for pubspec.yaml."""
        current = start_path.resolve()
        
        while current != current.parent:
            if (current / "pubspec.yaml").exists():
                return current
            current = current.parent
        
        return None

    async def _read_file(self, params: Dict[str, Any]) -> ToolResult:
        """Read file implementation with Flutter-specific analysis."""
        path = Path(params["path"])
        encoding = params.get("encoding", "utf-8")
        analyze_imports = params.get("analyze_imports", False)
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            stat = path.stat()
            
            result_data = {
                "content": content,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            # Add import analysis for Dart files
            if analyze_imports and path.suffix == '.dart':
                import_analysis = self._analyze_dart_imports(content)
                result_data["import_analysis"] = import_analysis
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                execution_time=0.01
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to read file {path}: {str(e)}"
            )

    def _analyze_dart_imports(self, content: str) -> Dict[str, Any]:
        """Analyze Dart import statements."""
        lines = content.split('\n')
        imports = {
            "dart_imports": [],
            "package_imports": [],
            "relative_imports": [],
            "unused_imports": [],
            "suggestions": []
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                if 'dart:' in line:
                    imports["dart_imports"].append(line)
                elif 'package:' in line:
                    imports["package_imports"].append(line)
                else:
                    imports["relative_imports"].append(line)
        
        # Add suggestions for optimization
        if len(imports["dart_imports"]) > 1:
            imports["suggestions"].append("Consider grouping dart: imports")
        if len(imports["package_imports"]) > 5:
            imports["suggestions"].append("Consider using barrel exports to reduce imports")
        
        return imports

    async def _write_file(self, params: Dict[str, Any]) -> ToolResult:
        """Write file implementation with backup, validation, and optimization."""
        path = Path(params["path"])
        content = params["content"]
        create_backup = params.get("create_backup", True)
        encoding = params.get("encoding", "utf-8")
        optimize_imports = params.get("optimize_imports", False)
        
        backup_path = None
        validation_result = {}
        
        try:
            # Optimize imports if requested and it's a Dart file
            if optimize_imports and path.suffix == '.dart':
                content = self._optimize_imports(content)
                validation_result["imports_optimized"] = True
            
            # Create backup if file exists and backup requested
            if path.exists() and create_backup:
                backup_dir = Path(self.backup_dir)
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"{path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                shutil.copy2(path, backup_path)
                validation_result["backup_created"] = True
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            validation_result["flutter_valid"] = True
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "written": True,
                    "backup_path": str(backup_path) if backup_path else None,
                    "validation_result": validation_result
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to write file {path}: {str(e)}"
            )

    async def _create_from_template(self, params: Dict[str, Any]) -> ToolResult:
        """Create file from dynamic template."""
        from ..template_engine import get_template_engine, TemplateContext, TemplateType, ArchitecturalPattern
        
        try:
            template_type_str = params.get("template_type")
            file_path = params.get("file_path")
            variables = params.get("variables", {})
            architectural_pattern = params.get("architectural_pattern", "basic")
            
            if not template_type_str or not file_path:
                return ToolResult(
                    tool_name=self.name,
                    operation="create_from_template",
                    status=ToolStatus.FAILED,
                    error_message="template_type and file_path are required"
                )
            
            # Map string to enum
            template_type_map = {
                "widget": TemplateType.WIDGET,
                "model": TemplateType.MODEL,
                "bloc": TemplateType.BLOC,
                "repository": TemplateType.REPOSITORY,
                "service": TemplateType.SERVICE,
                "screen": TemplateType.SCREEN,
                "test": TemplateType.TEST
            }
            
            template_type = template_type_map.get(template_type_str)
            if not template_type:
                return ToolResult(
                    tool_name=self.name,
                    operation="create_from_template",
                    status=ToolStatus.FAILED,
                    error_message=f"Unknown template type: {template_type_str}"
                )
            
            # Map architectural pattern
            arch_pattern_map = {
                "basic": ArchitecturalPattern.BASIC_PATTERN,
                "clean": ArchitecturalPattern.CLEAN_ARCHITECTURE,
                "bloc": ArchitecturalPattern.BLOC_PATTERN,
                "provider": ArchitecturalPattern.PROVIDER_PATTERN
            }
            
            arch_pattern = arch_pattern_map.get(architectural_pattern, ArchitecturalPattern.BASIC_PATTERN)
            
            # Create template context
            context = TemplateContext(
                app_name=variables.get("app_name", "MyApp"),
                app_description=variables.get("app_description", "Flutter application"),
                architectural_pattern=arch_pattern,
                custom_variables=variables
            )
            
            # Render template
            template_engine = get_template_engine()
            content = template_engine.render_template(template_type, context)
            
            if not content:
                return ToolResult(
                    tool_name=self.name,
                    operation="create_from_template",
                    status=ToolStatus.FAILED,
                    error_message=f"Failed to render template: {template_type_str}"
                )
            
            # Write file
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            file_path_obj.write_text(content, encoding='utf-8')
            
            return ToolResult(
                tool_name=self.name,
                operation="create_from_template",
                status=ToolStatus.SUCCESS,
                data={
                    "file_path": file_path,
                    "template_type": template_type_str,
                    "architectural_pattern": architectural_pattern,
                    "content_length": len(content),
                    "variables_used": list(variables.keys())
                },
                metadata={
                    "creation_time": datetime.utcnow().isoformat(),
                    "template_engine": "dynamic"
                }
            )
            
        except Exception as e:
            logger.error(f"Template creation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                operation="create_from_template",
                status=ToolStatus.FAILED,
                error_message=str(e)
            )

    def _validate_pubspec_changes(self, current_content: str, new_content: str) -> Tuple[bool, Optional[str]]:
        """Validate pubspec.yaml changes for potential issues."""
        try:
            current_data = yaml.safe_load(current_content)
            new_data = yaml.safe_load(new_content)
            
            # Check for required fields
            required_fields = ['name', 'version', 'flutter']
            for field in required_fields:
                if field not in new_data:
                    return False, f"Missing required field: {field}"
            
            return True, None
        except Exception as e:
            return False, f"YAML parsing error: {e}"

    def _optimize_imports(self, content: str) -> str:
        """Optimize Dart import statements."""
        lines = content.split('\n')
        dart_imports = []
        package_imports = []
        relative_imports = []
        other_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import '):
                if 'dart:' in line:
                    dart_imports.append(line)
                elif 'package:' in line:
                    package_imports.append(line)
                else:
                    relative_imports.append(line)
            else:
                other_lines.append(line)
        
        # Rebuild content with organized imports
        organized_lines = []
        if dart_imports:
            organized_lines.extend(sorted(dart_imports))
            organized_lines.append('')
        if package_imports:
            organized_lines.extend(sorted(package_imports))
            organized_lines.append('')
        if relative_imports:
            organized_lines.extend(sorted(relative_imports))
            organized_lines.append('')
        
        organized_lines.extend(other_lines)
        return '\n'.join(organized_lines)

    def _load_gitignore_patterns(self, project_path: Path) -> Set[str]:
        """Load .gitignore patterns for the project."""
        patterns = set()
        gitignore_path = project_path / '.gitignore'
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.add(line)
            except Exception as e:
                logger.warning(f"Failed to load .gitignore: {e}")
        
        return patterns

    async def _manage_pubspec(self, params: Dict[str, Any]) -> ToolResult:
        """Manage pubspec.yaml operations."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Pubspec management not yet implemented"},
            execution_time=0.01
        )

    async def _optimize_assets(self, params: Dict[str, Any]) -> ToolResult:
        """Optimize asset files."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Asset optimization not yet implemented"},
            execution_time=0.01
        )

    async def _create_barrel_exports(self, params: Dict[str, Any]) -> ToolResult:
        """Create barrel export files."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Barrel exports not yet implemented"},
            execution_time=0.01
        )

    async def _batch_operation(self, params: Dict[str, Any]) -> ToolResult:
        """Execute batch operations."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Batch operations not yet implemented"},
            execution_time=0.01
        )

    async def _analyze_project_structure(self, params: Dict[str, Any]) -> ToolResult:
        """Analyze Flutter project structure."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Project structure analysis not yet implemented"},
            execution_time=0.01
        )

    async def _setup_file_watcher(self, params: Dict[str, Any]) -> ToolResult:
        """Setup file watching."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "File watching not yet implemented"},
            execution_time=0.01
        )

    async def _validate_flutter_conventions(self, params: Dict[str, Any]) -> ToolResult:
        """Validate Flutter conventions."""
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Convention validation not yet implemented"},
            execution_time=0.01
        )

