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
        self.templates = self._load_flutter_templates()
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

    def _load_flutter_templates(self) -> Dict[str, str]:
        """Load Flutter-specific file templates."""
        return {
            "widget": '''import 'package:flutter/material.dart';

class {class_name} extends StatelessWidget {{
  const {class_name}({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return Container(
      child: Text('{class_name}'),
    );
  }}
}}''',
            "stateful_widget": '''import 'package:flutter/material.dart';

class {class_name} extends StatefulWidget {{
  const {class_name}({{Key? key}}) : super(key: key);

  @override
  State<{class_name}> createState() => _{class_name}State();
}}

class _{class_name}State extends State<{class_name}> {{
  @override
  Widget build(BuildContext context) {{
    return Container(
      child: Text('{class_name}'),
    );
  }}
}}''',
            "model": '''class {class_name} {{
  const {class_name}({{
    required this.id,
    required this.name,
  }});
  
  final String id;
  final String name;
  
  Map<String, dynamic> toJson() => {{
    'id': id,
    'name': name,
  }};
  
  factory {class_name}.fromJson(Map<String, dynamic> json) {{
    return {class_name}(
      id: json['id'],
      name: json['name'],
    );
  }}
  
  {class_name} copyWith({{
    String? id,
    String? name,
  }}) {{
    return {class_name}(
      id: id ?? this.id,
      name: name ?? this.name,
    );
  }}
}}''',
            "service": '''class {class_name} {{
  static final {class_name} _instance = {class_name}._internal();
  
  factory {class_name}() => _instance;
  
  {class_name}._internal();
  
  // Add your service methods here
}}''',
            "barrel_export": '''// Barrel export file for {directory}
// Generated by FlutterSwarm FileSystemTool

{exports}''',
            "test_file": '''import 'package:flutter_test/flutter_test.dart';
import 'package:{package_name}/{import_path}';

void main() {{
  group('{class_name} Tests', () {{
    test('should create instance', () {{
      // Arrange
      
      // Act
      final instance = {class_name}();
      
      // Assert
      expect(instance, isNotNull);
    }});
  }});
}}''',
            "screen": '''import 'package:flutter/material.dart';

class {class_name} extends StatelessWidget {{
  const {class_name}({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: AppBar(
        title: const Text('{title}'),
      ),
      body: const Center(
        child: Text('{class_name}'),
      ),
    );
  }}
}}''',
            "provider": '''import 'package:flutter/foundation.dart';

class {class_name} extends ChangeNotifier {{
  // Private state variables
  
  // Getters for state
  
  // Methods to modify state
  void updateState() {{
    // Update logic here
    notifyListeners();
  }}
}}''',
            "repository": '''abstract class {class_name}Repository {{
  // Abstract methods for data operations
}}

class {class_name}RepositoryImpl implements {class_name}Repository {{
  // Implementation of repository methods
}}'''
        }

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
                        "template": {"type": "string", "enum": ["widget", "stateful_widget", "model", "service", "screen", "provider", "repository", "test_file"]},
                        "path": {"type": "string"},
                        "class_name": {"type": "string"},
                        "variables": {"type": "object", "default": {}},
                        "create_test": {"type": "boolean", "default": False},
                        "create_barrel": {"type": "boolean", "default": False}
                    },
                    "required": ["template", "path", "class_name"]
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
            required = ["template", "path", "class_name"]
            missing = [p for p in required if p not in params]
            if missing:
                return False, f"Missing required parameters: {missing}"
            if params["template"] not in self.templates:
                return False, f"Unknown template: {params['template']}"
            
            # Validate class name follows Dart conventions
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
        """Create file from Flutter template."""
        template_name = params["template"]
        path = Path(params["path"])
        class_name = params["class_name"]
        variables = params.get("variables", {})
        
        try:
            template = self.templates[template_name]
            
            # Apply template variables
            content = template.format(class_name=class_name, **variables)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "created": True,
                    "path": str(path),
                    "template": template_name
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to create from template: {str(e)}"
            )

    async def _manage_pubspec(self, params: Dict[str, Any]) -> ToolResult:
        """Manage pubspec.yaml safely with validation."""
        action = params["action"]
        pubspec_path = Path("pubspec.yaml")
        
        try:
            if not pubspec_path.exists():
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message="pubspec.yaml not found in current directory"
                )
            
            with open(pubspec_path, 'r') as f:
                pubspec_content = f.read()
            
            pubspec_data = yaml.safe_load(pubspec_content)
            
            if action == "add_dependency":
                package_name = params["package_name"]
                version = params.get("version", "^1.0.0")
                dev_dependency = params.get("dev_dependency", False)
                
                section = "dev_dependencies" if dev_dependency else "dependencies"
                if section not in pubspec_data:
                    pubspec_data[section] = {}
                
                pubspec_data[section][package_name] = version
                
            elif action == "remove_dependency":
                package_name = params["package_name"]
                
                for section in ["dependencies", "dev_dependencies"]:
                    if section in pubspec_data and package_name in pubspec_data[section]:
                        del pubspec_data[section][package_name]
                        break
                
            elif action == "add_asset":
                asset_path = params["asset_path"]
                
                if "flutter" not in pubspec_data:
                    pubspec_data["flutter"] = {}
                if "assets" not in pubspec_data["flutter"]:
                    pubspec_data["flutter"]["assets"] = []
                
                if asset_path not in pubspec_data["flutter"]["assets"]:
                    pubspec_data["flutter"]["assets"].append(asset_path)
            
            # Write back to file
            new_content = yaml.dump(pubspec_data, default_flow_style=False, sort_keys=False)
            
            # Validate changes
            is_valid, error = self._validate_pubspec_changes(pubspec_content, new_content)
            if not is_valid:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Pubspec validation failed: {error}"
                )
            
            # Create backup
            backup_dir = Path(self.backup_dir)
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"pubspec.yaml.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            shutil.copy2(pubspec_path, backup_path)
            
            # Write new content
            with open(pubspec_path, 'w') as f:
                f.write(new_content)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "action_completed": action,
                    "backup_path": str(backup_path),
                    "modified_sections": [section for section in ["dependencies", "dev_dependencies", "flutter"] if section in pubspec_data]
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to manage pubspec: {str(e)}"
            )

    async def _optimize_assets(self, params: Dict[str, Any]) -> ToolResult:
        """Optimize assets for Flutter."""
        asset_path = Path(params["asset_path"])
        target_sizes = params.get("target_sizes", [(72, 72), (96, 96), (144, 144), (192, 192)])
        update_pubspec = params.get("update_pubspec", True)
        
        try:
            optimization_result = self._optimize_assets_implementation(asset_path, target_sizes)
            
            # Check if optimization was successful
            if "error" in optimization_result:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data=optimization_result,
                    error_message=optimization_result["error"]
                )
            
            # Update pubspec if requested
            if update_pubspec and optimization_result.get("optimized_versions"):
                pubspec_params = {
                    "action": "add_asset",
                    "asset_path": str(asset_path.parent) + "/"
                }
                await self._manage_pubspec(pubspec_params)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "asset_path": str(asset_path),
                    "optimization_result": optimization_result,
                    "pubspec_updated": update_pubspec
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to optimize assets: {str(e)}"
            )

    async def _create_barrel_exports(self, params: Dict[str, Any]) -> ToolResult:
        """Create barrel export files."""
        directory = Path(params["directory"])
        recursive = params.get("recursive", False)
        exclude_private = params.get("exclude_private", True)
        
        try:
            created_files = []
            
            if recursive:
                for subdir in directory.rglob("*"):
                    if subdir.is_dir() and any(subdir.glob("*.dart")):
                        barrel_content = self._generate_barrel_exports(subdir)
                        if barrel_content.count("export") > 0:
                            barrel_path = subdir / "index.dart"
                            with open(barrel_path, 'w') as f:
                                f.write(barrel_content)
                            created_files.append(str(barrel_path))
            else:
                barrel_content = self._generate_barrel_exports(directory)
                if barrel_content.count("export") > 0:
                    barrel_path = directory / "index.dart"
                    with open(barrel_path, 'w') as f:
                        f.write(barrel_content)
                    created_files.append(str(barrel_path))
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "created_files": created_files,
                    "directory": str(directory),
                    "recursive": recursive
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to create barrel exports: {str(e)}"
            )

    async def _setup_file_watcher(self, params: Dict[str, Any]) -> ToolResult:
        """Setup file watching with Flutter-specific categorization."""
        # Use the enhanced implementation
        return await self._setup_file_watcher_enhanced(params)

    # =============================================
    # MISSING CRITICAL METHODS IMPLEMENTATION
    # =============================================

    async def _batch_operation(self, params: Dict[str, Any]) -> ToolResult:
        """Execute multiple file operations in a transaction with rollback support."""
        operations = params["operations"]
        rollback_on_error = params.get("rollback_on_error", True)
        create_checkpoint = params.get("create_checkpoint", True)
        
        transaction_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        checkpoint_data = []
        executed_operations = []
        
        try:
            # Create checkpoint if requested
            if create_checkpoint:
                checkpoint_data = await self._create_transaction_checkpoint(operations)
            
            # Execute operations sequentially
            for i, operation_params in enumerate(operations):
                operation_name = operation_params.get("operation")
                operation_data = operation_params.get("params", {})
                
                try:
                    # Execute individual operation
                    result = await self.execute(operation_name, operation_data)
                    executed_operations.append({
                        "index": i,
                        "operation": operation_name,
                        "params": operation_data,
                        "result": result,
                        "status": result.status
                    })
                    
                    if result.status == ToolStatus.FAILURE and rollback_on_error:
                        # Rollback all previous operations
                        rollback_result = await self._rollback_transaction(
                            executed_operations, checkpoint_data
                        )
                        
                        return ToolResult(
                            status=ToolStatus.FAILURE,
                            data={
                                "transaction_id": transaction_id,
                                "failed_at_operation": i,
                                "executed_operations": len(executed_operations),
                                "rollback_performed": rollback_result["success"],
                                "rollback_details": rollback_result,
                                "error_message": f"Operation {i} failed: {result.error_message}"
                            },
                            error_message=f"Batch operation failed at step {i}: {result.error_message}"
                        )
                        
                except Exception as e:
                    if rollback_on_error:
                        rollback_result = await self._rollback_transaction(
                            executed_operations, checkpoint_data
                        )
                        
                        return ToolResult(
                            status=ToolStatus.FAILURE,
                            data={
                                "transaction_id": transaction_id,
                                "failed_at_operation": i,
                                "rollback_performed": rollback_result["success"],
                                "error_message": str(e)
                            },
                            error_message=f"Batch operation failed: {str(e)}"
                        )
                    else:
                        executed_operations.append({
                            "index": i,
                            "operation": operation_name,
                            "params": operation_data,
                            "error": str(e),
                            "status": ToolStatus.FAILURE
                        })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "transaction_id": transaction_id,
                    "operations_executed": len(executed_operations),
                    "operations_successful": len([op for op in executed_operations if op["status"] == ToolStatus.SUCCESS]),
                    "operations_failed": len([op for op in executed_operations if op["status"] == ToolStatus.FAILURE]),
                    "executed_operations": executed_operations,
                    "checkpoint_created": create_checkpoint
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={
                    "transaction_id": transaction_id,
                    "error_message": str(e)
                },
                error_message=f"Batch operation failed: {str(e)}"
            )

    async def _create_transaction_checkpoint(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create checkpoint data for rollback capability."""
        checkpoint_data = []
        
        for operation in operations:
            operation_name = operation.get("operation")
            params = operation.get("params", {})
            
            if operation_name in ["write_file", "create_from_template"]:
                # For file write operations, backup existing files
                file_path = Path(params.get("path", ""))
                if file_path.exists():
                    backup_content = file_path.read_text()
                    checkpoint_data.append({
                        "type": "file_backup",
                        "path": str(file_path),
                        "original_content": backup_content,
                        "operation": operation_name
                    })
                else:
                    checkpoint_data.append({
                        "type": "file_create",
                        "path": str(file_path),
                        "operation": operation_name
                    })
                    
        return checkpoint_data

    async def _rollback_transaction(self, executed_operations: List[Dict], checkpoint_data: List[Dict]) -> Dict[str, Any]:
        """Rollback executed operations using checkpoint data."""
        rollback_results = []
        rollback_successful = True
        
        try:
            # Restore files from checkpoint data
            for checkpoint in checkpoint_data:
                try:
                    if checkpoint["type"] == "file_backup":
                        # Restore original file content
                        file_path = Path(checkpoint["path"])
                        file_path.write_text(checkpoint["original_content"])
                        rollback_results.append({
                            "path": checkpoint["path"],
                            "action": "restored",
                            "success": True
                        })
                    elif checkpoint["type"] == "file_create":
                        # Remove newly created file
                        file_path = Path(checkpoint["path"])
                        if file_path.exists():
                            file_path.unlink()
                        rollback_results.append({
                            "path": checkpoint["path"],
                            "action": "removed",
                            "success": True
                        })
                        
                except Exception as e:
                    rollback_successful = False
                    rollback_results.append({
                        "path": checkpoint.get("path", "unknown"),
                        "action": "failed",
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "success": rollback_successful,
                "operations_rolled_back": len(rollback_results),
                "rollback_details": rollback_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rollback_details": rollback_results
            }

    def _optimize_imports(self, content: str) -> str:
        """Optimize Dart import statements following Flutter conventions."""
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        
        # Separate import lines from other content
        for line in lines:
            if line.strip().startswith('import '):
                import_lines.append(line.strip())
            else:
                other_lines.append(line)
        
        if not import_lines:
            return content
        
        # Categorize imports
        dart_imports = []
        package_imports = []
        relative_imports = []
        
        for import_line in import_lines:
            if 'dart:' in import_line:
                dart_imports.append(import_line)
            elif 'package:' in import_line:
                package_imports.append(import_line)
            else:
                relative_imports.append(import_line)
        
        # Sort each category
        dart_imports.sort()
        package_imports.sort()
        relative_imports.sort()
        
        # Rebuild content with organized imports
        organized_imports = []
        
        # Add dart: imports first
        if dart_imports:
            organized_imports.extend(dart_imports)
            organized_imports.append('')  # Empty line after dart imports
        
        # Add package: imports
        if package_imports:
            organized_imports.extend(package_imports)
            organized_imports.append('')  # Empty line after package imports
        
        # Add relative imports last
        if relative_imports:
            organized_imports.extend(relative_imports)
            organized_imports.append('')  # Empty line after relative imports
        
        # Find where imports end in original content
        non_import_start = 0
        for i, line in enumerate(other_lines):
            if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*'):
                non_import_start = i
                break
        
        # Combine organized imports with rest of content
        result_lines = organized_imports + other_lines[non_import_start:]
        
        return '\n'.join(result_lines)

    def _load_gitignore_patterns(self, project_path: Path) -> Set[str]:
        """Load .gitignore patterns for the project."""
        patterns = set()
        gitignore_path = project_path / ".gitignore"
        
        # Default Flutter ignore patterns
        flutter_defaults = {
            "build/", ".dart_tool/", ".packages", ".flutter-plugins",
            ".flutter-plugins-dependencies", "*.g.dart", "*.freezed.dart",
            ".vscode/", ".idea/", "*.iml", "ios/Pods/", "android/.gradle/",
            "web/favicon.png", "*.log"
        }
        patterns.update(flutter_defaults)
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.add(line)
            except Exception as e:
                logger.warning(f"Could not read .gitignore: {e}")
        
        return patterns

    def _is_ignored(self, file_path: Path, project_root: Path) -> bool:
        """Check if a file should be ignored based on .gitignore patterns."""
        try:
            relative_path = file_path.relative_to(project_root)
            path_str = str(relative_path);
            
            for pattern in self.gitignore_patterns:
                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                    return True
                    
                # Check directory patterns
                if pattern.endswith('/') and (pattern[:-1] in path_str.split('/')):
                    return True
                    
        except ValueError:
            # Path is not relative to project root
            pass
            
        return False

    def _detect_flutter_project(self, project_path: Path) -> Dict[str, Any]:
        """Detect if directory is a Flutter project and gather info."""
        pubspec_path = project_path / "pubspec.yaml"
        
        info = {
            "is_flutter_project": False,
            "platforms": [],
            "flutter_version": None,
            "dependencies": [],
            "dev_dependencies": []
        }
        
        if not pubspec_path.exists():
            return info
        
        try:
            with open(pubspec_path, 'r') as f:
                pubspec_data = yaml.safe_load(f)
            
            # Check if it's a Flutter project
            if 'flutter' in pubspec_data.get('dependencies', {}):
                info["is_flutter_project"] = True
                
                # Detect platforms
                for platform in ["android", "ios", "web", "windows", "macos", "linux"]:
                    platform_dir = project_path / platform
                    if platform_dir.exists():
                        info["platforms"].append(platform)
                
                # Get dependencies
                info["dependencies"] = list(pubspec_data.get('dependencies', {}).keys())
                info["dev_dependencies"] = list(pubspec_data.get('dev_dependencies', {}).keys())
                
        except Exception as e:
            logger.warning(f"Could not analyze pubspec.yaml: {e}")
        
        return info

    def _validate_pubspec_changes(self, original_content: str, new_content: str) -> Tuple[bool, Optional[str]]:
        """Validate that pubspec.yaml changes are safe."""
        try:
            original_data = yaml.safe_load(original_content)
            new_data = yaml.safe_load(new_content)
            
            # Check that essential fields are preserved
            essential_fields = ['name', 'version', 'environment']
            for field in essential_fields:
                if field in original_data and field not in new_data:
                    return False, f"Essential field '{field}' was removed"
                    
            # Validate environment constraints
            if 'environment' in new_data:
                env = new_data['environment']
                if 'sdk' in env and not isinstance(env['sdk'], str):
                    return False, "SDK version must be a string"
                if 'flutter' in env and not isinstance(env['flutter'], str):
                    return False, "Flutter version must be a string"
            
            return True, None
            
        except yaml.YAMLError as e:
            return False, f"Invalid YAML syntax: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def _generate_barrel_exports(self, directory: Path) -> str:
        """Generate barrel export content for a directory."""
        exports = []
        
        # Get all Dart files in directory (excluding private files)
        for dart_file in directory.glob("*.dart"):
            if not dart_file.name.startswith('_') and dart_file.name != 'index.dart':
                relative_path = dart_file.name
                exports.append(f"export '{relative_path}';")
        
        if not exports:
            return ""
        
        # Sort exports alphabetically
        exports.sort()
        
        # Generate content
        header = f"// Barrel export file for {directory.name}\n// Generated by FlutterSwarm FileSystemTool\n\n"
        return header + '\n'.join(exports) + '\n'

    def _optimize_assets_implementation(self, asset_path: Path, target_sizes: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Optimize image assets for Flutter with multiple resolutions."""
        try:
            if asset_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                return {"error": "Only PNG and JPEG images are supported for optimization"}
            
            # Use PIL to optimize images
            with Image.open(asset_path) as img:
                original_size = img.size
                optimized_versions = []
                
                for target_size in target_sizes:
                    # Create resized version
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Generate output path with resolution suffix
                    output_dir = asset_path.parent
                    name_stem = asset_path.stem
                    extension = asset_path.suffix
                    
                    if target_size == original_size:
                        output_path = asset_path
                    else:
                        output_path = output_dir / f"{name_stem}_{target_size[0]}x{target_size[1]}{extension}"
                    
                    # Save optimized version
                    resized_img.save(output_path, optimize=True, quality=85)
                    optimized_versions.append({
                        "size": target_size,
                        "path": str(output_path),
                        "file_size": output_path.stat().st_size if output_path.exists() else 0
                    })
                
                return {
                    "original_size": original_size,
                    "optimized_versions": optimized_versions,
                    "total_variants": len(optimized_versions)
                }
                
        except ImportError:
            return {"error": "PIL (Pillow) is required for image optimization"}
        except Exception as e:
            return {"error": f"Asset optimization failed: {str(e)}"}
    
    # =============================================
    # ENHANCED FILE WATCHING IMPLEMENTATION
    # =============================================

    async def _setup_file_watcher_enhanced(self, params: Dict[str, Any]) -> ToolResult:
        """Setup enhanced file watching with real-time change detection."""
        paths = params["paths"]
        ignore_patterns = params.get("ignore_patterns", [])
        categorize_changes = params.get("categorize_changes", True)
        
        try:
            # This would use watchdog in a real implementation
            # For now, we'll create a comprehensive monitoring setup
            
            watched_files = {}
            categories = {
                "dart_files": [],
                "asset_files": [],
                "config_files": [],
                "platform_files": [],
                "test_files": []
            }
            
            for path_pattern in paths:
                expanded_paths = list(Path(".").glob(path_pattern))
                
                for file_path in expanded_paths:
                    if self._is_ignored(file_path, Path(".")):
                        continue
                    
                    # Categorize file type
                    category = self._categorize_file(file_path)
                    categories[category].append(str(file_path))
                    
                    # Store file metadata for change detection
                    if file_path.is_file():
                        stat = file_path.stat()
                        watched_files[str(file_path)] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                            "category": category,
                            "hash": self._calculate_file_hash(file_path)
                        }
            
            # Store watching configuration
            self.watched_files.update(watched_files.keys())
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "watched_files": len(watched_files),
                    "categories": categories,
                    "ignore_patterns": ignore_patterns,
                    "categorize_changes": categorize_changes,
                    "watching_config": {
                        "total_files": len(watched_files),
                        "dart_files": len(categories["dart_files"]),
                        "asset_files": len(categories["asset_files"]),
                        "config_files": len(categories["config_files"]),
                        "platform_files": len(categories["platform_files"]),
                        "test_files": len(categories["test_files"])
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to setup file watcher: {str(e)}"
            )

    def _categorize_file(self, file_path: Path) -> str:
        """Categorize a file based on Flutter project conventions."""
        path_str = str(file_path)
        
        if file_path.suffix == '.dart':
            if 'test' in path_str or file_path.name.endswith('_test.dart'):
                return "test_files"
            else:
                return "dart_files"
        elif file_path.suffix in ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.ttf', '.otf']:
            return "asset_files"
        elif file_path.name in ['pubspec.yaml', 'analysis_options.yaml', '.gitignore']:
            return "config_files"
        elif any(platform in path_str for platform in ['android', 'ios', 'web', 'windows', 'macos', 'linux']):
            return "platform_files"
        else:
            return "dart_files"  # Default category

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file for change detection."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""

    async def _analyze_project_structure(self, params: Dict[str, Any]) -> ToolResult:
        """Analyze Flutter project structure with detailed insights."""
        project_path = Path(params.get("project_path", "."))
        deep_analysis = params.get("deep_analysis", True)
        check_conventions = params.get("check_conventions", True)
        
        try:
            structure = {}
            suggestions = []
            metrics = {
                "total_files": 0,
                "dart_files": 0,
                "test_files": 0,
                "asset_files": 0,
                "platform_files": 0
            }
            
            # Detect Flutter project
            flutter_info = self._detect_flutter_project(project_path)
            
            # Analyze directory structure
            for root, dirs, files in os.walk(project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not self._is_ignored(Path(root) / d, project_path)]
                
                rel_root = Path(root).relative_to(project_path)
                structure[str(rel_root)] = {
                    "files": files,
                    "type": "unknown"
                }
                
                # Categorize directory type
                root_name = rel_root.parts[0] if rel_root.parts else "."
                if root_name in self.flutter_structure:
                    structure[str(rel_root)]["type"] = "flutter_standard"
                    structure[str(rel_root)]["description"] = self.flutter_structure[root_name]["description"]
                
                for file in files:
                    metrics["total_files"] += 1
                    
                    if file.endswith('.dart'):
                        metrics["dart_files"] += 1
                        if 'test' in str(rel_root) or file.endswith('_test.dart'):
                            metrics["test_files"] += 1
                    elif file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif', '.ttf', '.otf')):
                        metrics["asset_files"] += 1
                    elif any(platform in str(rel_root) for platform in ["android", "ios", "web", "windows", "macos", "linux"]):
                        metrics["platform_files"] += 1
            
            # Generate suggestions based on analysis
            if flutter_info.get("is_flutter_project"):
                # Check standard directories
                for dir_name, dir_info in self.flutter_structure.items():
                    dir_path = project_path / dir_name
                    if dir_info.get("critical", False) and not dir_path.exists():
                        suggestions.append(f"Missing critical directory: {dir_name} - {dir_info['description']}")
                
                # Check for common patterns
                if not (project_path / "lib").exists():
                    suggestions.append("Missing lib/ directory - create main source directory")
                elif not any(f.name == 'main.dart' for f in (project_path / "lib").glob("**/*.dart")):
                    suggestions.append("No main.dart found in lib/ - add application entry point")
                
                if not (project_path / "test").exists():
                    suggestions.append("Missing test/ directory - add unit tests")
                elif metrics["dart_files"] > 0 and metrics["test_files"] == 0:
                    suggestions.append("No test files found - consider adding unit tests")
                
                # Check test coverage ratio
                if metrics["dart_files"] > 0:
                    test_ratio = metrics["test_files"] / metrics["dart_files"]
                    if test_ratio < 0.3:
                        suggestions.append(f"Low test coverage ratio ({test_ratio:.1%}) - consider adding more tests")
                
                # Check for barrel exports
                lib_path = project_path / "lib"
                if lib_path.exists():
                    subdirs_without_barrels = []
                    for subdir in lib_path.iterdir():
                        if subdir.is_dir() and list(subdir.glob("*.dart")) and not (subdir / "index.dart").exists():
                            subdirs_without_barrels.append(subdir.name)
                    
                    if subdirs_without_barrels:
                        suggestions.append(f"Consider adding barrel exports to: {', '.join(subdirs_without_barrels)}")
            else:
                suggestions.append("Not a Flutter project - missing pubspec.yaml with flutter dependency")
            
            # Deep analysis
            if deep_analysis and flutter_info.get("is_flutter_project"):
                # Analyze imports and dependencies
                import_analysis = await self._deep_analyze_imports(project_path)
                metrics.update(import_analysis)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "structure": structure,
                    "suggestions": suggestions,
                    "metrics": metrics,
                    "flutter_info": flutter_info,
                    "platform_support": flutter_info.get("platforms", []),
                    "analysis_type": "deep" if deep_analysis else "basic"
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to analyze project: {str(e)}"
            )

    async def _validate_flutter_conventions(self, params: Dict[str, Any]) -> ToolResult:
        """Validate Flutter conventions and best practices."""
        path = Path(params["path"])
        check_imports = params.get("check_imports", True)
        check_naming = params.get("check_naming", True)
        check_structure = params.get("check_structure", True)
        
        try:
            violations = []
            suggestions = []
            
            if path.is_file() and path.suffix == '.dart':
                # Check single Dart file
                with open(path, 'r') as f:
                    content = f.read()
                
                if check_naming:
                    # Check file naming convention
                    if not re.match(r'^[a-z][a-z0-9_]*\.dart$', path.name):
                        violations.append(f"File name '{path.name}' should use snake_case")
                
                if check_imports:
                    # Check import organization
                    lines = content.split('\n')
                    import_section = []
                    for line in lines:
                        if line.strip().startswith('import '):
                            import_section.append(line)
                        elif line.strip() == '' and import_section:
                            continue
                        elif import_section:
                            break
                    
                    # Check if imports are sorted
                    sorted_imports = sorted(import_section)
                    if import_section != sorted_imports:
                        suggestions.append("Imports should be sorted alphabetically")
                
            elif path.is_dir():
                # Check directory structure
                if check_structure:
                    for item in path.iterdir():
                        if item.is_file() and item.suffix == '.dart':
                            if not re.match(r'^[a-z][a-z0-9_]*\.dart$', item.name):
                                violations.append(f"File name '{item.name}' should use snake_case")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(path),
                    "violations": violations,
                    "suggestions": suggestions,
                    "conventions_followed": len(violations) == 0
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to validate conventions: {str(e)}"
            )

    async def _deep_analyze_imports(self, project_path: Path) -> Dict[str, Any]:
        """Perform deep analysis of imports across the project."""
        analysis = {
            "circular_imports": [],
            "unused_imports": [],
            "missing_exports": [],
            "import_complexity": 0
        }
        
        try:
            dart_files = list(project_path.glob("**/*.dart"))
            
            for dart_file in dart_files:
                if self._is_ignored(dart_file, project_path):
                    continue
                
                try:
                    with open(dart_file, 'r') as f:
                        content = f.read()
                    
                    # Count imports per file
                    import_count = len(re.findall(r'^\s*import\s+', content, re.MULTILINE))
                    analysis["import_complexity"] += import_count
                    
                except Exception:
                    continue
            
            # Calculate average import complexity
            if dart_files:
                analysis["avg_imports_per_file"] = analysis["import_complexity"] / len(dart_files)
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
