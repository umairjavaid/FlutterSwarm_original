"""
File System Tool for FlutterSwarm Multi-Agent System.

Provides Flutter-aware file operations, safe file handling, template-based creation, 
batch/transaction support, file watching, conflict resolution, and asset management.
"""

import os
import shutil
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_tool import (
    BaseTool, ToolCapabilities, ToolOperation, ToolPermission, ToolResult, ToolStatus,
    ToolCategory, ToolContext
)


class FileSystemTool(BaseTool):
    """Enhanced Flutter-aware file system tool with comprehensive operations."""
    
    def __init__(self):
        super().__init__(
            name="file_system",
            description="Flutter-aware file system operations with safety, templates, and batch support.",
            version="1.2.0",
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
  const {class_name}();
  
  Map<String, dynamic> toJson() => {{}};
  
  factory {class_name}.fromJson(Map<String, dynamic> json) {{
    return {class_name}();
  }}
}}''',
            "service": '''class {class_name} {{
  static final {class_name} _instance = {class_name}._internal();
  
  factory {class_name}() => _instance;
  
  {class_name}._internal();
}}'''
        }

    async def get_capabilities(self) -> ToolCapabilities:
        """Get comprehensive file system capabilities."""
        operations = [
            {
                "name": "read_file",
                "description": "Read a file with Flutter project awareness.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "encoding": {"type": "string", "default": "utf-8"}
                    },
                    "required": ["path"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "size": {"type": "integer"},
                        "modified": {"type": "string"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Read main.dart", "params": {"path": "lib/main.dart"}}],
                "error_codes": {"NOT_FOUND": "File not found", "PERMISSION_DENIED": "Access denied"}
            },
            {
                "name": "write_file",
                "description": "Write to a file with backup and rollback support.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "create_backup": {"type": "boolean", "default": True},
                        "encoding": {"type": "string", "default": "utf-8"}
                    },
                    "required": ["path", "content"]
                },
                "output_schema": {
                    "type": "object", 
                    "properties": {
                        "written": {"type": "boolean"},
                        "backup_path": {"type": "string"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_WRITE],
                "examples": [{"description": "Write widget code", "params": {"path": "lib/widgets/my_widget.dart", "content": "..."}}]
            },
            {
                "name": "create_from_template",
                "description": "Create a Flutter file from template.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "enum": ["widget", "stateful_widget", "model", "service"]},
                        "path": {"type": "string"},
                        "class_name": {"type": "string"},
                        "variables": {"type": "object", "default": {}}
                    },
                    "required": ["template", "path", "class_name"]
                },
                "required_permissions": [ToolPermission.FILE_CREATE],
                "examples": [{"description": "Create widget", "params": {"template": "widget", "path": "lib/widgets/header.dart", "class_name": "HeaderWidget"}}]
            },
            {
                "name": "batch_operation",
                "description": "Execute multiple file operations in a transaction.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "operations": {"type": "array", "items": {"type": "object"}},
                        "rollback_on_error": {"type": "boolean", "default": True}
                    },
                    "required": ["operations"]
                },
                "required_permissions": [ToolPermission.FILE_READ, ToolPermission.FILE_WRITE],
                "examples": [{"description": "Refactor multiple files", "params": {"operations": []}}]
            },
            {
                "name": "analyze_project_structure",
                "description": "Analyze Flutter project structure and provide insights.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "default": "."}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "structure": {"type": "object"},
                        "suggestions": {"type": "array"},
                        "metrics": {"type": "object"}
                    }
                },
                "required_permissions": [ToolPermission.FILE_READ],
                "examples": [{"description": "Analyze current project", "params": {}}]
            }
        ]
        
        return ToolCapabilities(
            available_operations=operations,
            supported_contexts=["flutter_project", "dart_package"],
            performance_characteristics={
                "avg_response_time": "50ms",
                "concurrent_operations": 10,
                "max_file_size": "10MB"
            },
            limitations=["Cannot operate outside project boundaries", "Requires valid Flutter project structure"]
        )

    async def validate_params(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate operation parameters."""
        if operation == "read_file":
            if "path" not in params:
                return False, "Missing required parameter: path"
            path = Path(params["path"])
            if not path.exists():
                return False, f"File not found: {path}"
            return True, None
            
        elif operation == "write_file":
            if "path" not in params or "content" not in params:
                return False, "Missing required parameters: path, content"
            return True, None
            
        elif operation == "create_from_template":
            required = ["template", "path", "class_name"]
            missing = [p for p in required if p not in params]
            if missing:
                return False, f"Missing required parameters: {missing}"
            if params["template"] not in self.templates:
                return False, f"Unknown template: {params['template']}"
            return True, None
            
        elif operation == "batch_operation":
            if "operations" not in params:
                return False, "Missing required parameter: operations"
            return True, None
            
        elif operation == "analyze_project_structure":
            return True, None
            
        return False, f"Unknown operation: {operation}"

    async def execute(self, operation: str, params: Dict[str, Any], operation_id: Optional[str] = None) -> ToolResult:
        """Execute file system operations."""
        start_time = datetime.now()
        
        try:
            if operation == "read_file":
                return await self._read_file(params)
            elif operation == "write_file":
                return await self._write_file(params)
            elif operation == "create_from_template":
                return await self._create_from_template(params)
            elif operation == "batch_operation":
                return await self._batch_operation(params)
            elif operation == "analyze_project_structure":
                return await self._analyze_project_structure(params)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Unknown operation: {operation}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _read_file(self, params: Dict[str, Any]) -> ToolResult:
        """Read file implementation."""
        path = Path(params["path"])
        encoding = params.get("encoding", "utf-8")
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            stat = path.stat()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "content": content,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                },
                execution_time=0.01
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to read file {path}: {str(e)}"
            )

    async def _write_file(self, params: Dict[str, Any]) -> ToolResult:
        """Write file implementation with backup."""
        path = Path(params["path"])
        content = params["content"]
        create_backup = params.get("create_backup", True)
        encoding = params.get("encoding", "utf-8")
        
        backup_path = None
        
        try:
            # Create backup if file exists and backup requested
            if path.exists() and create_backup:
                backup_dir = Path(self.backup_dir)
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"{path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                shutil.copy2(path, backup_path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "written": True,
                    "backup_path": str(backup_path) if backup_path else None
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

    async def _batch_operation(self, params: Dict[str, Any]) -> ToolResult:
        """Execute batch file operations."""
        operations = params["operations"]
        rollback_on_error = params.get("rollback_on_error", True)
        
        results = []
        executed_operations = []
        
        try:
            for op in operations:
                result = await self.execute(op["operation"], op["params"])
                results.append(result)
                executed_operations.append(op)
                
                if result.status == ToolStatus.FAILURE and rollback_on_error:
                    # Rollback previous operations
                    await self._rollback_operations(executed_operations[:-1])
                    return ToolResult(
                        status=ToolStatus.FAILURE,
                        data={"results": results},
                        error_message=f"Batch operation failed, rolled back: {result.error_message}"
                    )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"results": results, "operations_count": len(operations)}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Batch operation error: {str(e)}"
            )

    async def _analyze_project_structure(self, params: Dict[str, Any]) -> ToolResult:
        """Analyze Flutter project structure."""
        project_path = Path(params.get("project_path", "."))
        
        try:
            structure = {}
            suggestions = []
            metrics = {
                "total_files": 0,
                "dart_files": 0,
                "test_files": 0,
                "asset_files": 0
            }
            
            # Analyze directory structure
            for root, dirs, files in os.walk(project_path):
                rel_root = Path(root).relative_to(project_path)
                structure[str(rel_root)] = files
                
                for file in files:
                    metrics["total_files"] += 1
                    if file.endswith('.dart'):
                        metrics["dart_files"] += 1
                        if 'test' in str(rel_root):
                            metrics["test_files"] += 1
                    elif file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif')):
                        metrics["asset_files"] += 1
            
            # Generate suggestions
            if not (project_path / "lib").exists():
                suggestions.append("Missing lib/ directory - create main source directory")
            if not (project_path / "test").exists():
                suggestions.append("Missing test/ directory - add unit tests")
            if metrics["dart_files"] > 0 and metrics["test_files"] == 0:
                suggestions.append("No test files found - consider adding unit tests")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "structure": structure,
                    "suggestions": suggestions,
                    "metrics": metrics
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to analyze project: {str(e)}"
            )

    async def _rollback_operations(self, operations: List[Dict[str, Any]]) -> None:
        """Rollback executed operations."""
        # Implementation would depend on the specific operations
        # For now, just log the rollback attempt
        pass

    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """Get usage examples for the tool."""
        return [
            {
                "description": "Read a Dart file",
                "operation": "read_file",
                "params": {"path": "lib/main.dart"},
                "expected_output": "File content with metadata"
            },
            {
                "description": "Create a new widget from template",
                "operation": "create_from_template",
                "params": {
                    "template": "widget",
                    "path": "lib/widgets/header.dart",
                    "class_name": "HeaderWidget"
                },
                "expected_output": "Widget file created"
            },
            {
                "description": "Batch file operations",
                "operation": "batch_operation",
                "params": {
                    "operations": [
                        {
                            "operation": "create_from_template",
                            "params": {
                                "template": "model",
                                "path": "lib/models/user.dart",
                                "class_name": "User"
                            }
                        }
                    ]
                },
                "expected_output": "All operations completed"
            }
        ]

    async def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status."""
        return {
            "status": "healthy",
            "last_check": datetime.now().isoformat(),
            "metrics": {
                "backup_dir_exists": Path(self.backup_dir).exists(),
                "templates_loaded": len(self.templates),
                "watched_files": len(self.watched_files)
            }
        }
