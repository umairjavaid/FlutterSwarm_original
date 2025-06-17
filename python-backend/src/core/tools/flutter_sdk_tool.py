"""
Flutter SDK Tool for FlutterSwarm Multi-Agent System.

This module provides comprehensive Flutter SDK wrapper functionality for agents
to perform Flutter-specific operations through structured tool interface.
"""

import asyncio
import glob
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base_tool import (
    BaseTool, ToolCategory
)
from ...models.tool_models import (
    ToolCapabilities, ToolOperation, ToolPermission, 
    ToolResult, ToolStatus
)

logger = logging.getLogger(__name__)


class FlutterSDKTool(BaseTool):
    """
    Comprehensive Flutter SDK wrapper tool.
    
    Provides structured access to Flutter SDK operations including:
    - Project creation and management
    - Platform configuration
    - Build and run operations
    - Testing and analysis
    - Package management
    - Development server operations
    """
    
    def __init__(self):
        super().__init__(
            name="flutter_sdk",
            description="Flutter SDK operations wrapper for project management and development.",
            version="2.0.0",
            category=ToolCategory.DEVELOPMENT,
            required_permissions=[
                ToolPermission.PROCESS_SPAWN,
                ToolPermission.FILE_WRITE,
                ToolPermission.DIRECTORY_CREATE
            ]
        )
        self.flutter_executable = self._find_flutter_executable()
        self.dart_executable = self._find_dart_executable()
    
    def _find_flutter_executable(self) -> Optional[str]:
        """Find Flutter executable in system PATH."""
        import shutil
        flutter_path = shutil.which("flutter")
        logger.debug(f"Flutter executable found at: {flutter_path}")
        return flutter_path
    
    def _find_dart_executable(self) -> Optional[str]:
        """Find Dart executable in system PATH."""
        import shutil
        dart_path = shutil.which("dart")
        logger.debug(f"Dart executable found at: {dart_path}")
        return dart_path
    
    async def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities with dynamic examples."""
        return ToolCapabilities(
            operations=[
                ToolOperation(
                    name="create_project",
                    description="Create a new Flutter project based on description",
                    parameters={
                        "project_name": "string",
                        "description": "string", 
                        "platforms": "array",
                        "project_dir": "string"
                    },
                    required_params=["project_name", "description"],
                    example_usage=self._generate_dynamic_examples()
                ),
                ToolOperation(
                    name="add_platform",
                    description="Add platform support to existing Flutter project",
                    parameters={
                        "project_path": "string",
                        "platform": "string"
                    },
                    required_params=["project_path", "platform"]
                ),
                {
                    "name": "build_app",
                    "description": "Build Flutter application for specific platform with advanced configuration",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "platform": {"type": "string", "enum": ["android", "ios", "web", "windows", "macos", "linux"]},
                            "build_mode": {"type": "string", "enum": ["debug", "profile", "release"], "default": "debug"},
                            "target_file": {"type": "string", "description": "Target Dart file to build"},
                            "build_name": {"type": "string", "description": "Build name version"},
                            "build_number": {"type": "string", "description": "Build number"},
                            "flavor": {"type": "string", "description": "Product flavor"},
                            "dart_defines": {"type": "object", "description": "Dart compile-time variables"},
                            "obfuscate": {"type": "boolean", "default": False},
                            "split_debug_info": {"type": "string", "description": "Directory for debug info"}
                        },
                        "required": ["project_path", "platform"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "build_output": {"type": "string"},
                            "artifacts": {"type": "array"},
                            "build_time": {"type": "number"},
                            "warnings": {"type": "array"},
                            "size_analysis": {"type": "object"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.FILE_READ],
                    "examples": ["Build release APK", "Build iOS app", "Build web app"],
                    "error_codes": ["BUILD_FAILED", "PLATFORM_NOT_SUPPORTED", "PROJECT_NOT_FOUND"],
                    "estimated_duration": 180,
                    "supports_cancellation": True
                },
                {
                    "name": "run_app",
                    "description": "Run Flutter application on device/emulator with hot reload support",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "device_id": {"type": "string", "description": "Target device ID"},
                            "platform": {"type": "string", "enum": ["android", "ios", "web", "windows", "macos", "linux"]},
                            "build_mode": {"type": "string", "enum": ["debug", "profile", "release"], "default": "debug"},
                            "target_file": {"type": "string", "description": "Target Dart file to run"},
                            "dart_defines": {"type": "object", "description": "Dart compile-time variables"},
                            "web_port": {"type": "integer", "description": "Web server port"},
                            "web_hostname": {"type": "string", "description": "Web server hostname"},
                            "enable_software_rendering": {"type": "boolean", "default": False}
                        },
                        "required": ["project_path"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "process_id": {"type": "string"},
                            "device_info": {"type": "object"},
                            "app_url": {"type": "string"},
                            "debug_url": {"type": "string"},
                            "hot_reload_enabled": {"type": "boolean"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.NETWORK_ACCESS],
                    "examples": ["Run on emulator", "Run on physical device", "Run web app"],
                    "error_codes": ["RUN_FAILED", "DEVICE_NOT_FOUND", "PROJECT_NOT_FOUND"],
                    "estimated_duration": 60,
                    "supports_cancellation": True
                },
                {
                    "name": "test_app",
                    "description": "Run Flutter tests with coverage reporting and filtering options",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "test_type": {"type": "string", "enum": ["all", "unit", "widget", "integration"], "default": "all"},
                            "test_files": {"type": "array", "items": {"type": "string"}, "description": "Specific test files to run"},
                            "coverage": {"type": "boolean", "default": False, "description": "Generate coverage report"},
                            "coverage_path": {"type": "string", "description": "Custom coverage output path"},
                            "reporter": {"type": "string", "enum": ["compact", "expanded", "json"], "default": "compact"},
                            "concurrency": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4},
                            "update_goldens": {"type": "boolean", "default": False}
                        },
                        "required": ["project_path"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "test_results": {"type": "object"},
                            "coverage_report": {"type": "object"},
                            "failed_tests": {"type": "array"},
                            "execution_time": {"type": "number"},
                            "test_files_run": {"type": "array"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.FILE_READ],
                    "examples": ["Run unit tests", "Generate coverage report", "Run specific test files"],
                    "error_codes": ["TEST_FAILED", "PROJECT_NOT_FOUND", "NO_TESTS_FOUND"],
                    "estimated_duration": 120,
                    "supports_cancellation": True
                },
                {
                    "name": "analyze_code",
                    "description": "Analyze and format Flutter/Dart code with detailed results",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "fix": {"type": "boolean", "default": False, "description": "Apply automatic fixes"},
                            "format": {"type": "boolean", "default": False, "description": "Format code"},
                            "line_length": {"type": "integer", "minimum": 40, "maximum": 200, "default": 80},
                            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to analyze"}
                        },
                        "required": ["project_path"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "analysis_results": {"type": "object"},
                            "issues": {"type": "array"},
                            "formatted_files": {"type": "array"},
                            "fixed_issues": {"type": "array"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.FILE_WRITE],
                    "examples": ["Analyze project", "Format code", "Fix issues automatically"],
                    "error_codes": ["ANALYSIS_FAILED", "PROJECT_NOT_FOUND", "FORMAT_FAILED"],
                    "estimated_duration": 45,
                    "supports_cancellation": True
                },
                {
                    "name": "pub_operations",
                    "description": "Handle Flutter pub package management operations",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "operation": {"type": "string", "enum": ["get", "upgrade", "outdated", "add", "remove"], "description": "Pub operation to perform"},
                            "package_name": {"type": "string", "description": "Package name for add/remove operations"},
                            "version_constraint": {"type": "string", "description": "Version constraint for packages"},
                            "dev_dependency": {"type": "boolean", "default": False, "description": "Add as dev dependency"},
                            "offline": {"type": "boolean", "default": False, "description": "Use offline mode"},
                            "dry_run": {"type": "boolean", "default": False, "description": "Perform dry run"}
                        },
                        "required": ["project_path", "operation"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "operation_result": {"type": "object"},
                            "updated_dependencies": {"type": "array"},
                            "conflicts": {"type": "array"},
                            "pubspec_changes": {"type": "object"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.FILE_WRITE, ToolPermission.NETWORK_ACCESS],
                    "examples": ["Get dependencies", "Add package", "Upgrade packages", "Check outdated"],
                    "error_codes": ["PUB_FAILED", "DEPENDENCY_CONFLICT", "PACKAGE_NOT_FOUND", "PROJECT_NOT_FOUND"],
                    "estimated_duration": 90,
                    "supports_cancellation": True
                },
                {
                    "name": "clean_project",
                    "description": "Clean Flutter project build artifacts with selective options",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "project_path": {"type": "string", "description": "Path to Flutter project"},
                            "deep_clean": {"type": "boolean", "default": False, "description": "Perform deep clean including caches"}
                        },
                        "required": ["project_path"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "cleaned_directories": {"type": "array"},
                            "space_freed": {"type": "string"},
                            "cache_cleared": {"type": "boolean"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.FILE_DELETE],
                    "examples": ["Clean build artifacts", "Deep clean with caches", "Free disk space"],
                    "error_codes": ["CLEAN_FAILED", "PROJECT_NOT_FOUND", "PERMISSION_DENIED"],
                    "estimated_duration": 30,
                    "supports_cancellation": True
                },
                {
                    "name": "doctor",
                    "description": "Check Flutter environment health with detailed diagnostics",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "verbose": {"type": "boolean", "default": False, "description": "Show verbose output"},
                            "android_licenses": {"type": "boolean", "default": False, "description": "Check Android licenses"}
                        }
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "flutter_version": {"type": "string"},
                            "dart_version": {"type": "string"},
                            "platform_status": {"type": "object"},
                            "issues": {"type": "array"},
                            "environment_vars": {"type": "object"},
                            "connected_devices": {"type": "array"}
                        }
                    },
                    "required_permissions": [ToolPermission.PROCESS_SPAWN, ToolPermission.SYSTEM_INFO],
                    "examples": ["Check environment", "Validate setup", "List devices"],
                    "error_codes": ["DOCTOR_FAILED", "PERMISSION_DENIED"],
                    "estimated_duration": 20,
                    "supports_cancellation": True
                }
            ],
            supported_file_types=[".dart", ".yaml", ".json", ".gradle", ".xml"],
            performance_notes=[
                "Project creation time varies based on platform count and dependencies",
                "Platform addition requires network access for dependencies",
                "Build operations can be CPU intensive"
            ]
        )
    
    def _generate_dynamic_examples(self) -> List[Dict[str, Any]]:
        """Generate dynamic examples for project creation."""
        examples = [
            {
                "description": "Create a weather app",
                "parameters": {
                    "project_name": "weather_app",
                    "description": "A weather application that shows current weather and forecasts",
                    "platforms": ["android", "ios"]
                }
            },
            {
                "description": "Create a todo list app",
                "parameters": {
                    "project_name": "todo_app",
                    "description": "A task management app with todo lists and reminders",
                    "platforms": ["android", "ios", "web"]
                }
            },
            {
                "description": "Create an e-commerce app",
                "parameters": {
                    "project_name": "shopping_app",
                    "description": "E-commerce application with product catalog and shopping cart",
                    "platforms": ["android", "ios"]
                }
            }
        ]
        return examples

    async def validate_params(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate operation parameters dynamically."""
        if operation == "create_project":
            return self._validate_create_project_params(params)
        elif operation == "add_platform":
            return self._validate_platform_params(params)
        elif operation == "build":
            return self._validate_build_params(params)
        else:
            return True, None
    
    def _validate_create_project_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate project creation parameters."""
        project_name = params.get("project_name")
        if not project_name:
            return False, "project_name is required"
        
        description = params.get("description")
        if not description:
            return False, "description is required"
        
        # Validate project name format
        import re
        if not re.match(r'^[a-z][a-z0-9_]*$', project_name):
            return False, "project_name must be lowercase with underscores only"
        
        # Validate platforms if provided
        platforms = params.get("platforms", [])
        if platforms and not isinstance(platforms, list):
            return False, "platforms must be an array"
        
        supported_platforms = ["android", "ios", "web", "windows", "macos", "linux"]
        for platform in platforms:
            if platform not in supported_platforms:
                return False, f"platform '{platform}' is not supported. Must be one of: {', '.join(supported_platforms)}"
        
        return True, None
    
    def _validate_platform_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate platform addition parameters."""
        platform = params.get("platform")
        supported_platforms = ["android", "ios", "web", "windows", "macos", "linux"]
        
        if platform not in supported_platforms:
            return False, f"platform must be one of: {', '.join(supported_platforms)}"
        
        return True, None
    
    def _validate_build_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate build parameters."""
        project_path = params.get("project_path")
        if not project_path:
            return False, "project_path is required"
        
        return True, None
    
    async def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """Execute Flutter SDK operations dynamically."""
        
        # Validate parameters
        is_valid, error_msg = await self.validate_params(operation, params)
        if not is_valid:
            return ToolResult(
                status=ToolStatus.FAILURE,  # Fixed enum name
                error_message=error_msg,
                operation_id=operation_id
            )
        
        try:
            if operation == "create_project":
                return await self._create_project_dynamic(params, operation_id)
            elif operation == "add_platform":
                return await self._add_platform(params, operation_id)
            elif operation == "build":
                return await self._build_app(params, operation_id)
            elif operation == "run_app":
                return await self._run_app(params, operation_id)
            elif operation == "test_app":
                return await self._test_app(params, operation_id)
            elif operation == "analyze_code":
                return await self._analyze_code(params, operation_id)
            elif operation == "pub_operations":
                return await self._pub_operations(params, operation_id)
            elif operation == "clean_project":
                return await self._clean_project(params, operation_id)
            elif operation == "doctor":
                return await self._doctor(params, operation_id)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error_message=f"Unknown operation: {operation}",
                    operation_id=operation_id
                )
        except Exception as e:
            logger.error(f"Flutter SDK operation failed: {e}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                operation_id=operation_id
            )
    
    async def _create_project_dynamic(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Create Flutter project using standard Flutter commands without templates."""
        from pathlib import Path
        from ...config.settings import settings
        
        project_name = params["project_name"]
        description = params.get("description", f"A new Flutter project: {project_name}")
        platforms = params.get("platforms", ["android", "ios"])
        project_dir = params.get("project_dir")
        overwrite = params.get("overwrite", False)
        
        try:
            # Determine output directory - Use flutter_projects from workspace root
            if project_dir:
                output_dir = Path(project_dir)
            else:
                # Use the flutter_projects directory from the workspace root
                # Current file: .../python-backend/src/core/tools/flutter_sdk_tool.py
                # Workspace root: .../FlutterSwarm (4 levels up from src/core/tools/)
                workspace_root = Path(__file__).parent.parent.parent.parent
                flutter_projects_dir = workspace_root / "flutter_projects"
                output_dir = flutter_projects_dir
            
            output_dir.mkdir(parents=True, exist_ok=True)
            project_path = output_dir / project_name
            
            # Check if project already exists
            if project_path.exists():
                if overwrite:
                    logger.info(f"Project {project_name} already exists, but overwrite=True, so removing existing project")
                    import shutil
                    shutil.rmtree(str(project_path))
                else:
                    return ToolResult(
                        status=ToolStatus.FAILURE,
                        error_message=f"Project {project_name} already exists at {project_path}",
                        operation_id=operation_id
                    )
            
            # Create Flutter project using standard command WITHOUT templates
            create_cmd = [self.flutter_executable, "create"]
            
            # Add platforms if specified
            if platforms:
                platforms_str = ",".join(platforms)
                create_cmd.extend(["--platforms", platforms_str])
            
            # Add description
            create_cmd.extend(["--description", description])
            
            # Add project name
            create_cmd.append(project_name)
            
            logger.info(f"Creating Flutter project: {' '.join(create_cmd)}")
            logger.info(f"Working directory: {output_dir}")
            logger.info(f"Project will be created at: {project_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Ensure we're in the right directory
            os.chdir(str(output_dir))
            logger.info(f"Changed to directory: {os.getcwd()}")
            
            # Execute flutter create command
            result = await self._run_flutter_command(
                create_cmd,
                cwd=str(output_dir),
                operation_id=operation_id,
                timeout=120
            )
            
            if result.status != ToolStatus.SUCCESS:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error_message=f"Flutter create failed: {result.error_message}",
                    operation_id=operation_id
                )
            
            # Verify project was created
            if not project_path.exists() or not (project_path / "pubspec.yaml").exists():
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error_message=f"Project was not created successfully at {project_path}",
                    operation_id=operation_id
                )
            
            # Get project info
            project_info = {
                "project_path": str(project_path),
                "project_name": project_name,
                "description": description,
                "platforms": platforms,
                "pubspec_file": str(project_path / "pubspec.yaml"),
                "main_file": str(project_path / "lib" / "main.dart"),
                "output_directory": str(output_dir)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=project_info,
                metadata={
                    "flutter_version": self._get_flutter_version(),
                    "creation_method": "standard_flutter_create_no_templates",
                    "template_removed": True
                },
                operation_id=operation_id
            )
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                operation_id=operation_id
            )
    
    def _get_flutter_version(self) -> str:
        """Get Flutter version."""
        try:
            import subprocess
            result = subprocess.run(
                [self.flutter_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.split('\n')[0] if result.returncode == 0 else "unknown"
        except:
            return "unknown"

    async def _add_platform(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Add platform support to existing project."""
        project_path = params["project_path"]
        platforms = params["platforms"]
        
        cmd = [self.flutter_executable, "config", "--enable-web"]
        
        # Add platforms one by one
        results = []
        added_platforms = []
        
        for platform in platforms:
            platform_cmd = [self.flutter_executable, "create", "--platforms", platform, "."]
            result = await self._run_flutter_command(
                platform_cmd,
                cwd=project_path,
                operation_id=operation_id
            )
            
            if result.status == ToolStatus.SUCCESS:
                added_platforms.append(platform)
            results.append(result)
        
        # Check for configuration files
        config_files = []
        platform_dirs = {
            "android": "android",
            "ios": "ios",
            "web": "web",
            "windows": "windows",
            "macos": "macos",
            "linux": "linux"
        }
        
        for platform in added_platforms:
            platform_dir = os.path.join(project_path, platform_dirs.get(platform, platform))
            if os.path.exists(platform_dir):
                config_files.append(platform_dir)
        
        return self._create_success_result(
            "add_platform",
            {
                "added_platforms": added_platforms,
                "configuration_files": config_files,
                "required_setup": [
                    "Configure platform-specific settings",
                    "Set up signing for iOS/Android if needed",
                    "Install platform-specific dependencies"
                ]
            },
            operation_id=operation_id
        )
    
    async def _build_app(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Build Flutter application."""
        project_path = params["project_path"]
        platform = params["platform"]
        build_mode = params.get("build_mode", "debug")
        
        cmd = [self.flutter_executable, "build", platform]
        
        if build_mode != "debug":
            cmd.extend([f"--{build_mode}"])
        
        # Add optional parameters
        if params.get("target_file"):
            cmd.extend(["--target", params["target_file"]])
        
        if params.get("build_name"):
            cmd.extend(["--build-name", params["build_name"]])
        
        if params.get("build_number"):
            cmd.extend(["--build-number", params["build_number"]])
        
        if params.get("flavor"):
            cmd.extend(["--flavor", params["flavor"]])
        
        if params.get("dart_defines"):
            for key, value in params["dart_defines"].items():
                cmd.extend(["--dart-define", f"{key}={value}"])
        
        if params.get("obfuscate"):
            cmd.append("--obfuscate")
        
        if params.get("split_debug_info"):
            cmd.extend(["--split-debug-info", params["split_debug_info"]])
        
        # Execute build command
        start_time = datetime.utcnow()
        result = await self._run_flutter_command(
            cmd,
            cwd=project_path,
            operation_id=operation_id,
            timeout=300  # 5 minutes timeout for builds
        )
        
        if result.status == ToolStatus.SUCCESS:
            build_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Find build artifacts
            artifacts = self._find_build_artifacts(project_path, platform, build_mode)
            
            result.data = {
                "build_output": result.data.get("stdout", ""),
                "artifacts": artifacts,
                "build_time": build_time,
                "warnings": self._extract_warnings(result.data.get("stdout", "")),
                "size_analysis": self._analyze_build_size(artifacts)
            }
        
        return result
    
    async def _run_app(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Run Flutter application with hot reload."""
        project_path = params["project_path"]
        device_id = params.get("device_id")
        platform = params.get("platform")
        build_mode = params.get("build_mode", "debug")
        
        cmd = [self.flutter_executable, "run"]
        
        if device_id:
            cmd.extend(["-d", device_id])
        
        if build_mode != "debug":
            cmd.extend([f"--{build_mode}"])
        
        if params.get("target_file"):
            cmd.extend(["--target", params["target_file"]])
        
        if params.get("dart_defines"):
            for key, value in params["dart_defines"].items():
                cmd.extend(["--dart-define", f"{key}={value}"])
        
        if params.get("web_port"):
            cmd.extend(["--web-port", str(params["web_port"])])
        
        if params.get("web_hostname"):
            cmd.extend(["--web-hostname", params["web_hostname"]])
        
        if params.get("enable_software_rendering"):
            cmd.append("--enable-software-rendering")
        
        # Start the process
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            # Store process for later control
            proc_id = operation_id or str(uuid.uuid4())
            self.running_processes[proc_id] = process
            
            # Wait for initial startup and extract info
            startup_output = ""
            try:
                startup_data = await asyncio.wait_for(
                    process.stdout.read(4096),
                    timeout=30
                )
                startup_output = startup_data.decode('utf-8', errors='ignore')
            except asyncio.TimeoutError:
                pass
            
            # Extract connection info
            device_info = self._extract_device_info(startup_output)
            app_url = self._extract_app_url(startup_output)
            debug_url = self._extract_debug_url(startup_output)
            
            return self._create_success_result(
                "run_app",
                {
                    "process_id": proc_id,
                    "device_info": device_info,
                    "app_url": app_url,
                    "debug_url": debug_url,
                    "hot_reload_enabled": build_mode == "debug"
                },
                operation_id=operation_id
            )
        
        except Exception as e:
            return self._create_error_result(
                "run_app",
                f"Failed to start Flutter app: {str(e)}",
                "RUN_FAILED",
                operation_id
            )
    
    async def _test_app(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Run Flutter tests."""
        project_path = params["project_path"]
        test_type = params.get("test_type", "all")
        test_files = params.get("test_files", [])
        coverage = params.get("coverage", False)
        
        cmd = [self.flutter_executable, "test"]
        
        if test_files:
            cmd.extend(test_files)
        elif test_type != "all":
            # Add specific test directory based on type
            test_dirs = {
                "unit": "test/unit",
                "widget": "test/widget", 
                "integration": "test_driver"
            }
            if test_type in test_dirs:
                test_dir = os.path.join(project_path, test_dirs[test_type])
                if os.path.exists(test_dir):
                    cmd.append(test_dirs[test_type])
        
        if coverage:
            cmd.append("--coverage")
            if params.get("coverage_path"):
                cmd.extend(["--coverage-path", params["coverage_path"]])
        
        if params.get("reporter"):
            cmd.extend(["--reporter", params["reporter"]])
        
        if params.get("concurrency"):
            cmd.extend(["--concurrency", str(params["concurrency"])])
        
        if params.get("update_goldens"):
            cmd.append("--update-goldens")
        
        # Execute test command
        result = await self._run_flutter_command(
            cmd,
            cwd=project_path,
            operation_id=operation_id,
            timeout=180  # 3 minutes timeout for tests
        )
        
        if result.status == ToolStatus.SUCCESS:
            # Parse test output
            test_output = result.data.get("stdout", "")
            test_results = self._parse_test_results(test_output)
            
            # Read coverage report if generated
            coverage_report = {}
            if coverage:
                coverage_file = os.path.join(project_path, "coverage", "lcov.info")
                if os.path.exists(coverage_file):
                    coverage_report = self._parse_coverage_report(coverage_file)
            
            result.data = {
                "test_results": test_results,
                "coverage_report": coverage_report,
                "failed_tests": test_results.get("failed", []),
                "execution_time": test_results.get("total_time", 0),
                "test_files_run": test_results.get("files", [])
            }
        
        return result
    
    async def _analyze_code(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Analyze Flutter/Dart code."""
        project_path = params["project_path"]
        fix_issues = params.get("fix", False)
        format_code = params.get("format", False)
        files = params.get("files", [])
        
        results = {}
        
        # Run dart analyze
        analyze_cmd = [self.dart_executable, "analyze"]
        if files:
            analyze_cmd.extend(files)
        else:
            analyze_cmd.append(".")
        
        analyze_result = await self._run_flutter_command(
            analyze_cmd,
            cwd=project_path,
            operation_id=operation_id
        )
        
        if analyze_result.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS]:
            results["analysis_results"] = {
                "stdout": analyze_result.data.get("stdout", ""),
                "stderr": analyze_result.data.get("stderr", "")
            }
            results["issues"] = self._parse_analysis_issues(analyze_result.data.get("stdout", ""))
        
        # Format code if requested
        formatted_files = []
        if format_code:
            format_cmd = [self.dart_executable, "format"]
            if params.get("line_length"):
                format_cmd.extend(["--line-length", str(params["line_length"])])
            
            if files:
                format_cmd.extend(files)
            else:
                format_cmd.append(".")
            
            format_result = await self._run_flutter_command(
                format_cmd,
                cwd=project_path,
                operation_id=operation_id
            )
            
            if format_result.status == ToolStatus.SUCCESS:
                formatted_files = self._extract_formatted_files(format_result.data.get("stdout", ""))
        
        # Fix issues if requested
        fixed_issues = []
        if fix_issues:
            fix_cmd = [self.dart_executable, "fix", "--apply"]
            
            fix_result = await self._run_flutter_command(
                fix_cmd,
                cwd=project_path,
                operation_id=operation_id
            )
            
            if fix_result.status == ToolStatus.SUCCESS:
                fixed_issues = self._extract_fixed_issues(fix_result.data.get("stdout", ""))
        
        return self._create_success_result(
            "analyze_code",
            {
                "analysis_results": results.get("analysis_results", {}),
                "issues": results.get("issues", []),
                "formatted_files": formatted_files,
                "fixed_issues": fixed_issues
            },
            operation_id=operation_id
        )
    
    async def _pub_operations(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Handle Flutter pub operations."""
        project_path = params["project_path"]
        operation = params["operation"]
        
        cmd = [self.flutter_executable, "pub", operation]
        
        if operation in ["add", "remove"]:
            package_name = params["package_name"]
            if params.get("dev_dependency"):
                cmd.append("--dev")
            if params.get("version_constraint"):
                package_name += f":{params['version_constraint']}"
            cmd.append(package_name)
        
        if params.get("offline"):
            cmd.append("--offline")
        
        if params.get("dry_run"):
            cmd.append("--dry-run")
        
        # Execute pub command
        result = await self._run_flutter_command(
            cmd,
            cwd=project_path,
            operation_id=operation_id,
            timeout=120  # 2 minutes timeout for pub operations
        )
        
        if result.status == ToolStatus.SUCCESS:
            output = result.data.get("stdout", "")
            
            # Parse operation results
            updated_deps = self._extract_updated_dependencies(output)
            conflicts = self._extract_dependency_conflicts(output)
            
            # Read updated pubspec.yaml
            pubspec_changes = {}
            pubspec_path = os.path.join(project_path, "pubspec.yaml")
            if os.path.exists(pubspec_path):
                try:
                    import yaml
                    with open(pubspec_path, 'r') as f:
                        pubspec_changes = yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"Could not read updated pubspec.yaml: {e}")
            
            result.data = {
                "operation_result": {
                    "operation": operation,
                    "stdout": output,
                    "stderr": result.data.get("stderr", "")
                },
                "updated_dependencies": updated_deps,
                "conflicts": conflicts,
                "pubspec_changes": pubspec_changes
            }
        
        return result
    
    async def _clean_project(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Clean Flutter project build artifacts."""
        project_path = params["project_path"]
        deep_clean = params.get("deep_clean", False)
        
        # Standard flutter clean
        cmd = [self.flutter_executable, "clean"]
        
        result = await self._run_flutter_command(
            cmd,
            cwd=project_path,
            operation_id=operation_id
        )
        
        cleaned_dirs = ["build/"]
        space_freed = "Unknown"
        
        if result.status == ToolStatus.SUCCESS:
            # Additional cleaning if requested
            if deep_clean:
                additional_dirs = [
                    ".dart_tool",
                    "ios/Pods",
                    "ios/.symlinks",
                    "android/.gradle",
                    ".flutter-plugins",
                    ".flutter-plugins-dependencies"
                ]
                
                for dir_name in additional_dirs:
                    dir_path = os.path.join(project_path, dir_name)
                    if os.path.exists(dir_path):
                        try:
                            if os.path.isdir(dir_path):
                                shutil.rmtree(dir_path)
                            else:
                                os.remove(dir_path)
                            cleaned_dirs.append(dir_name)
                        except Exception as e:
                            logger.warning(f"Could not remove {dir_path}: {e}")
            
            result.data = {
                "cleaned_directories": cleaned_dirs,
                "space_freed": space_freed,
                "cache_cleared": True
            }
        
        return result
    
    async def _doctor(self, params: Dict[str, Any], operation_id: Optional[str]) -> ToolResult:
        """Check Flutter environment health."""
        cmd = [self.flutter_executable, "doctor"]
        
        if params.get("verbose"):
            cmd.append("-v")
        
        if params.get("android_licenses"):
            cmd.append("--android-licenses")
        
        result = await self._run_flutter_command(
            cmd,
            operation_id=operation_id,
            timeout=60
        )
        
        if result.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS]:
            doctor_output = result.data.get("stdout", "")
            
            # Parse doctor output
            flutter_version = self._extract_flutter_version(doctor_output)
            dart_version = self._extract_dart_version(doctor_output)
            platform_status = self._parse_platform_status(doctor_output)
            issues = self._extract_doctor_issues(doctor_output)
            
            # Get connected devices
            devices_cmd = [self.flutter_executable, "devices", "--machine"]
            devices_result = await self._run_flutter_command(devices_cmd, operation_id=operation_id)
            connected_devices = []
            
            if devices_result.status == ToolStatus.SUCCESS:
                try:
                    devices_json = devices_result.data.get("stdout", "[]")
                    connected_devices = json.loads(devices_json)
                except json.JSONDecodeError:
                    pass
            
            result.data = {
                "flutter_version": flutter_version,
                "dart_version": dart_version,
                "platform_status": platform_status,
                "issues": issues,
                "environment_vars": dict(os.environ),
                "connected_devices": connected_devices
            }
        
        return result
    
    async def _run_flutter_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        operation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ToolResult:
        """Run a Flutter command and return structured result."""
        op_id = operation_id or str(uuid.uuid4())
        
        try:
            logger.info(f"Executing Flutter command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                return self._create_error_result(
                    cmd[1] if len(cmd) > 1 else "flutter_command",
                    "Command timed out",
                    "PROCESS_TIMEOUT",
                    op_id
                )
            
            stdout_str = stdout.decode('utf-8', errors='ignore')
            stderr_str = stderr.decode('utf-8', errors='ignore')
            
            # Determine status based on exit code
            if process.returncode == 0:
                status = ToolStatus.SUCCESS
            else:
                status = ToolStatus.FAILURE
            
            result = ToolResult(
                status=status,
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": process.returncode,
                    "command": cmd
                },
                operation_id=op_id
            )
            
            if status == ToolStatus.FAILURE:
                result.error_message = stderr_str or "Command failed"
                result.error_code = f"EXIT_CODE_{process.returncode}"
            
            result.mark_completed()
            return result
            
        except Exception as e:
            logger.exception(f"Error running Flutter command: {cmd}")
            return self._create_error_result(
                cmd[1] if len(cmd) > 1 else "flutter_command",
                f"Failed to execute command: {str(e)}",
                "EXECUTION_ERROR",
                op_id
            )
    
    # Helper methods for parsing command outputs
    def _find_build_artifacts(self, project_path: str, platform: str, build_mode: str) -> List[str]:
        """Find build artifacts for the specified platform and mode."""
        artifacts = []
        build_dir = os.path.join(project_path, "build")
        
        if not os.path.exists(build_dir):
            return artifacts
        
        # Platform-specific artifact paths
        artifact_patterns = {
            "android": ["app/outputs/flutter-apk/*.apk", "app/outputs/bundle/release/*.aab"],
            "ios": ["ios/iphoneos/*.app", "ios/Runner.xcarchive"],
            "web": ["web/*"],
            "windows": ["windows/runner/Release/*"],
            "macos": ["macos/Build/Products/Release/*"],
            "linux": ["linux/release/bundle/*"]
        }
        
        import glob
        patterns = artifact_patterns.get(platform, [])
        
        for pattern in patterns:
            full_pattern = os.path.join(build_dir, pattern)
            artifacts.extend(glob.glob(full_pattern))
        
        return artifacts
    
    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warnings from build output."""
        warnings = []
        lines = output.split('\n')
        
        for line in lines:
            if 'warning:' in line.lower() or 'warn:' in line.lower():
                warnings.append(line.strip())
        
        return warnings
    
    def _analyze_build_size(self, artifacts: List[str]) -> Dict[str, Any]:
        """Analyze build artifact sizes."""
        size_info = {}
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                size = os.path.getsize(artifact)
                size_info[os.path.basename(artifact)] = {
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                }
        
        return size_info
    
    def _extract_device_info(self, output: str) -> Dict[str, Any]:
        """Extract device information from run output."""
        # Parse device info from Flutter run output
        device_info = {}
        
        lines = output.split('\n')
        for line in lines:
            if 'Launching' in line and 'on' in line:
                # Example: "Launching lib/main.dart on iPhone 12 Pro Max in debug mode..."
                parts = line.split(' on ')
                if len(parts) > 1:
                    device_info['device'] = parts[1].split(' in ')[0]
                    break
        
        return device_info
    
    def _extract_app_url(self, output: str) -> Optional[str]:
        """Extract app URL from run output (for web apps)."""
        lines = output.split('\n')
        for line in lines:
            if 'http://localhost:' in line or 'http://127.0.0.1:' in line:
                import re
                url_match = re.search(r'http://[^\s]+', line)
                if url_match:
                    return url_match.group(0)
        return None
    
    def _extract_debug_url(self, output: str) -> Optional[str]:
        """Extract debug URL from run output."""
        lines = output.split('\n')
        for line in lines:
            if 'Observatory' in line or 'DevTools' in line:
                import re
                url_match = re.search(r'http://[^\s]+', line)
                if url_match:
                    return url_match.group(0)
        return None
    
    def _parse_test_results(self, output: str) -> Dict[str, Any]:
        """Parse test execution results."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "files": [],
            "failed_tests": [],
            "total_time": 0
        }
        
        lines = output.split('\n')
        for line in lines:
            if 'All tests passed!' in line:
                results["passed"] = results["total"]
            elif 'tests passed' in line:
                import re
                match = re.search(r'(\d+) tests passed', line)
                if match:
                    results["passed"] = int(match.group(1))
            elif 'test failed' in line or 'tests failed' in line:
                import re
                match = re.search(r'(\d+) tests? failed', line)
                if match:
                    results["failed"] = int(match.group(1))
        
        results["total"] = results["passed"] + results["failed"]
        return results
    
    def _parse_coverage_report(self, coverage_file: str) -> Dict[str, Any]:
        """Parse LCOV coverage report."""
        coverage = {"files": [], "overall_percentage": 0}
        
        try:
            with open(coverage_file, 'r') as f:
                content = f.read()
            
            # Basic LCOV parsing
            files = content.split('end_of_record')
            for file_section in files:
                if 'SF:' in file_section:
                    lines = file_section.split('\n')
                    file_info = {}
                    
                    for line in lines:
                        if line.startswith('SF:'):
                            file_info['file'] = line[3:]
                        elif line.startswith('LF:'):
                            file_info['lines_total'] = int(line[3:])
                        elif line.startswith('LH:'):
                            file_info['lines_covered'] = int(line[3:])
                    
                    if 'lines_total' in file_info and file_info['lines_total'] > 0:
                        file_info['coverage_percentage'] = (
                            file_info['lines_covered'] / file_info['lines_total'] * 100
                        )
                        coverage["files"].append(file_info)
            
            # Calculate overall coverage
            if coverage["files"]:
                total_lines = sum(f.get('lines_total', 0) for f in coverage["files"])
                covered_lines = sum(f.get('lines_covered', 0) for f in coverage["files"])
                if total_lines > 0:
                    coverage["overall_percentage"] = covered_lines / total_lines * 100
        
        except Exception as e:
            logger.warning(f"Could not parse coverage report: {e}")
        
        return coverage
    
    def _parse_analysis_issues(self, output: str) -> List[Dict[str, Any]]:
        """Parse Dart analysis issues."""
        issues = []
        lines = output.split('\n')
        
        for line in lines:
            if ' • ' in line and (' error ' in line or ' warning ' in line or ' info ' in line):
                # Parse analysis issue format
                parts = line.split(' • ')
                if len(parts) >= 2:
                    issue = {
                        "message": parts[0].strip(),
                        "location": parts[1].strip() if len(parts) > 1 else "",
                        "type": "error" if "error" in line else "warning" if "warning" in line else "info"
                    }
                    issues.append(issue)
        
        return issues
    
    def _extract_formatted_files(self, output: str) -> List[str]:
        """Extract list of formatted files."""
        files = []
        lines = output.split('\n')
        
        for line in lines:
            if line.startswith('Formatted ') and line.endswith('.dart'):
                # Extract file path
                file_path = line.replace('Formatted ', '').strip()
                files.append(file_path)
        
        return files
    
    def _extract_fixed_issues(self, output: str) -> List[str]:
        """Extract list of fixed issues."""
        fixes = []
        lines = output.split('\n')
        
        for line in lines:
            if 'Fixed ' in line:
                fixes.append(line.strip())
        
        return fixes
    
    def _extract_updated_dependencies(self, output: str) -> List[str]:
        """Extract updated dependencies from pub output."""
        updated = []
        lines = output.split('\n')
        
        for line in lines:
            if '+ ' in line or '! ' in line or '> ' in line:
                # Parse dependency change line
                updated.append(line.strip())
        
        return updated
    
    def _extract_dependency_conflicts(self, output: str) -> List[str]:
        """Extract dependency conflicts from pub output."""
        conflicts = []
        lines = output.split('\n')
        
        for line in lines:
            if 'conflict' in line.lower() or 'incompatible' in line.lower():
                conflicts.append(line.strip())
        
        return conflicts
    
    def _extract_flutter_version(self, output: str) -> str:
        """Extract Flutter version from doctor output."""
        lines = output.split('\n')
        for line in lines:
            if 'Flutter' in line and ('channel' in line or 'version' in line):
                return line.strip()
        return "Unknown"
    
    def _extract_dart_version(self, output: str) -> str:
        """Extract Dart version from doctor output."""
        lines = output.split('\n')
        for line in lines:
            if 'Dart' in line and 'version' in line:
                return line.strip()
        return "Unknown"
    
    def _parse_platform_status(self, output: str) -> Dict[str, str]:
        """Parse platform status from doctor output."""
        status = {}
        lines = output.split('\n')
        
        for line in lines:
            if line.startswith('[✓]') or line.startswith('[✗]') or line.startswith('[!]'):
                # Parse status line
                status_symbol = line[1]
                platform = line[4:].split('•')[0].strip()
                
                if status_symbol == '✓':
                    status[platform] = "available"
                elif status_symbol == '✗':
                    status[platform] = "unavailable"
                elif status_symbol == '!':
                    status[platform] = "issues"
        
        return status
    
    def _extract_doctor_issues(self, output: str) -> List[str]:
        """Extract issues from doctor output."""
        issues = []
        lines = output.split('\n')
        
        for line in lines:
            if line.startswith('  ✗') or line.startswith('  !'):
                issues.append(line.strip())
        
        return issues
    
    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """Get comprehensive usage examples for Flutter SDK tool."""
        return [
            {
                "title": "Create a weather app",
                "description": "Create a Flutter weather application with real-time weather data",
                "operation": "create_project",
                "params": {
                    "project_name": "weather_app",
                    "description": "A beautiful weather application with real-time data and forecasts",
                    "platforms": ["android", "ios", "web"]
                },
                "expected_outcome": "New Flutter weather app project ready for development",
                "follow_up_actions": ["Implement weather API integration", "Design weather UI", "Add location services"]
            },
            {
                "title": "Create a todo list app",
                "description": "Create a Flutter todo list application with local storage",
                "operation": "create_project",
                "params": {
                    "project_name": "todo_app",
                    "description": "A productivity todo application with task management and reminders",
                    "platforms": ["android", "ios"]
                },
                "expected_outcome": "New Flutter todo app project with basic structure",
                "follow_up_actions": ["Implement task CRUD operations", "Add local database", "Design task UI"]
            },
            {
                "title": "Create a chat application",
                "description": "Create a Flutter real-time chat application",
                "operation": "create_project",
                "params": {
                    "project_name": "chat_app",
                    "description": "A real-time messaging application with Firebase integration",
                    "platforms": ["android", "ios", "web"]
                },
                "expected_outcome": "New Flutter chat app project structure created",
                "follow_up_actions": ["Set up Firebase", "Implement messaging UI", "Add real-time features"]
            },
            {
                "title": "Build release APK",
                "description": "Build optimized release APK for Android distribution",
                "operation": "build_app",
                "params": {
                    "project_path": "./weather_app",
                    "platform": "android",
                    "build_mode": "release",
                    "obfuscate": True,
                    "build_name": "1.0.0",
                    "build_number": "1"
                },
                "expected_outcome": "Optimized APK file ready for distribution",
                "follow_up_actions": ["Test APK on devices", "Upload to Play Store"]
            },
            {
                "title": "Run tests with coverage",
                "description": "Execute all tests and generate coverage report",
                "operation": "test_app",
                "params": {
                    "project_path": "./weather_app",
                    "test_type": "all",
                    "coverage": True,
                    "reporter": "expanded"
                },
                "expected_outcome": "Test results and coverage report",
                "follow_up_actions": ["Review failing tests", "Improve test coverage"]
            },
            {
                "title": "Add new dependency",
                "description": "Add a new package dependency to the project",
                "operation": "pub_operations",
                "params": {
                    "project_path": "./weather_app",
                    "operation": "add",
                    "package_name": "http",
                    "version_constraint": "^0.13.0"
                },
                "expected_outcome": "Package added to pubspec.yaml and downloaded",
                "follow_up_actions": ["Import package in code", "Use package features"]
            },
            {
                "title": "Check environment health",
                "description": "Verify Flutter installation and platform setup",
                "operation": "doctor",
                "params": {
                    "verbose": True
                },
                "expected_outcome": "Detailed environment status report",
                "follow_up_actions": ["Fix any reported issues", "Install missing SDKs"]
            }
        ]
    
    async def _health_check_impl(self) -> bool:
        """Implement Flutter-specific health check."""
        if not self.flutter_executable:
            return False
        
        try:
            # Quick flutter --version check
            process = await asyncio.create_subprocess_exec(
                self.flutter_executable, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                await asyncio.wait_for(process.communicate(), timeout=10)
                return process.returncode == 0
            except asyncio.TimeoutError:
                process.terminate()
                return False
        
        except Exception:
            return False
    
    async def _cancel_operation_impl(self, operation_id: str) -> bool:
        """Cancel a running Flutter operation."""
        if operation_id in self.running_processes:
            process = self.running_processes[operation_id]
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                del self.running_processes[operation_id]
                return True
            except Exception as e:
                logger.error(f"Error cancelling operation {operation_id}: {e}")
                return False
        
        return False
    
    def _create_success_result(
        self,
        operation: str,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """Create a successful tool result."""
        return ToolResult(
            operation_id=operation_id or str(uuid.uuid4()),
            status=ToolStatus.SUCCESS,
            data=data,
            metadata=metadata or {}
        )
    
    def _create_error_result(
        self,
        operation: str,
        error_message: str,
        error_code: str,
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """Create an error tool result."""
        return ToolResult(
            operation_id=operation_id or str(uuid.uuid4()),
            status=ToolStatus.FAILURE,
            error_message=error_message,
            metadata={"error_code": error_code, "operation": operation}
        )
    
    def _create_partial_success_result(
        self,
        operation: str,
        data: Any = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """Create a partial success tool result."""
        return ToolResult(
            operation_id=operation_id or str(uuid.uuid4()),
            status=ToolStatus.PARTIAL_SUCCESS,
            data=data,
            warnings=warnings or [],
            metadata=metadata or {}
        )

    def _validate_common_params(self, params: Dict[str, Any], required_params: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate common parameters across operations."""
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        return True, None
    
    def _calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, IOError):
            pass
        return total_size

    def _remove_directory_safely(self, directory: str) -> bool:
        """Safely remove directory and its contents."""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                return True
        except (OSError, IOError) as e:
            logger.warning(f"Could not remove directory {directory}: {e}")
        return False

    def _parse_lcov_file_section(self, file_section: str) -> Dict[str, Any]:
        """Parse individual file section from LCOV coverage report."""
        file_info = {'filename': '', 'lines_total': 0, 'lines_covered': 0}
        
        lines = file_section.split('\n')
        for line in lines:
            if line.startswith('SF:'):
                file_info['filename'] = line[3:].strip()
            elif line.startswith('LF:'):
                file_info['lines_total'] = int(line[3:])
            elif line.startswith('LH:'):
                file_info['lines_covered'] = int(line[3:])
        
        # Calculate coverage percentage
        if file_info['lines_total'] > 0:
            file_info['coverage_percentage'] = (file_info['lines_covered'] / file_info['lines_total']) * 100
        else:
            file_info['coverage_percentage'] = 0
            
        return file_info