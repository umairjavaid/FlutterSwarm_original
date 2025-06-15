"""
Process Management Tool for FlutterSwarm Multi-Agent System.

Handles long-running processes, Flutter dev servers, device management, hot reload, 
process health, logs, emulator lifecycle, and port management.
"""

import asyncio
import subprocess
import os
import signal
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import re

from .base_tool import (
    BaseTool, ToolCapabilities, ToolOperation, ToolPermission, ToolResult, ToolStatus,
    ToolCategory, ToolContext
)


class ProcessTool(BaseTool):
    """Enhanced process management tool for Flutter development."""
    
    def __init__(self):
        super().__init__(
            name="process_tool",
            description="Manage Flutter dev servers, emulators, hot reload, and process health.",
            version="1.2.0",
            category=ToolCategory.PROCESS,
            required_permissions=[
                ToolPermission.PROCESS_SPAWN,
                ToolPermission.PROCESS_KILL,
                ToolPermission.SYSTEM_INFO,
                ToolPermission.NETWORK_ACCESS
            ]
        )
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_metadata: Dict[str, Dict[str, Any]] = {}
        self.port_registry: Dict[int, str] = {}
        self.device_registry: Dict[str, Dict[str, Any]] = {}
        self.environment_cache: Dict[str, Any] = {}
        self.last_environment_check = None

    async def get_capabilities(self) -> ToolCapabilities:
        """Get comprehensive process management capabilities."""
        operations = [
            {
                "name": "start_dev_server",
                "description": "Start a Flutter development server for a project.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to Flutter project"},
                        "platform": {"type": "string", "enum": ["android", "ios", "web", "linux", "windows", "macos"]},
                        "device_id": {"type": "string", "description": "Specific device ID"},
                        "port": {"type": "integer", "description": "Port for web development"},
                        "hot_reload": {"type": "boolean", "default": True},
                        "debug_mode": {"type": "boolean", "default": True}
                    },
                    "required": ["project_path", "platform"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string"},
                        "port": {"type": "integer"},
                        "device_id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                },
                "required_permissions": [ToolPermission.PROCESS_SPAWN],
                "examples": [
                    {
                        "description": "Start dev server for web",
                        "params": {"project_path": "./myapp", "platform": "web", "port": 3000}
                    }
                ]
            },
            {
                "name": "stop_process",
                "description": "Stop a running process by ID.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string"},
                        "force": {"type": "boolean", "default": False}
                    },
                    "required": ["process_id"]
                },
                "required_permissions": [ToolPermission.PROCESS_KILL],
                "examples": [{"description": "Stop dev server", "params": {"process_id": "abc123"}}]
            },
            {
                "name": "hot_reload",
                "description": "Trigger hot reload on running Flutter app.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string"},
                        "restart": {"type": "boolean", "default": False}
                    },
                    "required": ["process_id"]
                },
                "required_permissions": [ToolPermission.PROCESS_SPAWN],
                "examples": [{"description": "Hot reload app", "params": {"process_id": "abc123"}}]
            },
            {
                "name": "list_devices",
                "description": "List available Flutter devices.",
                "parameters_schema": {"type": "object", "properties": {}},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "devices": {"type": "array", "items": {"type": "object"}}
                    }
                },
                "required_permissions": [ToolPermission.SYSTEM_INFO],
                "examples": [{"description": "List all devices", "params": {}}]
            },
            {
                "name": "start_emulator",
                "description": "Start an Android emulator.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "emulator_name": {"type": "string"},
                        "cold_boot": {"type": "boolean", "default": False}
                    },
                    "required": ["emulator_name"]
                },
                "required_permissions": [ToolPermission.PROCESS_SPAWN],
                "examples": [{"description": "Start emulator", "params": {"emulator_name": "Pixel_5_API_31"}}]
            },
            {
                "name": "monitor_process",
                "description": "Monitor process health and performance.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string"},
                        "metrics": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["process_id"]
                },
                "required_permissions": [ToolPermission.SYSTEM_INFO],
                "examples": [{"description": "Monitor dev server", "params": {"process_id": "abc123"}}]
            }
        ]
        
        return ToolCapabilities(
            available_operations=operations,
            supported_contexts=["flutter_development", "mobile_development"],
            performance_characteristics={
                "avg_startup_time": "2-5s",
                "max_concurrent_processes": 20,
                "hot_reload_time": "100-500ms"
            },
            limitations=["Platform-specific limitations", "Requires Flutter SDK"]
        )

    async def validate_params(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate operation parameters."""
        if operation == "start_dev_server":
            required = ["project_path", "platform"]
            missing = [p for p in required if p not in params]
            if missing:
                return False, f"Missing required parameters: {missing}"
            
            project_path = Path(params["project_path"])
            if not project_path.exists():
                return False, f"Project path does not exist: {project_path}"
            
            if not (project_path / "pubspec.yaml").exists():
                return False, f"Not a Flutter project: missing pubspec.yaml"
                
            return True, None
            
        elif operation in ["stop_process", "hot_reload", "monitor_process"]:
            if "process_id" not in params:
                return False, "Missing required parameter: process_id"
            if params["process_id"] not in self.processes:
                return False, f"Process not found: {params['process_id']}"
            return True, None
            
        elif operation == "start_emulator":
            if "emulator_name" not in params:
                return False, "Missing required parameter: emulator_name"
            return True, None
            
        elif operation == "list_devices":
            return True, None
            
        return False, f"Unknown operation: {operation}"

    async def execute(self, operation: str, params: Dict[str, Any], operation_id: Optional[str] = None) -> ToolResult:
        """Execute process management operations."""
        start_time = datetime.now()
        
        try:
            if operation == "start_dev_server":
                return await self._start_dev_server(params)
            elif operation == "stop_process":
                return await self._stop_process(params)
            elif operation == "hot_reload":
                return await self._hot_reload(params)
            elif operation == "list_devices":
                return await self._list_devices(params)
            elif operation == "start_emulator":
                return await self._start_emulator(params)
            elif operation == "monitor_process":
                return await self._monitor_process(params)
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

    async def _start_dev_server(self, params: Dict[str, Any]) -> ToolResult:
        """Start Flutter development server."""
        project_path = Path(params["project_path"])
        platform = params["platform"]
        device_id = params.get("device_id")
        port = params.get("port")
        hot_reload = params.get("hot_reload", True)
        debug_mode = params.get("debug_mode", True)
        
        try:
            # Build Flutter command
            cmd = ["flutter", "run"]
            
            if platform:
                cmd.extend(["-d", platform])
            if device_id:
                cmd.extend(["-d", device_id])
            if port and platform == "web":
                cmd.extend(["--web-port", str(port)])
            if not debug_mode:
                cmd.append("--release")
            if not hot_reload:
                cmd.append("--no-hot")
            
            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            process_id = str(uuid4())
            self.processes[process_id] = process
            self.process_metadata[process_id] = {
                "command": cmd,
                "project_path": str(project_path),
                "platform": platform,
                "port": port,
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }
            
            if port:
                self.port_registry[port] = process_id
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "process_id": process_id,
                    "port": port,
                    "device_id": device_id,
                    "status": "started"
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to start dev server: {str(e)}"
            )

    async def _stop_process(self, params: Dict[str, Any]) -> ToolResult:
        """Stop a running process."""
        process_id = params["process_id"]
        force = params.get("force", False)
        
        try:
            process = self.processes.get(process_id)
            if not process:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Process not found: {process_id}"
                )
            
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Wait for process to end
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            
            # Cleanup
            del self.processes[process_id]
            metadata = self.process_metadata.pop(process_id, {})
            
            # Remove from port registry
            port = metadata.get("port")
            if port and port in self.port_registry:
                del self.port_registry[port]
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"stopped": True, "process_id": process_id}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to stop process: {str(e)}"
            )

    async def _hot_reload(self, params: Dict[str, Any]) -> ToolResult:
        """Trigger hot reload."""
        process_id = params["process_id"]
        restart = params.get("restart", False)
        
        try:
            process = self.processes.get(process_id)
            if not process:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Process not found: {process_id}"
                )
            
            # Send hot reload command
            command = "R\n" if restart else "r\n"
            process.stdin.write(command.encode())
            await process.stdin.drain()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "reloaded": True,
                    "restart": restart,
                    "process_id": process_id
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to hot reload: {str(e)}"
            )

    async def _list_devices(self, params: Dict[str, Any]) -> ToolResult:
        """List available Flutter devices."""
        try:
            result = await asyncio.create_subprocess_exec(
                "flutter", "devices", "--machine",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Failed to list devices: {stderr.decode()}"
                )
            
            devices = json.loads(stdout.decode())
            
            # Update device registry
            for device in devices:
                self.device_registry[device["id"]] = device
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"devices": devices}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to list devices: {str(e)}"
            )

    async def _start_emulator(self, params: Dict[str, Any]) -> ToolResult:
        """Start Android emulator."""
        emulator_name = params["emulator_name"]
        cold_boot = params.get("cold_boot", False)
        
        try:
            cmd = ["emulator", "-avd", emulator_name]
            if cold_boot:
                cmd.append("-no-snapshot-load")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            process_id = str(uuid4())
            self.processes[process_id] = process
            self.process_metadata[process_id] = {
                "command": cmd,
                "emulator_name": emulator_name,
                "started_at": datetime.now().isoformat(),
                "status": "starting"
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "process_id": process_id,
                    "emulator_name": emulator_name,
                    "status": "starting"
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to start emulator: {str(e)}"
            )

    async def _monitor_process(self, params: Dict[str, Any]) -> ToolResult:
        """Monitor process health and performance."""
        process_id = params["process_id"]
        metrics = params.get("metrics", ["cpu", "memory", "status"])
        
        try:
            process = self.processes.get(process_id)
            if not process:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    data={},
                    error_message=f"Process not found: {process_id}"
                )
            
            monitoring_data = {
                "process_id": process_id,
                "status": "running" if process.poll() is None else "stopped",
                "pid": process.pid
            }
            
            # Get system metrics if process is running
            if process.poll() is None:
                try:
                    proc = psutil.Process(process.pid)
                    if "cpu" in metrics:
                        monitoring_data["cpu_percent"] = proc.cpu_percent()
                    if "memory" in metrics:
                        monitoring_data["memory_info"] = proc.memory_info()._asdict()
                    if "threads" in metrics:
                        monitoring_data["num_threads"] = proc.num_threads()
                except psutil.NoSuchProcess:
                    monitoring_data["status"] = "stopped"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=monitoring_data
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Failed to monitor process: {str(e)}"
            )

    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """Get usage examples for the tool."""
        return [
            {
                "description": "Start Flutter web development server",
                "operation": "start_dev_server",
                "params": {
                    "project_path": "./my_flutter_app",
                    "platform": "web",
                    "port": 3000
                },
                "expected_output": "Development server started"
            },
            {
                "description": "Hot reload running application",
                "operation": "hot_reload",
                "params": {"process_id": "abc123"},
                "expected_output": "Hot reload triggered"
            },
            {
                "description": "List available devices",
                "operation": "list_devices",
                "params": {},
                "expected_output": "List of Flutter-compatible devices"
            }
        ]

    async def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status."""
        active_processes = len([p for p in self.processes.values() if p.poll() is None])
        
        return {
            "status": "healthy",
            "last_check": datetime.now().isoformat(),
            "metrics": {
                "total_processes": len(self.processes),
                "active_processes": active_processes,
                "registered_ports": len(self.port_registry),
                "known_devices": len(self.device_registry)
            }
        }

    async def validate_environment(self, params: Dict[str, Any]) -> ToolResult:
        """Comprehensive environment validation for Flutter development."""
        start_time = datetime.now()
        
        try:
            validation_results = {
                "flutter_sdk": await self._validate_flutter_sdk(),
                "dart_sdk": await self._validate_dart_sdk(),
                "android_tools": await self._validate_android_tools(),
                "ios_tools": await self._validate_ios_tools() if self._is_macos() else {"available": False, "reason": "Not macOS"},
                "connected_devices": await self._get_connected_devices(),
                "environment_variables": await self._check_environment_variables(),
                "system_requirements": await self._check_system_requirements()
            }
            
            # Calculate health score
            health_score = self._calculate_environment_health(validation_results)
            
            # Generate recommendations
            recommendations = self._generate_environment_recommendations(validation_results)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "validation_results": validation_results,
                    "health_score": health_score,
                    "recommendations": recommendations,
                    "validation_time": (datetime.now() - start_time).total_seconds(),
                    "ready_for_development": health_score >= 0.7
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                data={},
                error_message=f"Environment validation failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _validate_flutter_sdk(self) -> Dict[str, Any]:
        """Validate Flutter SDK installation and configuration."""
        try:
            # Check flutter command availability
            result = await self._run_command(["flutter", "--version"], timeout=10)
            
            if result["exit_code"] != 0:
                return {
                    "available": False,
                    "issues": ["Flutter command not found in PATH"],
                    "version": None
                }
            
            # Parse version information
            output = result["stdout"]
            version_match = re.search(r'Flutter (\d+\.\d+\.\d+)', output)
            version = version_match.group(1) if version_match else "unknown"
            
            # Check doctor status
            doctor_result = await self._run_command(["flutter", "doctor", "--machine"], timeout=30)
            doctor_data = {}
            
            if doctor_result["exit_code"] == 0:
                try:
                    doctor_data = json.loads(doctor_result["stdout"])
                except json.JSONDecodeError:
                    pass
            
            return {
                "available": True,
                "version": version,
                "doctor_status": doctor_data,
                "issues": self._extract_flutter_issues(doctor_data),
                "path": await self._get_flutter_path()
            }
            
        except Exception as e:
            return {
                "available": False,
                "issues": [f"Flutter validation error: {str(e)}"],
                "version": None
            }

    async def _validate_dart_sdk(self) -> Dict[str, Any]:
        """Validate Dart SDK installation and configuration."""
        try:
            result = await self._run_command(["dart", "--version"], timeout=10)
            
            if result["exit_code"] != 0:
                return {
                    "available": False,
                    "issues": ["Dart command not found in PATH"],
                    "version": None
                }
            
            # Parse version
            output = result["stderr"] + result["stdout"]  # Dart outputs version to stderr
            version_match = re.search(r'Dart SDK version: (\d+\.\d+\.\d+)', output)
            version = version_match.group(1) if version_match else "unknown"
            
            return {
                "available": True,
                "version": version,
                "issues": [],
                "path": await self._get_dart_path()
            }
            
        except Exception as e:
            return {
                "available": False,
                "issues": [f"Dart validation error: {str(e)}"],
                "version": None
            }

    async def _validate_android_tools(self) -> Dict[str, Any]:
        """Validate Android development tools."""
        validation_result = {
            "available": False,
            "android_sdk": None,
            "android_studio": None,
            "build_tools": None,
            "emulators": [],
            "issues": []
        }
        
        try:
            # Check ANDROID_HOME environment variable
            android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
            
            if not android_home:
                validation_result["issues"].append("ANDROID_HOME environment variable not set")
                return validation_result
            
            android_sdk_path = Path(android_home)
            if not android_sdk_path.exists():
                validation_result["issues"].append(f"Android SDK path does not exist: {android_home}")
                return validation_result
            
            # Check SDK components
            platform_tools = android_sdk_path / "platform-tools"
            build_tools = android_sdk_path / "build-tools"
            
            validation_result["android_sdk"] = {
                "path": str(android_sdk_path),
                "platform_tools": platform_tools.exists(),
                "build_tools": build_tools.exists()
            }
            
            # Check for emulators
            emulator_cmd = platform_tools / "adb"
            if emulator_cmd.exists():
                devices_result = await self._run_command([str(emulator_cmd), "devices"], timeout=10)
                if devices_result["exit_code"] == 0:
                    validation_result["emulators"] = self._parse_adb_devices(devices_result["stdout"])
            
            # Check if Android Studio is installed
            validation_result["android_studio"] = await self._check_android_studio()
            
            validation_result["available"] = (
                android_sdk_path.exists() and
                platform_tools.exists() and
                build_tools.exists()
            )
            
        except Exception as e:
            validation_result["issues"].append(f"Android tools validation error: {str(e)}")
        
        return validation_result

    async def _validate_ios_tools(self) -> Dict[str, Any]:
        """Validate iOS development tools (macOS only)."""
        validation_result = {
            "available": False,
            "xcode": None,
            "simulators": [],
            "issues": []
        }
        
        try:
            # Check Xcode installation
            xcode_result = await self._run_command(["xcode-select", "--print-path"], timeout=10)
            
            if xcode_result["exit_code"] != 0:
                validation_result["issues"].append("Xcode not installed or not configured")
                return validation_result
            
            xcode_path = xcode_result["stdout"].strip()
            validation_result["xcode"] = {
                "path": xcode_path,
                "installed": True
            }
            
            # Check for iOS simulators
            simulators_result = await self._run_command(
                ["xcrun", "simctl", "list", "devices", "--json"], 
                timeout=15
            )
            
            if simulators_result["exit_code"] == 0:
                try:
                    sim_data = json.loads(simulators_result["stdout"])
                    validation_result["simulators"] = self._parse_ios_simulators(sim_data)
                except json.JSONDecodeError:
                    validation_result["issues"].append("Could not parse simulator list")
            
            validation_result["available"] = True
            
        except Exception as e:
            validation_result["issues"].append(f"iOS tools validation error: {str(e)}")
        
        return validation_result

    async def _get_connected_devices(self) -> List[Dict[str, Any]]:
        """Get list of connected devices for Flutter development."""
        devices = []
        
        try:
            # Use flutter devices command
            result = await self._run_command(["flutter", "devices", "--machine"], timeout=15)
            
            if result["exit_code"] == 0:
                try:
                    devices_data = json.loads(result["stdout"])
                    for device in devices_data:
                        devices.append({
                            "id": device.get("id", "unknown"),
                            "name": device.get("name", "Unknown Device"),
                            "platform": device.get("platform", "unknown"),
                            "type": device.get("type", "unknown"),
                            "is_emulator": device.get("emulator", False),
                            "available": device.get("available", False)
                        })
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.warning(f"Error getting connected devices: {e}")
        
        return devices

    async def _check_environment_variables(self) -> Dict[str, Any]:
        """Check important environment variables for Flutter development."""
        important_vars = [
            "FLUTTER_ROOT", "DART_SDK", "ANDROID_HOME", "ANDROID_SDK_ROOT",
            "JAVA_HOME", "PATH"
        ]
        
        env_status = {}
        
        for var in important_vars:
            value = os.environ.get(var)
            env_status[var] = {
                "set": value is not None,
                "value": value if value else None,
                "valid": self._validate_env_var(var, value) if value else False
            }
        
        return env_status

    def _validate_env_var(self, var_name: str, value: str) -> bool:
        """Validate if an environment variable points to a valid path."""
        if var_name in ["FLUTTER_ROOT", "DART_SDK", "ANDROID_HOME", "ANDROID_SDK_ROOT", "JAVA_HOME"]:
            return Path(value).exists() if value else False
        elif var_name == "PATH":
            return len(value.split(os.pathsep)) > 0 if value else False
        return True

    async def _check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for Flutter development."""
        import platform
        import psutil
        
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "disk_space_gb": round(psutil.disk_usage('/').free / (1024**3), 1),
            "meets_requirements": self._check_minimum_requirements()
        }

    def _check_minimum_requirements(self) -> bool:
        """Check if system meets minimum requirements for Flutter development."""
        try:
            import psutil
            
            # Minimum requirements
            min_memory_gb = 4
            min_disk_gb = 10
            min_cores = 2
            
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_gb = psutil.disk_usage('/').free / (1024**3)
            cores = psutil.cpu_count();
            
            return (
                memory_gb >= min_memory_gb and
                disk_gb >= min_disk_gb and
                cores >= min_cores
            )
        except:
            return True  # Assume OK if we can't check

    def _calculate_environment_health(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall environment health score."""
        score = 1.0
        
        # Critical components
        if not validation_results.get("flutter_sdk", {}).get("available", False):
            score -= 0.4
        
        if not validation_results.get("dart_sdk", {}).get("available", False):
            score -= 0.3
        
        # Platform tools
        android_available = validation_results.get("android_tools", {}).get("available", False)
        ios_available = validation_results.get("ios_tools", {}).get("available", False)
        
        if not android_available and not ios_available:
            score -= 0.2
        elif not android_available:
            score -= 0.1
        
        # Connected devices
        devices = validation_results.get("connected_devices", [])
        if not devices:
            score -= 0.1
        
        return max(0.0, score)

    def _generate_environment_recommendations(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Flutter SDK issues
        flutter_sdk = validation_results.get("flutter_sdk", {})
        if not flutter_sdk.get("available", False):
            recommendations.append({
                "priority": 1,
                "category": "installation",
                "title": "Install Flutter SDK",
                "description": "Flutter SDK is required for Flutter development",
                "steps": [
                    "Download Flutter SDK from https://flutter.dev",
                    "Extract to desired location",
                    "Add Flutter bin directory to PATH",
                    "Run 'flutter doctor' to verify installation"
                ],
                "automated": False,
                "estimated_time_minutes": 30
            })
        
        # Dart SDK issues
        dart_sdk = validation_results.get("dart_sdk", {})
        if not dart_sdk.get("available", False):
            recommendations.append({
                "priority": 1,
                "category": "installation",
                "title": "Install or Configure Dart SDK",
                "description": "Dart SDK is required and usually comes with Flutter",
                "steps": [
                    "Ensure Flutter SDK is properly installed",
                    "Add Flutter bin directory to PATH",
                    "Verify with 'dart --version'"
                ],
                "automated": False,
                "estimated_time_minutes": 10
            })
        
        # Android tools
        android_tools = validation_results.get("android_tools", {})
        if not android_tools.get("available", False):
            recommendations.append({
                "priority": 2,
                "category": "installation",
                "title": "Set up Android Development Environment",
                "description": "Android tools are needed for Android app development",
                "steps": [
                    "Install Android Studio",
                    "Set up Android SDK",
                    "Configure ANDROID_HOME environment variable",
                    "Accept Android licenses with 'flutter doctor --android-licenses'"
                ],
                "automated": False,
                "estimated_time_minutes": 45
            })
        
        # Device connectivity
        devices = validation_results.get("connected_devices", [])
        if not devices:
            recommendations.append({
                "priority": 3,
                "category": "configuration",
                "title": "Connect Development Device",
                "description": "No devices available for testing",
                "steps": [
                    "Connect a physical device via USB with debugging enabled",
                    "Or start an Android/iOS emulator",
                    "Verify with 'flutter devices'"
                ],
                "automated": False,
                "estimated_time_minutes": 15
            })
        
        return recommendations

    # Helper methods
    async def _run_command(self, cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run a system command and return result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }
            
        except asyncio.TimeoutError:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "Command timed out"
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }

    def _is_macos(self) -> bool:
        """Check if running on macOS."""
        return platform.system() == "Darwin"

    async def _get_flutter_path(self) -> Optional[str]:
        """Get Flutter installation path."""
        try:
            result = await self._run_command(["which", "flutter"], timeout=5)
            if result["exit_code"] == 0:
                return result["stdout"].strip()
        except:
            pass
        return None

    async def _get_dart_path(self) -> Optional[str]:
        """Get Dart installation path."""
        try:
            result = await self._run_command(["which", "dart"], timeout=5)
            if result["exit_code"] == 0:
                return result["stdout"].strip()
        except:
            pass
        return None

    def _extract_flutter_issues(self, doctor_data: Dict[str, Any]) -> List[str]:
        """Extract issues from Flutter doctor output."""
        issues = []
        
        if not doctor_data:
            return issues
        
        for item in doctor_data.get("doctorText", []):
            if item.get("type") == "error":
                issues.append(item.get("message", "Unknown error"))
        
        return issues

    def _parse_adb_devices(self, adb_output: str) -> List[Dict[str, Any]]:
        """Parse ADB devices output."""
        devices = []
        lines = adb_output.strip().split('\n')[1:]  # Skip header
        
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    devices.append({
                        "id": parts[0],
                        "status": parts[1],
                        "type": "device" if parts[1] == "device" else "emulator"
                    })
        
        return devices

    def _parse_ios_simulators(self, sim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse iOS simulator data."""
        simulators = []
        
        devices = sim_data.get("devices", {})
        for runtime, device_list in devices.items():
            for device in device_list:
                if device.get("isAvailable", False):
                    simulators.append({
                        "id": device.get("udid"),
                        "name": device.get("name"),
                        "runtime": runtime,
                        "state": device.get("state")
                    })
        
        return simulators

    async def _check_android_studio(self) -> Dict[str, Any]:
        """Check if Android Studio is installed."""
        # Common Android Studio paths
        studio_paths = [
            "/Applications/Android Studio.app",  # macOS
            "/opt/android-studio",  # Linux
            "C:\\Program Files\\Android\\Android Studio"  # Windows
        ]
        
        for path in studio_paths:
            if Path(path).exists():
                return {"installed": True, "path": path}
        
        return {"installed": False, "path": None}
