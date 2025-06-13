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
