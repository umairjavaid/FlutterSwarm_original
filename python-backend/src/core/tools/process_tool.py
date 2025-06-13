"""
Process Management Tool for FlutterSwarm Multi-Agent System.

Handles long-running processes, Flutter dev servers, device management, hot reload, process health, logs, emulator lifecycle, and port management.
"""

import asyncio
import subprocess
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

from .base_tool import (
    BaseTool, ToolCapabilities, ToolOperation, ToolPermission, ToolResult, ToolStatus
)

class ProcessTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="process_tool",
            description="Manage Flutter dev servers, emulators, hot reload, and process health.",
            version="1.0.0",
            required_permissions=[
                ToolPermission.PROCESS_SPAWN,
                ToolPermission.PROCESS_KILL,
                ToolPermission.SYSTEM_INFO,
                ToolPermission.NETWORK_ACCESS
            ]
        )
        self.processes: Dict[str, subprocess.Popen] = {}

    async def get_capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            available_operations=[
                ToolOperation(
                    name="start_dev_server",
                    description="Start a Flutter development server for a project.",
                    parameters_schema={"type": "object", "properties": {"project_path": {"type": "string"}, "platform": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"process_id": {"type": "string"}}},
                    required_permissions=[ToolPermission.PROCESS_SPAWN],
                    examples=[{"description": "Start dev server for linux", "params": {"project_path": "./myapp", "platform": "linux"}}],
                    error_codes={"SPAWN_FAILED": "Could not start process"}
                ),
                ToolOperation(
                    name="stop_process",
                    description="Stop a running process by ID.",
                    parameters_schema={"type": "object", "properties": {"process_id": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"stopped": {"type": "boolean"}}},
                    required_permissions=[ToolPermission.PROCESS_KILL],
                    examples=[{"description": "Stop dev server", "params": {"process_id": "abc123"}}],
                    error_codes={"NOT_FOUND": "Process not found"}
                ),
                # ... more: hot_reload, monitor, logs, emulator, port, etc.
            ]
        )

    async def validate_params(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if operation == "start_dev_server" and ("project_path" not in params or "platform" not in params):
            return False, "Missing 'project_path' or 'platform'."
        if operation == "stop_process" and "process_id" not in params:
            return False, "Missing 'process_id'."
        return True, None

    async def execute(self, operation: str, params: Dict[str, Any], operation_id: Optional[str] = None) -> ToolResult:
        try:
            if operation == "start_dev_server":
                cmd = ["flutter", "run", f"-d", params["platform"]]
                proc = subprocess.Popen(cmd, cwd=params["project_path"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pid = str(proc.pid)
                self.processes[pid] = proc
                return self._create_success_result(operation, {"process_id": pid}, operation_id=operation_id)
            if operation == "stop_process":
                pid = params["process_id"]
                proc = self.processes.get(pid)
                if not proc:
                    return self._create_error_result(operation, "Process not found", "NOT_FOUND", operation_id)
                proc.terminate()
                proc.wait(timeout=10)
                del self.processes[pid]
                return self._create_success_result(operation, {"stopped": True}, operation_id=operation_id)
            # ... implement other operations
            return self._create_error_result(operation, "Operation not implemented", "NOT_IMPLEMENTED", operation_id)
        except Exception as e:
            return self._create_error_result(operation, str(e), "EXCEPTION", operation_id)

    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        return [
            {"operation": "start_dev_server", "params": {"project_path": "./myapp", "platform": "linux"}},
            {"operation": "stop_process", "params": {"process_id": "12345"}}
        ]
