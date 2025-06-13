"""
Tool Registry for FlutterSwarm Multi-Agent System.

Singleton registry for tool discovery, registration, dependency resolution, version compatibility, metrics, and availability monitoring.
"""

from typing import Dict, Type, Optional, List
from threading import Lock

from .base_tool import BaseTool

class ToolRegistry:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.metrics: Dict[str, Dict] = {}

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_tool(self, tool: BaseTool, dependencies: Optional[List[str]] = None):
        self.tools[tool.name] = tool
        if dependencies:
            self.dependencies[tool.name] = dependencies

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

    def resolve_dependencies(self, tool_name: str) -> List[str]:
        return self.dependencies.get(tool_name, [])

    def check_version_compatibility(self, tool_name: str, version: str) -> bool:
        tool = self.get_tool(tool_name)
        return tool and tool.version == version

    def query_capabilities(self, tool_name: str):
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_capabilities()
        return None

    def update_metrics(self, tool_name: str, metrics: Dict):
        self.metrics[tool_name] = metrics

    def get_metrics(self, tool_name: str) -> Optional[Dict]:
        return self.metrics.get(tool_name)

    def is_available(self, tool_name: str) -> bool:
        tool = self.get_tool(tool_name)
        return tool.is_available if tool else False

    def graceful_degrade(self, tool_name: str):
        tool = self.get_tool(tool_name)
        if tool:
            tool.is_available = False
