"""
Tool Registry for FlutterSwarm Multi-Agent System.

Singleton registry for tool discovery, registration, dependency resolution, 
version compatibility, metrics, and availability monitoring.
"""

import asyncio
import logging
from threading import Lock
from typing import Dict, Type, Optional, List, Any
from datetime import datetime

from .base_tool import BaseTool, ToolCategory
from ...config import get_logger

logger = get_logger("tool_registry")


class ToolRegistry:
    """
    Singleton registry for managing all tools in the FlutterSwarm system.
    
    Features:
    - Tool discovery and registration
    - Dependency resolution
    - Version compatibility checking
    - Performance metrics tracking
    - Availability monitoring
    - Graceful degradation
    """
    
    _instance = None
    _lock = Lock()

    def __init__(self):
        if ToolRegistry._instance is not None:
            raise Exception("ToolRegistry is a singleton. Use instance() method.")
        
        self.tools: Dict[str, BaseTool] = {}
        self.tool_dependencies: Dict[str, List[str]] = {}
        self.tool_metrics: Dict[str, Dict[str, Any]] = {}
        self.availability_cache: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        
        logger.info("Tool Registry initialized")

    @classmethod
    def instance(cls) -> "ToolRegistry":
        """Get singleton instance of ToolRegistry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Alias for instance() method for compatibility."""
        return cls.instance()

    async def initialize(self):
        """Initialize the tool registry and discover available tools."""
        async with self.initialization_lock:
            if self.is_initialized:
                return
            
            logger.info("Initializing Tool Registry...")
            
            # Auto-discover and register tools
            await self._discover_tools()
            
            # Resolve dependencies
            await self._resolve_all_dependencies()
            
            # Initial health check
            await self._perform_health_checks()
            
            self.is_initialized = True
            logger.info(f"Tool Registry initialized with {len(self.tools)} tools")

    async def _discover_tools(self):
        """Automatically discover and instantiate available tools."""
        try:
            # Import and register available tools
            from .flutter_sdk_tool import FlutterSDKTool
            from .file_system_tool import FileSystemTool
            from .process_tool import ProcessTool
            # from .git_tool import GitTool  # When implemented
            
            # Register discovered tools
            tools_to_register = [
                (FlutterSDKTool(), []),
                (FileSystemTool(), []),
                (ProcessTool(), ["flutter_sdk_tool"]),  # Process tool depends on Flutter SDK
                # (GitTool(), []),
            ]
            
            for tool, dependencies in tools_to_register:
                await self.register_tool(tool, dependencies)
                
        except ImportError as e:
            logger.warning(f"Some tools could not be imported: {e}")
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")

    async def register_tool(self, tool: BaseTool, dependencies: Optional[List[str]] = None):
        """
        Register a tool with the registry.
        
        Args:
            tool: Tool instance to register
            dependencies: List of tool names this tool depends on
        """
        try:
            # Validate tool
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool must be instance of BaseTool, got {type(tool)}")
            
            if tool.name in self.tools:
                logger.warning(f"Tool {tool.name} already registered, replacing...")
            
            # Register tool
            self.tools[tool.name] = tool
            self.tool_dependencies[tool.name] = dependencies or []
            
            # Initialize metrics
            self.tool_metrics[tool.name] = {
                "registrations": self.tool_metrics.get(tool.name, {}).get("registrations", 0) + 1,
                "last_registered": datetime.now(),
                "total_uses": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0
            }
            
            # Check health
            health_status = await tool.check_health()
            self.availability_cache[tool.name] = health_status
            self.last_health_check[tool.name] = datetime.now()
            
            logger.info(
                f"Tool registered: {tool.name} v{tool.version} "
                f"(category: {tool.category.value}, healthy: {health_status})"
            )
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
            raise

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        tool = self.tools.get(name)
        if tool and self.is_available(name):
            return tool
        elif tool:
            logger.warning(f"Tool {name} is registered but not available")
        return None

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a specific category."""
        return [tool for tool in self.tools.values() if tool.category == category]

    def get_available_tools(self) -> List[BaseTool]:
        """Get all currently available tools."""
        return [tool for name, tool in self.tools.items() if self.is_available(name)]

    async def _resolve_all_dependencies(self):
        """Resolve dependencies for all registered tools."""
        for tool_name in self.tools.keys():
            try:
                resolved_deps = self.resolve_dependencies(tool_name)
                logger.debug(f"Dependencies for {tool_name}: {resolved_deps}")
            except Exception as e:
                logger.error(f"Failed to resolve dependencies for {tool_name}: {e}")

    def resolve_dependencies(self, tool_name: str) -> List[str]:
        """
        Resolve dependencies for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of dependency tool names in resolution order
        """
        if tool_name not in self.tool_dependencies:
            return []
        
        resolved = []
        visited = set()
        
        def _resolve_recursive(name: str):
            if name in visited:
                return  # Avoid circular dependencies
            
            visited.add(name)
            
            for dep_name in self.tool_dependencies.get(name, []):
                if dep_name not in resolved:
                    _resolve_recursive(dep_name)
                    resolved.append(dep_name)
        
        _resolve_recursive(tool_name)
        return resolved

    def check_version_compatibility(self, tool_name: str, required_version: str) -> bool:
        """
        Check if a tool version is compatible with requirements.
        
        Args:
            tool_name: Name of the tool
            required_version: Required version string
            
        Returns:
            True if compatible, False otherwise
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return False
        
        try:
            # Simple version comparison (could be enhanced with semantic versioning)
            tool_version = tool.version
            return tool_version >= required_version
        except Exception as e:
            logger.warning(f"Version compatibility check failed for {tool_name}: {e}")
            return False

    async def query_capabilities(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Query capabilities of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool capabilities dictionary or None
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        
        try:
            capabilities = await tool.get_capabilities()
            return {
                "tool_name": tool_name,
                "version": tool.version,
                "category": tool.category.value,
                "capabilities": capabilities,
                "performance": await tool.get_performance_summary()
            }
        except Exception as e:
            logger.error(f"Failed to query capabilities for {tool_name}: {e}")
            return None

    def update_metrics(self, tool_name: str, metrics: Dict[str, Any]):
        """
        Update metrics for a tool.
        
        Args:
            tool_name: Name of the tool
            metrics: Metrics dictionary to update
        """
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = {}
        
        self.tool_metrics[tool_name].update(metrics)
        self.tool_metrics[tool_name]["last_updated"] = datetime.now()

    def get_metrics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Metrics dictionary or None
        """
        return self.tool_metrics.get(tool_name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all tools."""
        return self.tool_metrics.copy()

    def is_available(self, tool_name: str) -> bool:
        """
        Check if a tool is currently available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if available, False otherwise
        """
        if tool_name not in self.tools:
            return False
        
        # Check cached availability (refresh if old)
        last_check = self.last_health_check.get(tool_name)
        if last_check and (datetime.now() - last_check).seconds < 300:  # 5 minute cache
            return self.availability_cache.get(tool_name, False)
        
        # Need fresh health check
        return self.availability_cache.get(tool_name, False)

    async def _perform_health_checks(self):
        """Perform health checks on all registered tools."""
        logger.info("Performing health checks on all tools...")
        
        for tool_name, tool in self.tools.items():
            try:
                health_status = await tool.check_health()
                self.availability_cache[tool_name] = health_status
                self.last_health_check[tool_name] = datetime.now()
                
                if not health_status:
                    logger.warning(f"Tool {tool_name} failed health check")
                else:
                    logger.debug(f"Tool {tool_name} health check passed")
                    
            except Exception as e:
                logger.error(f"Health check failed for {tool_name}: {e}")
                self.availability_cache[tool_name] = False

    async def refresh_health_checks(self):
        """Refresh health checks for all tools."""
        await self._perform_health_checks()

    def graceful_degrade(self, tool_name: str) -> Dict[str, Any]:
        """
        Handle graceful degradation when a tool becomes unavailable.
        
        Args:
            tool_name: Name of the unavailable tool
            
        Returns:
            Degradation strategy information
        """
        self.availability_cache[tool_name] = False
        
        # Find alternative tools or fallback strategies
        alternatives = []
        tool = self.tools.get(tool_name)
        
        if tool:
            # Find tools in same category
            same_category_tools = [
                t for t in self.tools.values() 
                if t.category == tool.category and t.name != tool_name and self.is_available(t.name)
            ]
            alternatives = [t.name for t in same_category_tools]
        
        degradation_info = {
            "unavailable_tool": tool_name,
            "alternatives": alternatives,
            "degradation_mode": "alternative_tools" if alternatives else "reduced_functionality",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Tool {tool_name} degraded: {degradation_info}")
        return degradation_info

    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status."""
        total_tools = len(self.tools)
        available_tools = len([name for name in self.tools.keys() if self.is_available(name)])
        
        return {
            "total_tools": total_tools,
            "available_tools": available_tools,
            "availability_rate": available_tools / total_tools if total_tools > 0 else 0,
            "categories": list(set(tool.category.value for tool in self.tools.values())),
            "last_health_check": max(self.last_health_check.values()) if self.last_health_check else None,
            "is_initialized": self.is_initialized
        }

    def update_metrics(self, tool_name: str, metrics: Dict):
        self.metrics[tool_name] = metrics

    def get_metrics(self, tool_name: str) -> Optional[Dict]:
        return self.metrics.get(tool_name)

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool is available without causing circular dependency."""
        tool = self.tools.get(tool_name)
        return hasattr(tool, 'is_available') and tool.is_available if tool else False

    def graceful_degrade(self, tool_name: str):
        """Mark a tool as unavailable."""
        tool = self.tools.get(tool_name)
        if tool and hasattr(tool, 'is_available'):
            tool.is_available = False
