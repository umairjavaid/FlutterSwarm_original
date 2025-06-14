"""
Tool Registry for FlutterSwarm Multi-Agent System.

Enhanced singleton registry for comprehensive tool management including:
- Tool discovery and automatic registration
- Dependency resolution with conflict detection
- Version compatibility checking
- Performance metrics collection
- Health monitoring with graceful degradation
- Agent-friendly API for tool selection
- Runtime loading/unloading capabilities
"""

import asyncio
import logging
import importlib
import importlib.util
import inspect
import re
from threading import Lock
from typing import Dict, Type, Optional, List, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path

from .base_tool import BaseTool, ToolCategory
from ...config import get_logger

logger = get_logger("tool_registry")


class ToolRegistry:
    """
    Enhanced singleton registry for managing all tools in the FlutterSwarm system.
    
    Features:
    - Tool discovery and automatic registration
    - Dependency resolution with conflict detection
    - Version compatibility checking with semantic versioning
    - Performance metrics collection and analysis
    - Health monitoring with graceful degradation
    - Agent-friendly API for intelligent tool selection
    - Runtime loading/unloading capabilities
    - Configuration management
    """
    
    _instance = None
    _lock = Lock()

    def __init__(self):
        if ToolRegistry._instance is not None:
            raise Exception("ToolRegistry is a singleton. Use instance() method.")
        
        # Core registry storage
        self.tools: Dict[str, BaseTool] = {}
        self.tool_classes: Dict[str, Type[BaseTool]] = {}
        self.tool_dependencies: Dict[str, List[str]] = {}
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        
        # Metrics and monitoring
        self.tool_metrics: Dict[str, Dict[str, Any]] = {}
        self.availability_cache: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Loading and discovery
        self.loaded_modules: Dict[str, Any] = {}
        self.discovery_paths: List[Path] = []
        self.auto_discover: bool = True
        
        # State management
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        self.health_check_interval = timedelta(minutes=5)
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Tool selection and compatibility
        self.compatibility_matrix: Dict[str, Dict[str, str]] = {}
        self.fallback_strategies: Dict[str, List[str]] = {}
        
        logger.info("Enhanced Tool Registry initialized")

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

    async def initialize(self, discovery_paths: Optional[List[str]] = None, auto_discover: bool = True):
        """
        Initialize the tool registry with enhanced discovery and management.
        
        Args:
            discovery_paths: Additional paths to search for tools
            auto_discover: Whether to automatically discover and register tools
        """
        async with self.initialization_lock:
            if self.is_initialized:
                logger.info("Tool Registry already initialized")
                return
            
            logger.info("Initializing Enhanced Tool Registry...")
            
            # Set discovery configuration
            self.auto_discover = auto_discover
            if discovery_paths:
                self.discovery_paths.extend([Path(p) for p in discovery_paths])
            
            try:
                # Auto-discover and register tools
                if self.auto_discover:
                    await self._discover_tools()
                
                # Load tool configurations
                await self._load_tool_configurations()
                
                # Resolve dependencies with conflict detection
                await self._resolve_all_dependencies()
                
                # Build compatibility matrix
                self._build_compatibility_matrix()
                
                # Setup fallback strategies
                self._setup_fallback_strategies()
                
                # Initial health check
                await self._perform_health_checks()
                
                # Start background health monitoring
                await self._start_health_monitoring()
                
                self.is_initialized = True
                logger.info(f"Enhanced Tool Registry initialized with {len(self.tools)} tools")
                
            except Exception as e:
                logger.error(f"Tool Registry initialization failed: {e}")
                raise

    async def _discover_tools(self):
        """Enhanced tool discovery with dynamic loading capabilities."""
        discovered_tools = []
        
        try:
            # Discover from core tools directory
            core_tools = await self._discover_core_tools()
            discovered_tools.extend(core_tools)
            
            # Discover from additional paths
            for path in self.discovery_paths:
                if path.exists():
                    path_tools = await self._discover_tools_in_path(path)
                    discovered_tools.extend(path_tools)
            
            # Register discovered tools
            for tool_class, dependencies, config in discovered_tools:
                try:
                    tool_instance = tool_class()
                    await self.register_tool(tool_instance, dependencies, config)
                except Exception as e:
                    logger.error(f"Failed to register tool {tool_class.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")

    async def _discover_core_tools(self) -> List[tuple]:
        """Discover core FlutterSwarm tools."""
        core_tools = []
        
        try:
            # Import and register available tools with their dependencies
            tool_configs = [
                ("flutter_sdk_tool", "FlutterSDKTool", [], {
                    "priority": 10,
                    "category": "DEVELOPMENT",
                    "required_for": ["build", "test", "run"]
                }),
                ("file_system_tool", "FileSystemTool", [], {
                    "priority": 9,
                    "category": "FILE_SYSTEM", 
                    "required_for": ["file_operations", "project_management"]
                }),
                ("process_tool", "ProcessTool", ["flutter_sdk_tool"], {
                    "priority": 8,
                    "category": "PROCESS",
                    "required_for": ["execution", "monitoring"]
                }),
                ("git_tool", "GitTool", [], {
                    "priority": 7,
                    "category": "VERSION_CONTROL",
                    "required_for": ["version_control"],
                    "optional": True
                })
            ]
            
            for module_name, class_name, dependencies, config in tool_configs:
                try:
                    module = importlib.import_module(f".{module_name}", package="src.core.tools")
                    tool_class = getattr(module, class_name)
                    
                    # Validate it's a proper tool class
                    if inspect.isclass(tool_class) and issubclass(tool_class, BaseTool):
                        core_tools.append((tool_class, dependencies, config))
                        self.tool_classes[class_name] = tool_class
                        logger.debug(f"Discovered core tool: {class_name}")
                    
                except ImportError as e:
                    if not config.get("optional", False):
                        logger.warning(f"Required tool {class_name} could not be imported: {e}")
                    else:
                        logger.debug(f"Optional tool {class_name} not available: {e}")
                except Exception as e:
                    logger.error(f"Error discovering tool {class_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Core tool discovery failed: {e}")
            
        return core_tools

    async def _discover_tools_in_path(self, path: Path) -> List[tuple]:
        """Discover tools in a specific path."""
        discovered = []
        
        try:
            for py_file in path.glob("*_tool.py"):
                if py_file.name.startswith("base_") or py_file.name.startswith("__"):
                    continue
                    
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find tool classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != BaseTool and 
                            issubclass(obj, BaseTool) and 
                            obj.__module__ == module.__name__):
                            discovered.append((obj, [], {}))
                            
        except Exception as e:
            logger.error(f"Error discovering tools in {path}: {e}")
            
        return discovered

    async def register_tool(self, tool: BaseTool, dependencies: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Enhanced tool registration with configuration and validation.
        
        Args:
            tool: Tool instance to register
            dependencies: List of tool names this tool depends on
            config: Tool-specific configuration
        """
        try:
            # Validate tool
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool must be instance of BaseTool, got {type(tool)}")
            
            # Check for conflicts
            if tool.name in self.tools:
                existing_tool = self.tools[tool.name]
                if not self._can_replace_tool(existing_tool, tool):
                    logger.warning(f"Tool {tool.name} registration blocked due to compatibility")
                    return
                logger.warning(f"Tool {tool.name} already registered, replacing...")
            
            # Validate dependencies
            missing_deps = await self._validate_dependencies(dependencies or [])
            if missing_deps:
                logger.error(f"Cannot register {tool.name}: missing dependencies {missing_deps}")
                raise ValueError(f"Missing dependencies: {missing_deps}")
            
            # Register tool
            self.tools[tool.name] = tool
            self.tool_dependencies[tool.name] = dependencies or []
            self.tool_configs[tool.name] = config or {}
            
            # Initialize comprehensive metrics
            self.tool_metrics[tool.name] = {
                "registrations": self.tool_metrics.get(tool.name, {}).get("registrations", 0) + 1,
                "last_registered": datetime.now(),
                "total_uses": 0,
                "successful_uses": 0,
                "failed_uses": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "total_response_time": 0.0,
                "min_response_time": float('inf'),
                "max_response_time": 0.0,
                "last_used": None,
                "category": tool.category.value,
                "version": tool.version,
                "priority": config.get("priority", 5) if config else 5
            }
            
            # Initialize performance history
            self.performance_history[tool.name] = []
            
            # Check health and availability
            try:
                health_status = await tool.check_health()
                self.availability_cache[tool.name] = health_status
                self.last_health_check[tool.name] = datetime.now()
            except Exception as e:
                logger.warning(f"Health check failed for {tool.name} during registration: {e}")
                self.availability_cache[tool.name] = False
            
            logger.info(
                f"Tool registered: {tool.name} v{tool.version} "
                f"(category: {tool.category.value}, healthy: {self.availability_cache.get(tool.name, False)}, "
                f"priority: {self.tool_metrics[tool.name]['priority']})"
            )
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
            raise

    async def unregister_tool(self, tool_name: str, force: bool = False) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            force: Whether to force removal even if other tools depend on it
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        try:
            if tool_name not in self.tools:
                logger.warning(f"Tool {tool_name} not found for unregistration")
                return False
            
            # Check for dependent tools
            dependent_tools = [
                name for name, deps in self.tool_dependencies.items() 
                if tool_name in deps
            ]
            
            if dependent_tools and not force:
                logger.error(f"Cannot unregister {tool_name}: tools {dependent_tools} depend on it")
                return False
            
            # Cleanup tool
            tool = self.tools[tool_name]
            try:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
            except Exception as e:
                logger.warning(f"Tool cleanup failed for {tool_name}: {e}")
            
            # Remove from registry
            del self.tools[tool_name]
            self.tool_dependencies.pop(tool_name, None)
            self.tool_configs.pop(tool_name, None)
            self.availability_cache.pop(tool_name, None)
            self.last_health_check.pop(tool_name, None)
            self.performance_history.pop(tool_name, None)
            
            # Archive metrics instead of deleting
            if tool_name in self.tool_metrics:
                self.tool_metrics[tool_name]["unregistered"] = datetime.now()
                self.tool_metrics[tool_name]["status"] = "unregistered"
            
            logger.info(f"Tool unregistered: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_name}: {e}")
            return False

    async def reload_tool(self, tool_name: str) -> bool:
        """
        Reload a tool from its class definition.
        
        Args:
            tool_name: Name of the tool to reload
            
        Returns:
            True if successfully reloaded, False otherwise
        """
        try:
            if tool_name not in self.tools:
                logger.error(f"Tool {tool_name} not found for reloading")
                return False
            
            # Get current tool info
            old_tool = self.tools[tool_name]
            dependencies = self.tool_dependencies.get(tool_name, [])
            config = self.tool_configs.get(tool_name, {})
            tool_class = type(old_tool)
            
            # Unregister current tool
            await self.unregister_tool(tool_name, force=True)
            
            # Create new instance
            new_tool = tool_class()
            
            # Re-register
            await self.register_tool(new_tool, dependencies, config)
            
            logger.info(f"Tool reloaded: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload tool {tool_name}: {e}")
            return False

    # =============================================
    # ENHANCED DEPENDENCY RESOLUTION & VERSION COMPATIBILITY
    # =============================================

    async def _resolve_all_dependencies(self):
        """Enhanced dependency resolution with conflict detection."""
        logger.info("Resolving tool dependencies...")
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph()
            
            # Detect circular dependencies
            cycles = self._detect_circular_dependencies(dependency_graph)
            if cycles:
                logger.error(f"Circular dependencies detected: {cycles}")
                raise ValueError(f"Circular dependencies: {cycles}")
            
            # Resolve in topological order
            resolution_order = self._topological_sort(dependency_graph)
            logger.debug(f"Dependency resolution order: {resolution_order}")
            
            # Validate all dependencies can be satisfied
            for tool_name in resolution_order:
                missing_deps = await self._validate_dependencies(
                    self.tool_dependencies.get(tool_name, [])
                )
                if missing_deps:
                    logger.error(f"Unresolvable dependencies for {tool_name}: {missing_deps}")
                    
        except Exception as e:
            logger.error(f"Dependency resolution failed: {e}")
            raise

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a dependency graph for all tools."""
        return {
            tool_name: deps.copy() 
            for tool_name, deps in self.tool_dependencies.items()
        }

    def _detect_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [neighbor])
                
            rec_stack.remove(node)
        
        for tool_name in graph:
            if tool_name not in visited:
                dfs(tool_name, [tool_name])
                
        return cycles

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph."""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Remove edges from this node
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result

    async def _validate_dependencies(self, dependencies: List[str]) -> List[str]:
        """Validate that all dependencies are available."""
        missing = []
        for dep in dependencies:
            if dep not in self.tools or not self.is_available(dep):
                missing.append(dep)
        return missing

    def check_version_compatibility(self, 
                                  tool_name: str, 
                                  required_version: str,
                                  comparison_type: str = ">=") -> bool:
        """
        Enhanced version compatibility checking with semantic versioning.
        
        Args:
            tool_name: Name of the tool
            required_version: Required version string
            comparison_type: Type of comparison (>=, ==, >, <, <=, !=)
            
        Returns:
            True if compatible, False otherwise
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return False
        
        try:
            tool_version = self._parse_version(tool.version)
            required_version_parsed = self._parse_version(required_version)
            
            if comparison_type == ">=":
                return tool_version >= required_version_parsed
            elif comparison_type == "==":
                return tool_version == required_version_parsed
            elif comparison_type == ">":
                return tool_version > required_version_parsed
            elif comparison_type == "<":
                return tool_version < required_version_parsed
            elif comparison_type == "<=":
                return tool_version <= required_version_parsed
            elif comparison_type == "!=":
                return tool_version != required_version_parsed
            else:
                logger.warning(f"Unknown comparison type: {comparison_type}")
                return False
                
        except Exception as e:
            logger.warning(f"Version compatibility check failed for {tool_name}: {e}")
            return False

    def _parse_version(self, version_str: str) -> tuple:
        """Parse version string into comparable tuple."""
        try:
            # Handle semantic versioning (e.g., "1.2.3", "2.0.0-beta")
            version_clean = re.sub(r'[^\d\.].*', '', version_str)
            parts = version_clean.split('.')
            return tuple(int(part) for part in parts if part.isdigit())
        except Exception:
            # Fallback to string comparison
            return (version_str,)

    def resolve_version_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve version conflicts between tools.
        
        Args:
            conflicts: List of conflict descriptions
            
        Returns:
            Resolution strategy and recommendations
        """
        try:
            resolution_strategies = []
            
            for conflict in conflicts:
                tool1 = conflict.get("tool1")
                tool2 = conflict.get("tool2")
                issue = conflict.get("issue")
                
                if "version" in issue.lower():
                    # Version conflict resolution
                    strategy = self._resolve_version_conflict(tool1, tool2, issue)
                    resolution_strategies.append(strategy)
                elif "dependency" in issue.lower():
                    # Dependency conflict resolution
                    strategy = self._resolve_dependency_conflict(tool1, tool2, issue)
                    resolution_strategies.append(strategy)
            
            return {
                "conflicts": conflicts,
                "resolutions": resolution_strategies,
                "auto_resolvable": all(s.get("auto_resolvable", False) for s in resolution_strategies)
            }
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return {"error": str(e)}

    def _resolve_version_conflict(self, tool1: str, tool2: str, issue: str) -> Dict[str, Any]:
        """Resolve version conflicts between two tools."""
        return {
            "type": "version_conflict",
            "tools": [tool1, tool2],
            "issue": issue,
            "strategy": "use_latest_compatible",
            "auto_resolvable": True,
            "action": f"Update {tool1} or {tool2} to compatible version"
        }

    def _resolve_dependency_conflict(self, tool1: str, tool2: str, issue: str) -> Dict[str, Any]:
        """Resolve dependency conflicts between two tools."""
        return {
            "type": "dependency_conflict", 
            "tools": [tool1, tool2],
            "issue": issue,
            "strategy": "alternative_tool",
            "auto_resolvable": False,
            "action": f"Consider alternative tools or modify dependencies"
        }

    # =============================================
    # AGENT-FRIENDLY API FOR TOOL SELECTION
    # =============================================

    async def select_best_tool(self, 
                             task_type: str, 
                             requirements: Optional[Dict[str, Any]] = None,
                             performance_weight: float = 0.4,
                             availability_weight: float = 0.6) -> Optional[BaseTool]:
        """
        Intelligently select the best tool for a given task.
        
        Args:
            task_type: Type of task (e.g., "file_operations", "build", "test")
            requirements: Specific requirements (version, capabilities, etc.)
            performance_weight: Weight for performance metrics in selection
            availability_weight: Weight for availability in selection
            
        Returns:
            Best tool for the task or None if none suitable
        """
        try:
            # Get candidate tools for task type
            candidates = await self._get_candidates_for_task(task_type, requirements)
            
            if not candidates:
                logger.warning(f"No tools found for task type: {task_type}")
                return None
            
            # Score each candidate
            scored_candidates = []
            for tool in candidates:
                score = await self._calculate_tool_score(
                    tool, task_type, requirements, performance_weight, availability_weight
                )
                scored_candidates.append((tool, score))
            
            # Sort by score (highest first)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            best_tool, best_score = scored_candidates[0]
            logger.debug(f"Selected tool {best_tool.name} with score {best_score:.2f} for task {task_type}")
            
            return best_tool
            
        except Exception as e:
            logger.error(f"Tool selection failed for task {task_type}: {e}")
            return None

    async def select_tools_for_workflow(self, 
                                      workflow_steps: List[Dict[str, Any]],
                                      optimize_for: str = "performance") -> Dict[str, BaseTool]:
        """
        Select optimal tools for a multi-step workflow.
        
        Args:
            workflow_steps: List of workflow steps with requirements
            optimize_for: Optimization criteria ("performance", "reliability", "speed")
            
        Returns:
            Dictionary mapping step names to selected tools
        """
        selected_tools = {}
        
        try:
            for step in workflow_steps:
                step_name = step.get("name", f"step_{len(selected_tools)}")
                task_type = step.get("task_type")
                requirements = step.get("requirements", {})
                
                # Adjust weights based on optimization criteria
                if optimize_for == "performance":
                    perf_weight, avail_weight = 0.7, 0.3
                elif optimize_for == "reliability":
                    perf_weight, avail_weight = 0.3, 0.7
                else:  # speed
                    perf_weight, avail_weight = 0.8, 0.2
                
                tool = await self.select_best_tool(
                    task_type, requirements, perf_weight, avail_weight
                )
                
                if tool:
                    selected_tools[step_name] = tool
                else:
                    logger.warning(f"No suitable tool found for workflow step: {step_name}")
            
            logger.info(f"Selected tools for {len(selected_tools)}/{len(workflow_steps)} workflow steps")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Workflow tool selection failed: {e}")
            return selected_tools

    async def get_tool_recommendations(self, 
                                     context: Dict[str, Any],
                                     limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on context.
        
        Args:
            context: Context information (project type, current task, etc.)
            limit: Maximum number of recommendations
            
        Returns:
            List of tool recommendations with reasoning
        """
        recommendations = []
        
        try:
            project_type = context.get("project_type", "flutter")
            current_task = context.get("current_task")
            available_tools = self.get_available_tools()
            
            for tool in available_tools:
                relevance_score = await self._calculate_relevance_score(tool, context)
                if relevance_score > 0.3:  # Threshold for relevance
                    recommendations.append({
                        "tool": tool,
                        "relevance_score": relevance_score,
                        "reasoning": await self._generate_recommendation_reasoning(tool, context),
                        "capabilities": await tool.get_capabilities(),
                        "metrics": self.get_metrics(tool.name)
                    })
            
            # Sort by relevance and limit
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Tool recommendation generation failed: {e}")
            return []

    async def check_tool_compatibility(self, 
                                     tool_names: List[str],
                                     operation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Check compatibility between multiple tools for joint operations.
        
        Args:
            tool_names: List of tool names to check
            operation_type: Specific operation type if applicable
            
        Returns:
            Compatibility analysis results
        """
        try:
            tools = [self.get_tool(name) for name in tool_names if self.get_tool(name)]
            
            if len(tools) != len(tool_names):
                missing = [name for name in tool_names if not self.get_tool(name)]
                return {
                    "compatible": False,
                    "error": f"Tools not available: {missing}"
                }
            
            compatibility_issues = []
            
            # Check version compatibility
            for i, tool1 in enumerate(tools):
                for tool2 in tools[i+1:]:
                    issues = await self._check_tool_pair_compatibility(tool1, tool2)
                    compatibility_issues.extend(issues)
            
            # Check for conflicts
            conflicts = await self._detect_tool_conflicts(tools, operation_type)
            
            return {
                "compatible": len(compatibility_issues) == 0 and len(conflicts) == 0,
                "tools": [tool.name for tool in tools],
                "issues": compatibility_issues,
                "conflicts": conflicts,
                "recommendations": await self._generate_compatibility_recommendations(tools)
            }
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return {"compatible": False, "error": str(e)}

    # =============================================
    # PERFORMANCE METRICS & MONITORING
    # =============================================

    async def record_tool_usage(self, 
                               tool_name: str, 
                               operation: str,
                               execution_time: float,
                               success: bool,
                               additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Record tool usage metrics for performance analysis.
        
        Args:
            tool_name: Name of the tool used
            operation: Operation performed
            execution_time: Time taken for execution
            success: Whether the operation was successful
            additional_metrics: Additional metrics to record
        """
        try:
            if tool_name not in self.tool_metrics:
                logger.warning(f"Recording metrics for unregistered tool: {tool_name}")
                return
            
            metrics = self.tool_metrics[tool_name]
            
            # Update basic counters
            metrics["total_uses"] += 1
            metrics["last_used"] = datetime.now()
            
            if success:
                metrics["successful_uses"] += 1
            else:
                metrics["failed_uses"] += 1
            
            # Update success rate
            metrics["success_rate"] = metrics["successful_uses"] / metrics["total_uses"]
            
            # Update response time metrics
            metrics["total_response_time"] += execution_time
            metrics["average_response_time"] = metrics["total_response_time"] / metrics["total_uses"]
            metrics["min_response_time"] = min(metrics["min_response_time"], execution_time)
            metrics["max_response_time"] = max(metrics["max_response_time"], execution_time)
            
            # Record in performance history
            history_entry = {
                "timestamp": datetime.now(),
                "operation": operation,
                "execution_time": execution_time,
                "success": success,
                "additional_metrics": additional_metrics or {}
            }
            
            self.performance_history[tool_name].append(history_entry)
            
            # Limit history size (keep last 1000 entries)
            if len(self.performance_history[tool_name]) > 1000:
                self.performance_history[tool_name] = self.performance_history[tool_name][-1000:]
            
            logger.debug(f"Recorded usage for {tool_name}: {operation} ({'success' if success else 'failure'}, {execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Failed to record tool usage for {tool_name}: {e}")

    def get_performance_analytics(self, 
                                tool_name: Optional[str] = None,
                                time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics.
        
        Args:
            tool_name: Specific tool name or None for all tools
            time_range_hours: Time range for analysis in hours
            
        Returns:
            Performance analytics data
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            if tool_name:
                return self._get_tool_analytics(tool_name, cutoff_time)
            else:
                return self._get_system_analytics(cutoff_time)
                
        except Exception as e:
            logger.error(f"Performance analytics generation failed: {e}")
            return {}

    def _get_tool_analytics(self, tool_name: str, cutoff_time: datetime) -> Dict[str, Any]:
        """Get analytics for a specific tool."""
        if tool_name not in self.tool_metrics:
            return {}
        
        metrics = self.tool_metrics[tool_name]
        history = self.performance_history.get(tool_name, [])
        
        # Filter recent history
        recent_history = [
            entry for entry in history 
            if entry["timestamp"] >= cutoff_time
        ]
        
        if not recent_history:
            return {
                "tool_name": tool_name,
                "no_recent_activity": True,
                "overall_metrics": metrics
            }
        
        # Calculate recent metrics
        recent_successes = sum(1 for entry in recent_history if entry["success"])
        recent_failures = len(recent_history) - recent_successes
        recent_avg_time = sum(entry["execution_time"] for entry in recent_history) / len(recent_history)
        
        # Operation breakdown
        operation_stats = {}
        for entry in recent_history:
            op = entry["operation"]
            if op not in operation_stats:
                operation_stats[op] = {"count": 0, "success_count": 0, "avg_time": 0, "total_time": 0}
            
            operation_stats[op]["count"] += 1
            operation_stats[op]["total_time"] += entry["execution_time"]
            if entry["success"]:
                operation_stats[op]["success_count"] += 1
        
        # Calculate averages
        for op_stats in operation_stats.values():
            op_stats["avg_time"] = op_stats["total_time"] / op_stats["count"]
            op_stats["success_rate"] = op_stats["success_count"] / op_stats["count"]
        
        return {
            "tool_name": tool_name,
            "time_range_hours": (datetime.now() - cutoff_time).total_seconds() / 3600,
            "recent_activity": {
                "total_operations": len(recent_history),
                "successful_operations": recent_successes,
                "failed_operations": recent_failures,
                "success_rate": recent_successes / len(recent_history),
                "average_response_time": recent_avg_time
            },
            "operation_breakdown": operation_stats,
            "overall_metrics": metrics,
            "availability": self.is_available(tool_name)
        }

    def _get_system_analytics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get system-wide analytics."""
        system_stats = {
            "total_tools": len(self.tools),
            "available_tools": len(self.get_available_tools()),
            "tool_analytics": {},
            "system_summary": {}
        }
        
        total_operations = 0
        total_successes = 0
        total_time = 0.0
        
        for tool_name in self.tools:
            tool_analytics = self._get_tool_analytics(tool_name, cutoff_time)
            system_stats["tool_analytics"][tool_name] = tool_analytics
            
            if "recent_activity" in tool_analytics:
                activity = tool_analytics["recent_activity"]
                total_operations += activity["total_operations"]
                total_successes += activity["successful_operations"]
                total_time += activity["average_response_time"] * activity["total_operations"]
        
        if total_operations > 0:
            system_stats["system_summary"] = {
                "total_operations": total_operations,
                "overall_success_rate": total_successes / total_operations,
                "average_response_time": total_time / total_operations,
                "most_used_tools": self._get_most_used_tools(cutoff_time),
                "performance_trends": self._calculate_performance_trends()
            }
        
        return system_stats

    def _get_most_used_tools(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get most used tools in the time range."""
        tool_usage = []
        
        for tool_name, history in self.performance_history.items():
            recent_count = sum(
                1 for entry in history 
                if entry["timestamp"] >= cutoff_time
            )
            if recent_count > 0:
                tool_usage.append({
                    "tool_name": tool_name,
                    "usage_count": recent_count
                })
        
        return sorted(tool_usage, key=lambda x: x["usage_count"], reverse=True)[:5]

    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends for the system."""
        # This is a simplified trend calculation
        # In a real implementation, you might use more sophisticated analysis
        
        trends = {}
        
        for tool_name, history in self.performance_history.items():
            if len(history) < 10:  # Need sufficient data
                continue
                
            recent_entries = history[-10:]
            older_entries = history[-20:-10] if len(history) >= 20 else []
            
            if older_entries:
                recent_avg = sum(e["execution_time"] for e in recent_entries) / len(recent_entries)
                older_avg = sum(e["execution_time"] for e in older_entries) / len(older_entries)
                
                if recent_avg < older_avg * 0.9:
                    trends[tool_name] = "improving"
                elif recent_avg > older_avg * 1.1:
                    trends[tool_name] = "degrading"
                else:
                    trends[tool_name] = "stable"
        
        return trends

    # =============================================
    # HEALTH MONITORING & CONFIGURATION MANAGEMENT
    # =============================================

    async def _start_health_monitoring(self):
        """Start background health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return  # Already running
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started background health monitoring")

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        try:
            while True:
                await asyncio.sleep(self.health_check_interval.total_seconds())
                await self._perform_health_checks()
                await self._update_availability_cache()
                
        except asyncio.CancelledError:
            logger.info("Health monitoring stopped")
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")

    async def _update_availability_cache(self):
        """Update availability cache and trigger degradation if needed."""
        for tool_name in self.tools:
            was_available = self.availability_cache.get(tool_name, True)
            is_available = self.availability_cache.get(tool_name, False)
            
            if was_available and not is_available:
                # Tool became unavailable
                logger.warning(f"Tool {tool_name} became unavailable")
                degradation_info = self.graceful_degrade(tool_name)
                await self._notify_tool_degradation(tool_name, degradation_info)

    async def _notify_tool_degradation(self, tool_name: str, degradation_info: Dict[str, Any]):
        """Notify about tool degradation (can be extended to send alerts)."""
        logger.warning(f"Tool degradation notification: {tool_name} - {degradation_info}")

    async def _load_tool_configurations(self):
        """Load tool-specific configurations."""
        try:
            # Default configurations for known tools
            default_configs = {
                "flutter_sdk_tool": {
                    "timeout": 300,
                    "retry_count": 3,
                    "cache_enabled": True
                },
                "file_system_tool": {
                    "backup_enabled": True,
                    "backup_retention_days": 7,
                    "max_file_size_mb": 10
                },
                "process_tool": {
                    "max_concurrent_processes": 5,
                    "process_timeout": 600
                }
            }
            
            # Merge with any existing configurations
            for tool_name, config in default_configs.items():
                if tool_name not in self.tool_configs:
                    self.tool_configs[tool_name] = {}
                self.tool_configs[tool_name].update(config)
                
            logger.debug("Tool configurations loaded")
            
        except Exception as e:
            logger.error(f"Failed to load tool configurations: {e}")

    def update_tool_config(self, tool_name: str, config_updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            config_updates: Configuration updates to apply
            
        Returns:
            True if successfully updated, False otherwise
        """
        try:
            if tool_name not in self.tools:
                logger.error(f"Cannot update config for unknown tool: {tool_name}")
                return False
            
            if tool_name not in self.tool_configs:
                self.tool_configs[tool_name] = {}
            
            self.tool_configs[tool_name].update(config_updates)
            
            # Notify tool of configuration change if it supports it
            tool = self.tools[tool_name]
            if hasattr(tool, 'update_config'):
                try:
                    tool.update_config(self.tool_configs[tool_name])
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed to update config: {e}")
            
            logger.info(f"Updated configuration for tool {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tool config for {tool_name}: {e}")
            return False

    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name)

    def _build_compatibility_matrix(self):
        """Build compatibility matrix for all tools."""
        try:
            self.compatibility_matrix = {}
            
            for tool1_name, tool1 in self.tools.items():
                self.compatibility_matrix[tool1_name] = {}
                
                for tool2_name, tool2 in self.tools.items():
                    if tool1_name == tool2_name:
                        continue
                    
                    # Check version compatibility
                    compatibility_score = self._calculate_compatibility_score(tool1, tool2)
                    self.compatibility_matrix[tool1_name][tool2_name] = compatibility_score
            
            logger.debug("Compatibility matrix built")
            
        except Exception as e:
            logger.error(f"Failed to build compatibility matrix: {e}")

    def _calculate_compatibility_score(self, tool1: BaseTool, tool2: BaseTool) -> float:
        """Calculate compatibility score between two tools."""
        score = 1.0  # Start with perfect compatibility
        
        try:
            # Category compatibility (tools in same category might conflict)
            if tool1.category == tool2.category:
                score -= 0.2
            
            # Version compatibility (simplified)
            version1 = self._parse_version(tool1.version)
            version2 = self._parse_version(tool2.version)
            
            if version1 and version2 and len(version1) > 0 and len(version2) > 0:
                major_diff = abs(version1[0] - version2[0]) if len(version1) > 0 and len(version2) > 0 else 0
                if major_diff > 1:
                    score -= 0.3
            
            # Dependency compatibility
            tool1_deps = self.tool_dependencies.get(tool1.name, [])
            tool2_deps = self.tool_dependencies.get(tool2.name, [])
            
            if tool1.name in tool2_deps or tool2.name in tool1_deps:
                score += 0.2  # Bonus for dependent tools
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Compatibility calculation failed for {tool1.name} and {tool2.name}: {e}")
            return 0.5  # Default neutral score

    def _setup_fallback_strategies(self):
        """Setup fallback strategies for tools."""
        try:
            # Group tools by category for fallback options
            category_tools = {}
            for tool_name, tool in self.tools.items():
                category = tool.category.value
                if category not in category_tools:
                    category_tools[category] = []
                category_tools[category].append(tool_name)
            
            # Setup fallbacks based on priority and availability
            for tool_name, tool in self.tools.items():
                category = tool.category.value
                same_category_tools = category_tools.get(category, [])
                
                # Remove self from fallback options
                fallbacks = [t for t in same_category_tools if t != tool_name]
                
                # Sort by priority (higher priority tools are better fallbacks)
                fallbacks.sort(key=lambda t: self.tool_metrics.get(t, {}).get("priority", 5), reverse=True)
                
                self.fallback_strategies[tool_name] = fallbacks[:3]  # Keep top 3 fallbacks
            
            logger.debug("Fallback strategies configured")
            
        except Exception as e:
            logger.error(f"Failed to setup fallback strategies: {e}")

    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")

    async def shutdown(self):
        """Shutdown the tool registry gracefully."""
        try:
            logger.info("Shutting down Tool Registry...")
            
            # Stop health monitoring
            await self.stop_health_monitoring()
            
            # Cleanup all tools
            for tool_name in list(self.tools.keys()):
                await self.unregister_tool(tool_name, force=True)
            
            # Clear all data
            self.tools.clear()
            self.tool_dependencies.clear()
            self.tool_metrics.clear()
            self.availability_cache.clear()
            self.last_health_check.clear()
            self.performance_history.clear()
            
            self.is_initialized = False
            logger.info("Tool Registry shutdown complete")
            
        except Exception as e:
            logger.error(f"Tool Registry shutdown failed: {e}")

    # =============================================
    # PUBLIC API METHODS
    # =============================================

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
            performance_summary = await tool.get_performance_summary() if hasattr(tool, 'get_performance_summary') else {}
            
            return {
                "tool_name": tool_name,
                "version": tool.version,
                "category": tool.category.value,
                "capabilities": capabilities,
                "performance": performance_summary,
                "availability": self.is_available(tool_name),
                "metrics": self.get_metrics(tool_name)
            }
        except Exception as e:
            logger.error(f"Failed to query capabilities for {tool_name}: {e}")
            return None

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
        
        # Return cached value (will be updated by background monitoring)
        return self.availability_cache.get(tool_name, False)

    async def _perform_health_checks(self):
        """Perform health checks on all registered tools."""
        logger.debug("Performing health checks on all tools...")
        
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
        alternatives = self.fallback_strategies.get(tool_name, [])
        
        # Filter alternatives to only available ones
        available_alternatives = [
            alt for alt in alternatives 
            if self.is_available(alt)
        ]
        
        degradation_info = {
            "unavailable_tool": tool_name,
            "alternatives": available_alternatives,
            "degradation_mode": "alternative_tools" if available_alternatives else "reduced_functionality",
            "timestamp": datetime.now().isoformat(),
            "fallback_strategy": self.fallback_strategies.get(tool_name, [])
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
            "unavailable_tools": total_tools - available_tools,
            "availability_rate": available_tools / total_tools if total_tools > 0 else 0,
            "categories": list(set(tool.category.value for tool in self.tools.values())),
            "last_health_check": max(self.last_health_check.values()) if self.last_health_check else None,
            "is_initialized": self.is_initialized,
            "monitoring_active": self._health_check_task is not None and not self._health_check_task.done(),
            "registry_uptime": datetime.now().isoformat()
        }

    async def execute_llm_task(self, user_prompt: str, context: dict, include_tools: bool = True) -> dict:
        """
        Execute an LLM-driven task with comprehensive tool integration.
        - Includes available tools/capabilities in the system prompt
        - Adds successful tool usage patterns from history
        - Parses tool usage intentions from LLM responses
        - Executes planned tool operations automatically
        - Feeds tool results back to LLM for interpretation
        - Supports iterative tool usage until task completion
        """
        llm_client = context.get("llm_client")
        if llm_client is None:
            raise ValueError("LLM client must be provided in context")

        # Step 1: Build tool-aware system prompt
        system_prompt = self._generate_tool_aware_prompt(user_prompt, context)
        conversation = []
        max_iterations = 5
        iteration = 0
        final_result = None
        tool_results = {}
        while iteration < max_iterations:
            # Step 2: Call LLM with current prompt and context
            llm_input = system_prompt
            if conversation:
                llm_input += "\n\n" + "\n".join(conversation)
            if tool_results:
                llm_input += f"\n\n[Tool Results]: {tool_results}"
            llm_response = await llm_client.generate(
                prompt=llm_input,
                model=context.get("llm_model"),
                temperature=context.get("temperature", 0.3),
                max_tokens=context.get("max_tokens", 2000),
            )
            conversation.append(f"[LLM]: {llm_response}")

            # Step 3: Parse tool usage intentions from LLM response
            tool_calls = self._parse_tool_usage_intentions(llm_response)
            if not tool_calls:
                # No more tool calls, assume task complete
                final_result = llm_response
                break

            # Step 4: Validate and execute tool calls
            tool_results = {}
            for call in tool_calls:
                tool_name = call.get("tool")
                operation = call.get("operation")
                params = call.get("parameters", {})
                if not self._validate_tool_call(tool_name, operation):
                    tool_results[tool_name] = {"error": f"Invalid tool or operation: {tool_name}.{operation}"}
                    continue
                tool = self.get_tool(tool_name)
                try:
                    op_method = getattr(tool, operation)
                    if asyncio.iscoroutinefunction(op_method):
                        result = await op_method(params, None)
                    else:
                        result = op_method(params, None)
                    tool_results[tool_name] = result if isinstance(result, dict) else result.__dict__
                except Exception as e:
                    tool_results[tool_name] = {"error": str(e)}
            # Feed tool results back to LLM for further reasoning
            iteration += 1
        return {"result": final_result, "conversation": conversation, "tool_results": tool_results}

    def _generate_tool_aware_prompt(self, user_prompt: str, context: dict) -> str:
        """
        Generate a system prompt that includes:
        - Available tool capabilities
        - Usage examples from history
        - Current project context
        - Resource constraints and performance considerations
        - Formatted for optimal LLM understanding and tool selection
        """
        available_tools = self.get_available_tools()
        tool_descriptions = []
        for tool in available_tools:
            try:
                capabilities = asyncio.run(tool.get_capabilities())
            except Exception:
                capabilities = {}
            tool_descriptions.append(f"- {tool.name} (v{tool.version}): {tool.description}\n  Capabilities: {capabilities}")
        # Add usage examples from history
        usage_examples = []
        for tool in available_tools:
            history = getattr(tool, "usage_history", [])
            for entry in history[-2:]:  # last 2 usages
                usage_examples.append(f"Tool: {tool.name}, Operation: {entry.get('operation')}, Params: {entry.get('params')}, Success: {entry.get('success')}")
        # Project context
        project_ctx = context.get("project_context", "[No project context provided]")
        # Resource/performance
        perf = context.get("performance", "Standard performance constraints apply.")
        prompt = (
            "You are an intelligent multi-agent system with access to the following tools.\n"
            "When planning, select the best tool(s) and operations for the user's task.\n"
            "If tool usage is needed, respond with a JSON array of tool calls.\n"
            "Otherwise, provide the final answer.\n\n"
            f"[User Task]: {user_prompt}\n\n"
            f"[Available Tools]:\n" + "\n".join(tool_descriptions) + "\n\n"
            f"[Usage Examples]:\n" + ("\n".join(usage_examples) if usage_examples else "None") + "\n\n"
            f"[Project Context]: {project_ctx}\n"
            f"[Performance]: {perf}\n"
            "Respond in this format if using tools: \n"
            "[TOOL_CALLS]: [\n  {\"tool\": \"tool_name\", \"operation\": \"operation_name\", \"parameters\": {...}}, ...\n]\n"
        )
        return prompt

    def _parse_tool_usage_intentions(self, llm_response: str) -> list:
        """
        Extract tool calls from LLM response. Expects a JSON array after [TOOL_CALLS]:
        """
        import json, re
        match = re.search(r"\[TOOL_CALLS\]:\s*(\[.*?\])", llm_response, re.DOTALL)
        if not match:
            return []
        try:
            tool_calls = json.loads(match.group(1))
            if isinstance(tool_calls, list):
                return tool_calls
        except Exception:
            return []
        return []

    def _validate_tool_call(self, tool_name: str, operation: str) -> bool:
        """
        Validate that the tool and operation exist and are available.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        if not hasattr(tool, operation):
            return False
        return True
