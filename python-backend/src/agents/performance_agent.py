"""
Performance Agent for FlutterSwarm Multi-Agent System.

This agent specializes in performance optimization, monitoring,
and performance analysis for Flutter applications.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.agents.base_agent import BaseAgent, AgentCapability, AgentConfig
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.models.agent_models import AgentMessage, TaskResult
from src.models.task_models import TaskContext
from src.models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from src.config import get_logger

logger = get_logger("performance_agent")

class PerformanceAgent(BaseAgent):
    """
    Specialized agent for Flutter application performance optimization and monitoring.
    """
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        performance_config = AgentConfig(
            agent_id=config.agent_id or f"performance_agent_{str(uuid.uuid4())[:8]}",
            agent_type="performance",
            capabilities=[
                AgentCapability.PERFORMANCE_OPTIMIZATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.MONITORING
            ],
            max_concurrent_tasks=3,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.3,
            max_tokens=7000,
            timeout=700,
            metadata=config.metadata
        )
        super().__init__(performance_config, llm_client, memory_manager, event_bus)
        self.performance_areas = [
            "ui_rendering", "memory_management", "network_optimization", 
            "battery_optimization", "startup_time", "build_performance",
            "animation_performance", "scroll_performance", "image_optimization"
        ]
        self.profiling_tools = {
            "flutter_tools": ["flutter_inspector", "performance_overlay", "widget_inspector"],
            "dart_tools": ["dart_devtools", "observatory", "timeline"],
            "platform_tools": ["xcode_instruments", "android_profiler"],
            "third_party": ["firebase_performance", "sentry_performance", "datadog_rum"]
        }
        self.optimization_techniques = {
            "widget_optimization": ["const_constructors", "widget_keys", "builder_patterns"],
            "memory_optimization": ["object_pooling", "weak_references", "disposal_patterns"],
            "rendering_optimization": ["repaint_boundaries", "clip_behavior", "opacity_optimization"],
            "network_optimization": ["caching", "compression", "connection_pooling"]
        }
        logger.info(f"Performance Agent {self.agent_id} initialized")

    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the performance agent."""
        return """
You are the Performance Agent in the FlutterSwarm multi-agent system, specializing in Flutter application performance optimization and monitoring.

CORE EXPERTISE:
- Flutter performance optimization techniques and best practices
- Dart language performance characteristics and optimization
- Cross-platform performance considerations (iOS, Android, Web, Desktop)
- UI rendering performance and optimization strategies

Always provide measurable performance improvements with specific optimization techniques and monitoring strategies.
"""

    async def get_capabilities(self) -> List[str]:
        return [
            "performance_analysis",
            "code_optimization",
            "memory_optimization",
            "ui_performance_tuning",
            "network_optimization",
            "battery_optimization",
            "performance_monitoring",
            "profiling_setup",
            "benchmark_creation",
            "performance_testing",
            "regression_detection",
            "optimization_recommendations"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            task_type = task_context.task_type.value
            if task_type == "performance_analysis":
                return await self._analyze_performance(task_context, llm_analysis)
            elif task_type == "optimization":
                return await self._optimize_performance(task_context, llm_analysis)
            elif task_type == "monitoring_setup":
                return await self._setup_monitoring(task_context, llm_analysis)
            elif task_type == "profiling_setup":
                return await self._setup_profiling(task_context, llm_analysis)
            elif task_type == "benchmark_creation":
                return await self._create_benchmarks(task_context, llm_analysis)
            else:
                return await self._process_performance_request(task_context, llm_analysis)
        except Exception as e:
            logger.error(f"PerformanceAgent error: {e}")
            return {"error": str(e)}

    async def _analyze_performance(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks and provide recommendations."""
        logger.info(f"Analyzing performance for task: {task_context.task_id}")
        # Example: Use LLM to analyze and return performance suggestions
        analysis_prompt = f"""
Analyze the following Flutter app code and context for performance bottlenecks:

CODE ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PERFORMANCE CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

REQUIREMENTS:
{task_context.description}
"""
        analysis = await self.execute_llm_task(
            user_prompt=analysis_prompt,
            context={
                "task": task_context.to_dict(),
                "performance_areas": self.performance_areas,
                "profiling_tools": self.profiling_tools
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Performance analysis: {json.dumps(analysis)}",
            metadata={
                "type": "performance_analysis",
                "target_file": task_context.metadata.get('target_file'),
                "bottlenecks": analysis.get('bottlenecks', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        return {
            "performance_analysis": analysis,
            "bottlenecks": analysis.get("bottlenecks", []),
            "recommendations": analysis.get("recommendations", [])
        }

    async def _optimize_performance(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code and provide performance improvements."""
        logger.info(f"Optimizing performance for task: {task_context.task_id}")
        optimization_prompt = f"""
Suggest and implement performance optimizations for the following Flutter code:

CODE ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PERFORMANCE CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

REQUIREMENTS:
{task_context.description}
"""
        optimizations = await self.execute_llm_task(
            user_prompt=optimization_prompt,
            context={
                "task": task_context.to_dict(),
                "optimization_techniques": self.optimization_techniques
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Performance optimizations: {json.dumps(optimizations)}",
            metadata={
                "type": "performance_optimization",
                "target_file": task_context.metadata.get('target_file'),
                "optimizations": optimizations.get('optimizations', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        return {
            "optimizations": optimizations.get("optimizations", []),
            "optimized_code": optimizations.get("optimized_code"),
            "notes": optimizations.get("notes", [])
        }

    async def _setup_monitoring(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Set up performance monitoring tools and strategies."""
        logger.info(f"Setting up monitoring for task: {task_context.task_id}")
        monitoring_prompt = f"""
Recommend and configure performance monitoring for the following Flutter app context:

APP CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

REQUIREMENTS:
{task_context.description}
"""
        monitoring = await self.execute_llm_task(
            user_prompt=monitoring_prompt,
            context={
                "task": task_context.to_dict(),
                "profiling_tools": self.profiling_tools
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Monitoring setup: {json.dumps(monitoring)}",
            metadata={
                "type": "monitoring_setup",
                "tools": monitoring.get('tools', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.7,
            long_term=True
        )
        return {
            "monitoring_tools": monitoring.get("tools", []),
            "setup_instructions": monitoring.get("setup", []),
            "alerting": monitoring.get("alerting", {})
        }

    async def _setup_profiling(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Set up profiling tools and provide profiling instructions."""
        logger.info(f"Setting up profiling for task: {task_context.task_id}")
        profiling_prompt = f"""
Recommend and configure profiling for the following Flutter app context:

APP CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

REQUIREMENTS:
{task_context.description}
"""
        profiling = await self.execute_llm_task(
            user_prompt=profiling_prompt,
            context={
                "task": task_context.to_dict(),
                "profiling_tools": self.profiling_tools
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Profiling setup: {json.dumps(profiling)}",
            metadata={
                "type": "profiling_setup",
                "tools": profiling.get('tools', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.7,
            long_term=True
        )
        return {
            "profiling_tools": profiling.get("tools", []),
            "setup_instructions": profiling.get("setup", []),
            "profiling_notes": profiling.get("notes", [])
        }

    async def _create_benchmarks(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance benchmarks and regression tests."""
        logger.info(f"Creating benchmarks for task: {task_context.task_id}")
        benchmark_prompt = f"""
Create performance benchmarks for the following Flutter app context:

APP CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

REQUIREMENTS:
{task_context.description}
"""
        benchmarks = await self.execute_llm_task(
            user_prompt=benchmark_prompt,
            context={
                "task": task_context.to_dict(),
                "performance_areas": self.performance_areas
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Benchmarks created: {json.dumps(benchmarks)}",
            metadata={
                "type": "benchmark_creation",
                "benchmarks": benchmarks.get('benchmarks', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.7,
            long_term=True
        )
        return {
            "benchmarks": benchmarks.get("benchmarks", []),
            "benchmark_code": benchmarks.get("benchmark_code"),
            "regression_tests": benchmarks.get("regression_tests", [])
        }

    async def _process_performance_request(self, task_context: TaskContext, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process generic performance-related requests."""
        logger.info(f"Processing generic performance request for task: {task_context.task_id}")
        generic_prompt = f"""
Process the following performance-related request for a Flutter application:

PERFORMANCE REQUEST:
{json.dumps(llm_analysis, indent=2)}

REQUEST DETAILS:
{task_context.description}

Please provide comprehensive performance guidance with actionable
recommendations and implementation details.
"""
        result = await self.execute_llm_task(
            user_prompt=generic_prompt,
            context={
                "task": task_context.to_dict(),
                "performance_areas": self.performance_areas
            },
            structured_output=True
        )
        await self.memory_manager.store_memory(
            content=f"Performance request processed: {json.dumps(result)}",
            metadata={
                "type": "performance_request",
                "request": task_context.description
            },
            correlation_id=task_context.correlation_id,
            importance=0.6,
            long_term=True
        )
        return {
            "performance_guidance": result.get("guidance", []),
            "implementation_details": result.get("implementation_details", {})
        }

    def _get_optimization_patterns(self) -> Dict[str, Any]:
        """Get optimization patterns and techniques."""
        return {
            "widget_patterns": {
                "const_constructors": "Use const constructors where possible",
                "builder_patterns": "Use builder patterns for expensive widgets",
                "repaint_boundaries": "Add RepaintBoundary for complex widgets",
                "widget_keys": "Use keys for stateful widgets in lists"
            },
            "memory_patterns": {
                "dispose_controllers": "Dispose controllers in dispose() method",
                "weak_references": "Use weak references for callbacks",
                "object_pooling": "Implement object pooling for frequently created objects",
                "stream_disposal": "Cancel streams and subscriptions"
            },
            "performance_patterns": {
                "lazy_loading": "Implement lazy loading for large datasets",
                "pagination": "Use pagination for long lists",
                "image_caching": "Implement proper image caching",
                "network_caching": "Cache network responses appropriately"
            }
        }

    # Prompt creation methods
    def _create_performance_analysis_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for performance analysis."""
        return f"""
Conduct comprehensive performance analysis for the following Flutter application:

APPLICATION ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PERFORMANCE SCOPE:
{task_context.description}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please provide detailed performance analysis including bottleneck identification,
optimization opportunities, and measurable performance metrics.
"""

    def _create_optimization_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for performance optimization."""
        return f"""
Implement performance optimizations for the following Flutter application:

OPTIMIZATION REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

OPTIMIZATION SCOPE:
{task_context.description}

Please provide comprehensive performance optimizations with optimized code
implementations and measurable improvements.
"""

    def _create_monitoring_setup_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for monitoring setup."""
        return f"""
Set up comprehensive performance monitoring for the following Flutter application:

MONITORING REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

MONITORING SCOPE:
{task_context.description}

Please provide complete monitoring setup with implementation code,
configuration files, and setup instructions.
"""

    def _create_profiling_setup_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for profiling setup."""
        return f"""
Set up performance profiling infrastructure for the following Flutter application:

PROFILING REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

PROFILING SCOPE:
{task_context.description}

Please provide comprehensive profiling setup with configuration files,
automation scripts, and analysis procedures.
"""

    def _create_benchmark_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for benchmark creation."""
        return f"""
Create comprehensive performance benchmarks for the following Flutter application:

BENCHMARK REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

BENCHMARK SCOPE:
{task_context.description}

Please provide complete benchmarking suite with test implementations,
automation scripts, and performance targets.
"""

    def _create_generic_performance_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for generic performance processing."""
        return f"""
Process the following performance-related request for a Flutter application:

PERFORMANCE REQUEST:
{json.dumps(llm_analysis, indent=2)}

REQUEST DETAILS:
{task_context.description}

Please provide comprehensive performance guidance with actionable
recommendations and implementation details.
"""
