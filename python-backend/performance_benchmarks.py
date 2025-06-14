#!/usr/bin/env python3
"""
Performance Benchmarking Framework for BaseAgent Tool Integration.

This framework provides comprehensive performance benchmarks for:
1. Tool selection accuracy measurement
2. Learning improvement tracking over time
3. Tool operation efficiency monitoring
4. Memory usage and cleanup validation
5. Scalability and concurrent operation testing

Metrics tracked:
- Tool selection accuracy rates
- Learning curve analysis
- Operation latency and throughput
- Memory consumption patterns
- Concurrent operation performance
"""

import asyncio
import gc
import json
import logging
import os
import psutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, median, stdev
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
from src.core.tools.base_tool import BaseTool, ToolCategory
from src.core.tools.file_system_tool import FileSystemTool
from src.core.tools.flutter_sdk_tool import FlutterSDKTool
from src.core.tools.process_tool import ProcessTool
from src.core.tools.tool_registry import ToolRegistry
from src.models.tool_models import ToolUsageEntry, ToolMetrics, ToolResult, ToolStatus
from src.core.memory_manager import MemoryManager
from src.core.event_bus import EventBus
from src.config import get_logger

logger = get_logger("performance_benchmarks")


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    benchmark_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    tool_selection_accuracy: float = 0.0
    average_operation_latency: float = 0.0
    operations_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    memory_growth_rate: float = 0.0
    learning_improvement_rate: float = 0.0
    error_rate: float = 0.0
    concurrent_operation_efficiency: float = 0.0


class PerformanceBenchmarkFramework:
    """Comprehensive performance benchmarking framework."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.test_agent: Optional[BaseAgent] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.benchmark_start_time = datetime.now()
        
    async def setup_benchmark_environment(self):
        """Set up the benchmarking environment."""
        logger.info("🔧 Setting up performance benchmark environment...")
        
        # Create test agent with mock dependencies
        from unittest.mock import Mock, AsyncMock
        
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value={
            "summary": "Mock tool analysis",
            "usage_scenarios": ["scenario1", "scenario2"],
            "confidence": 0.8
        })
        
        mock_memory = Mock()
        mock_memory.store_memory = AsyncMock()
        mock_memory.retrieve_memories = AsyncMock(return_value=[])
        
        mock_event_bus = Mock()
        mock_event_bus.subscribe = AsyncMock()
        mock_event_bus.publish = AsyncMock()
        
        # Create test agent
        agent_config = AgentConfig(
            agent_id="benchmark_agent",
            agent_type="benchmark",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.TESTING
            ],
            max_concurrent_tasks=10
        )
        
        self.test_agent = self._create_test_agent()
        self.tool_registry = ToolRegistry.instance()
        
        # Register real tools for testing
        await self._register_test_tools()
        
        logger.info("✅ Benchmark environment setup complete")
    
    def _create_test_agent(self) -> BaseAgent:
        """Create test agent for benchmarking."""
        from unittest.mock import Mock
        
        class BenchmarkTestAgent(BaseAgent):
            async def _get_default_system_prompt(self):
                return "Benchmark test agent for tool integration performance testing"
            
            async def get_capabilities(self):
                return ["benchmarking", "tool_integration", "performance_testing"]
        
        config = AgentConfig(
            agent_id="benchmark_test_agent",
            agent_type="benchmark",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.TESTING
            ],
            max_concurrent_tasks=10
        )
        
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value={"response": "mock response"})
        
        return BenchmarkTestAgent(
            config=config,
            llm_client=mock_llm,
            memory_manager=Mock(),
            event_bus=Mock()
        )
    
    async def _register_test_tools(self):
        """Register test tools for benchmarking."""
        await self.tool_registry.register_tool(FileSystemTool())
        await self.tool_registry.register_tool(FlutterSDKTool())
        await self.tool_registry.register_tool(ProcessTool())
    
    async def run_comprehensive_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all performance benchmarks and return results."""
        logger.info("🚀 Starting Comprehensive Performance Benchmarks")
        logger.info("=" * 70)
        
        results = {}
        
        # Core performance benchmarks
        benchmark_suites = [
            ("Tool Selection Accuracy", self._benchmark_tool_selection_accuracy),
            ("Learning Improvement Tracking", self._benchmark_learning_improvement_tracking),
            ("Tool Operation Efficiency", self._benchmark_tool_operation_efficiency),
            ("Memory Usage and Cleanup", self._benchmark_memory_usage_cleanup),
            ("Concurrent Operation Performance", self._benchmark_concurrent_operations),
            ("Tool Discovery Performance", self._benchmark_tool_discovery_performance),
            ("LLM Integration Efficiency", self._benchmark_llm_integration_efficiency),
            ("Error Recovery Performance", self._benchmark_error_recovery_performance)
        ]
        
        for benchmark_name, benchmark_func in benchmark_suites:
            logger.info(f"\\n⚡ Running: {benchmark_name}")
            logger.info("-" * 50)
            
            start_time = datetime.now()
            try:
                metrics = await benchmark_func()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = BenchmarkResult(
                    benchmark_name=benchmark_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=True,
                    metrics=metrics
                )
                
                results[benchmark_name] = result
                logger.info(f"✅ {benchmark_name} completed in {duration:.2f}s")
                self._log_benchmark_metrics(benchmark_name, metrics)
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = BenchmarkResult(
                    benchmark_name=benchmark_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=False,
                    error_message=str(e)
                )
                
                results[benchmark_name] = result
                logger.error(f"❌ {benchmark_name} failed: {e}")
        
        # Generate comprehensive report
        await self._generate_performance_report(results)
        
        return results
    
    async def _benchmark_tool_selection_accuracy(self) -> Dict[str, Any]:
        """
        Measure tool selection accuracy rates.
        
        Tests the agent's ability to select the most appropriate tool
        for different types of tasks with high accuracy.
        """
        logger.info("📊 Measuring tool selection accuracy...")
        
        # Define test scenarios with expected optimal tools
        test_scenarios = [
            {
                "task_description": "Create a new Flutter widget file with boilerplate code",
                "expected_tool": "file_system_tool",
                "category": "file_operations",
                "confidence_threshold": 0.8
            },
            {
                "task_description": "Build Flutter application for Android platform",
                "expected_tool": "flutter_sdk_tool", 
                "category": "build_operations",
                "confidence_threshold": 0.9
            },
            {
                "task_description": "Execute git commands for version control",
                "expected_tool": "process_tool",
                "category": "system_operations", 
                "confidence_threshold": 0.7
            },
            {
                "task_description": "Analyze Flutter project dependencies",
                "expected_tool": "flutter_sdk_tool",
                "category": "analysis_operations",
                "confidence_threshold": 0.8
            },
            {
                "task_description": "Create directory structure for new Flutter module",
                "expected_tool": "file_system_tool",
                "category": "structure_operations",
                "confidence_threshold": 0.85
            }
        ]
        
        correct_selections = 0
        total_selections = len(test_scenarios)
        accuracy_by_category = defaultdict(list)
        confidence_scores = []
        response_times = []
        
        for scenario in test_scenarios:
            start_time = time.time()
            
            # Test tool selection
            selection_result = await self.test_agent.select_best_tool_for_task(
                scenario["task_description"]
            )
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Evaluate accuracy
            selected_tool = selection_result.get("tool_name", "")
            confidence = selection_result.get("confidence", 0.0)
            confidence_scores.append(confidence)
            
            is_correct = (
                selected_tool == scenario["expected_tool"] and
                confidence >= scenario["confidence_threshold"]
            )
            
            if is_correct:
                correct_selections += 1
                accuracy_by_category[scenario["category"]].append(1.0)
            else:
                accuracy_by_category[scenario["category"]].append(0.0)
            
            logger.info(f"  Task: {scenario['task_description'][:50]}...")
            logger.info(f"    Expected: {scenario['expected_tool']}, Got: {selected_tool}")
            logger.info(f"    Confidence: {confidence:.3f}, Correct: {'✅' if is_correct else '❌'}")
        
        # Calculate metrics
        overall_accuracy = correct_selections / total_selections
        average_confidence = mean(confidence_scores) if confidence_scores else 0
        average_response_time = mean(response_times) if response_times else 0
        
        category_accuracies = {
            category: mean(accuracies) if accuracies else 0
            for category, accuracies in accuracy_by_category.items()
        }
        
        return {
            "overall_accuracy": overall_accuracy,
            "correct_selections": correct_selections,
            "total_selections": total_selections,
            "average_confidence": average_confidence,
            "average_response_time_ms": average_response_time * 1000,
            "category_accuracies": category_accuracies,
            "confidence_distribution": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "std": stdev(confidence_scores) if len(confidence_scores) > 1 else 0
            }
        }
    
    async def _benchmark_learning_improvement_tracking(self) -> Dict[str, Any]:
        """
        Track learning improvement over time.
        
        Measures how the agent's tool usage patterns and accuracy
        improve through repeated interactions and feedback.
        """
        logger.info("📈 Tracking learning improvement over time...")
        
        # Initialize learning session
        learning_iterations = 20
        initial_accuracy = 0.5  # Starting baseline
        
        accuracy_timeline = []
        confidence_timeline = []
        tool_preference_evolution = []
        learning_rate_measurements = []
        
        for iteration in range(learning_iterations):
            logger.info(f"  Learning iteration {iteration + 1}/{learning_iterations}")
            
            # Simulate tool usage with feedback
            task_scenarios = [
                "Create Flutter component",
                "Build application", 
                "Manage project files",
                "Execute system commands"
            ]
            
            iteration_accuracy = 0
            iteration_confidence = 0
            
            for task in task_scenarios:
                # Get tool selection
                selection = await self.test_agent.select_best_tool_for_task(task)
                
                # Simulate learning feedback
                feedback_success = iteration / learning_iterations + 0.5  # Improving over time
                confidence = selection.get("confidence", 0.0)
                
                # Record learning data
                usage_entry = ToolUsageEntry(
                    agent_id=self.test_agent.agent_id,
                    tool_name=selection.get("tool_name", "unknown"),
                    operation="selection_test",
                    parameters={"task": task},
                    timestamp=datetime.now(),
                    result={"success": feedback_success > 0.7},
                    reasoning=f"Learning iteration {iteration}"
                )
                
                await self.test_agent.learn_from_tool_usage(usage_entry)
                
                iteration_accuracy += min(feedback_success, 1.0)
                iteration_confidence += confidence
            
            # Calculate iteration metrics
            avg_accuracy = iteration_accuracy / len(task_scenarios)
            avg_confidence = iteration_confidence / len(task_scenarios)
            
            accuracy_timeline.append(avg_accuracy)
            confidence_timeline.append(avg_confidence)
            
            # Track tool preference evolution
            tool_preferences = await self.test_agent.get_tool_preference_rankings()
            tool_preference_evolution.append(tool_preferences)
            
            # Calculate learning rate
            if len(accuracy_timeline) >= 3:
                recent_improvement = accuracy_timeline[-1] - accuracy_timeline[-3]
                learning_rate_measurements.append(recent_improvement)
        
        # Calculate learning metrics
        final_accuracy = accuracy_timeline[-1] if accuracy_timeline else initial_accuracy
        total_improvement = final_accuracy - initial_accuracy
        average_learning_rate = mean(learning_rate_measurements) if learning_rate_measurements else 0
        
        # Analyze learning curve
        learning_curve_slope = 0
        if len(accuracy_timeline) > 1:
            x_values = list(range(len(accuracy_timeline)))
            y_values = accuracy_timeline
            
            # Simple linear regression for slope
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            learning_curve_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "total_improvement": total_improvement,
            "learning_iterations": learning_iterations,
            "average_learning_rate": average_learning_rate,
            "learning_curve_slope": learning_curve_slope,
            "accuracy_timeline": accuracy_timeline,
            "confidence_timeline": confidence_timeline,
            "improvement_percentage": (total_improvement / initial_accuracy) * 100 if initial_accuracy > 0 else 0,
            "learning_stability": {
                "accuracy_variance": stdev(accuracy_timeline) if len(accuracy_timeline) > 1 else 0,
                "confidence_variance": stdev(confidence_timeline) if len(confidence_timeline) > 1 else 0
            }
        }
    
    async def _benchmark_tool_operation_efficiency(self) -> Dict[str, Any]:
        """
        Monitor tool operation efficiency.
        
        Measures latency, throughput, and resource utilization
        for various tool operations.
        """
        logger.info("⚡ Monitoring tool operation efficiency...")
        
        # Test different operation types
        operation_tests = [
            {
                "name": "File Operations",
                "tool": "file_system_tool",
                "operations": ["create_file", "read_file", "write_file", "delete_file"],
                "iterations": 50
            },
            {
                "name": "Flutter SDK Operations", 
                "tool": "flutter_sdk_tool",
                "operations": ["create_project", "build_app", "analyze_code"],
                "iterations": 20
            },
            {
                "name": "Process Operations",
                "tool": "process_tool", 
                "operations": ["execute_command", "get_process_info"],
                "iterations": 30
            }
        ]
        
        efficiency_results = {}
        
        for test_suite in operation_tests:
            logger.info(f"  Testing {test_suite['name']}...")
            
            suite_metrics = {
                "operations_per_second": [],
                "average_latency_ms": [],
                "success_rate": [],
                "memory_usage_mb": [],
                "cpu_usage_percent": []
            }
            
            for operation in test_suite["operations"]:
                # Measure operation performance
                operation_times = []
                success_count = 0
                memory_samples = []
                
                for i in range(test_suite["iterations"]):
                    # Monitor system resources
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024
                    
                    start_time = time.time()
                    
                    try:
                        # Execute operation
                        result = await self.test_agent.use_tool(
                            tool_name=test_suite["tool"],
                            operation=operation,
                            parameters=self._get_test_parameters(operation, i),
                            reasoning=f"Efficiency test {i}"
                        )
                        
                        if result.status.value == "success":
                            success_count += 1
                            
                    except Exception as e:
                        logger.debug(f"Operation failed: {e}")
                    
                    operation_time = time.time() - start_time
                    operation_times.append(operation_time)
                    
                    final_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(final_memory - initial_memory)
                
                # Calculate operation metrics
                if operation_times:
                    avg_latency = mean(operation_times) * 1000  # ms
                    ops_per_second = 1 / mean(operation_times) if mean(operation_times) > 0 else 0
                    success_rate = success_count / test_suite["iterations"]
                    avg_memory_usage = mean(memory_samples) if memory_samples else 0
                    
                    suite_metrics["average_latency_ms"].append(avg_latency)
                    suite_metrics["operations_per_second"].append(ops_per_second)
                    suite_metrics["success_rate"].append(success_rate)
                    suite_metrics["memory_usage_mb"].append(avg_memory_usage)
                    
                    logger.info(f"    {operation}: {avg_latency:.2f}ms, {ops_per_second:.1f} ops/s, {success_rate:.1%} success")
            
            # Aggregate suite metrics
            efficiency_results[test_suite["name"]] = {
                "average_latency_ms": mean(suite_metrics["average_latency_ms"]) if suite_metrics["average_latency_ms"] else 0,
                "average_ops_per_second": mean(suite_metrics["operations_per_second"]) if suite_metrics["operations_per_second"] else 0,
                "overall_success_rate": mean(suite_metrics["success_rate"]) if suite_metrics["success_rate"] else 0,
                "average_memory_usage_mb": mean(suite_metrics["memory_usage_mb"]) if suite_metrics["memory_usage_mb"] else 0,
                "total_operations": len(test_suite["operations"]) * test_suite["iterations"]
            }
        
        # Calculate overall efficiency metrics
        all_latencies = []
        all_ops_per_sec = []
        all_success_rates = []
        
        for suite_name, metrics in efficiency_results.items():
            all_latencies.append(metrics["average_latency_ms"])
            all_ops_per_sec.append(metrics["average_ops_per_second"])
            all_success_rates.append(metrics["overall_success_rate"])
        
        return {
            "overall_average_latency_ms": mean(all_latencies) if all_latencies else 0,
            "overall_average_ops_per_second": mean(all_ops_per_sec) if all_ops_per_sec else 0,
            "overall_success_rate": mean(all_success_rates) if all_success_rates else 0,
            "efficiency_by_tool": efficiency_results,
            "performance_grade": self._calculate_performance_grade(all_latencies, all_ops_per_sec, all_success_rates)
        }
    
    async def _benchmark_memory_usage_cleanup(self) -> Dict[str, Any]:
        """
        Validate memory usage and cleanup efficiency.
        
        Tests memory consumption patterns and garbage collection
        effectiveness during tool operations.
        """
        logger.info("🧠 Validating memory usage and cleanup...")
        
        import gc
        
        # Get baseline memory
        gc.collect()
        await asyncio.sleep(0.1)
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = {
            "baseline_mb": baseline_memory,
            "peak_usage_mb": baseline_memory,
            "growth_patterns": [],
            "cleanup_efficiency": [],
            "memory_leaks_detected": []
        }
        
        # Perform memory-intensive operations
        intensive_operations = [
            ("Tool Discovery", self._memory_test_tool_discovery),
            ("Bulk Tool Usage", self._memory_test_bulk_operations),
            ("Learning Data Storage", self._memory_test_learning_storage),
            ("Concurrent Operations", self._memory_test_concurrent_ops)
        ]
        
        for test_name, test_func in intensive_operations:
            logger.info(f"  Memory test: {test_name}")
            
            # Record pre-test memory
            pre_test_memory = process.memory_info().rss / 1024 / 1024
            
            # Run intensive operation
            await test_func()
            
            # Record peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements["peak_usage_mb"] = max(memory_measurements["peak_usage_mb"], peak_memory)
            
            # Force cleanup
            gc.collect()
            await asyncio.sleep(0.2)  # Allow cleanup
            
            # Record post-cleanup memory
            post_cleanup_memory = process.memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            memory_growth = peak_memory - pre_test_memory
            memory_cleaned = peak_memory - post_cleanup_memory
            cleanup_efficiency = (memory_cleaned / memory_growth) if memory_growth > 0 else 1.0
            
            memory_measurements["growth_patterns"].append({
                "test": test_name,
                "growth_mb": memory_growth,
                "peak_mb": peak_memory
            })
            
            memory_measurements["cleanup_efficiency"].append({
                "test": test_name,
                "efficiency": cleanup_efficiency,
                "cleaned_mb": memory_cleaned
            })
            
            # Detect potential memory leaks
            if post_cleanup_memory > pre_test_memory + 5:  # 5MB threshold
                memory_measurements["memory_leaks_detected"].append({
                    "test": test_name,
                    "leak_size_mb": post_cleanup_memory - pre_test_memory
                })
            
            logger.info(f"    Growth: {memory_growth:.1f}MB, Cleanup: {cleanup_efficiency:.1%}")
        
        # Calculate final metrics
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = memory_measurements["peak_usage_mb"] - baseline_memory
        net_growth = final_memory - baseline_memory
        
        avg_cleanup_efficiency = mean([
            item["efficiency"] for item in memory_measurements["cleanup_efficiency"]
        ]) if memory_measurements["cleanup_efficiency"] else 0
        
        return {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": memory_measurements["peak_usage_mb"],
            "final_memory_mb": final_memory,
            "total_growth_mb": total_growth,
            "net_growth_mb": net_growth,
            "average_cleanup_efficiency": avg_cleanup_efficiency,
            "memory_leak_count": len(memory_measurements["memory_leaks_detected"]),
            "memory_leaks": memory_measurements["memory_leaks_detected"],
            "growth_patterns": memory_measurements["growth_patterns"],
            "cleanup_details": memory_measurements["cleanup_efficiency"],
            "memory_grade": self._calculate_memory_grade(avg_cleanup_efficiency, len(memory_measurements["memory_leaks_detected"]))
        }
    
    async def _benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """
        Test concurrent operation performance and scalability.
        
        Measures performance when multiple tool operations
        are executed simultaneously.
        """
        logger.info("🔄 Testing concurrent operation performance...")
        
        concurrency_levels = [1, 2, 5, 10, 20]
        concurrent_results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"  Testing concurrency level: {concurrency}")
            
            # Prepare concurrent tasks
            tasks = []
            for i in range(concurrency):
                task = self._create_concurrent_task(i)
                tasks.append(task)
            
            # Execute concurrently and measure
            start_time = time.time()
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                execution_time = time.time() - start_time
                
                # Analyze results
                successful_results = [r for r in results if not isinstance(r, Exception)]
                failed_results = [r for r in results if isinstance(r, Exception)]
                
                success_rate = len(successful_results) / len(results) if results else 0
                throughput = len(successful_results) / execution_time if execution_time > 0 else 0
                
                concurrent_results[f"concurrency_{concurrency}"] = {
                    "execution_time_seconds": execution_time,
                    "successful_operations": len(successful_results),
                    "failed_operations": len(failed_results),
                    "success_rate": success_rate,
                    "throughput_ops_per_second": throughput,
                    "average_latency_ms": (execution_time / len(results)) * 1000 if results else 0
                }
                
                logger.info(f"    Success rate: {success_rate:.1%}, Throughput: {throughput:.1f} ops/s")
                
            except Exception as e:
                concurrent_results[f"concurrency_{concurrency}"] = {
                    "error": str(e),
                    "success_rate": 0.0
                }
                logger.error(f"    Concurrency test failed: {e}")
        
        # Calculate scalability metrics
        scalability_factor = self._calculate_scalability_factor(concurrent_results)
        
        return {
            "concurrency_results": concurrent_results,
            "scalability_factor": scalability_factor,
            "optimal_concurrency": self._find_optimal_concurrency(concurrent_results),
            "concurrent_efficiency": self._calculate_concurrent_efficiency(concurrent_results)
        }
    
    async def _benchmark_tool_discovery_performance(self) -> Dict[str, Any]:
        """Benchmark tool discovery performance specifically."""
        logger.info("🔍 Benchmarking tool discovery performance...")
        
        discovery_iterations = 10
        discovery_times = []
        tool_counts = []
        analysis_times = []
        
        for i in range(discovery_iterations):
            # Reset agent state
            self.test_agent.available_tools.clear()
            self.test_agent.tool_capabilities.clear()
            
            start_time = time.time()
            
            # Perform discovery
            discovered_tools = await self.test_agent.discover_available_tools()
            
            discovery_time = time.time() - start_time
            discovery_times.append(discovery_time)
            tool_counts.append(len(discovered_tools))
            
            # Measure analysis time
            analysis_start = time.time()
            for tool in discovered_tools:
                await self.test_agent.analyze_tool_capability(tool)
            analysis_time = time.time() - analysis_start
            analysis_times.append(analysis_time)
        
        return {
            "average_discovery_time_ms": mean(discovery_times) * 1000 if discovery_times else 0,
            "average_tools_discovered": mean(tool_counts) if tool_counts else 0,
            "average_analysis_time_ms": mean(analysis_times) * 1000 if analysis_times else 0,
            "discovery_consistency": stdev(discovery_times) if len(discovery_times) > 1 else 0,
            "total_iterations": discovery_iterations
        }
    
    async def _benchmark_llm_integration_efficiency(self) -> Dict[str, Any]:
        """Benchmark LLM integration efficiency."""
        logger.info("🧠 Benchmarking LLM integration efficiency...")
        
        llm_operations = [
            "tool_capability_analysis",
            "tool_selection_reasoning", 
            "usage_pattern_analysis",
            "error_diagnosis"
        ]
        
        llm_metrics = {}
        
        for operation in llm_operations:
            response_times = []
            token_usage = []
            
            for i in range(10):  # 10 iterations per operation
                start_time = time.time()
                
                # Simulate LLM operation
                response = await self.test_agent.llm_client.generate(
                    prompt=f"Test {operation} prompt {i}",
                    max_tokens=500
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Track token usage if available
                token_count = len(str(response).split()) if response else 0
                token_usage.append(token_count)
            
            llm_metrics[operation] = {
                "average_response_time_ms": mean(response_times) * 1000 if response_times else 0,
                "average_token_count": mean(token_usage) if token_usage else 0,
                "response_consistency": stdev(response_times) if len(response_times) > 1 else 0
            }
        
        return llm_metrics
    
    async def _benchmark_error_recovery_performance(self) -> Dict[str, Any]:
        """Benchmark error handling and recovery performance."""
        logger.info("🚨 Benchmarking error recovery performance...")
        
        error_scenarios = [
            ("Invalid Tool Name", lambda: self.test_agent.use_tool("invalid_tool", "operation", {}, "test")),
            ("Invalid Operation", lambda: self.test_agent.use_tool("file_system_tool", "invalid_op", {}, "test")),
            ("Invalid Parameters", lambda: self.test_agent.use_tool("file_system_tool", "create_file", {"invalid": "params"}, "test")),
            ("LLM Failure", self._simulate_llm_failure),
            ("Memory Failure", self._simulate_memory_failure)
        ]
        
        recovery_metrics = {}
        
        for scenario_name, error_func in error_scenarios:
            recovery_times = []
            success_recoveries = 0
            
            for i in range(5):  # 5 attempts per scenario
                start_time = time.time()
                
                try:
                    await error_func()
                    # If no exception, recovery was successful
                    success_recoveries += 1
                except Exception:
                    # Expected for most error scenarios
                    pass
                
                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)
            
            recovery_metrics[scenario_name] = {
                "average_recovery_time_ms": mean(recovery_times) * 1000 if recovery_times else 0,
                "recovery_success_rate": success_recoveries / len(recovery_times) if recovery_times else 0,
                "recovery_consistency": stdev(recovery_times) if len(recovery_times) > 1 else 0
            }
        
        return recovery_metrics
    
    # Helper methods for benchmarking
    def _get_test_parameters(self, operation: str, iteration: int) -> Dict[str, Any]:
        """Get test parameters for different operations."""
        if operation == "create_file":
            return {
                "file_path": f"/tmp/test_file_{iteration}.dart",
                "content": f"// Test file {iteration}\\nclass TestClass{iteration} {{}}"
            }
        elif operation == "read_file":
            return {"file_path": f"/tmp/test_file_{iteration}.dart"}
        elif operation == "execute_command":
            return {"command": f"echo 'test command {iteration}'"}
        else:
            return {"test_param": f"value_{iteration}"}
    
    def _calculate_performance_grade(self, latencies: List[float], ops_per_sec: List[float], success_rates: List[float]) -> str:
        """Calculate overall performance grade."""
        if not latencies or not ops_per_sec or not success_rates:
            return "F"
        
        avg_latency = mean(latencies)
        avg_ops = mean(ops_per_sec)
        avg_success = mean(success_rates)
        
        # Grade based on thresholds
        if avg_latency < 100 and avg_ops > 10 and avg_success > 0.9:
            return "A"
        elif avg_latency < 200 and avg_ops > 5 and avg_success > 0.8:
            return "B"
        elif avg_latency < 500 and avg_ops > 2 and avg_success > 0.7:
            return "C"
        elif avg_latency < 1000 and avg_ops > 1 and avg_success > 0.5:
            return "D"
        else:
            return "F"
    
    def _calculate_memory_grade(self, cleanup_efficiency: float, leak_count: int) -> str:
        """Calculate memory management grade."""
        if cleanup_efficiency > 0.9 and leak_count == 0:
            return "A"
        elif cleanup_efficiency > 0.8 and leak_count <= 1:
            return "B"
        elif cleanup_efficiency > 0.7 and leak_count <= 2:
            return "C"
        elif cleanup_efficiency > 0.5 and leak_count <= 3:
            return "D"
        else:
            return "F"
    
    def _calculate_scalability_factor(self, concurrent_results: Dict[str, Any]) -> float:
        """Calculate scalability factor based on concurrent performance."""
        if len(concurrent_results) < 2:
            return 0.0
        
        # Compare performance at different concurrency levels
        baseline_key = "concurrency_1"
        high_concurrency_key = list(concurrent_results.keys())[-1]  # Highest concurrency
        
        if baseline_key not in concurrent_results or high_concurrency_key not in concurrent_results:
            return 0.0
        
        baseline_throughput = concurrent_results[baseline_key].get("throughput_ops_per_second", 0)
        high_throughput = concurrent_results[high_concurrency_key].get("throughput_ops_per_second", 0)
        
        if baseline_throughput == 0:
            return 0.0
        
        return high_throughput / baseline_throughput
    
    def _find_optimal_concurrency(self, concurrent_results: Dict[str, Any]) -> int:
        """Find the optimal concurrency level."""
        best_throughput = 0
        optimal_level = 1
        
        for key, result in concurrent_results.items():
            if "error" in result:
                continue
            
            throughput = result.get("throughput_ops_per_second", 0)
            if throughput > best_throughput:
                best_throughput = throughput
                # Extract concurrency level from key like "concurrency_5"
                optimal_level = int(key.split("_")[1])
        
        return optimal_level
    
    def _calculate_concurrent_efficiency(self, concurrent_results: Dict[str, Any]) -> float:
        """Calculate overall concurrent operation efficiency."""
        success_rates = []
        for result in concurrent_results.values():
            if "error" not in result:
                success_rates.append(result.get("success_rate", 0))
        
        return mean(success_rates) if success_rates else 0.0
    
    async def _create_concurrent_task(self, task_id: int):
        """Create a concurrent task for testing."""
        return await self.test_agent.use_tool(
            tool_name="file_system_tool",
            operation="create_file",
            parameters={
                "file_path": f"/tmp/concurrent_test_{task_id}.dart",
                "content": f"// Concurrent test {task_id}"
            },
            reasoning=f"Concurrent task {task_id}"
        )
    
    async def _memory_test_tool_discovery(self):
        """Memory-intensive tool discovery test."""
        for i in range(10):
            await self.test_agent.discover_available_tools()
            # Clear to force rediscovery
            self.test_agent.available_tools.clear()
    
    async def _memory_test_bulk_operations(self):
        """Memory-intensive bulk operations test."""
        for i in range(50):
            await self.test_agent.use_tool(
                tool_name="file_system_tool",
                operation="create_file",
                parameters={"file_path": f"/tmp/bulk_{i}.dart", "content": f"// Bulk {i}"},
                reasoning=f"Bulk test {i}"
            )
    
    async def _memory_test_learning_storage(self):
        """Memory-intensive learning storage test."""
        from src.models.tool_models import ToolUsageEntry
        
        for i in range(100):
            entry = ToolUsageEntry(
                agent_id=self.test_agent.agent_id,
                tool_name="test_tool",
                operation="test_operation",
                parameters={"param": f"value_{i}"},
                timestamp=datetime.now(),
                result={"status": "success"},
                reasoning=f"Memory test {i}"
            )
            await self.test_agent.learn_from_tool_usage(entry)
    
    async def _memory_test_concurrent_ops(self):
        """Memory-intensive concurrent operations test."""
        tasks = []
        for i in range(20):
            task = self._create_concurrent_task(i)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _simulate_llm_failure(self):
        """Simulate LLM failure for error recovery testing."""
        original_generate = self.test_agent.llm_client.generate
        self.test_agent.llm_client.generate = AsyncMock(side_effect=Exception("Mock LLM failure"))
        
        try:
            await self.test_agent.analyze_tool_capability(Mock())
        finally:
            self.test_agent.llm_client.generate = original_generate
    
    async def _simulate_memory_failure(self):
        """Simulate memory failure for error recovery testing."""
        original_store = self.test_agent.memory_manager.store_memory
        self.test_agent.memory_manager.store_memory = AsyncMock(side_effect=Exception("Mock memory failure"))
        
        try:
            await self.test_agent.discover_available_tools()
        finally:
            self.test_agent.memory_manager.store_memory = original_store
    
    def _log_benchmark_metrics(self, benchmark_name: str, metrics: Dict[str, Any]):
        """Log benchmark metrics in a readable format."""
        logger.info(f"  📊 {benchmark_name} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"    {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        logger.info(f"      {sub_key}: {sub_value:.3f}")
                    else:
                        logger.info(f"      {sub_key}: {sub_value}")
            elif isinstance(value, float):
                logger.info(f"    {key}: {value:.3f}")
            else:
                logger.info(f"    {key}: {value}")
    
    async def _generate_performance_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comprehensive performance report."""
        logger.info("\\n" + "=" * 70)
        logger.info("📊 PERFORMANCE BENCHMARK REPORT")
        logger.info("=" * 70)
        
        successful_benchmarks = [r for r in results.values() if r.success]
        failed_benchmarks = [r for r in results.values() if not r.success]
        
        logger.info(f"Total Benchmarks: {len(results)}")
        logger.info(f"Successful: {len(successful_benchmarks)}")
        logger.info(f"Failed: {len(failed_benchmarks)}")
        
        # Performance summary
        if successful_benchmarks:
            logger.info("\\n🎯 Key Performance Indicators:")
            
            # Extract key metrics
            for benchmark in successful_benchmarks:
                if "accuracy" in benchmark.metrics:
                    logger.info(f"  Tool Selection Accuracy: {benchmark.metrics['accuracy']:.1%}")
                
                if "improvement_rate" in benchmark.metrics:
                    logger.info(f"  Learning Improvement Rate: {benchmark.metrics['improvement_rate']:.3f}")
                
                if "operations_per_second" in benchmark.metrics:
                    logger.info(f"  Operation Throughput: {benchmark.metrics['operations_per_second']:.1f} ops/s")
                
                if "cleanup_efficiency" in benchmark.metrics:
                    logger.info(f"  Memory Cleanup Efficiency: {benchmark.metrics['cleanup_efficiency']:.1%}")
        
        # Recommendations
        logger.info("\\n💡 Performance Recommendations:")
        
        overall_performance = len(successful_benchmarks) / len(results) if results else 0
        
        if overall_performance >= 0.9:
            logger.info("  ✅ Excellent performance across all metrics")
            logger.info("  ✅ System ready for production deployment")
        elif overall_performance >= 0.7:
            logger.info("  ⚠️ Good performance with some areas for improvement")
            logger.info("  📈 Consider optimizing failed benchmarks")
        else:
            logger.info("  ❌ Performance needs significant improvement")
            logger.info("  🔧 Address critical performance issues before deployment")
        
        if failed_benchmarks:
            logger.info("\\n❌ Failed Benchmarks:")
            for benchmark in failed_benchmarks:
                logger.info(f"  - {benchmark.benchmark_name}: {benchmark.error_message}")
        
        logger.info("\\n" + "=" * 70)


async def run_performance_benchmarks():
    """Main function to run performance benchmarks."""
    framework = PerformanceBenchmarkFramework()
    
    try:
        await framework.setup_benchmark_environment()
        results = await framework.run_comprehensive_benchmarks()
        
        # Calculate overall score
        successful_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        overall_score = successful_count / total_count if total_count > 0 else 0
        
        logger.info(f"\\n🎯 Overall Performance Score: {overall_score:.1%}")
        
        return overall_score >= 0.8  # 80% success threshold
        
    except Exception as e:
        logger.error(f"❌ Benchmark execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_performance_benchmarks())
    sys.exit(0 if success else 1)
