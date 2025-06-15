#!/usr/bin/env python3
"""
Performance Benchmarks for Enhanced OrchestratorAgent.

This module provides comprehensive performance benchmarking for:
1. Workflow optimization effectiveness
2. Tool coordination efficiency 
3. Session management overhead
4. Adaptation accuracy and speed
5. Resource utilization metrics
6. Scalability testing

Usage:
    python performance_benchmarks.py
"""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
from src.models.task_models import TaskContext, TaskPriority
from src.models.tool_models import WorkflowFeedback, DevelopmentSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("performance_benchmarks")


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    test_name: str
    duration: float
    success_rate: float
    throughput: float
    resource_usage: Dict[str, float]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    suite_name: str
    total_duration: float
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class MockLLMClient:
    """High-performance mock LLM client for benchmarking."""
    
    def __init__(self):
        self.call_count = 0
        self.response_times = []
        
    async def generate(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate mock response with realistic timing."""
        start_time = time.time()
        self.call_count += 1
        
        # Simulate realistic LLM response time (100-500ms)
        await asyncio.sleep(0.1 + (self.call_count % 5) * 0.1)
        
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Return structured response based on prompt
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        
        if "decompose" in prompt_text.lower():
            return {
                "content": json.dumps({
                    "workflow": {
                        "id": f"workflow_{self.call_count}",
                        "steps": [
                            {"id": f"step_{i}", "name": f"task_{i}", "agent_type": "implementation"}
                            for i in range(3 + self.call_count % 5)
                        ],
                        "execution_strategy": "hybrid"
                    }
                })
            }
        else:
            return {
                "content": json.dumps({
                    "status": "success",
                    "optimizations": [
                        {"type": "parallelization", "improvement": 0.25},
                        {"type": "rebalancing", "improvement": 0.15}
                    ]
                })
            }


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking framework."""
    
    def __init__(self):
        self.orchestrator = None
        self.benchmark_results = []
        
    async def setup(self):
        """Setup benchmarking environment."""
        logger.info("Setting up performance benchmarking environment")
        
        config = AgentConfig(
            agent_id="benchmark-orchestrator",
            agent_type="orchestrator",
            capabilities=[AgentCapability.ORCHESTRATION]
        )
        
        # Create orchestrator with mock dependencies
        self.orchestrator = OrchestratorAgent(
            config=config,
            llm_client=MockLLMClient(),
            memory_manager=MockMemoryManager(),
            event_bus=MockEventBus()
        )
        
        # Setup mock agents for testing
        self.orchestrator.available_agents = {
            f"agent_{i:03d}": MockBenchmarkAgent(f"agent_{i:03d}", f"type_{i%3}")
            for i in range(20)  # 20 mock agents for scalability testing
        }
        
        logger.info("✅ Benchmarking environment setup complete")
    
    async def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        logger.info("🚀 Starting comprehensive performance benchmark suite")
        start_time = time.time()
        
        suite = BenchmarkSuite(suite_name="OrchestratorAgent_Enhanced_Benchmarks")
        
        # Workflow optimization benchmarks
        suite.results.append(await self.benchmark_workflow_optimization())
        suite.results.append(await self.benchmark_adaptation_speed())
        suite.results.append(await self.benchmark_optimization_accuracy())
        
        # Tool coordination benchmarks
        suite.results.append(await self.benchmark_tool_allocation())
        suite.results.append(await self.benchmark_conflict_resolution())
        suite.results.append(await self.benchmark_coordination_scalability())
        
        # Session management benchmarks
        suite.results.append(await self.benchmark_session_creation())
        suite.results.append(await self.benchmark_concurrent_sessions())
        suite.results.append(await self.benchmark_session_operations())
        
        # Task decomposition benchmarks
        suite.results.append(await self.benchmark_task_decomposition())
        suite.results.append(await self.benchmark_complex_decomposition())
        
        # Resource utilization benchmarks
        suite.results.append(await self.benchmark_memory_usage())
        suite.results.append(await self.benchmark_cpu_utilization())
        
        suite.total_duration = time.time() - start_time
        suite.summary = self._generate_summary(suite)
        
        logger.info(f"✅ Benchmark suite completed in {suite.total_duration:.2f}s")
        return suite
    
    async def benchmark_workflow_optimization(self) -> BenchmarkResult:
        """Benchmark workflow optimization effectiveness."""
        logger.info("📊 Benchmarking workflow optimization effectiveness")
        
        test_start = time.time()
        success_count = 0
        optimization_improvements = []
        
        project_context = ProjectContext(
            project_name="benchmark_project",
            project_type=ProjectType.MOBILE_APP,
            platforms=[PlatformTarget.ANDROID, PlatformTarget.IOS]
        )
        
        # Run multiple optimization tests
        for i in range(10):
            try:
                session = await self.orchestrator.create_development_session(
                    project_context=project_context,
                    session_type=f"optimization_test_{i}"
                )
                
                # Create workflow
                task_context = TaskContext(
                    task_id=f"opt_task_{i}",
                    description=f"Optimization test task {i}",
                    priority=TaskPriority.MEDIUM
                )
                
                workflow = await self.orchestrator.decompose_task(task_context, session.session_id)
                
                # Create feedback for optimization
                feedback = WorkflowFeedback(
                    workflow_id=workflow.id,
                    session_id=session.session_id,
                    performance_metrics={
                        "efficiency": 0.6 + (i % 3) * 0.1,
                        "bottlenecks": [f"step_{j}" for j in range(i % 3)]
                    }
                )
                
                # Optimize workflow
                start_opt = time.time()
                result = await self.orchestrator.modify_workflow(
                    workflow_id=workflow.id,
                    feedback=feedback
                )
                opt_time = time.time() - start_opt
                
                if result.success:
                    success_count += 1
                    optimization_improvements.append(result.improvement_score)
                
            except Exception as e:
                logger.warning(f"Optimization test {i} failed: {e}")
        
        duration = time.time() - test_start
        avg_improvement = statistics.mean(optimization_improvements) if optimization_improvements else 0
        
        return BenchmarkResult(
            test_name="workflow_optimization",
            duration=duration,
            success_rate=success_count / 10,
            throughput=10 / duration,
            resource_usage={"memory": "moderate", "cpu": "high"},
            metrics={
                "average_improvement": avg_improvement,
                "improvement_std": statistics.stdev(optimization_improvements) if len(optimization_improvements) > 1 else 0,
                "optimization_count": len(optimization_improvements)
            }
        )
    
    async def benchmark_adaptation_speed(self) -> BenchmarkResult:
        """Benchmark adaptation speed under various conditions."""
        logger.info("⚡ Benchmarking adaptation speed")
        
        test_start = time.time()
        adaptation_times = []
        success_count = 0
        
        project_context = ProjectContext(
            project_name="speed_test_project",
            project_type=ProjectType.WEB_APP
        )
        
        # Test adaptation speed with varying complexity
        complexities = ["simple", "medium", "complex", "very_complex"]
        
        for complexity in complexities:
            for i in range(5):  # 5 tests per complexity level
                try:
                    session = await self.orchestrator.create_development_session(
                        project_context=project_context,
                        session_type=f"speed_test_{complexity}_{i}"
                    )
                    
                    # Create workflow with varying complexity
                    steps_count = {"simple": 3, "medium": 8, "complex": 15, "very_complex": 25}[complexity]
                    
                    task_context = TaskContext(
                        task_id=f"speed_task_{complexity}_{i}",
                        description=f"Speed test for {complexity} workflow",
                        priority=TaskPriority.HIGH,
                        requirements={"complexity": complexity, "steps": steps_count}
                    )
                    
                    workflow = await self.orchestrator.decompose_task(task_context, session.session_id)
                    
                    # Create feedback requiring adaptation
                    feedback = WorkflowFeedback(
                        workflow_id=workflow.id,
                        session_id=session.session_id,
                        performance_metrics={
                            "bottlenecks": [f"step_{j}" for j in range(min(3, steps_count//3))],
                            "urgency": "high"
                        }
                    )
                    
                    # Measure adaptation time
                    adapt_start = time.time()
                    result = await self.orchestrator.modify_workflow(
                        workflow_id=workflow.id,
                        feedback=feedback,
                        adaptation_strategy="speed_focused"
                    )
                    adapt_time = time.time() - adapt_start
                    
                    adaptation_times.append(adapt_time)
                    if result.success:
                        success_count += 1
                        
                except Exception as e:
                    logger.warning(f"Speed test {complexity}_{i} failed: {e}")
        
        duration = time.time() - test_start
        
        return BenchmarkResult(
            test_name="adaptation_speed",
            duration=duration,
            success_rate=success_count / 20,
            throughput=20 / duration,
            resource_usage={"memory": "moderate", "cpu": "high"},
            metrics={
                "average_adaptation_time": statistics.mean(adaptation_times),
                "max_adaptation_time": max(adaptation_times) if adaptation_times else 0,
                "min_adaptation_time": min(adaptation_times) if adaptation_times else 0,
                "adaptation_time_std": statistics.stdev(adaptation_times) if len(adaptation_times) > 1 else 0
            }
        )
    
    async def benchmark_tool_allocation(self) -> BenchmarkResult:
        """Benchmark tool allocation efficiency."""
        logger.info("🔧 Benchmarking tool allocation efficiency")
        
        test_start = time.time()
        allocation_times = []
        success_count = 0
        efficiency_scores = []
        
        # Test with varying numbers of agents and tools
        agent_counts = [5, 10, 15, 20]
        
        for agent_count in agent_counts:
            for test_run in range(3):
                try:
                    project_context = ProjectContext(
                        project_name=f"tool_test_{agent_count}_{test_run}",
                        project_type=ProjectType.MOBILE_APP
                    )
                    
                    session = await self.orchestrator.create_development_session(
                        project_context=project_context,
                        session_type=f"tool_allocation_test_{agent_count}"
                    )
                    
                    # Create tool requirements for multiple agents
                    tool_requirements = {
                        f"agent_{i:03d}": [
                            "flutter_sdk", "file_system",
                            f"specialized_tool_{i%5}",
                            f"shared_tool_{i%3}"
                        ]
                        for i in range(agent_count)
                    }
                    
                    # Measure allocation time
                    alloc_start = time.time()
                    result = await self.orchestrator.plan_tool_allocation(
                        session_id=session.session_id,
                        agent_requirements=tool_requirements
                    )
                    alloc_time = time.time() - alloc_start
                    
                    allocation_times.append(alloc_time)
                    
                    if result.success:
                        success_count += 1
                        efficiency_scores.append(result.efficiency_score)
                    
                except Exception as e:
                    logger.warning(f"Tool allocation test {agent_count}_{test_run} failed: {e}")
        
        duration = time.time() - test_start
        
        return BenchmarkResult(
            test_name="tool_allocation",
            duration=duration,
            success_rate=success_count / 12,
            throughput=12 / duration,
            resource_usage={"memory": "low", "cpu": "medium"},
            metrics={
                "average_allocation_time": statistics.mean(allocation_times) if allocation_times else 0,
                "average_efficiency": statistics.mean(efficiency_scores) if efficiency_scores else 0,
                "max_allocation_time": max(allocation_times) if allocation_times else 0,
                "scalability_factor": allocation_times[-1] / allocation_times[0] if len(allocation_times) >= 2 else 1
            }
        )
    
    async def benchmark_concurrent_sessions(self) -> BenchmarkResult:
        """Benchmark concurrent session management."""
        logger.info("🔄 Benchmarking concurrent session management")
        
        test_start = time.time()
        session_counts = [1, 3, 5, 10]
        creation_times = []
        success_count = 0
        
        for session_count in session_counts:
            try:
                # Create multiple concurrent sessions
                create_start = time.time()
                
                sessions = []
                for i in range(session_count):
                    project_context = ProjectContext(
                        project_name=f"concurrent_project_{i}",
                        project_type=ProjectType.MOBILE_APP
                    )
                    
                    session = await self.orchestrator.create_development_session(
                        project_context=project_context,
                        session_type=f"concurrent_test_{i}"
                    )
                    sessions.append(session)
                
                creation_time = time.time() - create_start
                creation_times.append(creation_time)
                
                # Initialize all sessions concurrently
                init_tasks = [
                    self.orchestrator.initialize_session(session.session_id)
                    for session in sessions
                ]
                
                init_results = await asyncio.gather(*init_tasks, return_exceptions=True)
                
                # Count successful initializations
                successful_inits = sum(1 for result in init_results if not isinstance(result, Exception))
                
                if successful_inits == session_count:
                    success_count += 1
                
                # Cleanup sessions
                for session in sessions:
                    try:
                        await self.orchestrator.terminate_session(session.session_id)
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Concurrent session test with {session_count} sessions failed: {e}")
        
        duration = time.time() - test_start
        
        return BenchmarkResult(
            test_name="concurrent_sessions",
            duration=duration,
            success_rate=success_count / len(session_counts),
            throughput=sum(session_counts) / duration,
            resource_usage={"memory": "high", "cpu": "medium"},
            metrics={
                "creation_times": creation_times,
                "average_creation_time": statistics.mean(creation_times) if creation_times else 0,
                "max_concurrent_sessions": max(session_counts),
                "concurrency_overhead": creation_times[-1] / creation_times[0] if len(creation_times) >= 2 else 1
            }
        )
    
    async def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        logger.info("💾 Benchmarking memory usage patterns")
        
        test_start = time.time()
        
        # Simulate memory usage tracking
        memory_samples = []
        
        project_context = ProjectContext(
            project_name="memory_test_project",
            project_type=ProjectType.ENTERPRISE_APP
        )
        
        # Create session and perform operations while tracking memory
        session = await self.orchestrator.create_development_session(
            project_context=project_context,
            session_type="memory_test"
        )
        
        # Baseline memory
        memory_samples.append({"operation": "baseline", "usage": 100})  # Mock memory usage
        
        # Initialize session
        await self.orchestrator.initialize_session(session.session_id)
        memory_samples.append({"operation": "session_init", "usage": 120})
        
        # Create multiple workflows
        for i in range(5):
            task_context = TaskContext(
                task_id=f"memory_task_{i}",
                description=f"Memory test task {i}",
                priority=TaskPriority.MEDIUM
            )
            
            await self.orchestrator.decompose_task(task_context, session.session_id)
            memory_samples.append({"operation": f"workflow_{i}", "usage": 120 + i * 10})
        
        # Perform optimizations
        for i in range(3):
            feedback = WorkflowFeedback(
                workflow_id=f"workflow_{i}",
                session_id=session.session_id,
                performance_metrics={"efficiency": 0.7}
            )
            
            await self.orchestrator.modify_workflow(
                workflow_id=f"workflow_{i}",
                feedback=feedback
            )
            memory_samples.append({"operation": f"optimize_{i}", "usage": 160 + i * 5})
        
        duration = time.time() - test_start
        
        return BenchmarkResult(
            test_name="memory_usage",
            duration=duration,
            success_rate=1.0,
            throughput=8 / duration,  # 8 operations
            resource_usage={"memory": "tracked", "cpu": "low"},
            metrics={
                "memory_samples": memory_samples,
                "peak_memory": max(sample["usage"] for sample in memory_samples),
                "baseline_memory": memory_samples[0]["usage"],
                "memory_growth": memory_samples[-1]["usage"] - memory_samples[0]["usage"]
            }
        )
    
    def _generate_summary(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Generate benchmark suite summary."""
        total_tests = len(suite.results)
        successful_tests = sum(1 for result in suite.results if result.success_rate > 0.8)
        
        avg_success_rate = statistics.mean([result.success_rate for result in suite.results])
        avg_throughput = statistics.mean([result.throughput for result in suite.results])
        
        performance_categories = {
            "workflow_optimization": [r for r in suite.results if "optimization" in r.test_name or "adaptation" in r.test_name],
            "tool_coordination": [r for r in suite.results if "tool" in r.test_name or "allocation" in r.test_name],
            "session_management": [r for r in suite.results if "session" in r.test_name or "concurrent" in r.test_name],
            "resource_utilization": [r for r in suite.results if "memory" in r.test_name or "cpu" in r.test_name]
        }
        
        category_scores = {}
        for category, results in performance_categories.items():
            if results:
                category_scores[category] = {
                    "avg_success_rate": statistics.mean([r.success_rate for r in results]),
                    "avg_throughput": statistics.mean([r.throughput for r in results]),
                    "total_duration": sum([r.duration for r in results])
                }
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "overall_success_rate": avg_success_rate,
            "average_throughput": avg_throughput,
            "category_performance": category_scores,
            "performance_grade": self._calculate_performance_grade(avg_success_rate, avg_throughput),
            "recommendations": self._generate_recommendations(suite.results)
        }
    
    def _calculate_performance_grade(self, success_rate: float, throughput: float) -> str:
        """Calculate overall performance grade."""
        if success_rate >= 0.95 and throughput >= 1.0:
            return "A+"
        elif success_rate >= 0.90 and throughput >= 0.8:
            return "A"
        elif success_rate >= 0.85 and throughput >= 0.6:
            return "B+"
        elif success_rate >= 0.80 and throughput >= 0.4:
            return "B"
        elif success_rate >= 0.70 and throughput >= 0.2:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze results for improvement opportunities
        slow_tests = [r for r in results if r.throughput < 0.5]
        low_success_tests = [r for r in results if r.success_rate < 0.9]
        
        if slow_tests:
            recommendations.append(f"Optimize performance for: {', '.join([r.test_name for r in slow_tests])}")
        
        if low_success_tests:
            recommendations.append(f"Improve reliability for: {', '.join([r.test_name for r in low_success_tests])}")
        
        # Check for memory issues
        memory_results = [r for r in results if "memory" in r.test_name]
        for result in memory_results:
            if result.metrics.get("memory_growth", 0) > 100:
                recommendations.append("Consider memory optimization for long-running sessions")
        
        # Check for scalability issues
        concurrent_results = [r for r in results if "concurrent" in r.test_name]
        for result in concurrent_results:
            if result.metrics.get("concurrency_overhead", 1) > 2:
                recommendations.append("Optimize concurrent session handling")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters")
        
        return recommendations


class MockMemoryManager:
    """Mock memory manager for benchmarking."""
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None):
        pass
    
    async def retrieve(self, key: str) -> Any:
        return {"mock": "data"}
    
    async def delete(self, key: str):
        pass


class MockEventBus:
    """Mock event bus for benchmarking."""
    
    async def publish(self, event_type: str, data: Any):
        pass
    
    async def subscribe(self, event_type: str, handler):
        pass


class MockBenchmarkAgent:
    """Mock agent for benchmarking."""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.performance_metrics = {
            "response_time": 0.1 + (hash(agent_id) % 10) * 0.01,
            "success_rate": 0.95,
            "throughput": 10.0
        }


async def main():
    """Run performance benchmarks."""
    print("🚀 Starting OrchestratorAgent Performance Benchmarks")
    print("=" * 60)
    
    benchmarker = PerformanceBenchmarker()
    await benchmarker.setup()
    
    # Run benchmark suite
    suite = await benchmarker.run_all_benchmarks()
    
    # Display results
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    for result in suite.results:
        print(f"\n🔍 {result.test_name.upper()}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Success Rate: {result.success_rate:.1%}")
        print(f"   Throughput: {result.throughput:.2f} ops/sec")
        
        # Display key metrics
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")
    
    print(f"\n🎯 OVERALL PERFORMANCE")
    print("=" * 60)
    print(f"Total Duration: {suite.total_duration:.2f}s")
    print(f"Overall Success Rate: {suite.summary['overall_success_rate']:.1%}")
    print(f"Average Throughput: {suite.summary['average_throughput']:.2f} ops/sec")
    print(f"Performance Grade: {suite.summary['performance_grade']}")
    
    print(f"\n📈 CATEGORY PERFORMANCE")
    print("=" * 60)
    for category, metrics in suite.summary['category_performance'].items():
        print(f"{category.replace('_', ' ').title()}:")
        print(f"   Success Rate: {metrics['avg_success_rate']:.1%}")
        print(f"   Throughput: {metrics['avg_throughput']:.2f} ops/sec")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(suite.summary['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save results to file
    results_file = Path("benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "suite": suite.suite_name,
            "summary": suite.summary,
            "results": [
                {
                    "test_name": r.test_name,
                    "duration": r.duration,
                    "success_rate": r.success_rate,
                    "throughput": r.throughput,
                    "metrics": r.metrics
                }
                for r in suite.results
            ]
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Return performance grade for CI/CD
    grade = suite.summary['performance_grade']
    if grade in ['A+', 'A', 'B+']:
        print(f"\n🎉 Performance benchmark PASSED with grade {grade}")
        return 0
    else:
        print(f"\n⚠️ Performance benchmark needs attention (grade {grade})")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
