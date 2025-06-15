#!/usr/bin/env python3
"""
Comprehensive Test Runner for Enhanced OrchestratorAgent.

This script runs the complete test suite including:
1. Integration tests
2. Performance benchmarks  
3. Validation scenarios
4. Stress tests
5. Edge case testing

Usage:
    python run_complete_test_suite.py [--quick] [--performance] [--validation] [--all]
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_suite.log')
    ]
)
logger = logging.getLogger("test_suite")


class TestSuiteRunner:
    """Comprehensive test suite runner for OrchestratorAgent."""
    
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.summary = {}
        
    async def run_complete_suite(self, 
                                include_performance: bool = True,
                                include_validation: bool = True,
                                include_integration: bool = True,
                                quick_mode: bool = False) -> Dict[str, Any]:
        """Run the complete test suite."""
        
        logger.info("🚀 Starting Complete OrchestratorAgent Test Suite")
        logger.info("=" * 70)
        
        self.start_time = time.time()
        
        # Test categories to run
        test_categories = []
        
        if include_integration:
            test_categories.append("integration")
        if include_performance and not quick_mode:
            test_categories.append("performance")
        if include_validation and not quick_mode:
            test_categories.append("validation")
        
        # Always include basic functionality tests
        test_categories.append("basic_functionality")
        
        logger.info(f"Test categories to run: {', '.join(test_categories)}")
        
        # Run test categories
        for category in test_categories:
            logger.info(f"\n🔬 Running {category.upper()} tests...")
            result = await self._run_test_category(category, quick_mode)
            self.results[category] = result
        
        # Generate comprehensive summary
        self.summary = self._generate_comprehensive_summary()
        
        # Display results
        await self._display_results()
        
        # Save results
        await self._save_results()
        
        return {
            "summary": self.summary,
            "results": self.results,
            "overall_success": self.summary["overall_success"]
        }
    
    async def _run_test_category(self, category: str, quick_mode: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category."""
        
        category_start = time.time()
        
        try:
            if category == "integration":
                result = await self._run_integration_tests()
            elif category == "performance":
                result = await self._run_performance_tests(quick_mode)
            elif category == "validation":
                result = await self._run_validation_tests(quick_mode)
            elif category == "basic_functionality":
                result = await self._run_basic_functionality_tests()
            else:
                result = {"success": False, "error": f"Unknown category: {category}"}
        
        except Exception as e:
            logger.error(f"Error running {category} tests: {e}")
            result = {"success": False, "error": str(e)}
        
        result["duration"] = time.time() - category_start
        result["category"] = category
        
        return result
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests using pytest."""
        logger.info("🧪 Running integration tests with pytest...")
        
        try:
            # Run pytest on the integration test file
            cmd = [
                sys.executable, "-m", "pytest", 
                "tests/agents/test_orchestrator_agent_enhanced.py",
                "-v", "--tb=short", "--json-report", "--json-report-file=integration_results.json"
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            if Path("integration_results.json").exists():
                with open("integration_results.json", 'r') as f:
                    pytest_results = json.load(f)
                
                return {
                    "success": process.returncode == 0,
                    "tests_run": pytest_results.get("summary", {}).get("total", 0),
                    "tests_passed": pytest_results.get("summary", {}).get("passed", 0),
                    "tests_failed": pytest_results.get("summary", {}).get("failed", 0),
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "details": pytest_results
                }
            else:
                return {
                    "success": process.returncode == 0,
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Integration tests timed out"}
        except Exception as e:
            return {"success": False, "error": f"Integration test error: {e}"}
    
    async def _run_performance_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance benchmarks...")
        
        try:
            # Import and run performance benchmarks
            sys.path.insert(0, '.')
            from performance_benchmarks import PerformanceBenchmarker
            
            benchmarker = PerformanceBenchmarker()
            await benchmarker.setup()
            
            if quick_mode:
                # Run subset of benchmarks for quick mode
                result = await self._run_quick_performance_tests(benchmarker)
            else:
                # Run complete benchmark suite
                benchmark_suite = await benchmarker.run_all_benchmarks()
                result = {
                    "success": benchmark_suite.summary["performance_grade"] in ["A+", "A", "B+", "B"],
                    "performance_grade": benchmark_suite.summary["performance_grade"],
                    "overall_success_rate": benchmark_suite.summary["overall_success_rate"],
                    "average_throughput": benchmark_suite.summary["average_throughput"],
                    "benchmark_count": len(benchmark_suite.results),
                    "recommendations": benchmark_suite.summary["recommendations"]
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Performance test error: {e}")
            return {"success": False, "error": f"Performance test error: {e}"}
    
    async def _run_quick_performance_tests(self, benchmarker) -> Dict[str, Any]:
        """Run quick subset of performance tests."""
        
        # Run only essential performance tests
        quick_results = []
        
        # Test workflow optimization
        result = await benchmarker.benchmark_workflow_optimization()
        quick_results.append(result)
        
        # Test tool coordination
        result = await benchmarker.benchmark_tool_allocation()
        quick_results.append(result)
        
        # Test session management
        result = await benchmarker.benchmark_session_creation()
        quick_results.append(result)
        
        # Calculate quick summary
        avg_success_rate = sum(r.success_rate for r in quick_results) / len(quick_results)
        avg_throughput = sum(r.throughput for r in quick_results) / len(quick_results)
        
        grade = "A" if avg_success_rate > 0.9 and avg_throughput > 1.0 else "B"
        
        return {
            "success": avg_success_rate > 0.8,
            "performance_grade": grade,
            "overall_success_rate": avg_success_rate,
            "average_throughput": avg_throughput,
            "benchmark_count": len(quick_results),
            "mode": "quick"
        }
    
    async def _run_validation_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run validation scenarios."""
        logger.info("✅ Running validation scenarios...")
        
        try:
            # Import and run validation scenarios
            sys.path.insert(0, '.')
            from validation_scenarios import (
                ComplexFlutterProjectScenario,
                MultiAgentCollaborationScenario,
                WorkflowAdaptationScenario
            )
            
            scenarios = [
                ComplexFlutterProjectScenario(),
                MultiAgentCollaborationScenario(),
                WorkflowAdaptationScenario()
            ]
            
            if quick_mode:
                # Run only the first scenario for quick mode
                scenarios = scenarios[:1]
            
            scenario_results = []
            
            for scenario in scenarios:
                logger.info(f"Running scenario: {scenario.name}")
                result = await scenario.execute()
                scenario_results.append(result)
                await scenario.cleanup()
            
            # Calculate validation summary
            passed_scenarios = sum(1 for r in scenario_results if r["success"])
            total_scenarios = len(scenario_results)
            
            return {
                "success": passed_scenarios == total_scenarios,
                "scenarios_passed": passed_scenarios,
                "total_scenarios": total_scenarios,
                "success_rate": passed_scenarios / total_scenarios if total_scenarios > 0 else 0,
                "scenario_results": scenario_results
            }
        
        except Exception as e:
            logger.error(f"Validation test error: {e}")
            return {"success": False, "error": f"Validation test error: {e}"}
    
    async def _run_basic_functionality_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests."""
        logger.info("🔧 Running basic functionality tests...")
        
        try:
            # Import required modules to test basic functionality
            sys.path.insert(0, 'src')
            
            from src.agents.orchestrator_agent import OrchestratorAgent
            from src.agents.base_agent import AgentConfig, AgentCapability
            from src.models.project_models import ProjectContext, ProjectType
            from src.models.task_models import TaskContext, TaskPriority
            
            # Test basic imports
            logger.info("✓ Basic imports successful")
            
            # Test orchestrator creation
            config = AgentConfig(
                agent_id="test-basic",
                agent_type="orchestrator",
                capabilities=[AgentCapability.ORCHESTRATION]
            )
            
            # Create mock dependencies
            from unittest.mock import AsyncMock
            
            orchestrator = OrchestratorAgent(
                config=config,
                llm_client=AsyncMock(),
                memory_manager=AsyncMock(),
                event_bus=AsyncMock()
            )
            
            logger.info("✓ Orchestrator creation successful")
            
            # Test basic operations
            project_context = ProjectContext(
                project_name="basic_test",
                project_type=ProjectType.MOBILE_APP,
                description="Basic functionality test"
            )
            
            # Test session creation
            session = await orchestrator.create_development_session(
                project_context=project_context,
                session_type="basic_test"
            )
            
            logger.info("✓ Session creation successful")
            
            # Test task decomposition
            task_context = TaskContext(
                task_id="basic_task",
                description="Basic test task",
                priority=TaskPriority.MEDIUM
            )
            
            # Mock the LLM response for decomposition
            orchestrator.llm_client.generate.return_value = {
                "content": json.dumps({
                    "workflow": {
                        "id": "basic_workflow",
                        "steps": [{"id": "step1", "name": "test_step"}],
                        "execution_strategy": "sequential"
                    }
                })
            }
            
            workflow = await orchestrator.decompose_task(task_context, session.session_id)
            
            logger.info("✓ Task decomposition successful")
            
            return {
                "success": True,
                "tests_passed": 4,
                "operations_tested": [
                    "imports",
                    "orchestrator_creation", 
                    "session_creation",
                    "task_decomposition"
                ]
            }
        
        except Exception as e:
            logger.error(f"Basic functionality test error: {e}")
            return {"success": False, "error": f"Basic functionality error: {e}"}
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test suite summary."""
        
        total_duration = time.time() - self.start_time
        
        # Count successes and failures
        successful_categories = sum(1 for result in self.results.values() if result.get("success", False))
        total_categories = len(self.results)
        
        # Calculate overall metrics
        overall_success = successful_categories == total_categories
        success_rate = successful_categories / total_categories if total_categories > 0 else 0
        
        # Collect performance data
        performance_data = {}
        if "performance" in self.results:
            perf_result = self.results["performance"]
            performance_data = {
                "grade": perf_result.get("performance_grade", "N/A"),
                "success_rate": perf_result.get("overall_success_rate", 0),
                "throughput": perf_result.get("average_throughput", 0)
            }
        
        # Collect validation data
        validation_data = {}
        if "validation" in self.results:
            val_result = self.results["validation"]
            validation_data = {
                "scenarios_passed": val_result.get("scenarios_passed", 0),
                "total_scenarios": val_result.get("total_scenarios", 0),
                "success_rate": val_result.get("success_rate", 0)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "categories_tested": total_categories,
            "categories_passed": successful_categories,
            "overall_success": overall_success,
            "success_rate": success_rate,
            "performance": performance_data,
            "validation": validation_data,
            "recommendations": recommendations,
            "test_environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "test_runner_version": "1.0.0"
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check integration tests
        if "integration" in self.results:
            int_result = self.results["integration"]
            if not int_result.get("success", False):
                recommendations.append("Review integration test failures and fix underlying issues")
        
        # Check performance tests
        if "performance" in self.results:
            perf_result = self.results["performance"]
            grade = perf_result.get("performance_grade", "")
            if grade in ["C", "D"]:
                recommendations.append("Performance optimization needed - focus on bottlenecks")
            elif "recommendations" in perf_result:
                recommendations.extend(perf_result["recommendations"])
        
        # Check validation tests
        if "validation" in self.results:
            val_result = self.results["validation"]
            if val_result.get("success_rate", 0) < 1.0:
                recommendations.append("Address validation scenario failures")
        
        # General recommendations
        if self.summary.get("success_rate", 0) < 1.0:
            recommendations.append("Review failed test categories and address root causes")
        
        if not recommendations:
            recommendations.append("All tests passed - system is ready for production")
        
        return recommendations
    
    async def _display_results(self):
        """Display comprehensive test results."""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 70)
        
        # Overall summary
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"Total Duration: {self.summary['total_duration']:.2f}s")
        print(f"Categories Tested: {self.summary['categories_tested']}")
        print(f"Categories Passed: {self.summary['categories_passed']}")
        print(f"Overall Success Rate: {self.summary['success_rate']:.1%}")
        
        if self.summary["overall_success"]:
            print("🎉 OVERALL STATUS: PASSED")
        else:
            print("⚠️ OVERALL STATUS: FAILED")
        
        # Category details
        print(f"\n📋 CATEGORY RESULTS")
        print("-" * 50)
        
        for category, result in self.results.items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            duration = result.get("duration", 0)
            print(f"{category.upper():20} {status:10} ({duration:.2f}s)")
            
            # Show category-specific details
            if category == "integration" and "tests_run" in result:
                print(f"                     Tests: {result['tests_passed']}/{result['tests_run']}")
            elif category == "performance" and "performance_grade" in result:
                print(f"                     Grade: {result['performance_grade']}")
            elif category == "validation" and "scenarios_passed" in result:
                print(f"                     Scenarios: {result['scenarios_passed']}/{result['total_scenarios']}")
        
        # Performance details
        if "performance" in self.summary and self.summary["performance"]:
            print(f"\n⚡ PERFORMANCE METRICS")
            print("-" * 50)
            perf = self.summary["performance"]
            print(f"Grade: {perf.get('grade', 'N/A')}")
            print(f"Success Rate: {perf.get('success_rate', 0):.1%}")
            print(f"Throughput: {perf.get('throughput', 0):.2f} ops/sec")
        
        # Validation details
        if "validation" in self.summary and self.summary["validation"]:
            print(f"\n✅ VALIDATION METRICS")
            print("-" * 50)
            val = self.summary["validation"]
            print(f"Scenarios Passed: {val.get('scenarios_passed', 0)}/{val.get('total_scenarios', 0)}")
            print(f"Success Rate: {val.get('success_rate', 0):.1%}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(self.summary["recommendations"], 1):
            print(f"{i}. {rec}")
    
    async def _save_results(self):
        """Save test results to files."""
        
        # Save comprehensive results
        results_file = Path("comprehensive_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "summary": self.summary,
                "results": self.results
            }, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")
        
        # Save summary report
        report_file = Path("test_summary_report.md")
        with open(report_file, 'w') as f:
            f.write("# OrchestratorAgent Test Suite Report\n\n")
            f.write(f"**Generated:** {self.summary['timestamp']}\n\n")
            f.write(f"**Overall Result:** {'✅ PASSED' if self.summary['overall_success'] else '❌ FAILED'}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Duration:** {self.summary['total_duration']:.2f}s\n")
            f.write(f"- **Success Rate:** {self.summary['success_rate']:.1%}\n")
            f.write(f"- **Categories Tested:** {self.summary['categories_tested']}\n")
            f.write(f"- **Categories Passed:** {self.summary['categories_passed']}\n\n")
            
            f.write("## Category Results\n\n")
            for category, result in self.results.items():
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                f.write(f"- **{category.upper()}:** {status}\n")
            
            f.write("\n## Recommendations\n\n")
            for i, rec in enumerate(self.summary["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Summary report saved to {report_file}")


async def main():
    """Main test runner entry point."""
    
    parser = argparse.ArgumentParser(description="Run comprehensive OrchestratorAgent test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test mode")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--validation", action="store_true", help="Run only validation tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if not any([args.performance, args.validation, args.integration]):
        # Default: run all tests
        include_performance = True
        include_validation = True
        include_integration = True
    else:
        include_performance = args.performance
        include_validation = args.validation
        include_integration = args.integration
    
    # Create and run test suite
    runner = TestSuiteRunner()
    
    results = await runner.run_complete_suite(
        include_performance=include_performance,
        include_validation=include_validation,
        include_integration=include_integration,
        quick_mode=args.quick
    )
    
    # Return appropriate exit code
    if results["overall_success"]:
        print("\n🎉 All tests PASSED - OrchestratorAgent is ready!")
        return 0
    else:
        print("\n⚠️ Some tests FAILED - review results and fix issues")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
