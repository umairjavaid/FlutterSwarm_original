#!/usr/bin/env python3
"""
Test runner for OrchestratorAgent comprehensive test suite.

This script runs all orchestrator tests and generates performance reports.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspaces/FlutterSwarm/python-backend/logs/orchestrator_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator_test_runner")


async def run_orchestrator_tests():
    """Run comprehensive orchestrator test suite."""
    logger.info("Starting OrchestratorAgent comprehensive test suite")
    
    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_categories": {},
        "performance_metrics": {},
        "validation_results": {},
        "overall_status": "running"
    }
    
    try:
        # Import test functions
        from test_orchestrator_agent_enhanced import (
            test_complete_orchestrator_integration,
            TestOrchestratorSessionLifecycle,
            TestAdaptiveWorkflowModification,
            TestToolCoordination,
            TestSessionManagement,
            TestEnhancedTaskDecomposition,
            TestPerformanceBenchmarks
        )
        
        # 1. Run session lifecycle tests
        logger.info("=== Running Session Lifecycle Tests ===")
        session_tests = TestOrchestratorSessionLifecycle()
        session_results = await run_test_category(session_tests, [
            "test_session_creation_and_initialization",
            "test_session_environment_setup", 
            "test_session_task_execution"
        ])
        test_results["test_categories"]["session_lifecycle"] = session_results
        
        # 2. Run adaptive workflow tests
        logger.info("=== Running Adaptive Workflow Tests ===")
        workflow_tests = TestAdaptiveWorkflowModification()
        workflow_results = await run_test_category(workflow_tests, [
            "test_performance_analysis",
            "test_workflow_modification",
            "test_real_time_adaptation"
        ])
        test_results["test_categories"]["adaptive_workflow"] = workflow_results
        
        # 3. Run tool coordination tests
        logger.info("=== Running Tool Coordination Tests ===")
        coordination_tests = TestToolCoordination()
        coordination_results = await run_test_category(coordination_tests, [
            "test_tool_allocation_planning",
            "test_tool_conflict_resolution",
            "test_tool_sharing_optimization"
        ])
        test_results["test_categories"]["tool_coordination"] = coordination_results
        
        # 4. Run session management tests
        logger.info("=== Running Session Management Tests ===")
        session_mgmt_tests = TestSessionManagement()
        session_mgmt_results = await run_test_category(session_mgmt_tests, [
            "test_session_pause_resume",
            "test_session_interruption_recovery",
            "test_session_checkpoint_restore"
        ])
        test_results["test_categories"]["session_management"] = session_mgmt_results
        
        # 5. Run enhanced task decomposition tests
        logger.info("=== Running Enhanced Task Decomposition Tests ===")
        decomposition_tests = TestEnhancedTaskDecomposition()
        decomposition_results = await run_test_category(decomposition_tests, [
            "test_complex_flutter_project_decomposition",
            "test_intelligent_agent_assignment",
            "test_dependency_optimization"
        ])
        test_results["test_categories"]["task_decomposition"] = decomposition_results
        
        # 6. Run performance benchmarks
        logger.info("=== Running Performance Benchmarks ===")
        performance_tests = TestPerformanceBenchmarks()
        performance_results = await run_performance_benchmarks(performance_tests)
        test_results["performance_metrics"] = performance_results
        
        # 7. Run complete integration test
        logger.info("=== Running Complete Integration Test ===")
        integration_start = time.time()
        integration_result = await test_complete_orchestrator_integration()
        integration_time = time.time() - integration_start
        
        test_results["validation_results"]["integration_test"] = {
            "result": integration_result,
            "execution_time": integration_time,
            "status": "passed" if integration_result["overall_status"] == "passed" else "failed"
        }
        
        # Calculate overall status
        all_passed = all(
            category.get("overall_status") == "passed" 
            for category in test_results["test_categories"].values()
        )
        integration_passed = test_results["validation_results"]["integration_test"]["status"] == "passed"
        
        test_results["overall_status"] = "passed" if all_passed and integration_passed else "failed"
        
        # Generate test report
        await generate_test_report(test_results)
        
        logger.info(f"Test suite completed with status: {test_results['overall_status']}")
        return test_results
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)
        test_results["overall_status"] = "error"
        test_results["error"] = str(e)
        return test_results


async def run_test_category(test_instance, test_methods):
    """Run a category of tests and collect results."""
    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "execution_times": {},
        "errors": [],
        "overall_status": "passed"
    }
    
    for method_name in test_methods:
        try:
            logger.info(f"Running {method_name}...")
            start_time = time.time()
            
            # Get the method and run it
            method = getattr(test_instance, method_name)
            await method()
            
            execution_time = time.time() - start_time
            results["execution_times"][method_name] = execution_time
            results["tests_passed"] += 1
            
            logger.info(f"✓ {method_name} passed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results["execution_times"][method_name] = execution_time
            results["tests_failed"] += 1
            results["errors"].append({
                "test": method_name,
                "error": str(e),
                "execution_time": execution_time
            })
            results["overall_status"] = "failed"
            
            logger.error(f"✗ {method_name} failed: {e}")
        
        results["tests_run"] += 1
    
    return results


async def run_performance_benchmarks(performance_tests):
    """Run performance benchmarks and collect metrics."""
    logger.info("Running performance benchmarks...")
    
    benchmarks = {}
    
    try:
        # Workflow optimization effectiveness
        workflow_metrics = await performance_tests.test_workflow_optimization_effectiveness()
        benchmarks["workflow_optimization"] = workflow_metrics
        
        # Tool coordination efficiency  
        coordination_metrics = await performance_tests.test_tool_coordination_efficiency()
        benchmarks["tool_coordination"] = coordination_metrics
        
        # Session management overhead
        session_metrics = await performance_tests.test_session_management_overhead()
        benchmarks["session_management"] = session_metrics
        
        # Calculate overall performance score
        benchmarks["overall_performance"] = {
            "workflow_efficiency": workflow_metrics.get("improvement_percentage", 0),
            "coordination_efficiency": coordination_metrics.get("tool_utilization", 0) * 100,
            "session_efficiency": 100 - (session_metrics.get("memory_overhead_per_session", 0) / 1024 / 1024),
            "composite_score": 0  # Will be calculated
        }
        
        # Composite score (weighted average)
        weights = {"workflow": 0.4, "coordination": 0.3, "session": 0.3}
        composite = (
            benchmarks["overall_performance"]["workflow_efficiency"] * weights["workflow"] +
            benchmarks["overall_performance"]["coordination_efficiency"] * weights["coordination"] +
            benchmarks["overall_performance"]["session_efficiency"] * weights["session"]
        )
        benchmarks["overall_performance"]["composite_score"] = composite
        
        logger.info(f"Performance benchmarks completed. Composite score: {composite:.1f}")
        
    except Exception as e:
        logger.error(f"Performance benchmarks failed: {e}")
        benchmarks["error"] = str(e)
    
    return benchmarks


async def generate_test_report(test_results):
    """Generate comprehensive test report."""
    report_path = Path("/workspaces/FlutterSwarm/python-backend/logs/orchestrator_test_report.json")
    
    # Ensure logs directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON report
    with open(report_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Generate markdown summary
    markdown_report = generate_markdown_report(test_results)
    markdown_path = Path("/workspaces/FlutterSwarm/python-backend/logs/orchestrator_test_summary.md")
    
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    logger.info(f"Test reports generated:")
    logger.info(f"  JSON: {report_path}")
    logger.info(f"  Markdown: {markdown_path}")


def generate_markdown_report(test_results):
    """Generate markdown summary report."""
    status_emoji = "✅" if test_results["overall_status"] == "passed" else "❌"
    
    report = f"""# OrchestratorAgent Test Report {status_emoji}

**Test Execution Time**: {test_results['timestamp']}
**Overall Status**: {test_results['overall_status'].upper()}

## Test Categories Summary

"""
    
    for category, results in test_results.get("test_categories", {}).items():
        status_emoji = "✅" if results.get("overall_status") == "passed" else "❌"
        passed = results.get("tests_passed", 0)
        total = results.get("tests_run", 0)
        
        report += f"""### {category.title().replace('_', ' ')} {status_emoji}
- **Tests Run**: {total}
- **Tests Passed**: {passed}
- **Tests Failed**: {results.get("tests_failed", 0)}
- **Success Rate**: {(passed/total*100) if total > 0 else 0:.1f}%

"""
        
        if results.get("errors"):
            report += "**Errors:**\n"
            for error in results["errors"]:
                report += f"- `{error['test']}`: {error['error']}\n"
            report += "\n"
    
    # Performance metrics
    if "performance_metrics" in test_results:
        perf = test_results["performance_metrics"]
        if "overall_performance" in perf:
            overall = perf["overall_performance"]
            report += f"""## Performance Metrics

- **Composite Performance Score**: {overall.get('composite_score', 0):.1f}/100
- **Workflow Optimization**: {overall.get('workflow_efficiency', 0):.1f}% improvement
- **Tool Coordination**: {overall.get('coordination_efficiency', 0):.1f}% utilization
- **Session Management**: {overall.get('session_efficiency', 0):.1f}% efficiency

"""
    
    # Integration test results
    if "validation_results" in test_results:
        integration = test_results["validation_results"].get("integration_test", {})
        status_emoji = "✅" if integration.get("status") == "passed" else "❌"
        
        report += f"""## Integration Test {status_emoji}

- **Status**: {integration.get('status', 'unknown').upper()}
- **Execution Time**: {integration.get('execution_time', 0):.2f}s
- **Result**: {integration.get('result', {}).get('overall_status', 'unknown')}

"""
    
    # Recommendations
    report += """## Recommendations

Based on the test results:

"""
    
    if test_results["overall_status"] == "passed":
        report += "- ✅ All tests passed successfully\n"
        report += "- ✅ OrchestratorAgent is ready for production use\n"
        report += "- 📈 Consider monitoring performance metrics in production\n"
    else:
        report += "- ❌ Some tests failed - review error details above\n"
        report += "- 🔧 Fix failing tests before production deployment\n"
        report += "- 📊 Focus on improving performance metrics\n"
    
    return report


async def main():
    """Main test runner entry point."""
    print("🚀 Starting OrchestratorAgent Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        results = await run_orchestrator_tests()
        
        print("\n" + "=" * 60)
        print(f"🏁 Test Suite Complete: {results['overall_status'].upper()}")
        
        if results["overall_status"] == "passed":
            print("✅ All tests passed! OrchestratorAgent is ready.")
        else:
            print("❌ Some tests failed. Review the detailed report.")
            
        print(f"📊 Detailed reports available in /workspaces/FlutterSwarm/python-backend/logs/")
        
        return 0 if results["overall_status"] == "passed" else 1
        
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        logger.error(f"Test suite crashed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
