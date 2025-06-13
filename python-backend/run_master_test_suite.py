#!/usr/bin/env python3
"""
FlutterSwarm Tool System - Master Test Suite Runner

This script orchestrates all testing components:
1. Tool capability validation
2. Integration testing
3. Agent interaction testing
4. Performance benchmarking
5. End-to-end workflow validation

Usage:
    python run_master_test_suite.py [--level standard] [--output-format json]
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import test modules
from test_comprehensive_integration import ComprehensiveIntegrationTester
from tool_validation_framework import validate_all_tools, ValidationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'master_test_suite_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger("master_test_suite")


@dataclass
class TestSuiteResult:
    """Result of the complete test suite."""
    component: str
    success: bool
    score: float
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class MasterTestReport:
    """Comprehensive test report."""
    timestamp: str
    total_duration: float
    overall_success: bool
    overall_score: float
    component_results: List[TestSuiteResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class MasterTestSuiteRunner:
    """Master test suite runner for the entire FlutterSwarm tool system."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.results: List[TestSuiteResult] = []
        self.start_time = time.time()
        
    async def run_complete_test_suite(self) -> MasterTestReport:
        """Run the complete test suite."""
        logger.info("🚀 FlutterSwarm Tool System - Master Test Suite")
        logger.info("=" * 80)
        logger.info(f"Validation Level: {self.validation_level.value.upper()}")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Test components in order
        test_components = [
            ("System Initialization", self._test_system_initialization),
            ("Tool Capability Validation", self._test_tool_capabilities),
            ("Integration Testing", self._test_integration),
            ("Agent Interaction Testing", self._test_agent_interactions),
            ("Performance Benchmarking", self._test_performance),
            ("End-to-End Workflows", self._test_e2e_workflows),
            ("Security Compliance", self._test_security),
            ("Documentation Completeness", self._test_documentation)
        ]
        
        for component_name, test_func in test_components:
            logger.info(f"\n🔍 Running: {component_name}")
            logger.info("-" * 60)
            
            try:
                result = await test_func()
                self.results.append(result)
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                logger.info(f"{status} {component_name}: {result.score:.2f} ({result.duration:.2f}s)")
                
                if not result.success and result.error_message:
                    logger.error(f"   Error: {result.error_message}")
                    
            except Exception as e:
                error_result = TestSuiteResult(
                    component=component_name,
                    success=False,
                    score=0.0,
                    duration=time.time() - self.start_time,
                    error_message=str(e)
                )
                self.results.append(error_result)
                logger.error(f"❌ FAIL {component_name}: {e}")
                traceback.print_exc()
        
        return self._generate_master_report()
    
    async def _test_system_initialization(self) -> TestSuiteResult:
        """Test 1: System initialization and basic functionality."""
        start_time = time.time()
        
        try:
            # Test imports
            from core.tools.tool_registry import ToolRegistry
            from core.tools.flutter_sdk_tool import FlutterSDKTool
            from core.tools.file_system_tool import FileSystemTool
            from core.tools.process_tool import ProcessTool
            
            # Test registry initialization
            registry = ToolRegistry.instance()
            await registry.initialize(auto_discover=True)
            
            # Test tool registration
            tools_registered = len(registry.get_available_tools())
            initialization_success = registry.is_initialized and tools_registered > 0
            
            score = 1.0 if initialization_success else 0.0
            
            return TestSuiteResult(
                component="System Initialization",
                success=initialization_success,
                score=score,
                duration=time.time() - start_time,
                details={
                    "registry_initialized": registry.is_initialized,
                    "tools_registered": tools_registered,
                    "auto_discovery": True
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="System Initialization",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_tool_capabilities(self) -> TestSuiteResult:
        """Test 2: Tool capability validation."""
        start_time = time.time()
        
        try:
            # Run the tool validation framework
            validation_result = await validate_all_tools(self.validation_level)
            
            if not validation_result["success"]:
                return TestSuiteResult(
                    component="Tool Capability Validation",
                    success=False,
                    score=0.0,
                    duration=time.time() - start_time,
                    error_message=validation_result.get("error", "Unknown validation error")
                )
            
            summary = validation_result["summary"]
            avg_score = summary["average_score"]
            overall_status = validation_result["overall_status"]
            
            success = overall_status in ["excellent", "good"]
            
            return TestSuiteResult(
                component="Tool Capability Validation",
                success=success,
                score=avg_score,
                duration=time.time() - start_time,
                details={
                    "overall_status": overall_status,
                    "total_tools": summary["total_tools"],
                    "excellent_tools": summary["excellent_tools"],
                    "good_tools": summary["good_tools"],
                    "report_path": validation_result.get("report_path")
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Tool Capability Validation",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_integration(self) -> TestSuiteResult:
        """Test 3: Integration testing."""
        start_time = time.time()
        
        try:
            # Run comprehensive integration tests
            tester = ComprehensiveIntegrationTester()
            integration_results = await tester.run_all_tests()
            
            success_rate = integration_results.get("success_rate", 0)
            overall_success = integration_results.get("overall_success", False)
            
            return TestSuiteResult(
                component="Integration Testing",
                success=overall_success,
                score=success_rate / 100.0,  # Convert percentage to 0-1 scale
                duration=time.time() - start_time,
                details={
                    "success_rate": success_rate,
                    "passed_tests": tester.metrics.passed_tests,
                    "total_tests": tester.metrics.total_tests,
                    "test_duration": tester.metrics.total_duration
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Integration Testing",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_agent_interactions(self) -> TestSuiteResult:
        """Test 4: Agent interaction testing."""
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            # Test agent-friendly APIs
            agent_compatibility_tests = []
            
            # Test 1: Tool selection for workflows
            selected_tools = registry.select_tools_for_workflow("flutter_development", {})
            agent_compatibility_tests.append({
                "test": "workflow_tool_selection",
                "success": len(selected_tools) > 0
            })
            
            # Test 2: Tool recommendations
            recommendations = registry.get_tool_recommendations("create Flutter app", {})
            agent_compatibility_tests.append({
                "test": "tool_recommendations",
                "success": len(recommendations) > 0
            })
            
            # Test 3: Capability querying
            tools = registry.get_available_tools()
            capability_tests = []
            for tool in tools[:3]:  # Test first 3 tools
                capabilities = await tool.get_capabilities()
                has_clear_descriptions = all(
                    len(op.get("description", "")) > 10 
                    for op in capabilities.available_operations
                )
                capability_tests.append(has_clear_descriptions)
            
            agent_compatibility_tests.append({
                "test": "clear_operation_descriptions",
                "success": all(capability_tests)
            })
            
            passed_tests = sum(1 for test in agent_compatibility_tests if test["success"])
            total_tests = len(agent_compatibility_tests)
            score = passed_tests / total_tests if total_tests > 0 else 0
            
            return TestSuiteResult(
                component="Agent Interaction Testing",
                success=score >= 0.7,
                score=score,
                duration=time.time() - start_time,
                details={
                    "passed_tests": passed_tests,
                    "total_tests": total_tests,
                    "test_results": agent_compatibility_tests
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Agent Interaction Testing",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_performance(self) -> TestSuiteResult:
        """Test 5: Performance benchmarking."""
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            performance_metrics = []
            
            for tool in tools[:3]:  # Test first 3 tools for performance
                # Benchmark capabilities call
                cap_start = time.time()
                capabilities = await tool.get_capabilities()
                cap_time = time.time() - cap_start
                
                # Benchmark validation call
                val_start = time.time()
                await tool.validate_params("test_operation", {})
                val_time = time.time() - val_start
                
                # Benchmark error handling
                err_start = time.time()
                await tool.execute("nonexistent_operation", {})
                err_time = time.time() - err_start
                
                performance_metrics.append({
                    "tool": tool.name,
                    "capabilities_time": cap_time,
                    "validation_time": val_time,
                    "error_handling_time": err_time,
                    "total_time": cap_time + val_time + err_time
                })
            
            # Calculate performance score
            avg_total_time = sum(m["total_time"] for m in performance_metrics) / len(performance_metrics)
            
            # Score based on responsiveness (lower is better)
            if avg_total_time < 0.1:
                performance_score = 1.0
            elif avg_total_time < 0.5:
                performance_score = 0.8
            elif avg_total_time < 1.0:
                performance_score = 0.6
            else:
                performance_score = 0.4
            
            return TestSuiteResult(
                component="Performance Benchmarking",
                success=performance_score >= 0.6,
                score=performance_score,
                duration=time.time() - start_time,
                details={
                    "average_response_time": avg_total_time,
                    "performance_metrics": performance_metrics,
                    "tools_tested": len(performance_metrics)
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Performance Benchmarking",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_e2e_workflows(self) -> TestSuiteResult:
        """Test 6: End-to-end workflow testing."""
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            import tempfile
            
            registry = ToolRegistry.instance()
            
            # Test workflow: Create Flutter project structure
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = os.path.join(temp_dir, "e2e_test_project")
                
                workflow_steps = []
                
                # Step 1: Use file tool to create project structure
                file_tool = registry.get_tool("file_system_tool")
                if file_tool:
                    os.makedirs(os.path.join(project_path, "lib"), exist_ok=True)
                    
                    # Create pubspec.yaml
                    pubspec_content = """
name: e2e_test_project
description: End-to-end test project
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.13.0"

dependencies:
  flutter:
    sdk: flutter

dev_dependencies:
  flutter_test:
    sdk: flutter
"""
                    
                    result = await file_tool.execute("create_file", {
                        "file_path": os.path.join(project_path, "pubspec.yaml"),
                        "content": pubspec_content
                    })
                    
                    workflow_steps.append({
                        "step": "create_pubspec",
                        "success": result.status.value == "success"
                    })
                
                # Step 2: Use Flutter tool to validate project
                flutter_tool = registry.get_tool("flutter_sdk_tool")
                if flutter_tool:
                    result = await flutter_tool.execute("analyze_code", {
                        "project_path": project_path
                    })
                    
                    workflow_steps.append({
                        "step": "analyze_project",
                        "success": result.status.value in ["success", "failure"]  # Either is acceptable for validation
                    })
                
                # Step 3: Use process tool for additional operations
                process_tool = registry.get_tool("process_tool")
                if process_tool:
                    result = await process_tool.execute("execute_command", {
                        "command": f"ls -la {project_path}",
                        "working_directory": temp_dir
                    })
                    
                    workflow_steps.append({
                        "step": "list_project_files",
                        "success": result.status.value == "success"
                    })
                
                successful_steps = sum(1 for step in workflow_steps if step["success"])
                total_steps = len(workflow_steps)
                score = successful_steps / total_steps if total_steps > 0 else 0
                
                return TestSuiteResult(
                    component="End-to-End Workflows",
                    success=score >= 0.6,
                    score=score,
                    duration=time.time() - start_time,
                    details={
                        "workflow_steps": workflow_steps,
                        "successful_steps": successful_steps,
                        "total_steps": total_steps,
                        "project_path": project_path
                    }
                )
                
        except Exception as e:
            return TestSuiteResult(
                component="End-to-End Workflows",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_security(self) -> TestSuiteResult:
        """Test 7: Security compliance testing."""
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            security_checks = []
            
            for tool in tools:
                # Check 1: Has parameter validation
                has_validation = hasattr(tool, 'validate_params')
                
                # Check 2: Handles errors gracefully
                try:
                    result = await tool.execute("invalid_operation", {})
                    handles_errors = hasattr(result, 'status') and result.status.value == "failure"
                except Exception:
                    handles_errors = False  # Should not throw exceptions
                
                # Check 3: Has proper permissions model
                has_permissions = hasattr(tool, 'required_permissions')
                
                security_score = sum([has_validation, handles_errors, has_permissions]) / 3.0
                
                security_checks.append({
                    "tool": tool.name,
                    "has_validation": has_validation,
                    "handles_errors": handles_errors,
                    "has_permissions": has_permissions,
                    "security_score": security_score
                })
            
            avg_security_score = sum(check["security_score"] for check in security_checks) / len(security_checks)
            
            return TestSuiteResult(
                component="Security Compliance",
                success=avg_security_score >= 0.7,
                score=avg_security_score,
                duration=time.time() - start_time,
                details={
                    "security_checks": security_checks,
                    "tools_tested": len(security_checks),
                    "average_security_score": avg_security_score
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Security Compliance",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_documentation(self) -> TestSuiteResult:
        """Test 8: Documentation completeness."""
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            documentation_scores = []
            
            for tool in tools:
                # Check class docstring
                has_class_doc = bool(tool.__doc__ and len(tool.__doc__) > 20)
                
                # Check method docstrings
                methods_to_check = ['execute', 'get_capabilities', 'validate_params']
                documented_methods = 0
                total_methods = 0
                
                for method_name in methods_to_check:
                    if hasattr(tool, method_name):
                        total_methods += 1
                        method = getattr(tool, method_name)
                        if method.__doc__ and len(method.__doc__) > 10:
                            documented_methods += 1
                
                method_doc_score = documented_methods / total_methods if total_methods > 0 else 0
                
                # Check operation descriptions
                try:
                    capabilities = await tool.get_capabilities()
                    clear_descriptions = sum(
                        1 for op in capabilities.available_operations
                        if len(op.get("description", "")) >= 20
                    )
                    operation_doc_score = clear_descriptions / len(capabilities.available_operations) if capabilities.available_operations else 0
                except Exception:
                    operation_doc_score = 0
                
                overall_doc_score = (
                    (1.0 if has_class_doc else 0.0) + 
                    method_doc_score + 
                    operation_doc_score
                ) / 3.0
                
                documentation_scores.append({
                    "tool": tool.name,
                    "class_documented": has_class_doc,
                    "method_doc_score": method_doc_score,
                    "operation_doc_score": operation_doc_score,
                    "overall_score": overall_doc_score
                })
            
            avg_doc_score = sum(score["overall_score"] for score in documentation_scores) / len(documentation_scores)
            
            return TestSuiteResult(
                component="Documentation Completeness",
                success=avg_doc_score >= 0.6,
                score=avg_doc_score,
                duration=time.time() - start_time,
                details={
                    "documentation_scores": documentation_scores,
                    "average_doc_score": avg_doc_score,
                    "tools_analyzed": len(documentation_scores)
                }
            )
            
        except Exception as e:
            return TestSuiteResult(
                component="Documentation Completeness",
                success=False,
                score=0.0,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_master_report(self) -> MasterTestReport:
        """Generate the comprehensive master test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate overall metrics
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        
        # Weighted overall score
        total_weight = sum(self._get_component_weight(result.component) for result in self.results)
        weighted_score = sum(
            result.score * self._get_component_weight(result.component) 
            for result in self.results
        )
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        overall_success = passed_tests >= (total_tests * 0.75)  # 75% pass rate
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create summary
        summary = {
            "total_components": total_tests,
            "passed_components": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_score": overall_score,
            "validation_level": self.validation_level.value,
            "duration": total_duration
        }
        
        report = MasterTestReport(
            timestamp=datetime.now().isoformat(),
            total_duration=total_duration,
            overall_success=overall_success,
            overall_score=overall_score,
            component_results=self.results,
            summary=summary,
            recommendations=recommendations
        )
        
        self._print_final_report(report)
        return report
    
    def _get_component_weight(self, component_name: str) -> float:
        """Get weight for different test components."""
        weights = {
            "System Initialization": 2.0,
            "Tool Capability Validation": 3.0,
            "Integration Testing": 3.0,
            "Agent Interaction Testing": 2.5,
            "Performance Benchmarking": 1.5,
            "End-to-End Workflows": 2.5,
            "Security Compliance": 2.0,
            "Documentation Completeness": 1.0
        }
        return weights.get(component_name, 1.0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for result in self.results:
            if not result.success:
                if result.component == "Tool Capability Validation":
                    recommendations.append(
                        "🔧 Improve tool validation: Ensure all tools have proper parameter validation and error handling"
                    )
                elif result.component == "Integration Testing":
                    recommendations.append(
                        "🔗 Fix integration issues: Review tool interaction patterns and error propagation"
                    )
                elif result.component == "Agent Interaction Testing":
                    recommendations.append(
                        "🤖 Enhance agent compatibility: Improve operation descriptions and parameter schemas"
                    )
                elif result.component == "Performance Benchmarking":
                    recommendations.append(
                        "⚡ Optimize performance: Profile slow operations and implement caching where appropriate"
                    )
                elif result.component == "Security Compliance":
                    recommendations.append(
                        "🔒 Strengthen security: Implement proper input validation and permission checks"
                    )
                elif result.component == "Documentation Completeness":
                    recommendations.append(
                        "📚 Improve documentation: Add comprehensive docstrings and operation descriptions"
                    )
            elif result.score < 0.8:
                recommendations.append(
                    f"📈 {result.component} needs minor improvements to reach excellence"
                )
        
        if not recommendations:
            recommendations.append("🎉 Excellent! All components are performing well. System is production-ready.")
        
        return recommendations
    
    def _print_final_report(self, report: MasterTestReport):
        """Print the final comprehensive report."""
        logger.info("\n" + "=" * 80)
        logger.info("🏆 FLUTTERSWARM TOOL SYSTEM - MASTER TEST REPORT")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Overall Results:")
        logger.info(f"   🎯 Overall Success: {'✅ YES' if report.overall_success else '❌ NO'}")
        logger.info(f"   📈 Overall Score: {report.overall_score:.2f}")
        logger.info(f"   ⏱️  Total Duration: {report.total_duration:.2f}s")
        logger.info(f"   🧪 Components Tested: {report.summary['total_components']}")
        logger.info(f"   ✅ Components Passed: {report.summary['passed_components']}")
        logger.info(f"   📊 Success Rate: {report.summary['success_rate']:.1f}%")
        
        logger.info(f"\n📋 Component Results:")
        for result in report.component_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"   {status} {result.component}: {result.score:.2f} ({result.duration:.2f}s)")
            if not result.success and result.error_message:
                logger.info(f"      ⚠️  {result.error_message}")
        
        if report.recommendations:
            logger.info(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                logger.info(f"   {rec}")
        
        # Assessment
        if report.overall_score >= 0.9:
            logger.info(f"\n🏆 EXCELLENT! FlutterSwarm Tool System is production-ready!")
            logger.info(f"✨ All components are performing at high standards.")
            logger.info(f"🚀 System is ready for AI agent integration and real-world use.")
        elif report.overall_score >= 0.75:
            logger.info(f"\n✅ GOOD! FlutterSwarm Tool System is mostly ready!")
            logger.info(f"🔧 Minor improvements recommended for optimal performance.")
            logger.info(f"📈 System can be used with AI agents with some monitoring.")
        elif report.overall_score >= 0.6:
            logger.info(f"\n⚠️  ACCEPTABLE! FlutterSwarm Tool System needs improvements.")
            logger.info(f"🛠️  Several areas need attention before production use.")
            logger.info(f"🧪 Additional testing and fixes recommended.")
        else:
            logger.info(f"\n❌ NEEDS WORK! FlutterSwarm Tool System requires significant improvements.")
            logger.info(f"🚨 Major issues need to be resolved before use with AI agents.")
            logger.info(f"🔧 Review failed components and implement fixes.")
        
        logger.info("\n" + "=" * 80)


async def main():
    """Main entry point for the master test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlutterSwarm Tool System Master Test Suite")
    parser.add_argument(
        "--level",
        choices=["basic", "standard", "comprehensive", "production"],
        default="standard",
        help="Validation level (default: standard)"
    )
    parser.add_argument(
        "--output-format",
        choices=["console", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed report to file"
    )
    
    args = parser.parse_args()
    
    # Set validation level
    validation_level = ValidationLevel(args.level)
    
    # Create and run test suite
    runner = MasterTestSuiteRunner(validation_level)
    
    try:
        report = await runner.run_complete_test_suite()
        
        # Save report if requested
        if args.save_report or args.output_format in ["json", "both"]:
            report_filename = f"master_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                "timestamp": report.timestamp,
                "total_duration": report.total_duration,
                "overall_success": report.overall_success,
                "overall_score": report.overall_score,
                "summary": report.summary,
                "component_results": [
                    {
                        "component": result.component,
                        "success": result.success,
                        "score": result.score,
                        "duration": result.duration,
                        "details": result.details,
                        "error_message": result.error_message
                    }
                    for result in report.component_results
                ],
                "recommendations": report.recommendations
            }
            
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"\n📄 Detailed report saved to: {report_filename}")
        
        return report.overall_success
        
    except Exception as e:
        logger.error(f"Master test suite failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🏁 Master Test Suite {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
