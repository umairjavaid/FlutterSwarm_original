#!/usr/bin/env python3
"""
Comprehensive Validation Framework for BaseAgent Tool Integration System.

This validation framework ensures all requirements are properly implemented:
1. Tool discovery and understanding completeness
2. LLM integration functionality
3. Learning and adaptation mechanisms
4. Performance benchmarks and monitoring
5. Error handling and recovery
6. Inter-agent knowledge sharing
7. Backward compatibility
8. Documentation completeness

The framework generates detailed reports and recommendations for improvements.
"""

import asyncio
import inspect
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import importlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validation_framework")


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    PASS = "pass"


@dataclass
class ValidationResult:
    """Individual validation result."""
    test_name: str
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    framework_version: str
    validation_timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    critical_issues: int
    results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ToolIntegrationValidator:
    """Comprehensive validator for BaseAgent tool integration system."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = datetime.now()
        
    async def validate_all(self) -> ValidationReport:
        """Run comprehensive validation of tool integration system."""
        logger.info("🔍 Starting Comprehensive Tool Integration Validation")
        logger.info("=" * 70)
        
        # Run all validation categories
        await self._validate_tool_discovery()
        await self._validate_llm_integration()
        await self._validate_learning_mechanisms()
        await self._validate_performance_requirements()
        await self._validate_error_handling()
        await self._validate_inter_agent_sharing()
        await self._validate_backward_compatibility()
        await self._validate_documentation()
        
        return self._generate_report()
    
    async def _validate_tool_discovery(self) -> List[ValidationResult]:
        """Validate tool discovery and understanding capabilities."""
        logger.info("🔍 Validating Tool Discovery and Understanding...")
        
        try:
            # Test 1: BaseAgent class has tool discovery methods
            from src.agents.base_agent import BaseAgent
            
            required_methods = [
                'discover_available_tools',
                'analyze_tool_capabilities',
                'build_tool_understanding',
                'update_tool_preferences'
            ]
            
            for method in required_methods:
                if hasattr(BaseAgent, method):
                    self.results.append(ValidationResult(
                        test_name=f"BaseAgent.{method}_exists",
                        level=ValidationLevel.PASS,
                        message=f"Method {method} exists in BaseAgent"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"BaseAgent.{method}_missing",
                        level=ValidationLevel.CRITICAL,
                        message=f"Required method {method} missing from BaseAgent",
                        recommendations=[f"Implement {method} method in BaseAgent class"]
                    ))
            
            # Test 2: Tool understanding data structures
            from src.models.tool_models import ToolUnderstanding, ToolMetrics
            
            tool_understanding_fields = ['tool_name', 'capabilities', 'usage_patterns', 'confidence_score']
            understanding_instance = ToolUnderstanding.__annotations__ if hasattr(ToolUnderstanding, '__annotations__') else {}
            
            for field in tool_understanding_fields:
                if field in understanding_instance:
                    self.results.append(ValidationResult(
                        test_name=f"ToolUnderstanding.{field}_defined",
                        level=ValidationLevel.PASS,
                        message=f"ToolUnderstanding has {field} field"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"ToolUnderstanding.{field}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"ToolUnderstanding missing {field} field"
                    ))
            
            # Test 3: Tool registry integration
            from src.core.tools.tool_registry import ToolRegistry
            
            registry_methods = ['register_tool', 'get_available_tools', 'find_tools_by_capability']
            
            for method in registry_methods:
                if hasattr(ToolRegistry, method):
                    self.results.append(ValidationResult(
                        test_name=f"ToolRegistry.{method}_exists",
                        level=ValidationLevel.PASS,
                        message=f"ToolRegistry has {method} method"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"ToolRegistry.{method}_missing",
                        level=ValidationLevel.CRITICAL,
                        message=f"ToolRegistry missing {method} method"
                    ))
            
        except ImportError as e:
            self.results.append(ValidationResult(
                test_name="tool_discovery_import_error",
                level=ValidationLevel.CRITICAL,
                message=f"Cannot import required modules: {e}",
                recommendations=["Fix import paths and ensure all modules are accessible"]
            ))
        
        return self.results
    
    async def _validate_llm_integration(self) -> List[ValidationResult]:
        """Validate LLM integration with tool-aware prompts."""
        logger.info("🤖 Validating LLM Integration...")
        
        try:
            from src.agents.base_agent import BaseAgent
            
            # Test 1: LLM client integration
            if hasattr(BaseAgent, 'llm_client'):
                self.results.append(ValidationResult(
                    test_name="llm_client_integration",
                    level=ValidationLevel.PASS,
                    message="BaseAgent has LLM client integration"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="llm_client_missing",
                    level=ValidationLevel.CRITICAL,
                    message="BaseAgent missing LLM client integration"
                ))
            
            # Test 2: Tool-aware prompt generation
            prompt_methods = ['generate_tool_analysis_prompt', 'create_tool_selection_prompt']
            
            for method in prompt_methods:
                if hasattr(BaseAgent, method):
                    self.results.append(ValidationResult(
                        test_name=f"prompt_generation_{method}",
                        level=ValidationLevel.PASS,
                        message=f"Tool-aware prompt method {method} exists"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"prompt_generation_{method}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Tool-aware prompt method {method} missing"
                    ))
            
            # Test 3: Response parsing capabilities
            if hasattr(BaseAgent, 'parse_tool_analysis_response'):
                self.results.append(ValidationResult(
                    test_name="llm_response_parsing",
                    level=ValidationLevel.PASS,
                    message="LLM response parsing implemented"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="llm_response_parsing_missing",
                    level=ValidationLevel.WARNING,
                    message="LLM response parsing not implemented"
                ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="llm_integration_error",
                level=ValidationLevel.CRITICAL,
                message=f"Error validating LLM integration: {e}"
            ))
    
    async def _validate_learning_mechanisms(self) -> List[ValidationResult]:
        """Validate learning and adaptation mechanisms."""
        logger.info("📚 Validating Learning and Adaptation Mechanisms...")
        
        try:
            from src.models.learning_models import AdvancedToolWorkflowMixin
            
            # Test 1: Learning mixin integration
            learning_methods = [
                'update_tool_usage_patterns',
                'learn_from_tool_outcomes',
                'adapt_tool_selection_strategy'
            ]
            
            for method in learning_methods:
                if hasattr(AdvancedToolWorkflowMixin, method):
                    self.results.append(ValidationResult(
                        test_name=f"learning_{method}",
                        level=ValidationLevel.PASS,
                        message=f"Learning method {method} implemented"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"learning_{method}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Learning method {method} not implemented"
                    ))
            
            # Test 2: Tool usage tracking
            from src.models.tool_models import ToolUsageEntry
            
            usage_fields = ['tool_name', 'operation', 'timestamp', 'success_rate']
            
            for field in usage_fields:
                if hasattr(ToolUsageEntry, field) or field in getattr(ToolUsageEntry, '__annotations__', {}):
                    self.results.append(ValidationResult(
                        test_name=f"usage_tracking_{field}",
                        level=ValidationLevel.PASS,
                        message=f"Usage tracking field {field} defined"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"usage_tracking_{field}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Usage tracking field {field} missing"
                    ))
            
        except ImportError as e:
            self.results.append(ValidationResult(
                test_name="learning_import_error",
                level=ValidationLevel.CRITICAL,
                message=f"Cannot import learning modules: {e}"
            ))
    
    async def _validate_performance_requirements(self) -> List[ValidationResult]:
        """Validate performance benchmarks and monitoring."""
        logger.info("⚡ Validating Performance Requirements...")
        
        # Test 1: Performance metrics collection
        performance_benchmarks_exist = Path("performance_benchmarks.py").exists()
        
        if performance_benchmarks_exist:
            self.results.append(ValidationResult(
                test_name="performance_benchmarks_exist",
                level=ValidationLevel.PASS,
                message="Performance benchmarks framework exists"
            ))
        else:
            self.results.append(ValidationResult(
                test_name="performance_benchmarks_missing",
                level=ValidationLevel.WARNING,
                message="Performance benchmarks framework missing"
            ))
        
        # Test 2: Memory management validation
        try:
            from src.core.memory_manager import MemoryManager
            
            memory_methods = ['cleanup_expired_entries', 'optimize_storage']
            
            for method in memory_methods:
                if hasattr(MemoryManager, method):
                    self.results.append(ValidationResult(
                        test_name=f"memory_management_{method}",
                        level=ValidationLevel.PASS,
                        message=f"Memory management method {method} exists"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"memory_management_{method}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Memory management method {method} missing"
                    ))
                    
        except ImportError:
            self.results.append(ValidationResult(
                test_name="memory_manager_import_error",
                level=ValidationLevel.CRITICAL,
                message="Cannot import MemoryManager"
            ))
    
    async def _validate_error_handling(self) -> List[ValidationResult]:
        """Validate error handling and recovery scenarios."""
        logger.info("🚨 Validating Error Handling and Recovery...")
        
        try:
            from src.core.exceptions import AgentError, LLMError
            
            # Test 1: Custom exception classes exist
            exception_classes = [AgentError, LLMError]
            
            for exc_class in exception_classes:
                self.results.append(ValidationResult(
                    test_name=f"exception_class_{exc_class.__name__}",
                    level=ValidationLevel.PASS,
                    message=f"Exception class {exc_class.__name__} defined"
                ))
            
            # Test 2: Error recovery mechanisms
            from src.agents.base_agent import BaseAgent
            
            error_methods = ['handle_tool_error', 'recover_from_failure']
            
            for method in error_methods:
                if hasattr(BaseAgent, method):
                    self.results.append(ValidationResult(
                        test_name=f"error_handling_{method}",
                        level=ValidationLevel.PASS,
                        message=f"Error handling method {method} exists"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"error_handling_{method}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Error handling method {method} missing"
                    ))
                    
        except ImportError as e:
            self.results.append(ValidationResult(
                test_name="error_handling_import_error",
                level=ValidationLevel.CRITICAL,
                message=f"Cannot import error handling modules: {e}"
            ))
    
    async def _validate_inter_agent_sharing(self) -> List[ValidationResult]:
        """Validate inter-agent tool knowledge sharing."""
        logger.info("🤝 Validating Inter-Agent Knowledge Sharing...")
        
        try:
            from src.core.event_bus import EventBus
            
            # Test 1: Event bus integration
            if hasattr(EventBus, 'publish') and hasattr(EventBus, 'subscribe'):
                self.results.append(ValidationResult(
                    test_name="event_bus_integration",
                    level=ValidationLevel.PASS,
                    message="Event bus integration exists"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="event_bus_missing",
                    level=ValidationLevel.CRITICAL,
                    message="Event bus integration missing"
                ))
            
            # Test 2: Knowledge sharing events
            sharing_events = ['tool_discovery_event', 'tool_usage_event', 'tool_learning_event']
            
            # This is a placeholder check - in reality, we'd check event definitions
            for event in sharing_events:
                self.results.append(ValidationResult(
                    test_name=f"knowledge_sharing_{event}",
                    level=ValidationLevel.INFO,
                    message=f"Knowledge sharing event {event} should be implemented"
                ))
                
        except ImportError:
            self.results.append(ValidationResult(
                test_name="event_bus_import_error",
                level=ValidationLevel.CRITICAL,
                message="Cannot import EventBus"
            ))
    
    async def _validate_backward_compatibility(self) -> List[ValidationResult]:
        """Validate backward compatibility with existing BaseAgent functionality."""
        logger.info("🔄 Validating Backward Compatibility...")
        
        try:
            from src.agents.base_agent import BaseAgent
            
            # Test 1: Core BaseAgent methods still exist
            core_methods = ['process_task', 'get_capabilities', 'get_system_prompt']
            
            for method in core_methods:
                if hasattr(BaseAgent, method):
                    self.results.append(ValidationResult(
                        test_name=f"backward_compatibility_{method}",
                        level=ValidationLevel.PASS,
                        message=f"Core method {method} preserved"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"backward_compatibility_{method}_missing",
                        level=ValidationLevel.CRITICAL,
                        message=f"Core method {method} missing - breaks backward compatibility"
                    ))
            
            # Test 2: Configuration compatibility
            from src.agents.base_agent import AgentConfig
            
            config_fields = ['agent_id', 'agent_type', 'capabilities']
            
            for field in config_fields:
                if hasattr(AgentConfig, field) or field in getattr(AgentConfig, '__annotations__', {}):
                    self.results.append(ValidationResult(
                        test_name=f"config_compatibility_{field}",
                        level=ValidationLevel.PASS,
                        message=f"Config field {field} preserved"
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=f"config_compatibility_{field}_missing",
                        level=ValidationLevel.WARNING,
                        message=f"Config field {field} missing"
                    ))
                    
        except ImportError as e:
            self.results.append(ValidationResult(
                test_name="backward_compatibility_import_error",
                level=ValidationLevel.CRITICAL,
                message=f"Cannot import core modules for compatibility check: {e}"
            ))
    
    async def _validate_documentation(self) -> List[ValidationResult]:
        """Validate documentation completeness."""
        logger.info("📖 Validating Documentation Completeness...")
        
        # Test 1: Check for documentation files
        doc_files = [
            "TOOL_INTEGRATION_GUIDE.md",
            "API_REFERENCE.md",
            "TROUBLESHOOTING.md",
            "EXAMPLES.md"
        ]
        
        for doc_file in doc_files:
            if Path(doc_file).exists():
                self.results.append(ValidationResult(
                    test_name=f"documentation_{doc_file}",
                    level=ValidationLevel.PASS,
                    message=f"Documentation file {doc_file} exists"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=f"documentation_{doc_file}_missing",
                    level=ValidationLevel.WARNING,
                    message=f"Documentation file {doc_file} missing",
                    recommendations=[f"Create {doc_file} with comprehensive documentation"]
                ))
        
        # Test 2: Code documentation coverage
        try:
            from src.agents.base_agent import BaseAgent
            
            if BaseAgent.__doc__:
                self.results.append(ValidationResult(
                    test_name="base_agent_docstring",
                    level=ValidationLevel.PASS,
                    message="BaseAgent has documentation"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="base_agent_docstring_missing",
                    level=ValidationLevel.WARNING,
                    message="BaseAgent missing docstring"
                ))
                
        except ImportError:
            pass
    
    def _generate_report(self) -> ValidationReport:
        """Generate comprehensive validation report."""
        end_time = datetime.now()
        
        # Count results by level
        passed = len([r for r in self.results if r.level == ValidationLevel.PASS])
        warnings = len([r for r in self.results if r.level == ValidationLevel.WARNING])
        critical = len([r for r in self.results if r.level == ValidationLevel.CRITICAL])
        info = len([r for r in self.results if r.level == ValidationLevel.INFO])
        
        # Generate recommendations
        recommendations = []
        for result in self.results:
            recommendations.extend(result.recommendations)
        
        # Remove duplicates
        unique_recommendations = list(set(recommendations))
        
        return ValidationReport(
            framework_version="1.0.0",
            validation_timestamp=end_time,
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=critical,
            warnings=warnings,
            critical_issues=critical,
            results=self.results,
            performance_metrics={
                "validation_duration": (end_time - self.start_time).total_seconds(),
                "tests_per_second": len(self.results) / max((end_time - self.start_time).total_seconds(), 1)
            },
            recommendations=unique_recommendations
        )


async def run_comprehensive_validation() -> ValidationReport:
    """Run comprehensive validation of the tool integration system."""
    validator = ToolIntegrationValidator()
    return await validator.validate_all()


def generate_html_report(report: ValidationReport) -> str:
    """Generate HTML report from validation results."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tool Integration Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
            .metric {{ background: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
            .critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
            .warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
            .pass {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
            .info {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
            .result {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Tool Integration Validation Report</h1>
            <p>Generated: {report.validation_timestamp}</p>
            <p>Framework Version: {report.framework_version}</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>Total Tests</h3>
                <p>{report.total_tests}</p>
            </div>
            <div class="metric">
                <h3>Passed</h3>
                <p>{report.passed_tests}</p>
            </div>
            <div class="metric">
                <h3>Warnings</h3>
                <p>{report.warnings}</p>
            </div>
            <div class="metric">
                <h3>Critical Issues</h3>
                <p>{report.critical_issues}</p>
            </div>
        </div>
        
        <h2>Validation Results</h2>
    """
    
    for result in report.results:
        css_class = result.level.value
        html += f'''
        <div class="result {css_class}">
            <h4>{result.test_name}</h4>
            <p>{result.message}</p>
            {f"<ul>{''.join([f'<li>{rec}</li>' for rec in result.recommendations])}</ul>" if result.recommendations else ""}
        </div>
        '''
    
    if report.recommendations:
        html += f'''
        <h2>Overall Recommendations</h2>
        <ul>
            {''.join([f"<li>{rec}</li>" for rec in report.recommendations])}
        </ul>
        '''
    
    html += """
    </body>
    </html>
    """
    
    return html


async def main():
    """Main validation execution."""
    logger.info("🚀 Starting Comprehensive Tool Integration Validation")
    
    try:
        # Run validation
        report = await run_comprehensive_validation()
        
        # Generate reports
        logger.info("\n" + "=" * 70)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {report.total_tests}")
        logger.info(f"Passed: {report.passed_tests}")
        logger.info(f"Warnings: {report.warnings}")
        logger.info(f"Critical Issues: {report.critical_issues}")
        logger.info(f"Success Rate: {(report.passed_tests/report.total_tests)*100:.1f}%")
        
        # Save JSON report
        json_report = {
            "framework_version": report.framework_version,
            "validation_timestamp": report.validation_timestamp.isoformat(),
            "summary": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "warnings": report.warnings,
                "critical_issues": report.critical_issues
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details,
                    "recommendations": r.recommendations
                }
                for r in report.results
            ],
            "recommendations": report.recommendations,
            "performance_metrics": report.performance_metrics
        }
        
        with open("validation_report.json", "w") as f:
            json.dump(json_report, f, indent=2)
        
        # Save HTML report
        html_report = generate_html_report(report)
        with open("validation_report.html", "w") as f:
            f.write(html_report)
        
        logger.info("\n📄 Reports generated:")
        logger.info("  - validation_report.json")
        logger.info("  - validation_report.html")
        
        # Print critical issues
        if report.critical_issues > 0:
            logger.info("\n❌ CRITICAL ISSUES FOUND:")
            for result in report.results:
                if result.level == ValidationLevel.CRITICAL:
                    logger.info(f"  - {result.test_name}: {result.message}")
        
        # Print recommendations
        if report.recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in report.recommendations[:5]:  # Show first 5
                logger.info(f"  - {rec}")
            if len(report.recommendations) > 5:
                logger.info(f"  ... and {len(report.recommendations) - 5} more (see full report)")
        
        logger.info("\n✅ Validation completed successfully!")
        
        return report.critical_issues == 0
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
