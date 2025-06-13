#!/usr/bin/env python3
"""
Tool Capability Validation Framework for FlutterSwarm.

This framework provides:
1. Schema validation for all tool operations
2. Capability verification against requirements
3. Performance benchmarking
4. Compliance checking for AI agent integration
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger("tool_validation")


class ValidationLevel(Enum):
    """Validation levels for different requirements."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


@dataclass
class ValidationRequirement:
    """A validation requirement for tools."""
    name: str
    description: str
    validator: callable
    level: ValidationLevel
    mandatory: bool = True
    weight: float = 1.0


@dataclass
class ValidationResult:
    """Result of a validation check."""
    requirement_name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ToolValidationReport:
    """Comprehensive validation report for a tool."""
    tool_name: str
    overall_score: float
    passed_requirements: int
    total_requirements: int
    results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    compliance_status: str = "unknown"


class ToolCapabilityValidator:
    """Comprehensive validation framework for tool capabilities."""
    
    def __init__(self):
        self.requirements = self._setup_validation_requirements()
        self.schema_definitions = self._load_schema_definitions()
    
    def _setup_validation_requirements(self) -> List[ValidationRequirement]:
        """Setup validation requirements for all tools."""
        return [
            # Basic Requirements
            ValidationRequirement(
                name="has_name_and_description",
                description="Tool must have a name and description",
                validator=self._validate_basic_metadata,
                level=ValidationLevel.BASIC,
                mandatory=True,
                weight=1.0
            ),
            ValidationRequirement(
                name="has_capabilities",
                description="Tool must expose capabilities through get_capabilities()",
                validator=self._validate_capabilities_method,
                level=ValidationLevel.BASIC,
                mandatory=True,
                weight=2.0
            ),
            ValidationRequirement(
                name="has_execute_method",
                description="Tool must have an execute method",
                validator=self._validate_execute_method,
                level=ValidationLevel.BASIC,
                mandatory=True,
                weight=2.0
            ),
            ValidationRequirement(
                name="parameter_validation",
                description="Tool must validate parameters",
                validator=self._validate_parameter_validation,
                level=ValidationLevel.STANDARD,
                mandatory=True,
                weight=2.0
            ),
            ValidationRequirement(
                name="error_handling",
                description="Tool must handle errors gracefully",
                validator=self._validate_error_handling,
                level=ValidationLevel.STANDARD,
                mandatory=True,
                weight=2.0
            ),
            ValidationRequirement(
                name="operation_schema",
                description="All operations must have proper schema",
                validator=self._validate_operation_schemas,
                level=ValidationLevel.STANDARD,
                mandatory=True,
                weight=1.5
            ),
            ValidationRequirement(
                name="async_compliance",
                description="Tool must be async/await compliant",
                validator=self._validate_async_compliance,
                level=ValidationLevel.STANDARD,
                mandatory=True,
                weight=1.5
            ),
            ValidationRequirement(
                name="performance_tracking",
                description="Tool should track performance metrics",
                validator=self._validate_performance_tracking,
                level=ValidationLevel.COMPREHENSIVE,
                mandatory=False,
                weight=1.0
            ),
            ValidationRequirement(
                name="health_checks",
                description="Tool should support health checking",
                validator=self._validate_health_checks,
                level=ValidationLevel.COMPREHENSIVE,
                mandatory=False,
                weight=1.0
            ),
            ValidationRequirement(
                name="agent_compatibility",
                description="Tool must be compatible with AI agents",
                validator=self._validate_agent_compatibility,
                level=ValidationLevel.PRODUCTION,
                mandatory=True,
                weight=3.0
            ),
            ValidationRequirement(
                name="documentation_completeness",
                description="Tool must have complete documentation",
                validator=self._validate_documentation,
                level=ValidationLevel.PRODUCTION,
                mandatory=False,
                weight=1.0
            ),
            ValidationRequirement(
                name="security_compliance",
                description="Tool must follow security best practices",
                validator=self._validate_security_compliance,
                level=ValidationLevel.PRODUCTION,
                mandatory=True,
                weight=2.0
            )
        ]
    
    def _load_schema_definitions(self) -> Dict[str, Any]:
        """Load schema definitions for validation."""
        return {
            "tool_operation": {
                "type": "object",
                "required": ["name", "description", "parameters"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                    "description": {"type": "string", "minLength": 10},
                    "parameters": {"type": "object"},
                    "required_permissions": {"type": "array"},
                    "returns": {"type": "object"}
                }
            },
            "tool_result": {
                "type": "object",
                "required": ["status", "data"],
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failure", "running"]},
                    "data": {"type": "object"},
                    "error_message": {"type": "string"},
                    "execution_time": {"type": "number", "minimum": 0}
                }
            },
            "tool_capabilities": {
                "type": "object",
                "required": ["available_operations"],
                "properties": {
                    "available_operations": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/tool_operation"}
                    },
                    "supported_platforms": {"type": "array"},
                    "version_requirements": {"type": "object"}
                }
            }
        }
    
    async def validate_tool(self, tool, level: ValidationLevel = ValidationLevel.STANDARD) -> ToolValidationReport:
        """Validate a tool against all requirements."""
        logger.info(f"🔍 Validating tool: {tool.name}")
        
        relevant_requirements = [
            req for req in self.requirements 
            if req.level.value <= level.value or req.mandatory
        ]
        
        results = []
        total_weight = 0
        weighted_score = 0
        
        for requirement in relevant_requirements:
            try:
                start_time = time.time()
                result = await requirement.validator(tool)
                validation_time = time.time() - start_time
                
                if isinstance(result, bool):
                    result = ValidationResult(
                        requirement_name=requirement.name,
                        passed=result,
                        score=1.0 if result else 0.0,
                        details={"validation_time": validation_time}
                    )
                elif isinstance(result, ValidationResult):
                    result.details["validation_time"] = validation_time
                
                results.append(result)
                
                # Calculate weighted score
                total_weight += requirement.weight
                weighted_score += result.score * requirement.weight
                
                logger.debug(f"  ✓ {requirement.name}: {'PASS' if result.passed else 'FAIL'}")
                
            except Exception as e:
                error_result = ValidationResult(
                    requirement_name=requirement.name,
                    passed=False,
                    score=0.0,
                    error_message=str(e)
                )
                results.append(error_result)
                total_weight += requirement.weight
                logger.error(f"  ✗ {requirement.name}: ERROR - {e}")
        
        overall_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
        passed_count = sum(1 for result in results if result.passed)
        
        # Determine compliance status
        if overall_score >= 0.9:
            compliance_status = "excellent"
        elif overall_score >= 0.75:
            compliance_status = "good"
        elif overall_score >= 0.6:
            compliance_status = "acceptable"
        else:
            compliance_status = "needs_improvement"
        
        # Calculate performance metrics
        performance_metrics = await self._benchmark_tool_performance(tool)
        
        report = ToolValidationReport(
            tool_name=tool.name,
            overall_score=overall_score,
            passed_requirements=passed_count,
            total_requirements=len(relevant_requirements),
            results=results,
            performance_metrics=performance_metrics,
            compliance_status=compliance_status
        )
        
        logger.info(f"  📊 Score: {overall_score:.2f} ({compliance_status.upper()})")
        return report
    
    # Validation Methods
    
    async def _validate_basic_metadata(self, tool) -> ValidationResult:
        """Validate basic tool metadata."""
        has_name = hasattr(tool, 'name') and isinstance(tool.name, str) and len(tool.name) > 0
        has_description = hasattr(tool, 'description') and isinstance(tool.description, str) and len(tool.description) > 10
        has_version = hasattr(tool, 'version') and isinstance(tool.version, str)
        
        score = sum([has_name, has_description, has_version]) / 3.0
        
        return ValidationResult(
            requirement_name="has_name_and_description",
            passed=has_name and has_description,
            score=score,
            details={
                "has_name": has_name,
                "has_description": has_description,
                "has_version": has_version,
                "name_length": len(getattr(tool, 'name', '')) if has_name else 0,
                "description_length": len(getattr(tool, 'description', '')) if has_description else 0
            }
        )
    
    async def _validate_capabilities_method(self, tool) -> ValidationResult:
        """Validate capabilities method exists and works."""
        has_method = hasattr(tool, 'get_capabilities')
        is_async = asyncio.iscoroutinefunction(getattr(tool, 'get_capabilities', None))
        
        try:
            if has_method and is_async:
                capabilities = await tool.get_capabilities()
                has_operations = hasattr(capabilities, 'available_operations') and len(capabilities.available_operations) > 0
                valid_structure = all(
                    isinstance(op, dict) and 'name' in op and 'description' in op
                    for op in capabilities.available_operations
                )
            else:
                has_operations = False
                valid_structure = False
        except Exception:
            has_operations = False
            valid_structure = False
        
        score = sum([has_method, is_async, has_operations, valid_structure]) / 4.0
        
        return ValidationResult(
            requirement_name="has_capabilities",
            passed=has_method and is_async and has_operations,
            score=score,
            details={
                "has_method": has_method,
                "is_async": is_async,
                "has_operations": has_operations,
                "valid_structure": valid_structure
            }
        )
    
    async def _validate_execute_method(self, tool) -> ValidationResult:
        """Validate execute method exists and has proper signature."""
        has_method = hasattr(tool, 'execute')
        is_async = asyncio.iscoroutinefunction(getattr(tool, 'execute', None))
        
        # Check method signature
        import inspect
        try:
            if has_method:
                sig = inspect.signature(tool.execute)
                params = list(sig.parameters.keys())
                has_proper_signature = len(params) >= 2  # self, operation, parameters
            else:
                has_proper_signature = False
        except Exception:
            has_proper_signature = False
        
        score = sum([has_method, is_async, has_proper_signature]) / 3.0
        
        return ValidationResult(
            requirement_name="has_execute_method",
            passed=has_method and is_async and has_proper_signature,
            score=score,
            details={
                "has_method": has_method,
                "is_async": is_async,
                "has_proper_signature": has_proper_signature
            }
        )
    
    async def _validate_parameter_validation(self, tool) -> ValidationResult:
        """Validate parameter validation functionality."""
        has_validate_method = hasattr(tool, 'validate_params')
        
        if not has_validate_method:
            return ValidationResult(
                requirement_name="parameter_validation",
                passed=False,
                score=0.0,
                details={"has_validate_method": False}
            )
        
        try:
            # Test validation with empty parameters
            is_valid, error = await tool.validate_params("nonexistent_operation", {})
            validates_empty = isinstance(is_valid, bool) and (error is None or isinstance(error, str))
            
            # Test validation with invalid operation
            is_valid2, error2 = await tool.validate_params("invalid_op_name", {"test": "value"})
            validates_invalid = isinstance(is_valid2, bool)
            
            score = sum([has_validate_method, validates_empty, validates_invalid]) / 3.0
            
            return ValidationResult(
                requirement_name="parameter_validation",
                passed=has_validate_method and validates_empty,
                score=score,
                details={
                    "has_validate_method": has_validate_method,
                    "validates_empty": validates_empty,
                    "validates_invalid": validates_invalid
                }
            )
            
        except Exception as e:
            return ValidationResult(
                requirement_name="parameter_validation",
                passed=False,
                score=0.0,
                error_message=str(e)
            )
    
    async def _validate_error_handling(self, tool) -> ValidationResult:
        """Validate error handling capabilities."""
        try:
            # Test error handling with invalid operation
            result = await tool.execute("nonexistent_operation", {})
            
            has_status = hasattr(result, 'status')
            is_failure = has_status and result.status.value == "failure"
            has_error_message = hasattr(result, 'error_message') and result.error_message is not None
            
            score = sum([has_status, is_failure, has_error_message]) / 3.0
            
            return ValidationResult(
                requirement_name="error_handling",
                passed=is_failure and has_error_message,
                score=score,
                details={
                    "has_status": has_status,
                    "is_failure": is_failure,
                    "has_error_message": has_error_message,
                    "status_value": result.status.value if has_status else None
                }
            )
            
        except Exception as e:
            # If exception is thrown instead of returning error result, that's not ideal
            return ValidationResult(
                requirement_name="error_handling",
                passed=False,
                score=0.3,  # Partial credit for catching the error
                details={"exception_thrown": str(e)}
            )
    
    async def _validate_operation_schemas(self, tool) -> ValidationResult:
        """Validate operation schemas are well-defined."""
        try:
            capabilities = await tool.get_capabilities()
            operations = capabilities.available_operations
            
            valid_operations = 0
            total_operations = len(operations)
            
            for operation in operations:
                # Check required fields
                has_name = isinstance(operation.get('name'), str) and len(operation['name']) > 0
                has_description = isinstance(operation.get('description'), str) and len(operation['description']) > 5
                has_parameters = 'parameters' in operation
                
                if has_name and has_description and has_parameters:
                    valid_operations += 1
            
            score = valid_operations / total_operations if total_operations > 0 else 0.0
            
            return ValidationResult(
                requirement_name="operation_schema",
                passed=score >= 0.8,  # 80% of operations must be valid
                score=score,
                details={
                    "total_operations": total_operations,
                    "valid_operations": valid_operations,
                    "validity_rate": score
                }
            )
            
        except Exception as e:
            return ValidationResult(
                requirement_name="operation_schema",
                passed=False,
                score=0.0,
                error_message=str(e)
            )
    
    async def _validate_async_compliance(self, tool) -> ValidationResult:
        """Validate async/await compliance."""
        execute_is_async = asyncio.iscoroutinefunction(getattr(tool, 'execute', None))
        capabilities_is_async = asyncio.iscoroutinefunction(getattr(tool, 'get_capabilities', None))
        validate_is_async = asyncio.iscoroutinefunction(getattr(tool, 'validate_params', None))
        
        # Check if tool uses proper async patterns
        async_methods = sum([execute_is_async, capabilities_is_async, validate_is_async])
        score = async_methods / 3.0
        
        return ValidationResult(
            requirement_name="async_compliance",
            passed=execute_is_async and capabilities_is_async,
            score=score,
            details={
                "execute_is_async": execute_is_async,
                "capabilities_is_async": capabilities_is_async,
                "validate_is_async": validate_is_async
            }
        )
    
    async def _validate_performance_tracking(self, tool) -> ValidationResult:
        """Validate performance tracking capabilities."""
        has_metrics = hasattr(tool, 'usage_metrics') or hasattr(tool, 'metrics')
        has_timing = hasattr(tool, 'execution_time') or hasattr(tool, 'performance_data')
        
        # Test if tool records metrics during execution
        try:
            capabilities = await tool.get_capabilities()
            # Simple check - any form of performance tracking gets credit
            tracks_performance = has_metrics or has_timing
        except Exception:
            tracks_performance = False
        
        score = sum([has_metrics, has_timing, tracks_performance]) / 3.0
        
        return ValidationResult(
            requirement_name="performance_tracking",
            passed=tracks_performance,
            score=score,
            details={
                "has_metrics": has_metrics,
                "has_timing": has_timing,
                "tracks_performance": tracks_performance
            }
        )
    
    async def _validate_health_checks(self, tool) -> ValidationResult:
        """Validate health checking capabilities."""
        has_health_check = hasattr(tool, 'health_check') or hasattr(tool, 'is_healthy')
        has_status_method = hasattr(tool, 'get_status')
        
        # Test health check if available
        try:
            if has_health_check:
                if hasattr(tool, 'health_check'):
                    health_result = await tool.health_check()
                    health_works = health_result is not None
                else:
                    health_works = tool.is_healthy()
            else:
                health_works = False
        except Exception:
            health_works = False
        
        score = sum([has_health_check, has_status_method, health_works]) / 3.0
        
        return ValidationResult(
            requirement_name="health_checks",
            passed=health_works,
            score=score,
            details={
                "has_health_check": has_health_check,
                "has_status_method": has_status_method,
                "health_works": health_works
            }
        )
    
    async def _validate_agent_compatibility(self, tool) -> ValidationResult:
        """Validate compatibility with AI agents."""
        # Check if tool provides clear operation descriptions
        try:
            capabilities = await tool.get_capabilities()
            operations = capabilities.available_operations
            
            clear_descriptions = sum(
                1 for op in operations 
                if isinstance(op.get('description'), str) and len(op['description']) >= 20
            )
            
            description_score = clear_descriptions / len(operations) if operations else 0
            
            # Check if operations have proper parameter schemas
            has_param_schemas = sum(
                1 for op in operations 
                if 'parameters' in op and isinstance(op['parameters'], dict)
            )
            
            schema_score = has_param_schemas / len(operations) if operations else 0
            
            # Check if tool provides helpful error messages
            error_result = await tool.execute("invalid_operation", {})
            helpful_errors = (
                hasattr(error_result, 'error_message') and 
                error_result.error_message and 
                len(error_result.error_message) > 10
            )
            
            score = (description_score + schema_score + (1.0 if helpful_errors else 0.0)) / 3.0
            
            return ValidationResult(
                requirement_name="agent_compatibility",
                passed=score >= 0.7,
                score=score,
                details={
                    "description_score": description_score,
                    "schema_score": schema_score,
                    "helpful_errors": helpful_errors,
                    "total_operations": len(operations)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                requirement_name="agent_compatibility",
                passed=False,
                score=0.0,
                error_message=str(e)
            )
    
    async def _validate_documentation(self, tool) -> ValidationResult:
        """Validate documentation completeness."""
        has_docstring = bool(tool.__doc__)
        docstring_length = len(tool.__doc__ or "")
        
        # Check if methods have docstrings
        methods_with_docs = 0
        total_methods = 0
        
        for method_name in ['execute', 'get_capabilities', 'validate_params']:
            if hasattr(tool, method_name):
                total_methods += 1
                method = getattr(tool, method_name)
                if method.__doc__ and len(method.__doc__) > 10:
                    methods_with_docs += 1
        
        doc_coverage = methods_with_docs / total_methods if total_methods > 0 else 0
        
        score = (
            (1.0 if has_docstring else 0.0) +
            (1.0 if docstring_length > 50 else 0.0) +
            doc_coverage
        ) / 3.0
        
        return ValidationResult(
            requirement_name="documentation_completeness",
            passed=has_docstring and doc_coverage >= 0.5,
            score=score,
            details={
                "has_docstring": has_docstring,
                "docstring_length": docstring_length,
                "doc_coverage": doc_coverage,
                "methods_documented": methods_with_docs,
                "total_methods": total_methods
            }
        )
    
    async def _validate_security_compliance(self, tool) -> ValidationResult:
        """Validate security compliance."""
        # Check for input validation
        has_validation = hasattr(tool, 'validate_params')
        
        # Check for permission system
        has_permissions = hasattr(tool, 'required_permissions')
        
        # Check for safe execution patterns
        try:
            capabilities = await tool.get_capabilities()
            operations = capabilities.available_operations
            
            # Look for potentially dangerous operations without proper safeguards
            dangerous_patterns = ['delete', 'remove', 'kill', 'execute', 'run']
            dangerous_ops = [
                op for op in operations 
                if any(pattern in op.get('name', '').lower() for pattern in dangerous_patterns)
            ]
            
            has_safeguards = len(dangerous_ops) == 0 or has_validation
            
        except Exception:
            has_safeguards = False
        
        score = sum([has_validation, has_permissions, has_safeguards]) / 3.0
        
        return ValidationResult(
            requirement_name="security_compliance",
            passed=has_validation and has_safeguards,
            score=score,
            details={
                "has_validation": has_validation,
                "has_permissions": has_permissions,
                "has_safeguards": has_safeguards
            }
        )
    
    async def _benchmark_tool_performance(self, tool) -> Dict[str, float]:
        """Benchmark tool performance."""
        metrics = {}
        
        try:
            # Benchmark capabilities call
            start_time = time.time()
            await tool.get_capabilities()
            metrics['capabilities_time'] = time.time() - start_time
            
            # Benchmark validation call
            start_time = time.time()
            await tool.validate_params("test_operation", {})
            metrics['validation_time'] = time.time() - start_time
            
            # Benchmark error handling
            start_time = time.time()
            await tool.execute("nonexistent_operation", {})
            metrics['error_handling_time'] = time.time() - start_time
            
        except Exception:
            pass  # Benchmarking is best effort
        
        return metrics


async def validate_all_tools(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
    """Validate all registered tools."""
    print(f"🔍 Tool Capability Validation Framework")
    print(f"📊 Validation Level: {validation_level.value.upper()}")
    print("=" * 60)
    
    try:
        from core.tools.tool_registry import ToolRegistry
        
        # Initialize registry and get tools
        registry = ToolRegistry.instance()
        await registry.initialize(auto_discover=True)
        tools = registry.get_available_tools()
        
        if not tools:
            print("❌ No tools found for validation")
            return {"success": False, "error": "No tools found"}
        
        validator = ToolCapabilityValidator()
        reports = []
        
        print(f"\n🔧 Validating {len(tools)} tools...")
        
        for tool in tools:
            report = await validator.validate_tool(tool, validation_level)
            reports.append(report)
            
            status_emoji = "✅" if report.compliance_status in ["excellent", "good"] else "⚠️" if report.compliance_status == "acceptable" else "❌"
            print(f"  {status_emoji} {tool.name}: {report.overall_score:.2f} ({report.compliance_status})")
        
        # Generate summary
        total_tools = len(reports)
        excellent_tools = sum(1 for r in reports if r.compliance_status == "excellent")
        good_tools = sum(1 for r in reports if r.compliance_status == "good")
        acceptable_tools = sum(1 for r in reports if r.compliance_status == "acceptable")
        poor_tools = sum(1 for r in reports if r.compliance_status == "needs_improvement")
        
        avg_score = sum(r.overall_score for r in reports) / total_tools if total_tools > 0 else 0
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total Tools: {total_tools}")
        print(f"   Excellent: {excellent_tools} ({excellent_tools/total_tools*100:.1f}%)")
        print(f"   Good: {good_tools} ({good_tools/total_tools*100:.1f}%)")
        print(f"   Acceptable: {acceptable_tools} ({acceptable_tools/total_tools*100:.1f}%)")
        print(f"   Needs Improvement: {poor_tools} ({poor_tools/total_tools*100:.1f}%)")
        print(f"   Average Score: {avg_score:.2f}")
        
        # Overall assessment
        if avg_score >= 0.8 and poor_tools == 0:
            print(f"\n🎉 EXCELLENT! All tools meet validation standards")
            overall_status = "excellent"
        elif avg_score >= 0.7 and poor_tools <= 1:
            print(f"\n✅ GOOD! Most tools meet validation standards")
            overall_status = "good"
        elif avg_score >= 0.6:
            print(f"\n⚠️ ACCEPTABLE! Some tools need improvement")
            overall_status = "acceptable"
        else:
            print(f"\n❌ NEEDS WORK! Many tools require significant improvement")
            overall_status = "needs_improvement"
        
        # Save detailed report
        validation_report = {
            "timestamp": time.time(),
            "validation_level": validation_level.value,
            "summary": {
                "total_tools": total_tools,
                "average_score": avg_score,
                "excellent_tools": excellent_tools,
                "good_tools": good_tools,
                "acceptable_tools": acceptable_tools,
                "poor_tools": poor_tools,
                "overall_status": overall_status
            },
            "tool_reports": [
                {
                    "tool_name": report.tool_name,
                    "overall_score": report.overall_score,
                    "compliance_status": report.compliance_status,
                    "passed_requirements": report.passed_requirements,
                    "total_requirements": report.total_requirements,
                    "performance_metrics": report.performance_metrics,
                    "detailed_results": [
                        {
                            "requirement": result.requirement_name,
                            "passed": result.passed,
                            "score": result.score,
                            "details": result.details,
                            "error": result.error_message
                        }
                        for result in report.results
                    ]
                }
                for report in reports
            ]
        }
        
        report_path = f"tool_validation_report_{validation_level.value}.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"\n📄 Detailed validation report saved to: {report_path}")
        
        return {
            "success": True,
            "overall_status": overall_status,
            "average_score": avg_score,
            "report_path": report_path,
            "summary": validation_report["summary"]
        }
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def main():
    """Main validation runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlutterSwarm Tool Validation Framework")
    parser.add_argument(
        "--level", 
        choices=["basic", "standard", "comprehensive", "production"],
        default="standard",
        help="Validation level (default: standard)"
    )
    
    args = parser.parse_args()
    level = ValidationLevel(args.level)
    
    result = await validate_all_tools(level)
    return result["success"] if isinstance(result, dict) else False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
