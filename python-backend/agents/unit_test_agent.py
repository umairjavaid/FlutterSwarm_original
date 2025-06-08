"""
Unit Test Agent - Creates pure Dart unit tests for business logic.
Specializes in testing functions, classes, and business logic in isolation.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.agent_types import (
    AgentType, AgentMessage, MessageType, TaskDefinition, TaskStatus,
    AgentResponse, Priority, ProjectContext, WorkflowState
)
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class UnitTestAgent(BaseAgent):
    """
    Specialized agent for Dart unit testing.
    Focuses on testing business logic, models, services, and utilities in isolation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.UNIT_TEST, config)
        
        # Unit test configuration
        self.test_frameworks = {
            "test": "test",
            "mockito": "mockito", 
            "mocktail": "mocktail",
            "fake_async": "fake_async"
        }
        
        # Test generation settings
        self.coverage_threshold = config.get("unit_coverage_threshold", 90.0)
        self.generate_mocks = config.get("generate_mocks", True)
        self.test_data_generation = config.get("test_data_generation", True)
        
    def _define_capabilities(self) -> List[str]:
        """Define unit test agent capabilities."""
        return [
            "unit_test_generation",
            "mock_object_creation",
            "test_data_generation",
            "business_logic_testing",
            "repository_testing", 
            "service_testing",
            "edge_case_testing",
            "test_coverage_optimization",
            "dart_test_execution",
            "test_suite_organization",
            "parametrized_test_creation",
            "exception_testing"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process unit testing tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'unit_test_generation')
            
            if task_type == "unit_test_generation":
                return await self._handle_unit_test_generation(state)
            elif task_type == "mock_generation":
                return await self._handle_mock_generation(state)
            elif task_type == "test_execution":
                return await self._handle_test_execution(state)
            elif task_type == "coverage_analysis":
                return await self._handle_coverage_analysis(state)
            else:
                return await self._handle_default_unit_testing(state)
                
        except Exception as e:
            logger.error(f"Unit test agent error: {e}")
            return self._create_error_response(f"Unit testing failed: {str(e)}")
    
    async def _handle_unit_test_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate unit tests for Dart classes and functions."""
        try:
            project_context = state.project_context
            target_files = self.current_task.parameters.get("target_files", [])
            
            generated_tests = []
            
            for file_path in target_files:
                if file_path.endswith('.dart') and not file_path.endswith('_test.dart'):
                    test_content = await self._generate_unit_test_for_file(file_path, project_context)
                    if test_content:
                        test_file_path = self._get_test_file_path(file_path)
                        generated_tests.append({
                            "source_file": file_path,
                            "test_file": test_file_path,
                            "content": test_content
                        })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_tests)} unit test files",
                data={
                    "generated_tests": generated_tests,
                    "test_framework": "test",
                    "coverage_target": self.coverage_threshold
                },
                metadata={
                    "test_count": len(generated_tests),
                    "framework_used": "dart_test"
                }
            )
            
        except Exception as e:
            logger.error(f"Unit test generation failed: {e}")
            return self._create_error_response(f"Unit test generation failed: {str(e)}")
    
    async def _generate_unit_test_for_file(self, file_path: str, project_context: ProjectContext) -> str:
        """Generate unit test content for a specific Dart file."""
        try:
            # Read the source file
            with open(file_path, 'r') as f:
                source_content = f.read()
            
            # Analyze the source code to identify testable elements
            classes = self._extract_classes_from_dart(source_content)
            functions = self._extract_functions_from_dart(source_content)
            
            # Generate test content
            test_content = self._generate_test_file_template(file_path)
            
            # Add tests for each class
            for class_info in classes:
                class_tests = await self._generate_class_tests(class_info, project_context)
                test_content += class_tests
            
            # Add tests for standalone functions
            for function_info in functions:
                function_tests = await self._generate_function_tests(function_info, project_context)
                test_content += function_tests
            
            return test_content
            
        except Exception as e:
            logger.error(f"Failed to generate unit test for {file_path}: {e}")
            return ""
    
    def _generate_test_file_template(self, source_file_path: str) -> str:
        """Generate the basic template for a unit test file."""
        import_path = self._get_import_path(source_file_path)
        
        return f"""// Unit tests for {source_file_path}
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';

import '{import_path}';

// Generate mocks
@GenerateMocks([])
void main() {{
  group('{Path(source_file_path).stem} Unit Tests', () {{
    setUp(() {{
      // Setup code
    }});

    tearDown(() {{
      // Cleanup code
    }});

"""
    
    async def _generate_class_tests(self, class_info: Dict[str, Any], project_context: ProjectContext) -> str:
        """Generate unit tests for a Dart class."""
        class_name = class_info.get("name", "UnknownClass")
        methods = class_info.get("methods", [])
        properties = class_info.get("properties", [])
        
        test_content = f"""
    group('{class_name} Tests', () {{
      late {class_name} {class_name.lower()};
      
      setUp(() {{
        {class_name.lower()} = {class_name}();
      }});

"""
        
        # Generate tests for each method
        for method in methods:
            method_name = method.get("name", "")
            if method_name and not method_name.startswith("_"):  # Skip private methods
                test_content += f"""
      test('should {method_name} correctly', () {{
        // Arrange
        
        // Act
        final result = {class_name.lower()}.{method_name}();
        
        // Assert
        expect(result, isNotNull);
      }});

"""
        
        # Generate property tests
        for property in properties:
            prop_name = property.get("name", "")
            if prop_name and not prop_name.startswith("_"):
                test_content += f"""
      test('should set and get {prop_name}', () {{
        // Arrange
        const testValue = 'test';
        
        // Act
        {class_name.lower()}.{prop_name} = testValue;
        
        // Assert
        expect({class_name.lower()}.{prop_name}, equals(testValue));
      }});

"""
        
        test_content += "    });\n"
        return test_content
    
    async def _generate_function_tests(self, function_info: Dict[str, Any], project_context: ProjectContext) -> str:
        """Generate unit tests for standalone functions."""
        function_name = function_info.get("name", "")
        
        if not function_name or function_name.startswith("_"):
            return ""
        
        return f"""
    group('{function_name} Function Tests', () {{
      test('should {function_name} with valid input', () {{
        // Arrange
        
        // Act
        final result = {function_name}();
        
        // Assert
        expect(result, isNotNull);
      }});

      test('should handle edge cases for {function_name}', () {{
        // Arrange
        
        // Act & Assert
        expect(() => {function_name}(), returnsNormally);
      }});
    }});

"""
    
    async def _handle_mock_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate mock objects for testing."""
        try:
            target_classes = self.current_task.parameters.get("target_classes", [])
            generated_mocks = []
            
            for class_name in target_classes:
                mock_content = self._generate_mock_class(class_name)
                generated_mocks.append({
                    "class_name": class_name,
                    "mock_content": mock_content
                })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_mocks)} mock classes",
                data={
                    "generated_mocks": generated_mocks,
                    "framework": "mockito"
                }
            )
            
        except Exception as e:
            logger.error(f"Mock generation failed: {e}")
            return self._create_error_response(f"Mock generation failed: {str(e)}")
    
    def _generate_mock_class(self, class_name: str) -> str:
        """Generate a mock class using Mockito."""
        return f"""
class Mock{class_name} extends Mock implements {class_name} {{}}
"""
    
    async def _handle_test_execution(self, state: WorkflowState) -> AgentResponse:
        """Execute unit tests and return results."""
        try:
            test_files = self.current_task.parameters.get("test_files", [])
            
            if not test_files:
                # Run all unit tests
                cmd = ["flutter", "test", "test/unit/"]
            else:
                # Run specific test files
                cmd = ["flutter", "test"] + test_files
            
            result = await self._run_command(cmd)
            
            # Parse test results
            test_results = self._parse_test_results(result.stdout)
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED if result.returncode == 0 else TaskStatus.FAILED,
                content=f"Unit tests completed: {test_results['passed']} passed, {test_results['failed']} failed",
                data={
                    "test_results": test_results,
                    "exit_code": result.returncode,
                    "output": result.stdout
                }
            )
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return self._create_error_response(f"Test execution failed: {str(e)}")
    
    async def _handle_coverage_analysis(self, state: WorkflowState) -> AgentResponse:
        """Analyze test coverage and generate coverage reports."""
        try:
            # Run tests with coverage
            cmd = ["flutter", "test", "--coverage"]
            result = await self._run_command(cmd)
            
            if result.returncode != 0:
                return self._create_error_response("Coverage analysis failed during test execution")
            
            # Generate coverage report
            coverage_cmd = ["genhtml", "coverage/lcov.info", "-o", "coverage/html"]
            coverage_result = await self._run_command(coverage_cmd)
            
            # Parse coverage data
            coverage_data = await self._parse_coverage_data()
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Coverage analysis completed: {coverage_data['overall_coverage']:.1f}%",
                data={
                    "coverage_data": coverage_data,
                    "threshold_met": coverage_data['overall_coverage'] >= self.coverage_threshold,
                    "html_report": "coverage/html/index.html"
                }
            )
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return self._create_error_response(f"Coverage analysis failed: {str(e)}")
    
    def _extract_classes_from_dart(self, content: str) -> List[Dict[str, Any]]:
        """Extract class information from Dart source code."""
        # Simple regex-based extraction (could be enhanced with proper AST parsing)
        import re
        
        classes = []
        class_pattern = r'class\s+(\w+).*?\{'
        matches = re.finditer(class_pattern, content)
        
        for match in matches:
            class_name = match.group(1)
            classes.append({
                "name": class_name,
                "methods": [],  # Would need more sophisticated parsing
                "properties": []
            })
        
        return classes
    
    def _extract_functions_from_dart(self, content: str) -> List[Dict[str, Any]]:
        """Extract function information from Dart source code."""
        import re
        
        functions = []
        function_pattern = r'^(?!.*class).*?(\w+)\s*\([^)]*\)\s*(?:async\s*)?{'
        matches = re.finditer(function_pattern, content, re.MULTILINE)
        
        for match in matches:
            function_name = match.group(1)
            if function_name not in ['if', 'for', 'while', 'switch']:  # Filter out control structures
                functions.append({
                    "name": function_name
                })
        
        return functions
    
    def _get_test_file_path(self, source_file: str) -> str:
        """Get the corresponding test file path for a source file."""
        path = Path(source_file)
        return f"test/unit/{path.stem}_test.dart"
    
    def _get_import_path(self, source_file: str) -> str:
        """Get the import path for a source file."""
        path = Path(source_file)
        # Convert file path to package import
        if str(path).startswith("lib/"):
            return str(path)[4:]  # Remove 'lib/' prefix
        return str(path)
    
    def _parse_test_results(self, output: str) -> Dict[str, Any]:
        """Parse test execution output to extract results."""
        import re
        
        # Look for test summary in output
        passed_match = re.search(r'(\d+) tests? passed', output)
        failed_match = re.search(r'(\d+) tests? failed', output)
        
        return {
            "passed": int(passed_match.group(1)) if passed_match else 0,
            "failed": int(failed_match.group(1)) if failed_match else 0,
            "total": 0  # Would be calculated from passed + failed + skipped
        }
    
    async def _parse_coverage_data(self) -> Dict[str, Any]:
        """Parse coverage data from lcov.info file."""
        try:
            coverage_file = Path("coverage/lcov.info")
            if not coverage_file.exists():
                return {"overall_coverage": 0.0, "files": []}
            
            with open(coverage_file, 'r') as f:
                content = f.read()
            
            # Parse LCOV format (simplified)
            lines_found = content.count("LF:")
            lines_hit = content.count("LH:")
            
            overall_coverage = (lines_hit / lines_found * 100) if lines_found > 0 else 0.0
            
            return {
                "overall_coverage": overall_coverage,
                "lines_found": lines_found,
                "lines_hit": lines_hit,
                "files": []  # Would contain per-file coverage data
            }
            
        except Exception as e:
            logger.error(f"Failed to parse coverage data: {e}")
            return {"overall_coverage": 0.0, "files": []}
    
    async def _handle_default_unit_testing(self, state: WorkflowState) -> AgentResponse:
        """Handle default unit testing workflow."""
        try:
            # Default workflow: generate tests, run them, analyze coverage
            generation_result = await self._handle_unit_test_generation(state)
            
            if generation_result.status == TaskStatus.FAILED:
                return generation_result
            
            # Run the generated tests
            execution_result = await self._handle_test_execution(state)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Default unit testing failed: {e}")
            return self._create_error_response(f"Unit testing workflow failed: {str(e)}")
