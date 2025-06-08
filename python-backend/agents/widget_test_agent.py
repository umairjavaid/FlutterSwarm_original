"""
Widget Test Agent - Creates Flutter widget tests for UI components.
Specializes in testing widget behavior, interactions, and rendering.
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


class WidgetTestAgent(BaseAgent):
    """
    Specialized agent for Flutter widget testing.
    Focuses on testing UI components, user interactions, and widget behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.WIDGET_TEST, config)
        
        # Widget test configuration
        self.test_frameworks = {
            "flutter_test": "flutter_test",
            "mockito": "mockito",
            "golden_toolkit": "golden_toolkit",
            "patrol": "patrol"
        }
        
        # Test generation settings
        self.generate_golden_tests = config.get("generate_golden_tests", True)
        self.test_interactions = config.get("test_interactions", True)
        self.test_accessibility = config.get("test_accessibility", True)
        self.golden_threshold = config.get("golden_threshold", 0.0)
        
    def _define_capabilities(self) -> List[str]:
        """Define widget test agent capabilities."""
        return [
            "widget_test_creation",
            "ui_interaction_testing",
            "widget_finder_implementation",
            "gesture_simulation",
            "widget_state_testing",
            "accessibility_testing",
            "golden_file_testing",
            "responsive_ui_testing",
            "animation_testing",
            "theme_testing",
            "localization_testing",
            "custom_widget_testing"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process widget testing tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'widget_test_generation')
            
            if task_type == "widget_test_generation":
                return await self._handle_widget_test_generation(state)
            elif task_type == "golden_test_generation":
                return await self._handle_golden_test_generation(state)
            elif task_type == "interaction_test_generation":
                return await self._handle_interaction_test_generation(state)
            elif task_type == "accessibility_test_generation":
                return await self._handle_accessibility_test_generation(state)
            elif task_type == "widget_test_execution":
                return await self._handle_widget_test_execution(state)
            else:
                return await self._handle_default_widget_testing(state)
                
        except Exception as e:
            logger.error(f"Widget test agent error: {e}")
            return self._create_error_response(f"Widget testing failed: {str(e)}")
    
    async def _handle_widget_test_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate comprehensive widget tests."""
        try:
            project_context = state.project_context
            target_widgets = self.current_task.parameters.get("target_widgets", [])
            
            generated_tests = []
            
            for widget_file in target_widgets:
                if widget_file.endswith('.dart') and not widget_file.endswith('_test.dart'):
                    test_content = await self._generate_widget_test_for_file(widget_file, project_context)
                    if test_content:
                        test_file_path = self._get_widget_test_file_path(widget_file)
                        generated_tests.append({
                            "source_file": widget_file,
                            "test_file": test_file_path,
                            "content": test_content
                        })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_tests)} widget test files",
                data={
                    "generated_tests": generated_tests,
                    "test_framework": "flutter_test",
                    "includes_golden_tests": self.generate_golden_tests,
                    "includes_accessibility_tests": self.test_accessibility
                },
                metadata={
                    "test_count": len(generated_tests),
                    "framework_used": "flutter_test"
                }
            )
            
        except Exception as e:
            logger.error(f"Widget test generation failed: {e}")
            return self._create_error_response(f"Widget test generation failed: {str(e)}")
    
    async def _generate_widget_test_for_file(self, widget_file: str, project_context: ProjectContext) -> str:
        """Generate widget test content for a specific widget file."""
        try:
            # Read the widget file
            with open(widget_file, 'r') as f:
                source_content = f.read()
            
            # Analyze the widget file to identify widgets
            widgets = self._extract_widgets_from_dart(source_content)
            
            # Generate test content
            test_content = self._generate_widget_test_template(widget_file)
            
            # Add tests for each widget
            for widget_info in widgets:
                widget_tests = await self._generate_widget_tests(widget_info, project_context)
                test_content += widget_tests
            
            test_content += "\n  });\n}\n"
            
            return test_content
            
        except Exception as e:
            logger.error(f"Failed to generate widget test for {widget_file}: {e}")
            return ""
    
    def _generate_widget_test_template(self, source_file_path: str) -> str:
        """Generate the basic template for a widget test file."""
        import_path = self._get_import_path(source_file_path)
        
        return f"""// Widget tests for {source_file_path}
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';

import '{import_path}';

void main() {{
  group('{Path(source_file_path).stem} Widget Tests', () {{
    
"""
    
    async def _generate_widget_tests(self, widget_info: Dict[str, Any], project_context: ProjectContext) -> str:
        """Generate comprehensive tests for a widget."""
        widget_name = widget_info.get("name", "UnknownWidget")
        widget_type = widget_info.get("type", "StatelessWidget")
        
        test_content = f"""
    group('{widget_name} Tests', () {{
      testWidgets('should render {widget_name} correctly', (WidgetTester tester) async {{
        // Arrange
        const widget = {widget_name}();
        
        // Act
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: widget,
            ),
          ),
        );
        
        // Assert
        expect(find.byType({widget_name}), findsOneWidget);
      }});

"""
        
        # Add state-specific tests for StatefulWidgets
        if widget_type == "StatefulWidget":
            test_content += self._generate_stateful_widget_tests(widget_name)
        
        # Add interaction tests
        if self.test_interactions:
            test_content += self._generate_interaction_tests(widget_name, widget_info)
        
        # Add accessibility tests
        if self.test_accessibility:
            test_content += self._generate_accessibility_tests(widget_name)
        
        # Add golden tests
        if self.generate_golden_tests:
            test_content += self._generate_golden_tests(widget_name)
        
        test_content += "    });\n"
        return test_content
    
    def _generate_stateful_widget_tests(self, widget_name: str) -> str:
        """Generate tests specific to StatefulWidgets."""
        return f"""
      testWidgets('should manage state correctly in {widget_name}', (WidgetTester tester) async {{
        // Arrange
        const widget = {widget_name}();
        
        // Act
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: widget,
            ),
          ),
        );
        
        // Initial state test
        // Add specific state assertions here
        
        // State change test
        // Add interactions that change state
        // await tester.tap(find.byType(SomeButton));
        // await tester.pump();
        
        // Assert state change
        // Add assertions for changed state
      }});

"""
    
    def _generate_interaction_tests(self, widget_name: str, widget_info: Dict[str, Any]) -> str:
        """Generate user interaction tests."""
        return f"""
      testWidgets('should handle user interactions in {widget_name}', (WidgetTester tester) async {{
        // Arrange
        const widget = {widget_name}();
        
        // Act
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: widget,
            ),
          ),
        );
        
        // Test tap interactions
        final tapTargets = find.byType(GestureDetector);
        if (tapTargets.evaluate().isNotEmpty) {{
          await tester.tap(tapTargets.first);
          await tester.pump();
        }}
        
        // Test button interactions
        final buttons = find.byType(ElevatedButton);
        if (buttons.evaluate().isNotEmpty) {{
          await tester.tap(buttons.first);
          await tester.pump();
        }}
        
        // Test text field interactions
        final textFields = find.byType(TextField);
        if (textFields.evaluate().isNotEmpty) {{
          await tester.enterText(textFields.first, 'Test input');
          await tester.pump();
        }}
        
        // Assert interaction results
        // Add specific assertions based on expected behavior
      }});

"""
    
    def _generate_accessibility_tests(self, widget_name: str) -> str:
        """Generate accessibility tests."""
        return f"""
      testWidgets('should be accessible in {widget_name}', (WidgetTester tester) async {{
        // Arrange
        const widget = {widget_name}();
        
        // Act
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: widget,
            ),
          ),
        );
        
        // Assert accessibility
        final SemanticsHandle handle = tester.ensureSemantics();
        
        // Check for semantic labels
        expect(tester.getSemantics(find.byType({widget_name})), 
               matchesSemantics());
        
        // Test screen reader compatibility
        // Add specific semantic assertions
        
        handle.dispose();
      }});

"""
    
    def _generate_golden_tests(self, widget_name: str) -> str:
        """Generate golden file tests for visual regression testing."""
        return f"""
      testWidgets('should match golden file for {widget_name}', (WidgetTester tester) async {{
        // Arrange
        const widget = {widget_name}();
        
        // Act
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: widget,
            ),
          ),
        );
        
        // Assert golden file match
        await expectLater(
          find.byType({widget_name}),
          matchesGoldenFile('goldens/{widget_name.toLowerCase()}.png'),
        );
      }});

"""
    
    async def _handle_golden_test_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate golden file tests specifically."""
        try:
            target_widgets = self.current_task.parameters.get("target_widgets", [])
            generated_golden_tests = []
            
            for widget_file in target_widgets:
                widgets = self._extract_widgets_from_file(widget_file)
                
                for widget_info in widgets:
                    widget_name = widget_info.get("name")
                    golden_test = self._generate_comprehensive_golden_test(widget_name, widget_info)
                    
                    generated_golden_tests.append({
                        "widget_name": widget_name,
                        "test_content": golden_test,
                        "golden_file": f"test/goldens/{widget_name.lower()}.png"
                    })
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_golden_tests)} golden tests",
                data={
                    "generated_golden_tests": generated_golden_tests,
                    "threshold": self.golden_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Golden test generation failed: {e}")
            return self._create_error_response(f"Golden test generation failed: {str(e)}")
    
    def _generate_comprehensive_golden_test(self, widget_name: str, widget_info: Dict[str, Any]) -> str:
        """Generate comprehensive golden tests with different states/themes."""
        return f"""
    group('{widget_name} Golden Tests', () {{
      testWidgets('should match golden file - default theme', (WidgetTester tester) async {{
        await tester.pumpWidget(
          MaterialApp(
            theme: ThemeData.light(),
            home: Scaffold(body: const {widget_name}()),
          ),
        );
        
        await expectLater(
          find.byType({widget_name}),
          matchesGoldenFile('goldens/{widget_name.lower()}_light.png'),
        );
      }});

      testWidgets('should match golden file - dark theme', (WidgetTester tester) async {{
        await tester.pumpWidget(
          MaterialApp(
            theme: ThemeData.dark(),
            home: Scaffold(body: const {widget_name}()),
          ),
        );
        
        await expectLater(
          find.byType({widget_name}),
          matchesGoldenFile('goldens/{widget_name.lower()}_dark.png'),
        );
      }});
    }});

"""
    
    async def _handle_widget_test_execution(self, state: WorkflowState) -> AgentResponse:
        """Execute widget tests and return results."""
        try:
            test_files = self.current_task.parameters.get("test_files", [])
            
            if not test_files:
                # Run all widget tests
                cmd = ["flutter", "test", "test/widget/"]
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
                content=f"Widget tests completed: {test_results['passed']} passed, {test_results['failed']} failed",
                data={
                    "test_results": test_results,
                    "exit_code": result.returncode,
                    "output": result.stdout
                }
            )
            
        except Exception as e:
            logger.error(f"Widget test execution failed: {e}")
            return self._create_error_response(f"Widget test execution failed: {str(e)}")
    
    def _extract_widgets_from_dart(self, content: str) -> List[Dict[str, Any]]:
        """Extract widget information from Dart source code."""
        import re
        
        widgets = []
        
        # Pattern to match widget class declarations
        widget_pattern = r'class\s+(\w+)\s+extends\s+(StatelessWidget|StatefulWidget|InheritedWidget|RenderObjectWidget)'
        matches = re.finditer(widget_pattern, content)
        
        for match in matches:
            widget_name = match.group(1)
            widget_type = match.group(2)
            
            widgets.append({
                "name": widget_name,
                "type": widget_type,
                "methods": [],  # Could be enhanced with method extraction
                "properties": []
            })
        
        return widgets
    
    def _extract_widgets_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract widgets from a file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return self._extract_widgets_from_dart(content)
        except Exception as e:
            logger.error(f"Failed to extract widgets from {file_path}: {e}")
            return []
    
    def _get_widget_test_file_path(self, source_file: str) -> str:
        """Get the corresponding widget test file path."""
        path = Path(source_file)
        return f"test/widget/{path.stem}_test.dart"
    
    def _get_import_path(self, source_file: str) -> str:
        """Get the import path for a source file."""
        path = Path(source_file)
        if str(path).startswith("lib/"):
            return str(path)[4:]  # Remove 'lib/' prefix
        return str(path)
    
    def _parse_test_results(self, output: str) -> Dict[str, Any]:
        """Parse widget test execution output."""
        import re
        
        passed_match = re.search(r'(\d+) tests? passed', output)
        failed_match = re.search(r'(\d+) tests? failed', output)
        
        return {
            "passed": int(passed_match.group(1)) if passed_match else 0,
            "failed": int(failed_match.group(1)) if failed_match else 0,
            "total": 0
        }
    
    async def _handle_default_widget_testing(self, state: WorkflowState) -> AgentResponse:
        """Handle default widget testing workflow."""
        try:
            # Default workflow: generate widget tests and run them
            generation_result = await self._handle_widget_test_generation(state)
            
            if generation_result.status == TaskStatus.FAILED:
                return generation_result
            
            # Run the generated tests
            execution_result = await self._handle_widget_test_execution(state)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Default widget testing failed: {e}")
            return self._create_error_response(f"Widget testing workflow failed: {str(e)}")
