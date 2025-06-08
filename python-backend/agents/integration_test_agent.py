"""
Integration Test Agent - Creates end-to-end integration tests.
Handles full app flow testing and user journey validation.
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


class IntegrationTestAgent(BaseAgent):
    """
    Specialized agent for Flutter integration testing.
    Focuses on end-to-end testing, user journeys, and full app flow validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.INTEGRATION_TEST, config)
        
        # Integration test configuration
        self.test_frameworks = {
            "integration_test": "integration_test",
            "patrol": "patrol",
            "flutter_driver": "flutter_driver",
            "mockito": "mockito"
        }
        
        # Test execution settings
        self.test_timeout = config.get("integration_test_timeout", 300)  # 5 minutes
        self.device_configurations = config.get("device_configurations", ["android", "ios"])
        self.test_environments = config.get("test_environments", ["debug", "release"])
        self.parallel_execution = config.get("parallel_execution", False)
        
    def _define_capabilities(self) -> List[str]:
        """Define integration test agent capabilities."""
        return [
            "integration_test_setup",
            "end_to_end_flow_testing",
            "user_journey_automation",
            "cross_platform_testing",
            "performance_integration_testing",
            "api_integration_testing",
            "database_integration_testing",
            "navigation_flow_testing",
            "authentication_flow_testing",
            "data_persistence_testing",
            "offline_scenario_testing",
            "multi_screen_flow_testing"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process integration testing tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'integration_test_generation')
            
            if task_type == "integration_test_generation":
                return await self._handle_integration_test_generation(state)
            elif task_type == "user_journey_testing":
                return await self._handle_user_journey_testing(state)
            elif task_type == "api_integration_testing":
                return await self._handle_api_integration_testing(state)
            elif task_type == "performance_testing":
                return await self._handle_performance_testing(state)
            elif task_type == "integration_test_execution":
                return await self._handle_integration_test_execution(state)
            else:
                return await self._handle_default_integration_testing(state)
                
        except Exception as e:
            logger.error(f"Integration test agent error: {e}")
            return self._create_error_response(f"Integration testing failed: {str(e)}")
    
    async def _handle_integration_test_generation(self, state: WorkflowState) -> AgentResponse:
        """Generate comprehensive integration tests."""
        try:
            project_context = state.project_context
            user_journeys = self.current_task.parameters.get("user_journeys", [])
            
            generated_tests = []
            
            # Generate test for each user journey
            for journey in user_journeys:
                test_content = await self._generate_integration_test_for_journey(journey, project_context)
                if test_content:
                    test_file_path = f"integration_test/{journey['name'].lower().replace(' ', '_')}_test.dart"
                    generated_tests.append({
                        "journey_name": journey['name'],
                        "test_file": test_file_path,
                        "content": test_content
                    })
            
            # Generate general integration tests
            general_tests = await self._generate_general_integration_tests(project_context)
            generated_tests.extend(general_tests)
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Generated {len(generated_tests)} integration test files",
                data={
                    "generated_tests": generated_tests,
                    "test_framework": "integration_test",
                    "device_configurations": self.device_configurations
                },
                metadata={
                    "test_count": len(generated_tests),
                    "framework_used": "integration_test"
                }
            )
            
        except Exception as e:
            logger.error(f"Integration test generation failed: {e}")
            return self._create_error_response(f"Integration test generation failed: {str(e)}")
    
    async def _generate_integration_test_for_journey(self, journey: Dict[str, Any], project_context: ProjectContext) -> str:
        """Generate integration test for a specific user journey."""
        try:
            journey_name = journey.get("name", "Unknown Journey")
            steps = journey.get("steps", [])
            
            test_content = self._generate_integration_test_template(journey_name)
            
            # Add setup
            test_content += f"""
  group('{journey_name} Integration Tests', () {{
    late FlutterDriver driver;

    setUpAll(() async {{
      driver = await FlutterDriver.connect();
    }});

    tearDownAll(() async {{
      if (driver != null) {{
        driver.close();
      }}
    }});

"""
            
            # Generate test for complete journey
            test_content += f"""
    test('should complete {journey_name.lower()} journey successfully', () async {{
      // Test setup
      await driver.waitUntilFirstFrameRasterized();
      
"""
            
            # Add steps
            for i, step in enumerate(steps):
                step_code = self._generate_step_code(step, i)
                test_content += step_code
            
            test_content += """
      // Verify final state
      // Add final assertions here
    });
  });
}
"""
            
            return test_content
            
        except Exception as e:
            logger.error(f"Failed to generate integration test for journey {journey.get('name', 'Unknown')}: {e}")
            return ""
    
    def _generate_integration_test_template(self, journey_name: str) -> str:
        """Generate the basic template for an integration test file."""
        return f"""// Integration test for {journey_name}
import 'package:flutter_driver/flutter_driver.dart';
import 'package:test/test.dart';

void main() {{
"""
    
    def _generate_step_code(self, step: Dict[str, Any], step_number: int) -> str:
        """Generate code for a single test step."""
        action = step.get("action", "")
        target = step.get("target", "")
        value = step.get("value", "")
        
        step_code = f"""
      // Step {step_number + 1}: {step.get('description', action)}
"""
        
        if action == "tap":
            step_code += f"""
      await driver.tap(find.byValueKey('{target}'));
      await driver.waitFor(find.byValueKey('{target}'));
"""
        elif action == "enter_text":
            step_code += f"""
      await driver.tap(find.byValueKey('{target}'));
      await driver.enterText('{value}');
"""
        elif action == "scroll":
            step_code += f"""
      await driver.scroll(find.byValueKey('{target}'), 0, -300, Duration(milliseconds: 300));
"""
        elif action == "wait_for":
            step_code += f"""
      await driver.waitFor(find.byValueKey('{target}'));
"""
        elif action == "verify":
            step_code += f"""
      expect(await driver.getText(find.byValueKey('{target}')), '{value}');
"""
        else:
            step_code += f"""
      // Custom action: {action}
      // TODO: Implement custom action logic
"""
        
        return step_code
    
    async def _generate_general_integration_tests(self, project_context: ProjectContext) -> List[Dict[str, Any]]:
        """Generate general integration tests for common scenarios."""
        general_tests = []
        
        # App startup test
        startup_test_content = self._generate_app_startup_test()
        general_tests.append({
            "journey_name": "App Startup",
            "test_file": "integration_test/app_startup_test.dart",
            "content": startup_test_content
        })
        
        # Navigation test
        navigation_test_content = self._generate_navigation_test()
        general_tests.append({
            "journey_name": "Navigation Flow",
            "test_file": "integration_test/navigation_test.dart",
            "content": navigation_test_content
        })
        
        # Performance test
        performance_test_content = self._generate_performance_test()
        general_tests.append({
            "journey_name": "Performance Metrics",
            "test_file": "integration_test/performance_test.dart",
            "content": performance_test_content
        })
        
        return general_tests
    
    def _generate_app_startup_test(self) -> str:
        """Generate app startup integration test."""
        return """// App startup integration test
import 'package:flutter_driver/flutter_driver.dart';
import 'package:test/test.dart';

void main() {
  group('App Startup Tests', () {
    late FlutterDriver driver;

    setUpAll(() async {
      driver = await FlutterDriver.connect();
    });

    tearDownAll(() async {
      if (driver != null) {
        driver.close();
      }
    });

    test('should start app and show main screen', () async {
      // Wait for app to start
      await driver.waitUntilFirstFrameRasterized();
      
      // Verify main screen is loaded
      await driver.waitFor(find.byType('Scaffold'));
      
      // Take screenshot for verification
      final pixels = await driver.screenshot();
      expect(pixels, isNotNull);
    });

    test('should handle app lifecycle changes', () async {
      // Test app backgrounding and foregrounding
      await driver.waitUntilFirstFrameRasterized();
      
      // Background the app (platform specific)
      // await driver.requestData('background_app');
      // await Future.delayed(Duration(seconds: 2));
      // await driver.requestData('foreground_app');
      
      // Verify app state is maintained
      await driver.waitFor(find.byType('Scaffold'));
    });
  });
}
"""
    
    def _generate_navigation_test(self) -> str:
        """Generate navigation flow integration test."""
        return """// Navigation flow integration test
import 'package:flutter_driver/flutter_driver.dart';
import 'package:test/test.dart';

void main() {
  group('Navigation Flow Tests', () {
    late FlutterDriver driver;

    setUpAll(() async {
      driver = await FlutterDriver.connect();
    });

    tearDownAll(() async {
      if (driver != null) {
        driver.close();
      }
    });

    test('should navigate through main app screens', () async {
      await driver.waitUntilFirstFrameRasterized();
      
      // Navigate to different screens
      // Add navigation steps based on your app structure
      
      // Example navigation flow:
      // 1. From home to profile
      final profileButton = find.byValueKey('profile_button');
      if (await driver.waitFor(profileButton, timeout: Duration(seconds: 2))) {
        await driver.tap(profileButton);
        await driver.waitFor(find.text('Profile'));
      }
      
      // 2. Navigate back
      await driver.tap(find.pageBack());
      await driver.waitFor(find.byType('Scaffold'));
    });

    test('should handle deep links correctly', () async {
      // Test deep link navigation
      await driver.requestData('deep_link:/profile/123');
      await driver.waitFor(find.text('Profile'));
    });

    test('should maintain navigation stack', () async {
      await driver.waitUntilFirstFrameRasterized();
      
      // Build navigation stack
      // Navigate to screen 1
      // Navigate to screen 2
      // Navigate to screen 3
      
      // Test back navigation
      await driver.tap(find.pageBack());
      // Verify we're on screen 2
      
      await driver.tap(find.pageBack());
      // Verify we're on screen 1
    });
  });
}
"""
    
    def _generate_performance_test(self) -> str:
        """Generate performance integration test."""
        return """// Performance integration test
import 'package:flutter_driver/flutter_driver.dart';
import 'package:test/test.dart';

void main() {
  group('Performance Tests', () {
    late FlutterDriver driver;

    setUpAll(() async {
      driver = await FlutterDriver.connect();
    });

    tearDownAll(() async {
      if (driver != null) {
        driver.close();
      }
    });

    test('should meet performance benchmarks', () async {
      await driver.waitUntilFirstFrameRasterized();
      
      // Enable performance tracking
      final timeline = await driver.traceAction(() async {
        // Perform actions that need performance testing
        await driver.tap(find.byValueKey('performance_test_button'));
        await driver.waitFor(find.text('Performance Test Complete'));
      });
      
      // Analyze timeline
      final summary = new TimelineSummary.summarize(timeline);
      
      // Assert performance metrics
      expect(summary.averageFrameBuildTimeMillis, lessThan(16.0));
      expect(summary.worstFrameBuildTimeMillis, lessThan(100.0));
      expect(summary.missedFrameBuildBudgetCount, equals(0));
    });

    test('should handle memory efficiently', () async {
      await driver.waitUntilFirstFrameRasterized();
      
      // Monitor memory usage during operations
      final memoryInfo = await driver.requestData('memory_info');
      final initialMemory = double.parse(memoryInfo);
      
      // Perform memory-intensive operations
      for (int i = 0; i < 10; i++) {
        await driver.tap(find.byValueKey('create_heavy_widget'));
        await driver.waitFor(find.byValueKey('heavy_widget_$i'));
      }
      
      final finalMemoryInfo = await driver.requestData('memory_info');
      final finalMemory = double.parse(finalMemoryInfo);
      
      // Assert memory growth is reasonable
      expect(finalMemory - initialMemory, lessThan(50.0)); // Less than 50MB growth
    });
  });
}
"""
    
    async def _handle_user_journey_testing(self, state: WorkflowState) -> AgentResponse:
        """Handle specific user journey testing."""
        try:
            journeys = self.current_task.parameters.get("journeys", [])
            results = []
            
            for journey in journeys:
                test_result = await self._execute_user_journey_test(journey)
                results.append(test_result)
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED,
                content=f"Completed testing {len(journeys)} user journeys",
                data={
                    "journey_results": results,
                    "success_rate": sum(1 for r in results if r["passed"]) / len(results) * 100
                }
            )
            
        except Exception as e:
            logger.error(f"User journey testing failed: {e}")
            return self._create_error_response(f"User journey testing failed: {str(e)}")
    
    async def _execute_user_journey_test(self, journey: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single user journey test."""
        try:
            journey_name = journey.get("name", "Unknown Journey")
            
            # Execute the journey test
            cmd = ["flutter", "drive", "--target=integration_test/app.dart", 
                   "--driver=test_driver/integration_test.dart"]
            
            result = await self._run_command(cmd)
            
            return {
                "journey_name": journey_name,
                "passed": result.returncode == 0,
                "execution_time": 0,  # Would be calculated from actual execution
                "errors": [] if result.returncode == 0 else [result.stderr]
            }
            
        except Exception as e:
            logger.error(f"Failed to execute journey test {journey.get('name', 'Unknown')}: {e}")
            return {
                "journey_name": journey.get("name", "Unknown"),
                "passed": False,
                "execution_time": 0,
                "errors": [str(e)]
            }
    
    async def _handle_integration_test_execution(self, state: WorkflowState) -> AgentResponse:
        """Execute integration tests and return results."""
        try:
            test_target = self.current_task.parameters.get("test_target", "integration_test/app.dart")
            device_config = self.current_task.parameters.get("device", "android")
            
            # Execute integration tests
            cmd = [
                "flutter", "drive",
                f"--target={test_target}",
                "--driver=test_driver/integration_test.dart",
                f"--device-id={device_config}"
            ]
            
            result = await self._run_command(cmd)
            
            # Parse results
            test_results = self._parse_integration_test_results(result.stdout)
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=self.current_task.task_id,
                status=TaskStatus.COMPLETED if result.returncode == 0 else TaskStatus.FAILED,
                content=f"Integration tests completed: {test_results['passed']} passed, {test_results['failed']} failed",
                data={
                    "test_results": test_results,
                    "exit_code": result.returncode,
                    "device_config": device_config,
                    "execution_time": test_results.get("execution_time", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Integration test execution failed: {e}")
            return self._create_error_response(f"Integration test execution failed: {str(e)}")
    
    def _parse_integration_test_results(self, output: str) -> Dict[str, Any]:
        """Parse integration test execution output."""
        import re
        
        # Parse test results from flutter drive output
        passed_match = re.search(r'All tests passed!', output)
        failed_match = re.search(r'(\d+) test[s]? failed', output)
        
        return {
            "passed": 1 if passed_match else 0,
            "failed": int(failed_match.group(1)) if failed_match else 0,
            "total": 1,
            "execution_time": 0  # Would extract from actual output
        }
    
    async def _handle_default_integration_testing(self, state: WorkflowState) -> AgentResponse:
        """Handle default integration testing workflow."""
        try:
            # Default workflow: generate and execute integration tests
            generation_result = await self._handle_integration_test_generation(state)
            
            if generation_result.status == TaskStatus.FAILED:
                return generation_result
            
            # Execute the generated tests
            execution_result = await self._handle_integration_test_execution(state)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Default integration testing failed: {e}")
            return self._create_error_response(f"Integration testing workflow failed: {str(e)}")
