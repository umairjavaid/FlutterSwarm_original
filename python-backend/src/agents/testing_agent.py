"""
Testing Agent for FlutterSwarm Multi-Agent System.

This agent specializes in test creation, quality assurance,
and testing strategy for Flutter applications.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..config import get_logger

logger = get_logger("testing_agent")


class TestingAgent(BaseAgent):
    """
    Specialized agent for Flutter application testing and quality assurance.
    
    This agent handles:
    - Unit test creation and optimization
    - Widget test development and maintenance
    - Integration test planning and execution
    - Test automation and CI/CD integration
    - Code coverage analysis and improvement
    - Security testing and vulnerability assessment
    - Test strategy planning and best practices
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        # Override config for testing-specific settings
        testing_config = AgentConfig(
            agent_id=config.agent_id or f"testing_agent_{str(uuid.uuid4())[:8]}",
            agent_type="testing",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.TESTING,
                AgentCapability.QUALITY_ASSURANCE
            ],
            max_concurrent_tasks=4,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.3,  # Moderate temperature for balanced test generation
            max_tokens=6000,
            timeout=600,
            metadata=config.metadata
        )
        
        super().__init__(testing_config, llm_client, memory_manager, event_bus)
        
        # Testing-specific state
        self.test_types = [
            "unit_tests", "widget_tests", "integration_tests", 
            "golden_tests", "performance_tests", "security_tests",
            "accessibility_tests", "e2e_tests"
        ]
        
        self.testing_frameworks = {
            "unit_testing": ["flutter_test", "mockito", "test"],
            "widget_testing": ["flutter_test", "flutter_driver"],
            "integration_testing": ["integration_test", "flutter_driver"],
            "mocking": ["mockito", "mocktail", "fake_async"],
            "test_data": ["test_api", "faker", "json_fixtures"],
            "performance": ["flutter_benchmark", "trace", "profile"]
        }
        
        self.test_patterns = {
            "arrange_act_assert": "Standard unit test pattern",
            "given_when_then": "BDD-style test pattern",
            "page_object_model": "UI test organization",
            "test_doubles": "Mocks, stubs, and fakes",
            "test_fixtures": "Reusable test data"
        }
        
        logger.info(f"Testing Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the testing agent."""
        return """
You are the Testing Agent in the FlutterSwarm multi-agent system, specializing in Flutter application testing and quality assurance.

CORE EXPERTISE:
- Flutter testing framework and best practices
- Unit testing with flutter_test and mockito
- Widget testing for UI components and interactions
- Integration testing for end-to-end workflows
- Test automation and CI/CD integration
- Code coverage analysis and optimization
- Performance testing and benchmarking
- Security testing and vulnerability assessment
- Test-driven development (TDD) methodologies

TESTING RESPONSIBILITIES:
1. Test Strategy: Design comprehensive testing strategies for Flutter applications
2. Unit Testing: Create thorough unit tests for business logic and utilities
3. Widget Testing: Develop widget tests for UI components and user interactions
4. Integration Testing: Plan and implement end-to-end integration tests
5. Test Automation: Set up automated testing pipelines and CI/CD integration
6. Quality Assurance: Ensure code quality through systematic testing approaches
7. Performance Testing: Create performance benchmarks and stress tests
8. Security Testing: Identify and test for security vulnerabilities

TESTING PRINCIPLES:
- Write clear, maintainable, and reliable tests
- Follow the testing pyramid: more unit tests, fewer integration tests
- Use appropriate test doubles (mocks, stubs, fakes) for isolation
- Implement comprehensive error scenario testing
- Ensure high code coverage with meaningful tests
- Follow arrange-act-assert or given-when-then patterns
- Create deterministic and repeatable tests
- Test edge cases and boundary conditions

FLUTTER TESTING APPROACH:
1. Unit Tests: Test business logic, utilities, and data models in isolation
2. Widget Tests: Test individual widgets and their behavior
3. Integration Tests: Test complete user workflows and app functionality
4. Golden Tests: Visual regression testing for UI consistency
5. Performance Tests: Benchmark critical paths and memory usage
6. Accessibility Tests: Ensure app accessibility compliance

TEST GENERATION STRATEGY:
1. Analyze code structure and identify testable components
2. Create comprehensive test suites covering all scenarios
3. Generate appropriate mocks and test data
4. Include positive, negative, and edge case scenarios
5. Ensure proper test isolation and cleanup
6. Add meaningful assertions and error messages
7. Consider maintainability and test readability

Always generate complete, executable test code with proper imports, setup, and teardown procedures.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of testing-specific capabilities."""
        return [
            "unit_test_generation",
            "widget_test_creation",
            "integration_test_planning",
            "test_strategy_development",
            "code_coverage_analysis",
            "performance_testing",
            "security_testing",
            "accessibility_testing",
            "test_automation_setup",
            "mock_generation",
            "test_data_creation",
            "golden_test_creation",
            "ci_cd_integration"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute testing-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "unit_testing":
                return await self._generate_unit_tests(task_context, llm_analysis)
            elif task_type == "widget_testing":
                return await self._generate_widget_tests(task_context, llm_analysis)
            elif task_type == "integration_testing":
                return await self._plan_integration_tests(task_context, llm_analysis)
            elif task_type == "test_strategy":
                return await self._develop_test_strategy(task_context, llm_analysis)
            elif task_type == "performance_testing":
                return await self._create_performance_tests(task_context, llm_analysis)
            elif task_type == "security_testing":
                return await self._create_security_tests(task_context, llm_analysis)
            else:
                # Generic testing processing
                return await self._process_testing_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"Testing processing failed: {e}")
            return {
                "error": str(e),
                "test_files": {},
                "testing_notes": []
            }

    async def _generate_unit_tests(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive unit tests for the specified code."""
        logger.info(f"Generating unit tests for task: {task_context.task_id}")
        
        unit_test_prompt = self._create_unit_test_prompt(task_context, llm_analysis)
        
        unit_tests = await self.execute_llm_task(
            user_prompt=unit_test_prompt,
            context={
                "task": task_context.to_dict(),
                "testing_frameworks": self.testing_frameworks,
                "test_patterns": self.test_patterns
            },
            structured_output=True
        )
        
        # Store test generation details
        await self.memory_manager.store_memory(
            content=f"Unit tests generated: {json.dumps(unit_tests)}",
            metadata={
                "type": "unit_test_generation",
                "target_code": task_context.metadata.get('target_file'),
                "test_count": len(unit_tests.get('test_cases', []))
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        
        return {
            "unit_tests": unit_tests,
            "test_files": unit_tests.get("test_files", {}),
            "mock_files": unit_tests.get("mock_files", {}),
            "test_utilities": unit_tests.get("utilities", {}),
            "coverage_expectations": unit_tests.get("coverage", {}),
            "setup_instructions": unit_tests.get("setup", [])
        }

    async def _generate_widget_tests(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate widget tests for UI components."""
        logger.info(f"Generating widget tests for task: {task_context.task_id}")
        
        widget_test_prompt = self._create_widget_test_prompt(task_context, llm_analysis)
        
        widget_tests = await self.execute_llm_task(
            user_prompt=widget_test_prompt,
            context={
                "task": task_context.to_dict(),
                "widget_testing_patterns": self._get_widget_testing_patterns()
            },
            structured_output=True
        )
        
        # Store widget test details
        await self.memory_manager.store_memory(
            content=f"Widget tests generated: {json.dumps(widget_tests)}",
            metadata={
                "type": "widget_test_generation",
                "target_widgets": widget_tests.get('tested_widgets', []),
                "interaction_tests": len(widget_tests.get('interaction_tests', []))
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        
        return {
            "widget_tests": widget_tests,
            "test_files": widget_tests.get("test_files", {}),
            "golden_tests": widget_tests.get("golden_tests", {}),
            "interaction_tests": widget_tests.get("interaction_tests", []),
            "accessibility_tests": widget_tests.get("accessibility_tests", []),
            "performance_tests": widget_tests.get("performance_tests", [])
        }

    async def _plan_integration_tests(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan and create integration tests for end-to-end workflows."""
        logger.info(f"Planning integration tests for task: {task_context.task_id}")
        
        integration_test_prompt = self._create_integration_test_prompt(task_context, llm_analysis)
        
        integration_plan = await self.execute_llm_task(
            user_prompt=integration_test_prompt,
            context={
                "task": task_context.to_dict(),
                "integration_patterns": self._get_integration_patterns()
            },
            structured_output=True
        )
        
        return {
            "integration_plan": integration_plan,
            "test_scenarios": integration_plan.get("scenarios", []),
            "test_files": integration_plan.get("test_files", {}),
            "setup_scripts": integration_plan.get("setup_scripts", {}),
            "data_fixtures": integration_plan.get("fixtures", {}),
            "ci_cd_config": integration_plan.get("ci_config", {})
        }

    # Prompt creation methods
    def _create_unit_test_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for unit test generation."""
        return f"""
Generate comprehensive unit tests for the following Flutter code:

TARGET CODE ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

TESTING REQUIREMENTS:
{task_context.description}

CODE CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please generate complete unit tests including:

1. TEST STRUCTURE:
   - Well-organized test suites with descriptive names
   - Proper test group organization and hierarchy
   - Clear arrange-act-assert pattern implementation
   - Comprehensive setup and teardown procedures

2. TEST COVERAGE:
   - Test all public methods and functions
   - Cover positive and negative scenarios
   - Test edge cases and boundary conditions
   - Include error handling and exception scenarios
   - Test state changes and side effects

3. MOCKING AND ISOLATION:
   - Create appropriate mocks for dependencies
   - Use test doubles for external services
   - Implement proper test isolation
   - Mock complex objects and APIs

4. TEST DATA:
   - Create realistic test data and fixtures
   - Use builders or factories for complex objects
   - Include various data scenarios and edge cases
   - Implement data cleanup strategies

5. ASSERTIONS AND VALIDATION:
   - Use meaningful and specific assertions
   - Verify expected outcomes and state changes
   - Check for proper error handling
   - Validate side effects and interactions

6. PERFORMANCE CONSIDERATIONS:
   - Ensure tests run quickly and efficiently
   - Avoid unnecessary setup and teardown
   - Use appropriate test lifecycle methods
   - Consider parallel test execution

Provide complete, executable Dart test files with proper imports and documentation.
"""

    def _create_widget_test_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for widget test generation."""
        return f"""
Generate comprehensive widget tests for the following Flutter UI components:

WIDGET ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

TESTING REQUIREMENTS:
{task_context.description}

Please generate complete widget tests including:

1. WIDGET RENDERING TESTS:
   - Test widget builds without errors
   - Verify correct widget tree structure
   - Check for expected child widgets
   - Validate widget properties and styling

2. INTERACTION TESTS:
   - Test user interactions (taps, swipes, input)
   - Verify navigation and routing behavior
   - Test form validation and submission
   - Check gesture recognition and responses

3. STATE MANAGEMENT TESTS:
   - Test widget state changes
   - Verify state provider integration
   - Test reactive updates and rebuilds
   - Check state persistence scenarios

4. GOLDEN TESTS:
   - Create visual regression tests
   - Generate golden file comparisons
   - Test different screen sizes and orientations
   - Verify theming and styling consistency

5. ACCESSIBILITY TESTS:
   - Test screen reader compatibility
   - Verify semantic labels and descriptions
   - Check focus management and navigation
   - Test high contrast and accessibility features

6. PERFORMANCE TESTS:
   - Measure widget build performance
   - Test memory usage and leaks
   - Verify efficient rebuilds and optimizations
   - Check animation performance

Provide complete widget test implementations with proper test utilities and helpers.
"""

    def _create_integration_test_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for integration test planning."""
        return f"""
Plan and create integration tests for the following Flutter application scenarios:

APPLICATION ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

INTEGRATION REQUIREMENTS:
{task_context.description}

Please provide comprehensive integration testing plan including:

1. TEST SCENARIOS:
   - End-to-end user workflows
   - Cross-feature integration points
   - API integration and data flow
   - Platform-specific functionality

2. TEST IMPLEMENTATION:
   - Complete integration test files
   - Driver setup and configuration
   - Page object model implementation
   - Test data management and cleanup

3. ENVIRONMENT SETUP:
   - Test environment configuration
   - Mock services and API responses
   - Database seeding and cleanup
   - External service integration

4. CI/CD INTEGRATION:
   - Automated test execution pipeline
   - Test reporting and notifications
   - Performance monitoring integration
   - Failure handling and retry logic

5. CROSS-PLATFORM TESTING:
   - Platform-specific test scenarios
   - Device and screen size variations
   - Browser compatibility (for web)
   - Performance across platforms

Provide complete integration test suite with setup and execution instructions.
"""

    # Helper methods for patterns and guidelines
    def _get_widget_testing_patterns(self) -> Dict[str, Any]:
        """Get widget testing patterns and best practices."""
        return {
            "test_widgets": ["testWidgets", "WidgetTester", "pumpWidget"],
            "finders": ["find.text", "find.byKey", "find.byType", "find.byIcon"],
            "interactions": ["tap", "enterText", "drag", "pinch", "scroll"],
            "verification": ["expect", "findsOneWidget", "findsNothing", "matchesGoldenFile"]
        }

    def _get_integration_patterns(self) -> Dict[str, Any]:
        """Get integration testing patterns."""
        return {
            "test_driver": ["FlutterDriver", "integration_test", "flutter_driver"],
            "page_objects": ["Page Object Model", "Element locators", "Action methods"],
            "test_flows": ["User journeys", "Feature workflows", "Cross-platform flows"],
            "data_management": ["Test fixtures", "Mock services", "Database seeding"]
        }
