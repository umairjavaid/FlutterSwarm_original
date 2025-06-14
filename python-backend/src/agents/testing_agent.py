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
from ..models.tool_models import ToolOperation
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
    
    Tool-Driven Capabilities:
    - Automated test file generation and management
    - Test execution and reporting
    - Code coverage analysis
    - Test data management and fixtures
    - Performance testing and benchmarking
    - Mock and stub generation
    - Test environment setup and teardown
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
        
        # Test quality metrics
        self.quality_metrics = {
            "code_coverage": 0.0,
            "test_count": 0,
            "test_pass_rate": 0.0,
            "mutation_score": 0.0,
            "test_execution_time": 0.0
        }
        
        logger.info(f"Testing Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the testing agent."""
        return """
You are the Testing Agent in the FlutterSwarm multi-agent system, specializing in Flutter application testing and quality assurance.

CORE EXPERTISE:
- Flutter testing framework and best practices
- Unit testing with flutter_test and mockito
- Widget testing for UI components
- Integration testing for complete workflows
- Performance testing and benchmarking
- Security testing and vulnerability assessment
- Test automation and CI/CD integration
- Code coverage analysis and improvement

TOOL-DRIVEN APPROACH:
You use specialized tools to:
- Generate comprehensive test files and test data
- Execute tests and analyze results
- Manage test environments and dependencies
- Create mocks, stubs, and test fixtures
- Measure code coverage and quality metrics
- Set up automated testing pipelines
- Monitor test performance and reliability

RESPONSIBILITIES:
1. Test Strategy Development
   - Analyze project requirements and create test plans
   - Define testing scope and priorities
   - Establish testing standards and guidelines
   - Plan test automation strategies

2. Test Creation and Management
   - Generate unit tests for business logic
   - Create widget tests for UI components
   - Develop integration tests for user workflows
   - Implement performance and security tests
   - Maintain test data and fixtures

3. Quality Assurance
   - Monitor code coverage and test quality
   - Identify testing gaps and blind spots
   - Recommend improvements to test suite
   - Ensure test maintainability and reliability

4. Test Automation
   - Set up CI/CD testing pipelines
   - Configure automated test execution
   - Implement test reporting and notifications
   - Manage test environments and dependencies

Always reason through testing decisions using LLM analysis, then use tools to implement your testing strategies.
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

    # Tool-driven test management methods
    async def create_comprehensive_test_suite(
        self, 
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Create a comprehensive test suite for the Flutter project."""
        logger.info(f"Creating comprehensive test suite for project: {project_context.project_name}")
        
        try:
            # Analyze project structure for test planning
            analysis_prompt = f"""
            Analyze this Flutter project for comprehensive testing:
            
            Project: {project_context.project_name}
            Type: {project_context.project_type}
            Architecture: {project_context.architecture_pattern}
            Platforms: {project_context.target_platforms}
            
            Analyze the project structure and determine:
            1. What types of tests are needed (unit, widget, integration)
            2. Which components require testing priority
            3. Test data and fixture requirements
            4. Mock and stub generation needs
            5. Performance testing requirements
            6. Security testing considerations
            
            Provide a detailed test strategy with specific recommendations.
            """
            
            strategy_response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            
            test_strategy = strategy_response.choices[0].message.content
            
            # Use file system tool to analyze existing code structure
            if self.tool_registry:
                file_tool = await self.tool_registry.get_tool("file_system_tool")
                if file_tool:
                    # Analyze project structure
                    analysis_result = await file_tool.execute(ToolOperation(
                        operation_type="analyze_project_structure",
                        parameters={
                            "project_path": project_context.project_path,
                            "include_test_analysis": True
                        }
                    ))
                    
                    if analysis_result.success:
                        project_structure = analysis_result.result
                        
                        # Generate test files based on analysis
                        test_results = await self._generate_test_files(
                            project_context, project_structure, test_strategy
                        )
                        
                        # Set up test environment
                        env_result = await self._setup_test_environment(project_context)
                        
                        # Create test data and fixtures
                        fixtures_result = await self._create_test_fixtures(project_context)
                        
                        return {
                            "strategy": test_strategy,
                            "project_analysis": project_structure,
                            "test_files_created": test_results,
                            "environment_setup": env_result,
                            "fixtures_created": fixtures_result,
                            "status": "comprehensive_test_suite_created"
                        }
            
            return {"error": "File system tool not available", "status": "failed"}
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive test suite: {e}")
            return {"error": str(e), "status": "failed"}

    async def _generate_test_files(
        self, 
        project_context: ProjectContext,
        project_structure: Dict[str, Any],
        test_strategy: str
    ) -> Dict[str, Any]:
        """Generate test files based on project analysis and strategy."""
        generated_tests = {}
        
        if not self.tool_registry:
            return {"error": "Tool registry not available"}
        
        file_tool = await self.tool_registry.get_tool("file_system_tool")
        if not file_tool:
            return {"error": "File system tool not available"}
        
        # Generate unit tests for business logic
        lib_files = project_structure.get("lib_files", [])
        for file_info in lib_files:
            if self._should_create_unit_test(file_info):
                test_content = await self._generate_unit_test_content(
                    file_info, test_strategy
                )
                
                test_file_path = self._get_test_file_path(
                    file_info["path"], "unit"
                )
                
                result = await file_tool.execute(ToolOperation(
                    operation_type="create_file",
                    parameters={
                        "file_path": test_file_path,
                        "content": test_content,
                        "template_type": "dart_test",
                        "ensure_directory": True
                    }
                ))
                
                if result.success:
                    generated_tests[f"unit_{file_info['name']}"] = test_file_path
        
        # Generate widget tests for UI components
        widget_files = [f for f in lib_files if self._is_widget_file(f)]
        for widget_file in widget_files:
            test_content = await self._generate_widget_test_content(
                widget_file, test_strategy
            )
            
            test_file_path = self._get_test_file_path(
                widget_file["path"], "widget"
            )
            
            result = await file_tool.execute(ToolOperation(
                operation_type="create_file",
                parameters={
                    "file_path": test_file_path,
                    "content": test_content,
                    "template_type": "dart_test",
                    "ensure_directory": True
                }
            ))
            
            if result.success:
                generated_tests[f"widget_{widget_file['name']}"] = test_file_path
        
        # Generate integration tests for main workflows
        integration_tests = await self._generate_integration_tests(
            project_context, test_strategy
        )
        generated_tests.update(integration_tests)
        
        return generated_tests

    async def _generate_unit_test_content(
        self, 
        file_info: Dict[str, Any], 
        strategy: str
    ) -> str:
        """Generate unit test content for a Dart file."""
        prompt = f"""
        Generate comprehensive unit tests for this Dart file:
        
        File: {file_info['path']}
        Content Preview: {file_info.get('content_preview', 'N/A')}
        
        Test Strategy Context: {strategy}
        
        Create unit tests that:
        1. Test all public methods and functions
        2. Cover edge cases and error conditions
        3. Use appropriate mocking for dependencies
        4. Follow Flutter testing best practices
        5. Include proper test descriptions and organization
        6. Use arrange-act-assert pattern
        
        Generate the complete Dart test file content.
        """
        
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content

    async def _generate_widget_test_content(
        self, 
        widget_file: Dict[str, Any], 
        strategy: str
    ) -> str:
        """Generate widget test content for a Flutter widget."""
        prompt = f"""
        Generate comprehensive widget tests for this Flutter widget file:
        
        File: {widget_file['path']}
        Content Preview: {widget_file.get('content_preview', 'N/A')}
        
        Test Strategy Context: {strategy}
        
        Create widget tests that:
        1. Test widget rendering and appearance
        2. Test user interactions (taps, gestures, input)
        3. Test state changes and updates
        4. Test accessibility features
        5. Test error states and edge cases
        6. Use appropriate test helpers and utilities
        7. Include golden tests if applicable
        
        Generate the complete Dart widget test file content.
        """
        
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content

    async def _generate_integration_tests(
        self, 
        project_context: ProjectContext, 
        strategy: str
    ) -> Dict[str, Any]:
        """Generate integration tests for main application workflows."""
        integration_tests = {}
        
        if not self.tool_registry:
            return integration_tests
        
        file_tool = await self.tool_registry.get_tool("file_system_tool")
        if not file_tool:
            return integration_tests
        
        # Define common integration test scenarios
        test_scenarios = [
            "app_startup_flow",
            "user_authentication_flow",
            "main_navigation_flow",
            "data_persistence_flow",
            "network_operations_flow"
        ]
        
        for scenario in test_scenarios:
            test_content = await self._generate_integration_test_scenario(
                scenario, project_context, strategy
            )
            
            test_file_path = f"{project_context.project_path}/integration_test/{scenario}_test.dart"
            
            result = await file_tool.execute(ToolOperation(
                operation_type="create_file",
                parameters={
                    "file_path": test_file_path,
                    "content": test_content,
                    "template_type": "dart_test",
                    "ensure_directory": True
                }
            ))
            
            if result.success:
                integration_tests[scenario] = test_file_path
        
        return integration_tests

    async def _generate_integration_test_scenario(
        self, 
        scenario: str, 
        project_context: ProjectContext, 
        strategy: str
    ) -> str:
        """Generate integration test content for a specific scenario."""
        prompt = f"""
        Generate a comprehensive integration test for this scenario:
        
        Scenario: {scenario}
        Project: {project_context.project_name}
        Type: {project_context.project_type}
        
        Test Strategy Context: {strategy}
        
        Create an integration test that:
        1. Tests the complete user workflow for this scenario
        2. Includes proper setup and teardown
        3. Tests across multiple screens/widgets
        4. Verifies data persistence and state management
        5. Handles network operations if applicable
        6. Tests error handling and recovery
        7. Uses flutter_driver or integration_test framework
        
        Generate the complete Dart integration test file content.
        """
        
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content

    async def _setup_test_environment(
        self, 
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Set up the testing environment and dependencies."""
        if not self.tool_registry:
            return {"error": "Tool registry not available"}
        
        file_tool = await self.tool_registry.get_tool("file_system_tool")
        if not file_tool:
            return {"error": "File system tool not available"}
        
        # Create test configuration files
        test_configs = {
            "test_helper.dart": await self._generate_test_helper_content(project_context),
            "mock_data.dart": await self._generate_mock_data_content(project_context),
            "test_utilities.dart": await self._generate_test_utilities_content(project_context)
        }
        
        created_files = {}
        for filename, content in test_configs.items():
            file_path = f"{project_context.project_path}/test/{filename}"
            
            result = await file_tool.execute(ToolOperation(
                operation_type="create_file",
                parameters={
                    "file_path": file_path,
                    "content": content,
                    "template_type": "dart_helper",
                    "ensure_directory": True
                }
            ))
            
            if result.success:
                created_files[filename] = file_path
        
        return {"created_files": created_files}

    async def _create_test_fixtures(
        self, 
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Create test fixtures and sample data."""
        if not self.tool_registry:
            return {"error": "Tool registry not available"}
        
        file_tool = await self.tool_registry.get_tool("file_system_tool")
        if not file_tool:
            return {"error": "File system tool not available"}
        
        # Generate test data fixtures
        fixtures = {
            "sample_users.json": await self._generate_user_fixtures(),
            "test_responses.json": await self._generate_api_fixtures(),
            "widget_test_data.dart": await self._generate_widget_fixtures()
        }
        
        created_fixtures = {}
        for filename, content in fixtures.items():
            file_path = f"{project_context.project_path}/test/fixtures/{filename}"
            
            result = await file_tool.execute(ToolOperation(
                operation_type="create_file",
                parameters={
                    "file_path": file_path,
                    "content": content,
                    "ensure_directory": True
                }
            ))
            
            if result.success:
                created_fixtures[filename] = file_path
        
        return {"created_fixtures": created_fixtures}

    async def run_test_suite(
        self, 
        project_context: ProjectContext,
        test_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run the complete test suite and analyze results."""
        logger.info(f"Running test suite for project: {project_context.project_name}")
        
        if not self.tool_registry:
            return {"error": "Tool registry not available"}
        
        # Use process tool to run tests
        process_tool = await self.tool_registry.get_tool("process_tool")
        flutter_tool = await self.tool_registry.get_tool("flutter_sdk_tool")
        
        if not process_tool or not flutter_tool:
            return {"error": "Required tools not available"}
        
        test_results = {}
        
        # Run different types of tests
        test_commands = {
            "unit_tests": "flutter test",
            "widget_tests": "flutter test test/widget",
            "integration_tests": "flutter test integration_test",
            "coverage": "flutter test --coverage"
        }
        
        target_tests = test_types or list(test_commands.keys())
        
        for test_type in target_tests:
            if test_type in test_commands:
                result = await flutter_tool.execute(ToolOperation(
                    operation_type="test",
                    parameters={
                        "project_path": project_context.project_path,
                        "test_type": test_type,
                        "with_coverage": test_type == "coverage"
                    }
                ))
                
                test_results[test_type] = {
                    "success": result.success,
                    "output": result.result,
                    "execution_time": result.execution_time
                }
        
        # Analyze test results and coverage
        analysis = await self._analyze_test_results(test_results, project_context)
        
        return {
            "test_results": test_results,
            "analysis": analysis,
            "recommendations": await self._generate_test_recommendations(analysis)
        }

    async def _analyze_test_results(
        self, 
        test_results: Dict[str, Any],
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Analyze test results and extract metrics."""
        analysis_prompt = f"""
        Analyze these Flutter test results:
        
        Project: {project_context.project_name}
        Test Results: {json.dumps(test_results, indent=2)}
        
        Provide analysis covering:
        1. Overall test success rate
        2. Code coverage metrics
        3. Performance metrics
        4. Test reliability and flakiness
        5. Areas needing attention
        6. Quality improvements needed
        
        Return structured analysis with specific metrics and recommendations.
        """
        
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.2
        )
        
        return {"llm_analysis": response.choices[0].message.content}

    async def _generate_test_recommendations(
        self, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific recommendations for test improvements."""
        recommendations_prompt = f"""
        Based on this test analysis, provide specific actionable recommendations:
        
        Analysis: {json.dumps(analysis, indent=2)}
        
        Generate 5-10 specific recommendations for:
        1. Improving test coverage
        2. Enhancing test quality
        3. Optimizing test performance
        4. Reducing test flakiness
        5. Better test organization
        
        Make recommendations specific and actionable.
        """
        
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": recommendations_prompt}],
            temperature=0.3
        )
        
        recommendations_text = response.choices[0].message.content
        return [rec.strip() for rec in recommendations_text.split('\n') if rec.strip()]

    # Helper methods for test file analysis
    def _should_create_unit_test(self, file_info: Dict[str, Any]) -> bool:
        """Determine if a file should have unit tests."""
        file_path = file_info.get("path", "")
        file_name = file_info.get("name", "")
        
        # Skip test files, generated files, and UI-only files
        if (file_path.endswith("_test.dart") or 
            file_path.endswith(".g.dart") or
            file_name.startswith("main.dart")):
            return False
        
        # Include business logic, models, services, utilities
        include_patterns = [
            "/models/", "/services/", "/utils/", "/repositories/",
            "/controllers/", "/providers/", "/blocs/", "/cubits/"
        ]
        
        return any(pattern in file_path for pattern in include_patterns)

    def _is_widget_file(self, file_info: Dict[str, Any]) -> bool:
        """Determine if a file contains Flutter widgets."""
        file_path = file_info.get("path", "")
        content_preview = file_info.get("content_preview", "")
        
        # Check for widget-related patterns
        widget_patterns = [
            "extends StatelessWidget", "extends StatefulWidget",
            "extends Widget", "class.*Widget", "/widgets/", "/screens/", "/pages/"
        ]
        
        return any(pattern in content_preview or pattern in file_path 
                  for pattern in widget_patterns)

    def _get_test_file_path(self, source_file_path: str, test_type: str) -> str:
        """Generate test file path for a source file."""
        # Convert lib path to test path
        test_path = source_file_path.replace("/lib/", f"/test/{test_type}/")
        
        # Add _test suffix
        if test_path.endswith(".dart"):
            test_path = test_path[:-5] + "_test.dart"
        
        return test_path

    # Test content generation helper methods
    async def _generate_test_helper_content(self, project_context: ProjectContext) -> str:
        """Generate test helper utilities."""
        return """
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// Test helper utilities for Flutter tests
class TestHelper {
  /// Create a testable widget with MaterialApp wrapper
  static Widget makeTestableWidget(Widget child) {
    return MaterialApp(
      home: child,
    );
  }
  
  /// Pump and settle with custom duration
  static Future<void> pumpAndSettleWithDelay(
    WidgetTester tester, [
    Duration delay = const Duration(milliseconds: 100)
  ]) async {
    await tester.pump(delay);
    await tester.pumpAndSettle();
  }
  
  /// Find widget by key safely
  static Finder findByKeyString(String key) {
    return find.byKey(Key(key));
  }
  
  /// Verify widget exists and is visible
  static void verifyWidgetExists(Finder finder) {
    expect(finder, findsOneWidget);
  }
  
  /// Verify text content
  static void verifyTextContent(String text) {
    expect(find.text(text), findsOneWidget);
  }
}
"""

    async def _generate_mock_data_content(self, project_context: ProjectContext) -> str:
        """Generate mock data utilities."""
        return """
/// Mock data utilities for testing
class MockData {
  static const Map<String, dynamic> sampleUser = {
    'id': '123',
    'name': 'Test User',
    'email': 'test@example.com',
    'avatar': 'https://example.com/avatar.jpg',
  };
  
  static const List<Map<String, dynamic>> sampleUsers = [
    {
      'id': '1',
      'name': 'Alice Johnson',
      'email': 'alice@example.com',
    },
    {
      'id': '2', 
      'name': 'Bob Smith',
      'email': 'bob@example.com',
    },
  ];
  
  static const Map<String, dynamic> sampleApiResponse = {
    'status': 'success',
    'data': sampleUser,
    'message': 'User fetched successfully',
  };
  
  static const Map<String, dynamic> sampleErrorResponse = {
    'status': 'error',
    'message': 'User not found',
    'code': 404,
  };
}
"""

    async def _generate_test_utilities_content(self, project_context: ProjectContext) -> str:
        """Generate test utilities and custom matchers."""
        return """
import 'package:flutter_test/flutter_test.dart';

/// Custom matchers for Flutter tests
Matcher hasWidgetCount(int count) {
  return _HasWidgetCount(count);
}

class _HasWidgetCount extends Matcher {
  final int expectedCount;
  
  _HasWidgetCount(this.expectedCount);
  
  @override
  bool matches(item, Map matchState) {
    if (item is Finder) {
      return item.evaluate().length == expectedCount;
    }
    return false;
  }
  
  @override
  Description describe(Description description) {
    return description.add('has exactly $expectedCount widgets');
  }
}

/// Test utilities for common operations
class TestUtils {
  /// Wait for animation to complete
  static Future<void> waitForAnimation(WidgetTester tester) async {
    await tester.pumpAndSettle(const Duration(seconds: 1));
  }
  
  /// Simulate network delay
  static Future<void> simulateNetworkDelay() async {
    await Future.delayed(const Duration(milliseconds: 500));
  }
  
  /// Create mock HTTP response
  static Map<String, dynamic> createMockResponse({
    required int statusCode,
    required Map<String, dynamic> body,
  }) {
    return {
      'statusCode': statusCode,
      'body': body,
      'headers': {'content-type': 'application/json'},
    };
  }
}
"""
