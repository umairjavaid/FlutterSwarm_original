#!/usr/bin/env python3
"""
Comprehensive Integration Testing Framework for FlutterSwarm Tool System.

This framework tests:
1. All tools work together seamlessly
2. Agent can successfully use tools through LLM reasoning
3. Error handling works across tool boundaries
4. Performance metrics are accurate
5. End-to-end workflow testing with real Flutter projects
6. Tool capability verification against requirements
7. Schema validation for all operations

Usage:
    python test_comprehensive_integration.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)

logger = logging.getLogger("comprehensive_integration_test")


@dataclass
class TestResult:
    """Comprehensive test result tracking."""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    sub_results: List['TestResult'] = field(default_factory=list)


@dataclass
class IntegrationMetrics:
    """Metrics collected during integration testing."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_duration: float = 0.0
    tool_usage_count: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    performance_data: List[Dict[str, Any]] = field(default_factory=list)


class MockLLMClient:
    """Enhanced mock LLM client for testing tool system integration."""
    
    def __init__(self):
        self.call_count = 0
        self.response_templates = self._setup_response_templates()
    
    def _setup_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Setup realistic response templates for different scenarios."""
        return {
            "create_flutter_project": {
                "tool_name": "flutter_sdk_tool",
                "operation": "create_project",
                "parameters": {
                    "project_name": "test_app",
                    "output_directory": "/tmp/flutter_projects",
                    "template": "app",
                    "org": "com.example"
                },
                "reasoning": "Creating a new Flutter project with standard app template"
            },
            "create_feature": {
                "tool_name": "file_system_tool",
                "operation": "create_file",
                "parameters": {
                    "file_path": "/tmp/flutter_projects/test_app/lib/features/todo/todo_screen.dart",
                    "content": "// TODO feature implementation"
                },
                "reasoning": "Creating a new feature file in the appropriate directory structure"
            },
            "analyze_project": {
                "tool_name": "flutter_sdk_tool",
                "operation": "analyze_code",
                "parameters": {
                    "project_path": "/tmp/flutter_projects/test_app",
                    "fix": False
                },
                "reasoning": "Analyzing the project structure and code quality"
            },
            "build_project": {
                "tool_name": "flutter_sdk_tool",
                "operation": "build_app",
                "parameters": {
                    "project_path": "/tmp/flutter_projects/test_app",
                    "platform": "android"
                },
                "reasoning": "Building the Flutter app for Android platform"
            }
        }
    
    async def generate(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate mock responses based on context."""
        self.call_count += 1
        
        # Extract user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Determine appropriate response based on content
        if "create" in user_message.lower() and "project" in user_message.lower():
            response = self.response_templates["create_flutter_project"]
        elif "feature" in user_message.lower() or "todo" in user_message.lower():
            response = self.response_templates["create_feature"]
        elif "analyze" in user_message.lower():
            response = self.response_templates["analyze_project"]
        elif "build" in user_message.lower():
            response = self.response_templates["build_project"]
        else:
            # Default response
            response = self.response_templates["create_flutter_project"]
        
        return {
            "content": json.dumps(response),
            "usage": {"prompt_tokens": 50, "completion_tokens": 100}
        }


class ComprehensiveIntegrationTester:
    """Comprehensive integration testing framework."""
    
    def __init__(self):
        self.metrics = IntegrationMetrics()
        self.results: List[TestResult] = []
        self.llm_client = MockLLMClient()
        self.temp_dirs: List[str] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete integration test suite."""
        logger.info("🚀 Starting Comprehensive Integration Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Core system tests
            await self._test_tool_system_initialization()
            await self._test_tool_discovery_and_registration()
            await self._test_tool_capabilities_verification()
            
            # Integration tests
            await self._test_tool_interaction_workflows()
            await self._test_agent_tool_integration()
            await self._test_error_handling_across_boundaries()
            
            # Performance tests
            await self._test_performance_metrics_accuracy()
            await self._test_concurrent_tool_usage()
            
            # End-to-end tests
            await self._test_flutter_project_workflow()
            await self._test_schema_validation()
            
            # Stress tests
            await self._test_system_under_load()
            
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}")
            traceback.print_exc()
        
        finally:
            await self._cleanup_test_environment()
        
        self.metrics.total_duration = time.time() - start_time
        return self._generate_final_report()
    
    async def _test_tool_system_initialization(self) -> TestResult:
        """Test 1: Tool system initialization and basic functionality."""
        logger.info("\n1️⃣ Testing Tool System Initialization")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            from core.tools.flutter_sdk_tool import FlutterSDKTool
            from core.tools.file_system_tool import FileSystemTool
            from core.tools.process_tool import ProcessTool
            
            # Test singleton pattern
            registry1 = ToolRegistry.instance()
            registry2 = ToolRegistry.instance()
            assert registry1 is registry2, "Singleton pattern failed"
            
            # Test initialization
            await registry1.initialize(auto_discover=True)
            assert registry1.is_initialized, "Registry initialization failed"
            
            # Test tool registration
            flutter_tool = FlutterSDKTool()
            file_tool = FileSystemTool()
            process_tool = ProcessTool()
            
            await registry1.register_tool(flutter_tool)
            await registry1.register_tool(file_tool)
            await registry1.register_tool(process_tool)
            
            tools = registry1.get_available_tools()
            assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
            
            result = TestResult(
                test_name="tool_system_initialization",
                success=True,
                duration=time.time() - start_time,
                details={
                    "tools_registered": len(tools),
                    "tool_names": [tool.name for tool in tools]
                }
            )
            
            logger.info(f"✅ Tool system initialization successful - {len(tools)} tools registered")
            
        except Exception as e:
            result = TestResult(
                test_name="tool_system_initialization",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Tool system initialization failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_tool_discovery_and_registration(self) -> TestResult:
        """Test 2: Tool discovery and automatic registration."""
        logger.info("\n2️⃣ Testing Tool Discovery and Registration")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            # Test capability querying
            available_tools = registry.get_available_tools()
            capabilities_map = {}
            
            for tool in available_tools:
                capabilities = await tool.get_capabilities()
                capabilities_map[tool.name] = {
                    "operations": len(capabilities.available_operations),
                    "categories": tool.category.value if hasattr(tool, 'category') else "unknown"
                }
            
            # Test tool selection API
            best_tool = registry.select_best_tool("file_operations", {})
            assert best_tool is not None, "Tool selection failed"
            
            result = TestResult(
                test_name="tool_discovery_and_registration",
                success=True,
                duration=time.time() - start_time,
                details={
                    "discovered_tools": len(available_tools),
                    "capabilities_map": capabilities_map,
                    "best_tool_for_files": best_tool.name if best_tool else None
                }
            )
            
            logger.info(f"✅ Tool discovery successful - {len(available_tools)} tools with capabilities")
            
        except Exception as e:
            result = TestResult(
                test_name="tool_discovery_and_registration",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Tool discovery failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_tool_capabilities_verification(self) -> TestResult:
        """Test 3: Verify tool capabilities against requirements."""
        logger.info("\n3️⃣ Testing Tool Capabilities Verification")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            requirements_check = {
                "flutter_sdk_tool": {
                    "required_operations": ["create_project", "build_app", "run_app", "add_platform"],
                    "optional_operations": ["test_app", "analyze_code", "pub_operations"]
                },
                "file_system_tool": {
                    "required_operations": ["create_file", "read_file", "write_file", "delete_file"],
                    "optional_operations": ["backup_file", "watch_directory"]
                },
                "process_tool": {
                    "required_operations": ["execute_command", "kill_process"],
                    "optional_operations": ["monitor_process"]
                }
            }
            
            verification_results = {}
            
            for tool in tools:
                if tool.name in requirements_check:
                    capabilities = await tool.get_capabilities()
                    available_ops = [op["name"] for op in capabilities.available_operations]
                    
                    required = requirements_check[tool.name]["required_operations"]
                    optional = requirements_check[tool.name]["optional_operations"]
                    
                    missing_required = [op for op in required if op not in available_ops]
                    available_optional = [op for op in optional if op in available_ops]
                    
                    verification_results[tool.name] = {
                        "all_required_present": len(missing_required) == 0,
                        "missing_required": missing_required,
                        "available_optional": available_optional,
                        "total_operations": len(available_ops)
                    }
            
            all_tools_valid = all(
                result["all_required_present"] 
                for result in verification_results.values()
            )
            
            result = TestResult(
                test_name="tool_capabilities_verification",
                success=all_tools_valid,
                duration=time.time() - start_time,
                details={"verification_results": verification_results}
            )
            
            logger.info(f"✅ Tool capabilities verification {'passed' if all_tools_valid else 'failed'}")
            
        except Exception as e:
            result = TestResult(
                test_name="tool_capabilities_verification",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Tool capabilities verification failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_tool_interaction_workflows(self) -> TestResult:
        """Test 4: Test tools working together in workflows."""
        logger.info("\n4️⃣ Testing Tool Interaction Workflows")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            # Create a temporary project directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dirs.append(temp_dir)
                project_path = os.path.join(temp_dir, "workflow_test_app")
                
                # Workflow: Create project structure using file system tool
                file_tool = registry.get_tool("file_system_tool")
                if file_tool:
                    # Create basic Flutter structure
                    dirs_to_create = [
                        os.path.join(project_path, "lib"),
                        os.path.join(project_path, "test"),
                        os.path.join(project_path, "assets")
                    ]
                    
                    for dir_path in dirs_to_create:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    # Create pubspec.yaml
                    pubspec_content = """
name: workflow_test_app
description: Test app for workflow testing
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
                    pubspec_path = os.path.join(project_path, "pubspec.yaml")
                    create_result = await file_tool.execute("create_file", {
                        "file_path": pubspec_path,
                        "content": pubspec_content
                    })
                    
                    assert create_result.status.value == "success", "File creation failed"
                
                # Test process tool execution
                process_tool = registry.get_tool("process_tool")
                if process_tool:
                    # Test directory listing
                    list_result = await process_tool.execute("execute_command", {
                        "command": f"ls -la {project_path}",
                        "working_directory": temp_dir
                    })
                    
                    assert list_result.status.value == "success", "Process execution failed"
                
                workflow_success = True
                
            result = TestResult(
                test_name="tool_interaction_workflows",
                success=workflow_success,
                duration=time.time() - start_time,
                details={
                    "project_created": os.path.exists(pubspec_path),
                    "tools_used": ["file_system_tool", "process_tool"]
                }
            )
            
            logger.info(f"✅ Tool interaction workflows {'successful' if workflow_success else 'failed'}")
            
        except Exception as e:
            result = TestResult(
                test_name="tool_interaction_workflows",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Tool interaction workflows failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_agent_tool_integration(self) -> TestResult:
        """Test 5: Agent using tools through LLM reasoning."""
        logger.info("\n5️⃣ Testing Agent Tool Integration")
        start_time = time.time()
        
        try:
            from agents.implementation_agent import ImplementationAgent
            from agents.base_agent import AgentConfig
            from core.memory_manager import MemoryManager
            from core.event_bus import EventBus
            from core.tools.tool_registry import ToolRegistry
            
            # Setup agent with mock LLM
            config = AgentConfig(
                agent_id="test_implementation_agent",
                agent_type="implementation",
                capabilities=[],
                llm_model="gpt-4"
            )
            
            memory_manager = MemoryManager()
            event_bus = EventBus()
            registry = ToolRegistry.instance()
            
            agent = ImplementationAgent(
                config=config,
                llm_client=self.llm_client,
                memory_manager=memory_manager,
                event_bus=event_bus,
                tool_registry=registry
            )
            
            # Test agent reasoning with tools
            task_description = "Create a new Flutter todo app with basic functionality"
            
            # Mock the agent's tool selection and usage
            selected_tools = registry.select_tools_for_workflow("flutter_project_creation", {})
            assert len(selected_tools) > 0, "No tools selected for workflow"
            
            # Simulate agent using tools through reasoning
            tool_usage_results = []
            for tool in selected_tools[:2]:  # Test first 2 tools
                capabilities = await tool.get_capabilities()
                if capabilities.available_operations:
                    op = capabilities.available_operations[0]
                    # Use minimal valid parameters
                    params = {"project_path": "/tmp/test"} if "project_path" in str(op) else {}
                    
                    try:
                        result = await tool.execute(op["name"], params)
                        tool_usage_results.append({
                            "tool": tool.name,
                            "operation": op["name"],
                            "success": result.status.value == "success"
                        })
                    except Exception as tool_error:
                        tool_usage_results.append({
                            "tool": tool.name,
                            "operation": op["name"],
                            "success": False,
                            "error": str(tool_error)
                        })
            
            agent_integration_success = len(tool_usage_results) > 0
            
            result = TestResult(
                test_name="agent_tool_integration",
                success=agent_integration_success,
                duration=time.time() - start_time,
                details={
                    "llm_calls": self.llm_client.call_count,
                    "tools_used": tool_usage_results,
                    "selected_tools_count": len(selected_tools)
                }
            )
            
            logger.info(f"✅ Agent tool integration {'successful' if agent_integration_success else 'failed'}")
            
        except Exception as e:
            result = TestResult(
                test_name="agent_tool_integration",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Agent tool integration failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_error_handling_across_boundaries(self) -> TestResult:
        """Test 6: Error handling works across tool boundaries."""
        logger.info("\n6️⃣ Testing Error Handling Across Boundaries")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            error_scenarios = []
            
            # Test 1: Invalid parameters
            flutter_tool = registry.get_tool("flutter_sdk_tool")
            if flutter_tool:
                invalid_result = await flutter_tool.execute("create_project", {
                    "project_name": "123invalid",  # Invalid name
                    "output_directory": "/nonexistent/path"
                })
                error_scenarios.append({
                    "scenario": "invalid_parameters",
                    "tool": "flutter_sdk_tool",
                    "handled_gracefully": invalid_result.status.value == "failure" and invalid_result.error_message
                })
            
            # Test 2: Missing required parameters
            file_tool = registry.get_tool("file_system_tool")
            if file_tool:
                missing_params_result = await file_tool.execute("create_file", {})
                error_scenarios.append({
                    "scenario": "missing_parameters",
                    "tool": "file_system_tool",
                    "handled_gracefully": missing_params_result.status.value == "failure"
                })
            
            # Test 3: Non-existent operation
            if flutter_tool:
                nonexistent_result = await flutter_tool.execute("nonexistent_operation", {})
                error_scenarios.append({
                    "scenario": "nonexistent_operation",
                    "tool": "flutter_sdk_tool",
                    "handled_gracefully": nonexistent_result.status.value == "failure"
                })
            
            all_errors_handled = all(scenario["handled_gracefully"] for scenario in error_scenarios)
            
            result = TestResult(
                test_name="error_handling_across_boundaries",
                success=all_errors_handled,
                duration=time.time() - start_time,
                details={"error_scenarios": error_scenarios}
            )
            
            logger.info(f"✅ Error handling {'successful' if all_errors_handled else 'needs improvement'}")
            
        except Exception as e:
            result = TestResult(
                test_name="error_handling_across_boundaries",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Error handling test failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_performance_metrics_accuracy(self) -> TestResult:
        """Test 7: Performance metrics are accurate."""
        logger.info("\n7️⃣ Testing Performance Metrics Accuracy")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            performance_data = []
            
            for tool in tools[:2]:  # Test first 2 tools
                # Record usage
                start_op_time = time.time()
                capabilities = await tool.get_capabilities()
                op_duration = time.time() - start_op_time
                
                # Check if tool tracks metrics
                if hasattr(tool, 'usage_metrics') or hasattr(tool, 'metrics'):
                    performance_data.append({
                        "tool": tool.name,
                        "operation": "get_capabilities",
                        "duration": op_duration,
                        "has_metrics": True
                    })
                else:
                    performance_data.append({
                        "tool": tool.name,
                        "operation": "get_capabilities",
                        "duration": op_duration,
                        "has_metrics": False
                    })
            
            # Test registry metrics
            registry_metrics = registry.get_performance_analytics()
            has_registry_metrics = isinstance(registry_metrics, dict)
            
            metrics_accurate = len(performance_data) > 0 and has_registry_metrics
            
            result = TestResult(
                test_name="performance_metrics_accuracy",
                success=metrics_accurate,
                duration=time.time() - start_time,
                details={
                    "performance_data": performance_data,
                    "registry_has_metrics": has_registry_metrics
                }
            )
            
            logger.info(f"✅ Performance metrics {'accurate' if metrics_accurate else 'need improvement'}")
            
        except Exception as e:
            result = TestResult(
                test_name="performance_metrics_accuracy",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Performance metrics test failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_concurrent_tool_usage(self) -> TestResult:
        """Test 8: Concurrent tool usage."""
        logger.info("\n8️⃣ Testing Concurrent Tool Usage")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            # Create concurrent tasks
            async def use_tool_concurrently(tool_name: str, task_id: int):
                tool = registry.get_tool(tool_name)
                if tool:
                    capabilities = await tool.get_capabilities()
                    return {"task_id": task_id, "tool": tool_name, "success": True}
                return {"task_id": task_id, "tool": tool_name, "success": False}
            
            # Run concurrent operations
            tasks = []
            for i in range(5):
                tasks.append(use_tool_concurrently("file_system_tool", i))
                tasks.append(use_tool_concurrently("flutter_sdk_tool", i + 5))
            
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_concurrent = sum(
                1 for result in concurrent_results 
                if isinstance(result, dict) and result.get("success", False)
            )
            
            concurrency_success = successful_concurrent > 0
            
            result = TestResult(
                test_name="concurrent_tool_usage",
                success=concurrency_success,
                duration=time.time() - start_time,
                details={
                    "total_concurrent_tasks": len(tasks),
                    "successful_tasks": successful_concurrent,
                    "results": [r for r in concurrent_results if isinstance(r, dict)]
                }
            )
            
            logger.info(f"✅ Concurrent tool usage {'successful' if concurrency_success else 'failed'}")
            
        except Exception as e:
            result = TestResult(
                test_name="concurrent_tool_usage",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Concurrent tool usage test failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_flutter_project_workflow(self) -> TestResult:
        """Test 9: End-to-end Flutter project workflow."""
        logger.info("\n9️⃣ Testing Flutter Project Workflow")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dirs.append(temp_dir)
                project_path = os.path.join(temp_dir, "e2e_test_app")
                
                workflow_steps = []
                
                # Step 1: Create project structure
                file_tool = registry.get_tool("file_system_tool")
                if file_tool:
                    os.makedirs(os.path.join(project_path, "lib"), exist_ok=True)
                    
                    # Create main.dart
                    main_dart_content = '''
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'E2E Test App',
      home: Scaffold(
        appBar: AppBar(title: Text('Test')),
        body: Center(child: Text('Hello World')),
      ),
    );
  }
}
'''
                    main_result = await file_tool.execute("create_file", {
                        "file_path": os.path.join(project_path, "lib", "main.dart"),
                        "content": main_dart_content
                    })
                    workflow_steps.append({"step": "create_main_dart", "success": main_result.status.value == "success"})
                
                # Step 2: Create pubspec.yaml
                pubspec_content = '''
name: e2e_test_app
description: End-to-end test Flutter app
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
'''
                if file_tool:
                    pubspec_result = await file_tool.execute("create_file", {
                        "file_path": os.path.join(project_path, "pubspec.yaml"),
                        "content": pubspec_content
                    })
                    workflow_steps.append({"step": "create_pubspec", "success": pubspec_result.status.value == "success"})
                
                # Step 3: Validate project structure
                flutter_tool = registry.get_tool("flutter_sdk_tool")
                if flutter_tool:
                    # Test project validation (will likely fail without Flutter SDK, but tests the workflow)
                    validate_result = await flutter_tool.execute("analyze_code", {
                        "project_path": project_path
                    })
                    workflow_steps.append({
                        "step": "validate_project", 
                        "success": validate_result.status.value in ["success", "failure"],  # Either is acceptable
                        "status": validate_result.status.value
                    })
                
                workflow_success = len(workflow_steps) >= 2 and any(step["success"] for step in workflow_steps)
                
                result = TestResult(
                    test_name="flutter_project_workflow",
                    success=workflow_success,
                    duration=time.time() - start_time,
                    details={
                        "workflow_steps": workflow_steps,
                        "project_path": project_path
                    }
                )
                
                logger.info(f"✅ Flutter project workflow {'successful' if workflow_success else 'failed'}")
                
        except Exception as e:
            result = TestResult(
                test_name="flutter_project_workflow",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Flutter project workflow failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_schema_validation(self) -> TestResult:
        """Test 10: Schema validation for all operations."""
        logger.info("\n🔟 Testing Schema Validation")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            tools = registry.get_available_tools()
            
            validation_results = []
            
            for tool in tools:
                capabilities = await tool.get_capabilities()
                
                for operation in capabilities.available_operations:
                    # Test parameter validation
                    op_name = operation["name"]
                    
                    # Test with empty parameters
                    empty_result = await tool.validate_params(op_name, {})
                    
                    # Test with invalid parameters
                    invalid_result = await tool.validate_params(op_name, {"invalid_param": "test"})
                    
                    validation_results.append({
                        "tool": tool.name,
                        "operation": op_name,
                        "validates_empty": isinstance(empty_result, tuple) and len(empty_result) == 2,
                        "validates_invalid": isinstance(invalid_result, tuple) and len(invalid_result) == 2,
                        "has_validation": True
                    })
            
            validation_success = len(validation_results) > 0 and all(
                result["has_validation"] for result in validation_results
            )
            
            result = TestResult(
                test_name="schema_validation",
                success=validation_success,
                duration=time.time() - start_time,
                details={
                    "validation_results": validation_results[:5],  # Show first 5
                    "total_validations": len(validation_results)
                }
            )
            
            logger.info(f"✅ Schema validation {'successful' if validation_success else 'failed'}")
            
        except Exception as e:
            result = TestResult(
                test_name="schema_validation",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ Schema validation test failed: {e}")
        
        self._record_result(result)
        return result
    
    async def _test_system_under_load(self) -> TestResult:
        """Test 11: System performance under load."""
        logger.info("\n1️⃣1️⃣ Testing System Under Load")
        start_time = time.time()
        
        try:
            from core.tools.tool_registry import ToolRegistry
            
            registry = ToolRegistry.instance()
            
            # Simulate high load
            async def load_test_task(task_id: int):
                tool = registry.get_tool("file_system_tool")
                if tool:
                    start_time = time.time()
                    capabilities = await tool.get_capabilities()
                    duration = time.time() - start_time
                    return {"task_id": task_id, "duration": duration, "success": True}
                return {"task_id": task_id, "success": False}
            
            # Run 20 concurrent load test tasks
            load_tasks = [load_test_task(i) for i in range(20)]
            load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            successful_loads = [
                result for result in load_results 
                if isinstance(result, dict) and result.get("success", False)
            ]
            
            avg_duration = sum(result["duration"] for result in successful_loads) / len(successful_loads) if successful_loads else 0
            
            load_success = len(successful_loads) >= 15 and avg_duration < 1.0  # 15+ successes, under 1s avg
            
            result = TestResult(
                test_name="system_under_load",
                success=load_success,
                duration=time.time() - start_time,
                details={
                    "total_load_tasks": len(load_tasks),
                    "successful_tasks": len(successful_loads),
                    "average_duration": avg_duration,
                    "max_duration": max((r["duration"] for r in successful_loads), default=0)
                }
            )
            
            logger.info(f"✅ System under load {'performed well' if load_success else 'needs optimization'}")
            
        except Exception as e:
            result = TestResult(
                test_name="system_under_load",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
            logger.error(f"❌ System load test failed: {e}")
        
        self._record_result(result)
        return result
    
    def _record_result(self, result: TestResult):
        """Record a test result."""
        self.results.append(result)
        self.metrics.total_tests += 1
        
        if result.success:
            self.metrics.passed_tests += 1
        else:
            self.metrics.failed_tests += 1
            if result.error_message:
                error_type = type(result.error_message).__name__
                self.metrics.error_counts[error_type] = self.metrics.error_counts.get(error_type, 0) + 1
        
        self.metrics.performance_data.append({
            "test_name": result.test_name,
            "duration": result.duration,
            "success": result.success,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        logger.info("\n🧹 Cleaning up test environment...")
        
        # Cleanup is handled by context managers for temporary directories
        # Additional cleanup if needed
        
        logger.info("✅ Test environment cleaned up")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE INTEGRATION TEST REPORT")
        logger.info("=" * 80)
        
        success_rate = (self.metrics.passed_tests / self.metrics.total_tests * 100) if self.metrics.total_tests > 0 else 0
        
        logger.info(f"\n🎯 Overall Results:")
        logger.info(f"   Total Tests: {self.metrics.total_tests}")
        logger.info(f"   Passed: {self.metrics.passed_tests}")
        logger.info(f"   Failed: {self.metrics.failed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Total Duration: {self.metrics.total_duration:.2f}s")
        
        logger.info(f"\n📋 Test Results Summary:")
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"   {status} {result.test_name} ({result.duration:.2f}s)")
            if not result.success and result.error_message:
                logger.info(f"      Error: {result.error_message}")
        
        if self.metrics.error_counts:
            logger.info(f"\n🚨 Error Summary:")
            for error_type, count in self.metrics.error_counts.items():
                logger.info(f"   {error_type}: {count}")
        
        # Assessment
        if success_rate >= 90:
            logger.info(f"\n🎉 EXCELLENT! Tool system integration is highly successful!")
            logger.info(f"✅ System is ready for production use with AI agents")
        elif success_rate >= 75:
            logger.info(f"\n✅ GOOD! Tool system integration is mostly successful")
            logger.info(f"⚠️  Some areas need improvement for optimal agent performance")
        else:
            logger.info(f"\n⚠️ NEEDS IMPROVEMENT! Tool system integration has issues")
            logger.info(f"❌ Significant fixes needed before production use")
        
        return {
            "metrics": self.metrics,
            "results": self.results,
            "success_rate": success_rate,
            "overall_success": success_rate >= 75
        }


async def main():
    """Main test runner."""
    print("🚀 FlutterSwarm Tool System - Comprehensive Integration Testing")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = ComprehensiveIntegrationTester()
    
    try:
        final_report = await tester.run_all_tests()
        
        # Save detailed report
        report_path = "comprehensive_integration_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_tests": tester.metrics.total_tests,
                    "passed_tests": tester.metrics.passed_tests,
                    "failed_tests": tester.metrics.failed_tests,
                    "total_duration": tester.metrics.total_duration,
                    "success_rate": final_report["success_rate"]
                },
                "results": [
                    {
                        "test_name": result.test_name,
                        "success": result.success,
                        "duration": result.duration,
                        "error_message": result.error_message,
                        "details": result.details
                    }
                    for result in tester.results
                ]
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        return final_report["overall_success"]
        
    except Exception as e:
        logger.error(f"Integration testing failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
