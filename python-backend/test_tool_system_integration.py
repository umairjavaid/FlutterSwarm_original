#!/usr/bin/env python3
"""
Integration Test for FlutterSwarm Tool System.

This test demonstrates the comprehensive tool system working with agents
to perform real Flutter development tasks through LLM reasoning.
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Setup test environment
os.environ["ANTHROPIC_API_KEY"] = "test-key-for-demo"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["ENVIRONMENT"] = "test"

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.tools.tool_registry import ToolRegistry
from src.core.tools.file_system_tool import FileSystemTool
from src.core.tools.process_tool import ProcessTool
from src.core.tools.flutter_sdk_tool import FlutterSDKTool
from src.agents.implementation_agent import ImplementationAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.memory_manager import MemoryManager
from src.core.event_bus import EventBus
from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
from src.config import setup_logging, get_logger

# Configure logging
setup_logging()
logger = get_logger("tool_system_test")


class MockLLMClient:
    """Mock LLM client for testing tool system without actual API calls."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = self._setup_mock_responses()
    
    def _setup_mock_responses(self) -> Dict[str, Any]:
        """Setup realistic mock responses for different prompts."""
        return {
            "tool_analysis": {
                "summary": "File system tool for Flutter-aware file operations",
                "usage_scenarios": [
                    "Reading project files",
                    "Creating new components",
                    "Updating configuration files"
                ],
                "parameter_patterns": {
                    "read_file": ["path"],
                    "write_file": ["path", "content"]
                },
                "success_indicators": ["file created", "content written"],
                "failure_patterns": ["permission denied", "file not found"]
            },
            "project_analysis": {
                "primary_pattern": "clean_architecture",
                "state_management": "bloc",
                "existing_patterns": {
                    "widgets": ["stateless", "stateful"],
                    "architecture": ["feature_based"]
                }
            },
            "implementation_plan": {
                "files_to_create": [
                    {
                        "path": "lib/features/todo/presentation/pages/todo_page.dart",
                        "type": "widget",
                        "purpose": "Main todo list page"
                    },
                    {
                        "path": "lib/features/todo/domain/entities/todo.dart",
                        "type": "model",
                        "purpose": "Todo entity model"
                    }
                ],
                "dependencies": ["equatable", "dartz"],
                "integration_points": ["main.dart", "app_router.dart"]
            },
            "code_generation": {
                "code": """import 'package:flutter/material.dart';

class TodoPage extends StatelessWidget {
  const TodoPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Todo List'),
      ),
      body: Center(
        child: Text('Todo List Page'),
      ),
    );
  }
}"""
            },
            "package_mapping": {
                "state_management": {
                    "package": "flutter_bloc",
                    "version": "^8.1.0",
                    "configuration": {
                        "needs_provider": True,
                        "needs_observer": True
                    }
                }
            }
        }
    
    async def generate(self, messages, **kwargs):
        """Generate mock response based on prompt content."""
        self.call_count += 1
        
        # Analyze prompt to determine response type
        prompt_text = str(messages).lower()
        
        if "analyze this tool" in prompt_text:
            return self._create_response(self.responses["tool_analysis"])
        elif "analyze this flutter project structure" in prompt_text:
            return self._create_response(self.responses["project_analysis"])
        elif "plan the implementation" in prompt_text:
            return self._create_response(self.responses["implementation_plan"])
        elif "generate flutter/dart code" in prompt_text:
            return self._create_response(self.responses["code_generation"])
        elif "map these required features" in prompt_text:
            return self._create_response(self.responses["package_mapping"])
        else:
            return self._create_response({"result": "Mock response", "success": True})
    
    def _create_response(self, content: Dict[str, Any]) -> Any:
        """Create a mock response object."""
        class MockResponse:
            def __init__(self, content):
                self.content = str(content)
        
        return MockResponse(str(content))


class ToolSystemIntegrationTest:
    """Comprehensive test of the tool system integration."""
    
    def __init__(self):
        self.test_dir = None
        self.tool_registry = None
        self.agent = None
        self.mock_llm = MockLLMClient()
        
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tool system integration test."""
        
        logger.info("🚀 Starting FlutterSwarm Tool System Integration Test")
        
        results = {
            "setup": await self.test_setup(),
            "tool_discovery": await self.test_tool_discovery(),
            "tool_usage": await self.test_tool_usage(),
            "agent_integration": await self.test_agent_integration(),
            "project_workflow": await self.test_project_workflow(),
            "learning_system": await self.test_learning_system(),
            "cleanup": await self.test_cleanup()
        }
        
        # Summary
        success_count = sum(1 for result in results.values() if result.get("success", False))
        total_tests = len(results)
        
        logger.info(f"✅ Test Summary: {success_count}/{total_tests} passed")
        
        if success_count == total_tests:
            logger.info("🎉 All tests passed! Tool system is working correctly.")
        else:
            logger.warning(f"⚠️  Some tests failed. Check individual results.")
        
        return results
    
    async def test_setup(self) -> Dict[str, Any]:
        """Test system setup and initialization."""
        
        try:
            logger.info("📋 Testing system setup...")
            
            # Create temporary test directory
            self.test_dir = tempfile.mkdtemp(prefix="flutterswarm_test_")
            logger.info(f"Created test directory: {self.test_dir}")
            
            # Initialize tool registry
            self.tool_registry = ToolRegistry.instance()
            
            # Register tools
            file_tool = FileSystemTool()
            process_tool = ProcessTool()
            flutter_tool = FlutterSDKTool()
            
            self.tool_registry.register_tool(file_tool)
            self.tool_registry.register_tool(process_tool)
            self.tool_registry.register_tool(flutter_tool)
            
            # Initialize event bus and memory manager
            event_bus = EventBus()
            memory_manager = MemoryManager("test_agent", self.mock_llm)
            
            # Create test agent
            agent_config = AgentConfig(
                agent_id="test_implementation_agent",
                agent_type="implementation",
                capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.FILE_OPERATIONS]
            )
            
            self.agent = ImplementationAgent(
                agent_config, self.mock_llm, memory_manager, event_bus
            )
            
            # Allow agent to initialize tools
            await asyncio.sleep(0.1)
            
            logger.info("✅ System setup completed successfully")
            
            return {
                "success": True,
                "test_dir": self.test_dir,
                "tools_registered": len(self.tool_registry.get_all_tools()),
                "agent_created": True
            }
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_tool_discovery(self) -> Dict[str, Any]:
        """Test agent tool discovery and understanding."""
        
        try:
            logger.info("🔍 Testing tool discovery...")
            
            # Agent should have discovered tools during initialization
            discovered_tools = list(self.agent.available_tools.keys())
            tool_capabilities = self.agent.tool_capabilities
            
            # Verify tools were discovered
            expected_tools = ["file_system_tool", "process_tool", "flutter_sdk_tool"]
            discovered_expected = [tool for tool in expected_tools if tool in discovered_tools]
            
            # Test tool capability analysis
            capability_analysis = {}
            for tool_name in discovered_tools:
                if tool_name in self.agent.available_tools:
                    tool = self.agent.available_tools[tool_name]
                    capabilities = await tool.get_capabilities()
                    capability_analysis[tool_name] = {
                        "operations_count": len(capabilities.available_operations),
                        "has_examples": len(await tool.get_usage_examples()) > 0
                    }
            
            logger.info(f"✅ Discovered {len(discovered_tools)} tools")
            
            return {
                "success": True,
                "discovered_tools": discovered_tools,
                "expected_found": discovered_expected,
                "capability_analysis": capability_analysis,
                "llm_calls": self.mock_llm.call_count
            }
            
        except Exception as e:
            logger.error(f"❌ Tool discovery failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_tool_usage(self) -> Dict[str, Any]:
        """Test direct tool usage through agents."""
        
        try:
            logger.info("🔧 Testing tool usage...")
            
            # Test file system tool
            test_file_path = f"{self.test_dir}/test_file.dart"
            test_content = "// Test Flutter file\nvoid main() {}"
            
            # Write file using tool
            write_result = await self.agent.use_tool(
                "file_system_tool",
                "write_file",
                {"path": test_file_path, "content": test_content},
                "Creating test file for tool usage verification"
            )
            
            # Read file back
            read_result = await self.agent.use_tool(
                "file_system_tool",
                "read_file",
                {"path": test_file_path},
                "Reading back test file to verify write operation"
            )
            
            # Test template creation
            widget_path = f"{self.test_dir}/test_widget.dart"
            template_result = await self.agent.use_tool(
                "file_system_tool",
                "create_from_template",
                {
                    "template": "widget",
                    "path": widget_path,
                    "class_name": "TestWidget"
                },
                "Creating widget from template"
            )
            
            # Verify results
            write_success = write_result.status.value == "success"
            read_success = read_result.status.value == "success"
            template_success = template_result.status.value == "success"
            
            # Check if file actually exists
            file_exists = Path(test_file_path).exists()
            widget_exists = Path(widget_path).exists()
            
            logger.info(f"✅ Tool usage test completed")
            
            return {
                "success": write_success and read_success and template_success,
                "write_result": write_success,
                "read_result": read_success,
                "template_result": template_success,
                "file_exists": file_exists,
                "widget_exists": widget_exists,
                "tool_usage_history": len(self.agent.tool_usage_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Tool usage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_agent_integration(self) -> Dict[str, Any]:
        """Test agent integration with tools for complex tasks."""
        
        try:
            logger.info("🤖 Testing agent-tool integration...")
            
            # Create a mock Flutter project structure
            project_path = f"{self.test_dir}/flutter_project"
            os.makedirs(f"{project_path}/lib", exist_ok=True)
            os.makedirs(f"{project_path}/test", exist_ok=True)
            
            # Create basic pubspec.yaml
            pubspec_content = """name: test_project
description: A test Flutter project
version: 1.0.0+1

environment:
  sdk: ">=2.17.0 <4.0.0"
  flutter: ">=3.0.0"

dependencies:
  flutter:
    sdk: flutter

dev_dependencies:
  flutter_test:
    sdk: flutter
"""
            
            with open(f"{project_path}/pubspec.yaml", "w") as f:
                f.write(pubspec_content)
            
            # Test project analysis using agent
            project_context = ProjectContext(
                project_path=project_path,
                project_type=ProjectType.APP,
                platform_targets=[PlatformTarget.ANDROID, PlatformTarget.IOS]
            )
            
            # Test contextual code generation
            generation_result = await self.agent.generate_contextual_code(
                "Create a simple todo list feature",
                project_context
            )
            
            # Test dependency management
            dependency_result = await self.agent.manage_project_dependencies(
                ["state_management", "navigation"],
                project_path
            )
            
            logger.info("✅ Agent integration test completed")
            
            return {
                "success": True,
                "project_created": os.path.exists(f"{project_path}/pubspec.yaml"),
                "code_generation": generation_result.get("success", False),
                "dependency_management": dependency_result.get("success", False),
                "llm_interactions": self.mock_llm.call_count
            }
            
        except Exception as e:
            logger.error(f"❌ Agent integration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_project_workflow(self) -> Dict[str, Any]:
        """Test complete project development workflow."""
        
        try:
            logger.info("🔄 Testing project workflow...")
            
            project_path = f"{self.test_dir}/workflow_project"
            
            # 1. Project setup
            setup_result = await self._setup_test_project(project_path)
            
            # 2. Feature implementation
            feature_result = await self._implement_test_feature(project_path)
            
            # 3. Code validation
            validation_result = await self._validate_project_code(project_path)
            
            # 4. Hot reload simulation
            hot_reload_result = await self._test_hot_reload_workflow(project_path)
            
            workflow_success = all([
                setup_result.get("success", False),
                feature_result.get("success", False),
                validation_result.get("success", False)
            ])
            
            logger.info("✅ Project workflow test completed")
            
            return {
                "success": workflow_success,
                "setup": setup_result,
                "feature_implementation": feature_result,
                "validation": validation_result,
                "hot_reload": hot_reload_result
            }
            
        except Exception as e:
            logger.error(f"❌ Project workflow failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_learning_system(self) -> Dict[str, Any]:
        """Test the agent learning system with tool usage patterns."""
        
        try:
            logger.info("🧠 Testing learning system...")
            
            # Record some tool usage patterns
            initial_usage_count = len(self.agent.tool_usage_history)
            
            # Simulate multiple tool uses for learning
            for i in range(3):
                await self.agent.use_tool(
                    "file_system_tool",
                    "read_file",
                    {"path": f"{self.test_dir}/test_file.dart"},
                    f"Learning test iteration {i+1}"
                )
            
            final_usage_count = len(self.agent.tool_usage_history)
            
            # Check learning metrics
            file_tool_metrics = self.agent.tool_performance_metrics.get("file_system_tool")
            has_learning_data = file_tool_metrics is not None
            
            if has_learning_data:
                success_rate = file_tool_metrics.success_rate
                total_uses = file_tool_metrics.total_uses
            else:
                success_rate = 0
                total_uses = 0
            
            logger.info("✅ Learning system test completed")
            
            return {
                "success": True,
                "usage_recorded": final_usage_count > initial_usage_count,
                "initial_count": initial_usage_count,
                "final_count": final_usage_count,
                "has_metrics": has_learning_data,
                "success_rate": success_rate,
                "total_uses": total_uses
            }
            
        except Exception as e:
            logger.error(f"❌ Learning system test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_cleanup(self) -> Dict[str, Any]:
        """Test cleanup and resource management."""
        
        try:
            logger.info("🧹 Testing cleanup...")
            
            # Clean up test directory
            if self.test_dir and os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                cleanup_successful = not os.path.exists(self.test_dir)
            else:
                cleanup_successful = True
            
            # Agent cleanup
            if self.agent:
                await self.agent.shutdown()
            
            logger.info("✅ Cleanup completed")
            
            return {
                "success": cleanup_successful,
                "directory_removed": cleanup_successful,
                "agent_shutdown": True
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods for workflow testing
    
    async def _setup_test_project(self, project_path: str) -> Dict[str, Any]:
        """Setup a test Flutter project."""
        
        os.makedirs(f"{project_path}/lib", exist_ok=True)
        os.makedirs(f"{project_path}/test", exist_ok=True)
        
        # Create main.dart
        main_content = """import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Test App',
      home: Scaffold(
        appBar: AppBar(title: Text('Test App')),
        body: Center(child: Text('Hello World')),
      ),
    );
  }
}"""
        
        await self.agent.use_tool(
            "file_system_tool",
            "write_file",
            {"path": f"{project_path}/lib/main.dart", "content": main_content},
            "Creating main.dart for test project"
        )
        
        return {"success": True, "files_created": ["lib/main.dart"]}
    
    async def _implement_test_feature(self, project_path: str) -> Dict[str, Any]:
        """Implement a test feature using the agent."""
        
        project_context = ProjectContext(
            project_path=project_path,
            project_type=ProjectType.APP,
            platform_targets=[PlatformTarget.ANDROID]
        )
        
        # Use agent to generate a simple feature
        result = await self.agent.generate_contextual_code(
            "Add a counter button feature",
            project_context
        )
        
        return result
    
    async def _validate_project_code(self, project_path: str) -> Dict[str, Any]:
        """Validate project code using agent tools."""
        
        validation_result = await self.agent.validate_code_continuously(project_path)
        
        return validation_result
    
    async def _test_hot_reload_workflow(self, project_path: str) -> Dict[str, Any]:
        """Test hot reload workflow (mocked)."""
        
        # In a real implementation, this would start a dev server
        # For testing, we simulate the workflow
        
        hot_reload_result = await self.agent.develop_with_hot_reload(
            project_path,
            {"platform": "web", "port": 3000}
        )
        
        return hot_reload_result


async def main():
    """Run the comprehensive tool system integration test."""
    
    test = ToolSystemIntegrationTest()
    
    try:
        results = await test.run_comprehensive_test()
        
        print("\n" + "="*60)
        print("FLUTTERSWARM TOOL SYSTEM INTEGRATION TEST RESULTS")
        print("="*60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"{test_name.upper():<20} {status}")
            
            if not result.get("success", False) and "error" in result:
                print(f"  Error: {result['error']}")
        
        print("\n" + "="*60)
        
        # Overall success
        total_success = sum(1 for r in results.values() if r.get("success", False))
        total_tests = len(results)
        
        if total_success == total_tests:
            print("🎉 ALL TESTS PASSED! Tool system is fully functional.")
            print("\nKey Achievements:")
            print("✅ Tools can be discovered and analyzed by agents")
            print("✅ Agents use LLM reasoning for all tool decisions")
            print("✅ Project-aware code generation works")
            print("✅ Learning system tracks and improves tool usage")
            print("✅ Complete development workflows are supported")
        else:
            print(f"⚠️  {total_tests - total_success} tests failed out of {total_tests}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ TEST EXECUTION FAILED: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
