#!/usr/bin/env python3
"""
Comprehensive Integration Tests for BaseAgent Tool Integration System.

This test suite validates:
1. Complete tool discovery and understanding process
2. Tool usage with real Flutter tools
3. LLM integration with tool-aware prompts
4. Learning and adaptation mechanisms
5. Workflow execution and monitoring

Requirements verified:
- Tool discovery and capability analysis
- LLM-powered tool understanding
- Tool usage learning and optimization
- Inter-agent tool knowledge sharing
- Error handling and recovery
- Performance monitoring and metrics
"""

import asyncio
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
from src.core.tools.base_tool import BaseTool, ToolCategory
from src.core.tools.file_system_tool import FileSystemTool
from src.core.tools.flutter_sdk_tool import FlutterSDKTool
from src.core.tools.process_tool import ProcessTool
from src.core.tools.tool_registry import ToolRegistry
from src.models.tool_models import (
    ToolUsageEntry, ToolMetrics, ToolUnderstanding, ToolLearningModel,
    ToolCapabilities, ToolResult, ToolStatus, AsyncTask, TaskOutcome
)
from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
from src.core.memory_manager import MemoryManager
from src.core.event_bus import EventBus
from src.config import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger("test_base_agent_tools")


class TestBaseAgentToolIntegration(unittest.TestCase):
    """Comprehensive test suite for BaseAgent tool integration."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_llm_client = self._create_mock_llm_client()
        self.mock_memory_manager = self._create_mock_memory_manager()
        self.mock_event_bus = self._create_mock_event_bus()
        
        # Create test agent
        self.agent_config = AgentConfig(
            agent_id="test_agent_integration",
            agent_type="implementation",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.TESTING
            ],
            max_concurrent_tasks=5
        )
        
        self.test_agent = TestImplementationAgent(
            config=self.agent_config,
            llm_client=self.mock_llm_client,
            memory_manager=self.mock_memory_manager,
            event_bus=self.mock_event_bus
        )
        
        # Initialize tool registry with real tools
        self.tool_registry = ToolRegistry.instance()
        await self._setup_real_tools()

    async def asyncTearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    async def test_complete_tool_discovery_workflow(self):
        """Test complete tool discovery and understanding process."""
        logger.info("🔍 Testing Complete Tool Discovery Workflow")
        
        # Test discovery process
        available_tools = await self.test_agent.discover_available_tools()
        self.assertGreater(len(available_tools), 0, "Should discover available tools")
        
        # Test tool understanding for each discovered tool
        for tool in available_tools:
            understanding = await self.test_agent.analyze_tool_capabilities(tool)
            self.assertIsNotNone(understanding, f"Should understand {tool.name}")
            self.assertGreater(understanding.confidence_score, 0.5, "Understanding confidence should be reasonable")
            
            # Verify understanding contains required fields
            self.assertIn('usage_scenarios', understanding.analysis_result)
            self.assertIn('parameter_patterns', understanding.analysis_result)
            self.assertIn('decision_factors', understanding.analysis_result)
        
        logger.info(f"✅ Successfully analyzed {len(available_tools)} tools")
    
    async def test_tool_usage_with_real_flutter_tools(self):
        """Test tool usage with real Flutter development tools."""
        logger.info("🛠️ Testing Tool Usage with Real Flutter Tools")
        
        # Test FileSystemTool usage
        file_tool = FileSystemTool()
        
        # Create a test Flutter file
        test_file_path = os.path.join(self.test_dir, "test_widget.dart")
        dart_content = '''
import 'package:flutter/material.dart';

class TestWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      child: Text('Test Widget'),
    );
  }
}
'''
        
        # Test file creation through agent
        create_result = await self.test_agent.use_tool(
            tool=file_tool,
            operation="create_file",
            parameters={
                "file_path": test_file_path,
                "content": dart_content,
                "template": "dart_widget"
            }
        )
        
        self.assertEqual(create_result.status, ToolStatus.SUCCESS)
        self.assertTrue(os.path.exists(test_file_path))
        
        # Test FlutterSDKTool usage
        flutter_tool = FlutterSDKTool()
        
        # Test parameter validation
        project_params = {
            "project_name": "test_flutter_app",
            "output_directory": self.test_dir,
            "template": "app"
        }
        
        validation_result = await flutter_tool.validate_params("create_project", project_params)
        self.assertTrue(validation_result[0], "Flutter project parameters should be valid")
        
        logger.info("✅ Tool usage with real Flutter tools successful")
    
    async def test_llm_integration_with_tool_aware_prompts(self):
        """Test LLM integration with tool-aware prompts."""
        logger.info("🤖 Testing LLM Integration with Tool-Aware Prompts")
        
        # Test tool analysis prompt generation
        test_tool = FileSystemTool()
        analysis_prompt = await self.test_agent.generate_tool_analysis_prompt(test_tool)
        
        self.assertIn("tool capabilities", analysis_prompt.lower())
        self.assertIn("usage scenarios", analysis_prompt.lower())
        self.assertIn("flutter", analysis_prompt.lower())
        
        # Test tool selection prompt
        task_context = {
            "task_type": "create_flutter_widget",
            "requirements": ["file creation", "dart code generation"],
            "project_context": {"type": "mobile_app", "platform": "flutter"}
        }
        
        selection_prompt = await self.test_agent.generate_tool_selection_prompt(
            task_context, 
            [FileSystemTool(), FlutterSDKTool()]
        )
        
        self.assertIn("appropriate tool", selection_prompt.lower())
        self.assertIn("flutter", selection_prompt.lower())
        
        # Test LLM response parsing
        mock_response = {
            "selected_tool": "FileSystemTool",
            "confidence": 0.9,
            "reasoning": "File creation is required for widget development"
        }
        
        parsed_result = await self.test_agent.parse_tool_selection_response(mock_response)
        self.assertEqual(parsed_result["tool_name"], "FileSystemTool")
        self.assertGreater(parsed_result["confidence"], 0.8)
        
        logger.info("✅ LLM integration with tool-aware prompts successful")
    
    async def test_learning_and_adaptation_mechanisms(self):
        """Test learning and adaptation mechanisms."""
        logger.info("📚 Testing Learning and Adaptation Mechanisms")
        
        # Simulate tool usage history
        usage_entries = [
            ToolUsageEntry(
                tool_name="FileSystemTool",
                operation="create_file",
                timestamp=datetime.now() - timedelta(days=1),
                success_rate=0.95,
                performance_metrics={"latency": 0.1, "memory_usage": 1024}
            ),
            ToolUsageEntry(
                tool_name="FlutterSDKTool", 
                operation="create_project",
                timestamp=datetime.now() - timedelta(days=2),
                success_rate=0.8,
                performance_metrics={"latency": 5.0, "memory_usage": 2048}
            )
        ]
        
        # Test usage pattern learning
        for entry in usage_entries:
            await self.test_agent.update_tool_usage_patterns(entry)
        
        # Test adaptation based on outcomes
        successful_outcome = TaskOutcome(
            task_id="test_task_1",
            success=True,
            tools_used=["FileSystemTool"],
            execution_time=1.0,
            quality_score=0.9
        )
        
        await self.test_agent.learn_from_tool_outcomes([successful_outcome])
        
        # Test strategy adaptation
        old_strategy = self.test_agent.get_tool_selection_strategy()
        await self.test_agent.adapt_tool_selection_strategy(usage_entries)
        new_strategy = self.test_agent.get_tool_selection_strategy()
        
        # Strategy should have evolved based on usage patterns
        self.assertNotEqual(old_strategy, new_strategy)
        
        logger.info("✅ Learning and adaptation mechanisms working correctly")
    
    async def test_workflow_execution_and_monitoring(self):
        """Test workflow execution and monitoring."""
        logger.info("⚙️ Testing Workflow Execution and Monitoring")
        
        # Create a complex workflow task
        workflow_task = {
            "task_id": "create_flutter_feature",
            "steps": [
                {
                    "step": "create_model_file",
                    "tool": "FileSystemTool",
                    "operation": "create_file",
                    "parameters": {
                        "file_path": os.path.join(self.test_dir, "models", "user.dart"),
                        "template": "dart_model"
                    }
                },
                {
                    "step": "create_service_file", 
                    "tool": "FileSystemTool",
                    "operation": "create_file",
                    "parameters": {
                        "file_path": os.path.join(self.test_dir, "services", "user_service.dart"),
                        "template": "dart_service"
                    }
                }
            ]
        }
        
        # Execute workflow with monitoring
        workflow_result = await self.test_agent.execute_monitored_workflow(workflow_task)
        
        self.assertTrue(workflow_result["success"])
        self.assertEqual(len(workflow_result["completed_steps"]), 2)
        self.assertGreater(workflow_result["total_execution_time"], 0)
        
        # Verify monitoring data
        monitoring_data = workflow_result["monitoring_data"]
        self.assertIn("step_durations", monitoring_data)
        self.assertIn("tool_performance", monitoring_data)
        self.assertIn("resource_usage", monitoring_data)
        
        logger.info("✅ Workflow execution and monitoring successful")
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        logger.info("🚨 Testing Error Handling and Recovery")
        
        # Test tool execution error handling
        invalid_params = {
            "file_path": "/invalid/path/that/does/not/exist",
            "content": "test content"
        }
        
        file_tool = FileSystemTool()
        error_result = await self.test_agent.use_tool_with_recovery(
            tool=file_tool,
            operation="create_file",
            parameters=invalid_params
        )
        
        # Should handle error gracefully
        self.assertFalse(error_result["success"])
        self.assertIsNotNone(error_result["error_message"])
        self.assertIn("recovery_attempted", error_result)
        
        # Test LLM error recovery
        self.mock_llm_client.generate.side_effect = Exception("Mock LLM failure")
        
        recovery_result = await self.test_agent.analyze_tool_capabilities_with_fallback(file_tool)
        
        # Should fall back to cached or default understanding
        self.assertIsNotNone(recovery_result)
        self.assertGreater(recovery_result.confidence_score, 0)
        
        # Reset mock
        self.mock_llm_client.generate.side_effect = None
        
        logger.info("✅ Error handling and recovery working correctly")
    
    async def test_inter_agent_knowledge_sharing(self):
        """Test inter-agent tool knowledge sharing."""
        logger.info("🤝 Testing Inter-Agent Knowledge Sharing")
        
        # Create second agent
        agent2_config = AgentConfig(
            agent_id="test_agent_2",
            agent_type="testing",
            capabilities=[AgentCapability.TESTING, AgentCapability.QUALITY_ASSURANCE]
        )
        
        agent2 = TestImplementationAgent(
            config=agent2_config,
            llm_client=self.mock_llm_client,
            memory_manager=self.mock_memory_manager,
            event_bus=self.mock_event_bus
        )
        
        # Agent 1 learns about a tool
        tool_understanding = ToolUnderstanding(
            tool_name="TestTool",
            agent_id=self.test_agent.config.agent_id,
            confidence_score=0.9,
            analysis_result={
                "usage_scenarios": ["testing", "validation"],
                "best_practices": ["always validate input", "handle errors gracefully"]
            }
        )
        
        await self.test_agent.share_tool_knowledge(tool_understanding)
        
        # Agent 2 should receive the shared knowledge
        shared_knowledge = await agent2.get_shared_tool_knowledge("TestTool")
        
        self.assertIsNotNone(shared_knowledge)
        self.assertEqual(shared_knowledge.tool_name, "TestTool")
        self.assertGreater(shared_knowledge.confidence_score, 0.8)
        
        logger.info("✅ Inter-agent knowledge sharing working correctly")
    
    async def test_performance_monitoring_and_metrics(self):
        """Test performance monitoring and metrics collection."""
        logger.info("⚡ Testing Performance Monitoring and Metrics")
        
        # Execute multiple tool operations to collect metrics
        file_tool = FileSystemTool()
        
        start_time = time.time()
        
        for i in range(5):
            test_file = os.path.join(self.test_dir, f"test_file_{i}.dart")
            await self.test_agent.use_tool(
                tool=file_tool,
                operation="create_file",
                parameters={
                    "file_path": test_file,
                    "content": f"// Test file {i}\nclass TestClass{i} {{}}"
                }
            )
        
        end_time = time.time()
        
        # Get performance metrics
        metrics = await self.test_agent.get_tool_performance_metrics("FileSystemTool")
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_operations, 0)
        self.assertGreater(metrics.average_latency, 0)
        self.assertLessEqual(metrics.error_rate, 0.1)  # Low error rate expected
        
        # Test memory usage tracking
        memory_metrics = await self.test_agent.get_memory_usage_metrics()
        
        self.assertIn("current_usage", memory_metrics)
        self.assertIn("peak_usage", memory_metrics)
        self.assertIn("cleanup_events", memory_metrics)
        
        logger.info("✅ Performance monitoring and metrics collection successful")
    
    async def test_backward_compatibility(self):
        """Test backward compatibility with existing BaseAgent functionality."""
        logger.info("🔄 Testing Backward Compatibility")
        
        # Test that core BaseAgent methods still work
        capabilities = await self.test_agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertGreater(len(capabilities), 0)
        
        # Test system prompt generation
        system_prompt = await self.test_agent.get_system_prompt()
        self.assertIsInstance(system_prompt, str)
        self.assertGreater(len(system_prompt), 0)
        
        # Test task processing (mock)
        from src.models.task_models import TaskContext, TaskType
        
        task_context = TaskContext(
            task_id="test_task",
            task_type=TaskType.CODE_GENERATION,
            description="Create a simple Flutter widget",
            requirements=["Create widget file", "Add basic structure"]
        )
        
        # This should work without breaking existing functionality
        result = await self.test_agent.process_task(task_context)
        
        self.assertIsNotNone(result)
        self.assertIn("status", result)
        
        logger.info("✅ Backward compatibility maintained")

    def _create_mock_llm_client(self):
        """Create comprehensive mock LLM client."""
        mock_llm = Mock()
        
        # Comprehensive LLM responses for different scenarios
        def llm_response_generator(prompt=None, context=None, **kwargs):
            prompt_str = str(prompt).lower() if prompt else ""
            
            # Tool capability analysis responses
            if "analyze tool capability" in prompt_str or "tool understanding" in prompt_str:
                if "file" in prompt_str:
                    return {
                        "summary": "File system management tool for Flutter project file operations",
                        "usage_scenarios": [
                            "Creating Flutter project structure",
                            "Managing Dart source files",
                            "Handling asset files and resources",
                            "Creating test files and fixtures",
                            "Managing configuration files"
                        ],
                        "parameter_patterns": {
                            "file_path": "Absolute or relative path to file",
                            "content": "File content for write operations",
                            "template": "Template type for code generation"
                        },
                        "success_indicators": ["File created successfully", "Content written"],
                        "failure_patterns": ["Permission denied", "Path not found"],
                        "responsibility_mapping": {
                            "file_management": "Primary responsibility",
                            "code_generation": "Supporting role"
                        },
                        "decision_factors": ["File system operations needed"]
                    }
                elif "flutter" in prompt_str:
                    return {
                        "summary": "Flutter SDK tool for project creation and development operations",
                        "usage_scenarios": [
                            "Creating new Flutter projects",
                            "Building applications for different platforms",
                            "Running tests and analysis",
                            "Managing dependencies",
                            "Hot reload during development"
                        ],
                        "parameter_patterns": {
                            "project_name": "Name for new Flutter project",
                            "platform": "Target platform (android, ios, web)",
                            "project_path": "Path for project creation"
                        },
                        "success_indicators": ["Project created", "Build successful"],
                        "failure_patterns": ["SDK not found", "Build failed"],
                        "responsibility_mapping": {
                            "project_creation": "Primary responsibility",
                            "build_management": "Core feature"
                        },
                        "decision_factors": ["Flutter development tasks needed"]
                    }
                else:
                    return {
                        "summary": "General development tool",
                        "usage_scenarios": ["Development tasks"],
                        "parameter_patterns": {},
                        "success_indicators": ["Operation successful"],
                        "failure_patterns": ["Operation failed"],
                        "responsibility_mapping": {"general": "General tasks"},
                        "decision_factors": ["Development support needed"]
                    }
            
            # Tool preference building
            elif "tool preference" in prompt_str or "prioritize tools" in prompt_str:
                return {
                    "primary_tools": ["file_system_tool", "flutter_sdk_tool"],
                    "secondary_tools": ["process_tool"],
                    "specialized_tools": [],
                    "preference_scores": {
                        "file_system_tool": 0.95,
                        "flutter_sdk_tool": 0.90,
                        "process_tool": 0.75
                    },
                    "usage_priorities": {
                        "file_system_tool": "Always use for file operations",
                        "flutter_sdk_tool": "Primary for Flutter development",
                        "process_tool": "Use for system commands"
                    }
                }
            
            # Task planning and execution
            elif "plan task" in prompt_str or "execution strategy" in prompt_str:
                return {
                    "execution_plan": {
                        "steps": [
                            {"step": 1, "action": "Analyze project structure", "tool": "file_system_tool"},
                            {"step": 2, "action": "Create necessary files", "tool": "file_system_tool"},
                            {"step": 3, "action": "Run Flutter operations", "tool": "flutter_sdk_tool"}
                        ],
                        "estimated_duration": 300,
                        "confidence": 0.85,
                        "risk_factors": ["File permission issues", "SDK availability"]
                    }
                }
            
            # Tool usage reasoning
            elif "should use tool" in prompt_str or "tool selection" in prompt_str:
                return {
                    "recommended_tool": "file_system_tool",
                    "reasoning": "File operations are required for this task",
                    "confidence": 0.9,
                    "alternative_tools": ["process_tool"],
                    "parameters_suggestion": {
                        "file_path": "/project/lib/main.dart",
                        "content": "// Generated Flutter code"
                    }
                }
            
            # Default response
            return {"response": "Mock LLM response", "confidence": 0.8}
        
        mock_llm.generate = AsyncMock(side_effect=llm_response_generator)
        return mock_llm

    def _create_mock_memory_manager(self):
        """Create mock memory manager."""
        mock_memory = Mock()
        mock_memory.store_memory = AsyncMock()
        mock_memory.retrieve_memories = AsyncMock(return_value=[])
        mock_memory.search_memories = AsyncMock(return_value=[])
        return mock_memory

    def _create_mock_event_bus(self):
        """Create mock event bus."""
        mock_event_bus = Mock()
        mock_event_bus.subscribe = AsyncMock()
        mock_event_bus.publish = AsyncMock()
        mock_event_bus.unsubscribe = AsyncMock()
        return mock_event_bus

    async def _setup_real_tools(self):
        """Set up real tools for testing."""
        # Register real tools
        file_tool = FileSystemTool()
        flutter_tool = FlutterSDKTool()
        process_tool = ProcessTool()
        
        await self.tool_registry.register_tool(file_tool)
        await self.tool_registry.register_tool(flutter_tool)
        await self.tool_registry.register_tool(process_tool)

    async def test_complete_tool_discovery_process(self):
        """Test 1: Complete tool discovery and understanding process."""
        logger.info("🔍 Testing complete tool discovery process...")
        
        # Initialize tool discovery
        await self.test_agent.discover_available_tools()
        
        # Verify tools were discovered
        self.assertGreater(len(self.test_agent.available_tools), 0)
        self.assertIn("file_system_tool", self.test_agent.available_tools)
        
        # Verify tool capabilities were analyzed
        self.assertGreater(len(self.test_agent.tool_capabilities), 0)
        self.assertIn("file_system_tool", self.test_agent.tool_capabilities)
        
        # Verify performance metrics were initialized
        self.assertGreater(len(self.test_agent.tool_performance_metrics), 0)
        self.assertIn("file_system_tool", self.test_agent.tool_performance_metrics)
        
        # Verify memory storage was called
        self.mock_memory_manager.store_memory.assert_called()
        
        # Verify event subscriptions
        self.mock_event_bus.subscribe.assert_called()
        
        logger.info("✅ Tool discovery process completed successfully")

    async def test_tool_usage_with_real_flutter_tools(self):
        """Test 2: Tool usage with real Flutter tools."""
        logger.info("🛠️ Testing tool usage with real Flutter tools...")
        
        # Initialize tools first
        await self.test_agent.discover_available_tools()
        
        # Test file system tool usage
        test_file_path = f"{self.test_dir}/test_widget.dart"
        test_content = '''
import 'package:flutter/material.dart';

class TestWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      child: Text('Hello World'),
    );
  }
}
'''
        
        # Use tool through agent
        result = await self.test_agent.use_tool(
            tool_name="file_system_tool",
            operation="create_file",
            parameters={
                "file_path": test_file_path,
                "content": test_content,
                "template_type": "dart_widget"
            },
            reasoning="Creating a test Flutter widget for validation"
        )
        
        # Verify tool usage
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        
        # Verify file was created
        self.assertTrue(Path(test_file_path).exists())
        
        # Verify usage was recorded
        self.assertGreater(len(self.test_agent.tool_usage_history), 0)
        
        # Test tool usage metrics update
        metrics = self.test_agent.tool_performance_metrics["file_system_tool"]
        self.assertGreater(metrics.total_uses, 0)
        
        logger.info("✅ Real tool usage completed successfully")

    async def test_llm_integration_tool_aware_prompts(self):
        """Test 3: LLM integration with tool-aware prompts."""
        logger.info("🧠 Testing LLM integration with tool-aware prompts...")
        
        # Initialize tools
        await self.test_agent.discover_available_tools()
        
        # Test tool selection reasoning
        task_description = "Create a new Flutter widget with state management"
        
        # Agent should use LLM to reason about tool selection
        plan = await self.test_agent.plan_tool_usage(task_description)
        
        # Verify LLM was called for planning
        self.mock_llm_client.generate.assert_called()
        
        # Verify plan structure
        self.assertIn("execution_plan", plan)
        self.assertIn("steps", plan["execution_plan"])
        self.assertIn("confidence", plan["execution_plan"])
        
        # Test tool-specific prompt adaptation
        tool = self.test_agent.available_tools["file_system_tool"]
        understanding = await self.test_agent.analyze_tool_capability(tool)
        
        # Verify understanding was generated
        self.assertIsInstance(understanding, ToolUnderstanding)
        self.assertEqual(understanding.tool_name, "file_system_tool")
        self.assertGreater(len(understanding.usage_scenarios), 0)
        self.assertGreater(understanding.confidence_level, 0.5)
        
        logger.info("✅ LLM integration with tool-aware prompts working correctly")

    async def test_learning_and_adaptation_mechanisms(self):
        """Test 4: Learning and adaptation mechanisms."""
        logger.info("📚 Testing learning and adaptation mechanisms...")
        
        # Initialize tools
        await self.test_agent.discover_available_tools()
        
        # Simulate multiple tool usages for learning
        for i in range(5):
            result = await self.test_agent.use_tool(
                tool_name="file_system_tool",
                operation="create_file",
                parameters={
                    "file_path": f"{self.test_dir}/test_file_{i}.dart",
                    "content": f"// Test file {i}",
                },
                reasoning=f"Learning iteration {i}"
            )
            
            # Simulate different success rates
            if i % 2 == 0:
                await self.test_agent.record_tool_success("file_system_tool", 0.1)
            else:
                await self.test_agent.record_tool_failure("file_system_tool", "Simulated failure", 0.1)
        
        # Verify learning metrics
        metrics = self.test_agent.tool_performance_metrics["file_system_tool"]
        self.assertEqual(metrics.total_uses, 5)
        self.assertLess(metrics.success_rate, 1.0)  # Should reflect the failures
        
        # Test learning insights generation
        insights = await self.test_agent.generate_tool_insights()
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Verify learning data is stored
        learning_data = self.test_agent.tool_learning_models.get("file_system_tool")
        if learning_data:
            self.assertGreater(len(learning_data.usage_patterns), 0)
        
        logger.info("✅ Learning and adaptation mechanisms working correctly")

    async def test_workflow_execution_and_monitoring(self):
        """Test 5: Workflow execution and monitoring."""
        logger.info("⚙️ Testing workflow execution and monitoring...")
        
        # Initialize tools
        await self.test_agent.discover_available_tools()
        
        # Create a complex workflow
        workflow_tasks = [
            {
                "task": "create_project_structure",
                "tool": "file_system_tool",
                "operation": "create_directory",
                "parameters": {"directory_path": f"{self.test_dir}/flutter_app/lib"}
            },
            {
                "task": "create_main_file",
                "tool": "file_system_tool", 
                "operation": "create_file",
                "parameters": {
                    "file_path": f"{self.test_dir}/flutter_app/lib/main.dart",
                    "content": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(MyApp());"
                }
            },
            {
                "task": "create_widget",
                "tool": "file_system_tool",
                "operation": "create_from_template", 
                "parameters": {
                    "template": "stateful_widget",
                    "path": f"{self.test_dir}/flutter_app/lib/my_widget.dart",
                    "class_name": "MyWidget"
                }
            }
        ]
        
        # Execute workflow with monitoring
        workflow_results = []
        for task in workflow_tasks:
            start_time = datetime.now()
            
            result = await self.test_agent.use_tool(
                tool_name=task["tool"],
                operation=task["operation"],
                parameters=task["parameters"],
                reasoning=f"Executing workflow task: {task['task']}"
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_results.append({
                "task": task["task"],
                "result": result,
                "execution_time": execution_time,
                "success": result.status == ToolStatus.SUCCESS
            })
        
        # Verify workflow execution
        self.assertEqual(len(workflow_results), 3)
        successful_tasks = [r for r in workflow_results if r["success"]]
        self.assertGreater(len(successful_tasks), 0)
        
        # Verify monitoring data
        total_execution_time = sum(r["execution_time"] for r in workflow_results)
        self.assertGreater(total_execution_time, 0)
        
        # Verify async task tracking
        active_tasks = self.test_agent.active_tool_operations
        self.assertIsInstance(active_tasks, dict)
        
        logger.info("✅ Workflow execution and monitoring working correctly")

    async def test_inter_agent_tool_knowledge_sharing(self):
        """Test 6: Inter-agent tool knowledge sharing."""
        logger.info("🤝 Testing inter-agent tool knowledge sharing...")
        
        # Create a second agent
        second_agent_config = AgentConfig(
            agent_id="test_agent_secondary",
            agent_type="architecture",
            capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
            max_concurrent_tasks=3
        )
        
        second_agent = TestImplementationAgent(
            config=second_agent_config,
            llm_client=self.mock_llm_client,
            memory_manager=self.mock_memory_manager,
            event_bus=self.mock_event_bus
        )
        
        # First agent discovers tools
        await self.test_agent.discover_available_tools()
        
        # Simulate knowledge sharing through event bus
        tool_discovery_event = {
            "agent_id": self.test_agent.agent_id,
            "discovered_tools": list(self.test_agent.available_tools.keys()),
            "tool_capabilities": self.test_agent.tool_capabilities,
            "insights": await self.test_agent.generate_tool_insights()
        }
        
        # Publish discovery to event bus
        await self.test_agent.share_tool_discovery()
        
        # Verify event was published
        self.mock_event_bus.publish.assert_called()
        
        # Second agent receives shared knowledge
        await second_agent.receive_shared_tool_knowledge(tool_discovery_event)
        
        # Verify knowledge transfer
        self.assertGreater(len(second_agent.available_tools), 0)
        
        logger.info("✅ Inter-agent tool knowledge sharing working correctly")

    async def test_error_handling_and_recovery(self):
        """Test 7: Error handling and recovery scenarios."""
        logger.info("🚨 Testing error handling and recovery...")
        
        # Initialize tools
        await self.test_agent.discover_available_tools()
        
        # Test invalid tool usage
        result = await self.test_agent.use_tool(
            tool_name="nonexistent_tool",
            operation="invalid_operation",
            parameters={},
            reasoning="Testing error handling"
        )
        
        # Verify error handling
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertIn("error", result.data)
        
        # Test tool failure recovery
        with patch.object(self.test_agent.available_tools.get("file_system_tool", Mock()), 
                         'execute', side_effect=Exception("Simulated tool failure")):
            
            result = await self.test_agent.use_tool(
                tool_name="file_system_tool",
                operation="create_file",
                parameters={"file_path": "/invalid/path", "content": "test"},
                reasoning="Testing failure recovery"
            )
            
            # Verify graceful failure handling
            self.assertEqual(result.status, ToolStatus.FAILED)
        
        # Test retry mechanism
        retry_result = await self.test_agent.retry_failed_operation(
            tool_name="file_system_tool",
            operation="create_file",
            parameters={"file_path": f"{self.test_dir}/retry_test.dart", "content": "test"},
            max_retries=2
        )
        
        # Verify retry attempts
        self.assertIsInstance(retry_result, ToolResult)
        
        logger.info("✅ Error handling and recovery working correctly")

    async def test_performance_monitoring_and_optimization(self):
        """Test 8: Performance monitoring and optimization."""
        logger.info("📊 Testing performance monitoring and optimization...")
        
        # Initialize tools
        await self.test_agent.discover_available_tools()
        
        # Perform multiple operations for performance data
        operations = [
            ("create_file", {"file_path": f"{self.test_dir}/perf_test_{i}.dart", "content": f"// File {i}"})
            for i in range(10)
        ]
        
        start_time = datetime.now()
        for operation, params in operations:
            await self.test_agent.use_tool(
                tool_name="file_system_tool",
                operation=operation,
                parameters=params,
                reasoning="Performance testing"
            )
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Verify performance metrics
        metrics = self.test_agent.tool_performance_metrics["file_system_tool"]
        self.assertEqual(metrics.total_uses, 10)
        self.assertGreater(metrics.average_execution_time, 0)
        
        # Test performance optimization suggestions
        optimization_report = await self.test_agent.generate_performance_report()
        self.assertIn("tool_performance", optimization_report)
        self.assertIn("optimization_suggestions", optimization_report)
        
        # Test performance threshold monitoring
        if metrics.average_execution_time > 1.0:  # If operations are slow
            suggestions = await self.test_agent.get_performance_suggestions("file_system_tool")
            self.assertIsInstance(suggestions, list)
        
        logger.info("✅ Performance monitoring and optimization working correctly")

    async def test_tool_usage_patterns_and_learning(self):
        """Test 9: Tool usage patterns and learning optimization."""
        logger.info("🔄 Testing tool usage patterns and learning...")
        
        await self.test_agent.discover_available_tools()
        
        # Simulate consistent usage patterns
        pattern_tasks = [
            ("file_system_tool", "create_file", "morning_routine"),
            ("file_system_tool", "read_file", "morning_routine"),
            ("flutter_sdk_tool", "build_app", "build_phase"),
            ("file_system_tool", "create_file", "development"),
            ("flutter_sdk_tool", "test_app", "testing_phase")
        ]
        
        for tool_name, operation, pattern in pattern_tasks:
            await self.test_agent.use_tool(
                tool_name=tool_name,
                operation=operation,
                parameters={"pattern": pattern},
                reasoning=f"Pattern learning: {pattern}"
            )
        
        # Analyze usage patterns
        patterns = await self.test_agent.analyze_usage_patterns()
        self.assertIsInstance(patterns, dict)
        
        # Test predictive suggestions
        suggestions = await self.test_agent.suggest_next_tools("morning_routine")
        self.assertIsInstance(suggestions, list)
        
        logger.info("✅ Tool usage patterns and learning working correctly")

    async def test_tool_integration_completeness(self):
        """Test 10: Complete tool integration requirements verification."""
        logger.info("🔍 Testing tool integration completeness...")
        
        # Verify all required attributes exist
        required_attributes = [
            'available_tools',
            'tool_capabilities', 
            'tool_usage_history',
            'tool_performance_metrics',
            'tool_understanding_cache',
            'tool_learning_models',
            'active_tool_operations'
        ]
        
        for attr in required_attributes:
            self.assertTrue(hasattr(self.test_agent, attr), f"Missing attribute: {attr}")
        
        # Verify all required methods exist
        required_methods = [
            'discover_available_tools',
            'analyze_tool_capability',
            'use_tool',
            'plan_tool_usage',
            'learn_from_tool_usage',
            'share_tool_discovery',
            'generate_tool_insights'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(self.test_agent, method), f"Missing method: {method}")
            self.assertTrue(callable(getattr(self.test_agent, method)), f"Method not callable: {method}")
        
        # Test integration with existing functionality
        capabilities = await self.test_agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertGreater(len(capabilities), 0)
        
        logger.info("✅ Tool integration completeness verified")


class TestImplementationAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    async def _get_default_system_prompt(self) -> str:
        return "You are a test implementation agent for tool integration testing."
    
    async def get_capabilities(self) -> List[str]:
        return ["tool_integration_testing", "file_operations", "flutter_development"]


# Performance benchmark tests
class ToolIntegrationPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for tool integration system."""
    
    async def asyncSetUp(self):
        """Set up performance testing environment."""
        self.test_agent = TestImplementationAgent(
            config=AgentConfig(
                agent_id="perf_test_agent",
                agent_type="performance_test",
                capabilities=[AgentCapability.CODE_GENERATION],
                max_concurrent_tasks=10
            ),
            llm_client=Mock(),
            memory_manager=Mock(),
            event_bus=Mock()
        )
    
    async def test_tool_discovery_performance(self):
        """Benchmark tool discovery performance."""
        logger.info("⏱️ Benchmarking tool discovery performance...")
        
        start_time = datetime.now()
        await self.test_agent.discover_available_tools()
        discovery_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        self.assertLess(discovery_time, 5.0, "Tool discovery should complete within 5 seconds")
        
        logger.info(f"✅ Tool discovery completed in {discovery_time:.2f} seconds")
    
    async def test_tool_usage_throughput(self):
        """Benchmark tool usage throughput."""
        logger.info("⏱️ Benchmarking tool usage throughput...")
        
        await self.test_agent.discover_available_tools()
        
        # Simulate high-frequency tool usage
        start_time = datetime.now()
        operations_count = 50
        
        for i in range(operations_count):
            await self.test_agent.use_tool(
                tool_name="file_system_tool",
                operation="create_file",
                parameters={"file_path": f"/tmp/perf_test_{i}.dart", "content": "test"},
                reasoning="Performance test"
            )
        
        throughput_time = (datetime.now() - start_time).total_seconds()
        operations_per_second = operations_count / throughput_time
        
        # Performance assertions
        self.assertGreater(operations_per_second, 5, "Should handle at least 5 operations per second")
        
        logger.info(f"✅ Tool usage throughput: {operations_per_second:.2f} ops/sec")
    
    async def test_memory_usage_efficiency(self):
        """Benchmark memory usage efficiency."""
        logger.info("⏱️ Benchmarking memory usage efficiency...")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        await self.test_agent.discover_available_tools()
        
        for i in range(100):
            await self.test_agent.analyze_tool_capability(
                Mock(name=f"test_tool_{i}", category=ToolCategory.DEVELOPMENT)
            )
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        self.assertLess(memory_increase, 100, "Memory increase should be less than 100MB")
        
        logger.info(f"✅ Memory usage increase: {memory_increase:.2f} MB")


def create_test_suite():
    """Create comprehensive test suite."""
    suite = unittest.TestSuite()
    
    # Add main integration tests
    integration_tests = [
        'test_complete_tool_discovery_process',
        'test_tool_usage_with_real_flutter_tools', 
        'test_llm_integration_tool_aware_prompts',
        'test_learning_and_adaptation_mechanisms',
        'test_workflow_execution_and_monitoring',
        'test_inter_agent_tool_knowledge_sharing',
        'test_error_handling_and_recovery',
        'test_performance_monitoring_and_optimization',
        'test_tool_usage_patterns_and_learning',
        'test_tool_integration_completeness'
    ]
    
    for test_name in integration_tests:
        suite.addTest(TestBaseAgentToolIntegration(test_name))
    
    # Add performance benchmarks
    performance_tests = [
        'test_tool_discovery_performance',
        'test_tool_usage_throughput',
        'test_memory_usage_efficiency'
    ]
    
    for test_name in performance_tests:
        suite.addTest(ToolIntegrationPerformanceBenchmarks(test_name))
    
    return suite


async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting Comprehensive BaseAgent Tool Integration Tests")
    logger.info("=" * 70)
    
    # Create and run test suite
    suite = create_test_suite()
    
    # Custom test runner for async tests
    class AsyncTestRunner:
        def __init__(self):
            self.results = []
        
        async def run_test(self, test):
            try:
                if hasattr(test, 'asyncSetUp'):
                    await test.asyncSetUp()
                
                test_method = getattr(test, test._testMethodName)
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                if hasattr(test, 'asyncTearDown'):
                    await test.asyncTearDown()
                
                self.results.append({"test": test._testMethodName, "status": "PASSED"})
                logger.info(f"✅ {test._testMethodName} PASSED")
                
            except Exception as e:
                self.results.append({"test": test._testMethodName, "status": "FAILED", "error": str(e)})
                logger.error(f"❌ {test._testMethodName} FAILED: {e}")
        
        async def run_suite(self, test_suite):
            for test in test_suite:
                await self.run_test(test)
    
    # Run tests
    runner = AsyncTestRunner()
    await runner.run_suite(suite)
    
    # Generate summary report
    passed_tests = [r for r in runner.results if r["status"] == "PASSED"]
    failed_tests = [r for r in runner.results if r["status"] == "FAILED"]
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY REPORT")
    logger.info("=" * 70)
    logger.info(f"Total Tests: {len(runner.results)}")
    logger.info(f"Passed: {len(passed_tests)}")
    logger.info(f"Failed: {len(failed_tests)}")
    logger.info(f"Success Rate: {len(passed_tests)/len(runner.results)*100:.1f}%")
    
    if failed_tests:
        logger.info("\n❌ FAILED TESTS:")
        for test in failed_tests:
            logger.info(f"  - {test['test']}: {test.get('error', 'Unknown error')}")
    
    logger.info("\n✅ Comprehensive tool integration testing completed!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)
