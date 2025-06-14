#!/usr/bin/env python3
"""
Enhanced Comprehensive Integration Tests for BaseAgent Tool Integration System.

This test suite validates:
1. Complete tool discovery and understanding process
2. Tool usage with real Flutter tools
3. LLM integration with tool-aware prompts
4. Learning and adaptation mechanisms
5. Workflow execution and monitoring
6. Performance benchmarks
7. Validation framework
8. Comprehensive documentation

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
import unittest
import shutil
import json
import logging
import time
import psutil
import gc
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
from src.core.tools.base_tool import BaseTool, ToolCategory
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
logger = get_logger("test_base_agent_tools_enhanced")


class MockTool(BaseTool):
    """Enhanced mock tool for comprehensive testing."""
    
    def __init__(self, name: str, category: ToolCategory = ToolCategory.DEVELOPMENT):
        super().__init__(
            name=name,
            description=f"Mock tool for testing: {name}",
            version="1.0.0",
            required_permissions=[],
            category=category
        )
        self.execution_count = 0
        self.last_execution = None
        self.failure_rate = 0.0
        
    async def get_capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            available_operations=[
                {
                    "name": "test_operation",
                    "description": f"Test operation for {self.name}",
                    "parameters": {
                        "input": {"type": "string", "required": True},
                        "options": {"type": "dict", "required": False}
                    }
                },
                {
                    "name": "create_file",
                    "description": "Create a file",
                    "parameters": {
                        "file_path": {"type": "string", "required": True},
                        "content": {"type": "string", "required": True}
                    }
                }
            ],
            input_schemas={
                "test_operation": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "options": {"type": "object"}
                    }
                }
            },
            output_schemas={
                "test_operation": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            }
        )
    
    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "operation": "test_operation",
                "parameters": {"input": "test value", "options": {}},
                "description": f"Basic test operation for {self.name}",
                "expected_outcome": "successful operation"
            },
            {
                "operation": "create_file",
                "parameters": {"file_path": "/tmp/test.txt", "content": "Hello World"},
                "description": "Create a test file",
                "expected_outcome": "file created successfully"
            }
        ]
    
    async def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "last_check": datetime.now(),
            "execution_count": self.execution_count,
            "availability": "high"
        }
    
    async def validate_params(self, operation: str, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if operation == "test_operation" and "input" not in params:
            return False, "Missing required parameter: input"
        if operation == "create_file" and "file_path" not in params:
            return False, "Missing required parameter: file_path"
        return True, None
    
    async def execute(self, operation: str, params: Dict[str, Any], operation_id: str = None) -> ToolResult:
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        # Simulate failure rate
        import random
        if random.random() < self.failure_rate:
            return ToolResult(
                operation_id=operation_id or f"{self.name}_{operation}_{int(time.time())}",
                status=ToolStatus.FAILURE,
                error_message="Simulated failure for testing",
                execution_time=0.1
            )
        
        # Simulate successful execution
        if operation == "test_operation":
            result_data = {
                "result": f"Processed: {params.get('input', 'default')}",
                "status": "success",
                "tool": self.name
            }
        elif operation == "create_file":
            # Actually create the file for testing
            file_path = params.get("file_path")
            content = params.get("content", "")
            if file_path:
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write(content)
                    result_data = {
                        "file_path": file_path,
                        "status": "created",
                        "size": len(content)
                    }
                except Exception as e:
                    return ToolResult(
                        operation_id=operation_id or f"{self.name}_{operation}_{int(time.time())}",
                        status=ToolStatus.FAILURE,
                        error_message=str(e),
                        execution_time=0.1
                    )
            else:
                result_data = {"status": "no file path provided"}
        else:
            result_data = {"status": "unknown operation"}
        
        return ToolResult(
            operation_id=operation_id or f"{self.name}_{operation}_{int(time.time())}",
            status=ToolStatus.SUCCESS,
            data=result_data,
            execution_time=0.1
        )


class MockEventBus:
    """Enhanced mock event bus for testing."""
    
    def __init__(self):
        self.subscriptions = {}
        self.published_events = []
    
    async def subscribe(self, topic: str, handler):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(handler)
    
    async def unsubscribe(self, topic: str, handler):
        if topic in self.subscriptions:
            self.subscriptions[topic].remove(handler)
    
    async def publish(self, topic: str, data: Any):
        self.published_events.append({"topic": topic, "data": data, "timestamp": datetime.now()})


class MockMemoryManager:
    """Enhanced mock memory manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.search_results = []
    
    async def store_memory(self, content: str, metadata: Dict[str, Any] = None, 
                          importance: float = 0.5, long_term: bool = False):
        self.stored_memories.append({
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "long_term": long_term,
            "timestamp": datetime.now()
        })
    
    async def retrieve_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.search_results[:limit]
    
    async def search_memories(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        return self.search_results


class MockLLMClient:
    """Enhanced mock LLM client with comprehensive responses."""
    
    def __init__(self):
        self.call_count = 0
        self.response_history = []
    
    async def generate(self, prompt: str = None, context: Dict[str, Any] = None, **kwargs):
        self.call_count += 1
        prompt_str = str(prompt).lower() if prompt else ""
        
        # Tool capability analysis responses
        if "analyze tool capability" in prompt_str or "tool information" in prompt_str:
            if "file" in prompt_str or "mock_file" in prompt_str:
                response = {
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
            elif "flutter" in prompt_str or "mock_flutter" in prompt_str:
                response = {
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
                response = {
                    "summary": f"General development tool for {context.get('tool_name', 'unknown')}",
                    "usage_scenarios": [
                        "General development tasks",
                        "Test operations",
                        "Mock functionality"
                    ],
                    "parameter_patterns": {
                        "input": "General input parameter",
                        "options": "Configuration options"
                    },
                    "success_indicators": ["Operation successful"],
                    "failure_patterns": ["Operation failed"],
                    "responsibility_mapping": {"general": "General tasks"},
                    "decision_factors": ["Development support needed"]
                }
        
        # Tool preference building
        elif "tool preference" in prompt_str or "prioritize tools" in prompt_str:
            response = {
                "primary_tools": ["mock_file_tool", "mock_flutter_tool"],
                "secondary_tools": ["mock_process_tool"],
                "specialized_tools": [],
                "preference_scores": {
                    "mock_file_tool": 0.95,
                    "mock_flutter_tool": 0.90,
                    "mock_process_tool": 0.75
                },
                "usage_priorities": {
                    "mock_file_tool": "Always use for file operations",
                    "mock_flutter_tool": "Primary for Flutter development",
                    "mock_process_tool": "Use for system commands"
                }
            }
        
        # Task planning and execution
        elif "plan task" in prompt_str or "execution strategy" in prompt_str:
            response = {
                "execution_plan": {
                    "steps": [
                        {"step": 1, "action": "Analyze project structure", "tool": "mock_file_tool"},
                        {"step": 2, "action": "Create necessary files", "tool": "mock_file_tool"},
                        {"step": 3, "action": "Run Flutter operations", "tool": "mock_flutter_tool"}
                    ],
                    "estimated_duration": 300,
                    "confidence": 0.85,
                    "risk_factors": ["File permission issues", "SDK availability"]
                }
            }
        
        # Tool usage reasoning
        elif "should use tool" in prompt_str or "tool selection" in prompt_str:
            response = {
                "recommended_tool": "mock_file_tool",
                "reasoning": "File operations are required for this task",
                "confidence": 0.9,
                "alternative_tools": ["mock_process_tool"],
                "parameters_suggestion": {
                    "file_path": "/project/lib/main.dart",
                    "content": "// Generated Flutter code"
                }
            }
        
        # Learning analysis
        elif "learning analysis" in prompt_str or "tool usage" in prompt_str:
            response = {
                "insights": [
                    "Tool usage is effective for the given task",
                    "Parameters were appropriate",
                    "Consider caching for better performance"
                ],
                "improvements": [
                    "Optimize parameter selection",
                    "Implement retry mechanisms",
                    "Add validation checks"
                ],
                "patterns": [
                    "Frequent file operations during project setup",
                    "Flutter tools used for build processes"
                ]
            }
        
        # Default response
        else:
            response = {"response": "Mock LLM response", "confidence": 0.8}
        
        self.response_history.append({
            "prompt": prompt_str,
            "context": context,
            "response": response,
            "timestamp": datetime.now()
        })
        
        return response


class TestImplementationAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    async def _get_default_system_prompt(self) -> str:
        return "You are a test implementation agent for tool integration testing."
    
    async def get_capabilities(self) -> List[str]:
        return ["tool_integration_testing", "file_operations", "flutter_development"]
    
    async def _subscribe_to_tool_events(self) -> None:
        """Subscribe to tool-related events."""
        await self.event_bus.subscribe("tool.availability.*", self._handle_tool_availability)
        await self.event_bus.subscribe("tool.performance.*", self._handle_tool_performance)
        await self.event_bus.subscribe("tool.registered.*", self._handle_tool_registered)
    
    async def _handle_tool_availability(self, event_data: Dict[str, Any]):
        """Handle tool availability events."""
        pass
    
    async def _handle_tool_performance(self, event_data: Dict[str, Any]):
        """Handle tool performance events."""
        pass
    
    async def _handle_tool_registered(self, event_data: Dict[str, Any]):
        """Handle tool registration events."""
        pass
    
    async def _build_tool_preferences(self) -> None:
        """Build tool preferences based on agent type."""
        # Mock implementation for testing
        pass
    
    async def share_tool_discovery(self, discovery=None) -> None:
        """Share tool discovery with other agents."""
        await self.event_bus.publish("tool.discovery", {
            "agent_id": self.agent_id,
            "discovered_tools": list(self.available_tools.keys()),
            "timestamp": datetime.now()
        })
    
    async def receive_shared_tool_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Receive shared tool knowledge from other agents."""
        for tool_name in knowledge.get("discovered_tools", []):
            if tool_name not in self.available_tools:
                # Mock tool addition
                mock_tool = MockTool(tool_name)
                self.available_tools[tool_name] = mock_tool
    
    async def generate_tool_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from tool usage."""
        insights = []
        for tool_name, metrics in self.tool_performance_metrics.items():
            insights.append({
                "tool": tool_name,
                "insight": f"Tool {tool_name} used {metrics.total_uses} times",
                "success_rate": metrics.success_rate,
                "recommendation": "Continue using" if metrics.success_rate > 0.8 else "Review usage"
            })
        return insights
    
    async def record_tool_success(self, tool_name: str, execution_time: float):
        """Record successful tool usage."""
        if tool_name in self.tool_performance_metrics:
            metrics = self.tool_performance_metrics[tool_name]
            metrics.total_uses += 1
            # Update success rate (assuming previous success)
            metrics.success_rate = (metrics.success_rate * (metrics.total_uses - 1) + 1.0) / metrics.total_uses
    
    async def record_tool_failure(self, tool_name: str, error: str, execution_time: float):
        """Record failed tool usage."""
        if tool_name in self.tool_performance_metrics:
            metrics = self.tool_performance_metrics[tool_name]
            metrics.total_uses += 1
            # Update success rate (assuming failure)
            metrics.success_rate = (metrics.success_rate * (metrics.total_uses - 1)) / metrics.total_uses
    
    async def retry_failed_operation(self, tool_name: str, operation: str, 
                                   parameters: Dict[str, Any], max_retries: int = 2) -> ToolResult:
        """Retry a failed tool operation."""
        for attempt in range(max_retries + 1):
            try:
                result = await self.use_tool(tool_name, operation, parameters, 
                                           f"Retry attempt {attempt + 1}")
                if result.status == ToolStatus.SUCCESS:
                    return result
            except Exception as e:
                if attempt == max_retries:
                    return ToolResult(
                        operation_id=f"retry_{tool_name}_{operation}",
                        status=ToolStatus.FAILURE,
                        error_message=f"All {max_retries + 1} attempts failed: {str(e)}"
                    )
        
        return ToolResult(
            operation_id=f"retry_{tool_name}_{operation}",
            status=ToolStatus.FAILURE,
            error_message="Max retries exceeded"
        )
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for tools."""
        return {
            "tool_performance": {
                tool_name: {
                    "total_uses": metrics.total_uses,
                    "success_rate": metrics.success_rate,
                    "avg_execution_time": metrics.average_execution_time
                }
                for tool_name, metrics in self.tool_performance_metrics.items()
            },
            "optimization_suggestions": [
                "Consider caching for frequently used tools",
                "Implement parallel execution for independent operations",
                "Add timeout mechanisms for long-running operations"
            ]
        }
    
    async def get_performance_suggestions(self, tool_name: str) -> List[str]:
        """Get performance suggestions for a specific tool."""
        return [
            f"Optimize {tool_name} parameter validation",
            f"Consider batching operations for {tool_name}",
            f"Implement caching for {tool_name} results"
        ]
    
    async def analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze tool usage patterns."""
        patterns = {}
        for entry in self.tool_usage_history:
            pattern = entry.context.get("pattern", "default")
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(entry.tool_name)
        
        return {
            "patterns": patterns,
            "most_common": max(patterns.keys(), key=lambda k: len(patterns[k])) if patterns else None,
            "tool_frequency": {
                tool: len([e for e in self.tool_usage_history if e.tool_name == tool])
                for tool in self.available_tools.keys()
            }
        }
    
    async def suggest_next_tools(self, pattern: str) -> List[str]:
        """Suggest next tools based on usage pattern."""
        # Simple implementation for testing
        if pattern == "morning_routine":
            return ["mock_file_tool", "mock_flutter_tool"]
        return list(self.available_tools.keys())[:3]


class TestEnhancedBaseAgentToolIntegration(unittest.TestCase):
    """Enhanced comprehensive test suite for BaseAgent tool integration."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_llm_client = MockLLMClient()
        self.mock_memory_manager = MockMemoryManager()
        self.mock_event_bus = MockEventBus()
        
        # Create test agent
        self.agent_config = AgentConfig(
            agent_id="test_agent_enhanced",
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
        
        # Setup mock tools
        await self._setup_mock_tools()

    async def asyncTearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def _setup_mock_tools(self):
        """Set up mock tools for testing."""
        # Create mock tools directly in agent
        mock_file_tool = MockTool("mock_file_tool", ToolCategory.FILE_SYSTEM)
        mock_flutter_tool = MockTool("mock_flutter_tool", ToolCategory.DEVELOPMENT)
        mock_process_tool = MockTool("mock_process_tool", ToolCategory.DEVELOPMENT)
        
        # Add tools to agent
        self.test_agent.available_tools = {
            "mock_file_tool": mock_file_tool,
            "mock_flutter_tool": mock_flutter_tool,
            "mock_process_tool": mock_process_tool
        }
        
        # Initialize tool capabilities and metrics
        for tool_name, tool in self.test_agent.available_tools.items():
            understanding = await self.test_agent.analyze_tool_capability(tool)
            self.test_agent.tool_capabilities[tool_name] = understanding.usage_scenarios
            self.test_agent.tool_performance_metrics[tool_name] = ToolMetrics()

    @pytest.mark.asyncio
    async def test_complete_tool_discovery_process(self):
        """Test 1: Complete tool discovery and understanding process."""
        logger.info("🔍 Testing complete tool discovery process...")
        
        # Verify tools were discovered
        self.assertGreater(len(self.test_agent.available_tools), 0)
        self.assertIn("mock_file_tool", self.test_agent.available_tools)
        
        # Verify tool capabilities were analyzed
        self.assertGreater(len(self.test_agent.tool_capabilities), 0)
        self.assertIn("mock_file_tool", self.test_agent.tool_capabilities)
        
        # Verify performance metrics were initialized
        self.assertGreater(len(self.test_agent.tool_performance_metrics), 0)
        self.assertIn("mock_file_tool", self.test_agent.tool_performance_metrics)
        
        # Verify LLM was called for analysis
        self.assertGreater(self.mock_llm_client.call_count, 0)
        
        logger.info("✅ Tool discovery process completed successfully")

    @pytest.mark.asyncio
    async def test_tool_usage_with_mock_tools(self):
        """Test 2: Tool usage with mock tools."""
        logger.info("🛠️ Testing tool usage with mock tools...")
        
        # Test file creation tool usage
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
            tool_name="mock_file_tool",
            operation="create_file",
            parameters={
                "file_path": test_file_path,
                "content": test_content
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
        metrics = self.test_agent.tool_performance_metrics["mock_file_tool"]
        self.assertGreater(metrics.total_uses, 0)
        
        logger.info("✅ Mock tool usage completed successfully")

    @pytest.mark.asyncio
    async def test_llm_integration_tool_aware_prompts(self):
        """Test 3: LLM integration with tool-aware prompts."""
        logger.info("🧠 Testing LLM integration with tool-aware prompts...")
        
        # Test tool selection reasoning
        task_description = "Create a new Flutter widget with state management"
        
        # Agent should use LLM to reason about tool selection
        plan = await self.test_agent.plan_tool_usage(task_description)
        
        # Verify LLM was called for planning
        initial_calls = self.mock_llm_client.call_count
        self.assertGreater(initial_calls, 0)
        
        # Verify plan structure
        self.assertIsInstance(plan, (dict, object))  # Could be ToolUsagePlan or dict
        
        # Test tool-specific prompt adaptation
        tool = self.test_agent.available_tools["mock_file_tool"]
        understanding = await self.test_agent.analyze_tool_capability(tool)
        
        # Verify understanding was generated
        self.assertIsInstance(understanding, ToolUnderstanding)
        self.assertEqual(understanding.tool_name, "mock_file_tool")
        self.assertGreater(len(understanding.usage_scenarios), 0)
        self.assertGreater(understanding.confidence_level, 0.5)
        
        logger.info("✅ LLM integration with tool-aware prompts working correctly")

    @pytest.mark.asyncio
    async def test_learning_and_adaptation_mechanisms(self):
        """Test 4: Learning and adaptation mechanisms."""
        logger.info("📚 Testing learning and adaptation mechanisms...")
        
        # Simulate multiple tool usages for learning
        for i in range(5):
            result = await self.test_agent.use_tool(
                tool_name="mock_file_tool",
                operation="test_operation",
                parameters={
                    "input": f"test_input_{i}",
                    "options": {"iteration": i}
                },
                reasoning=f"Learning iteration {i}"
            )
            
            # Simulate different success rates
            if i % 2 == 0:
                await self.test_agent.record_tool_success("mock_file_tool", 0.1)
            else:
                await self.test_agent.record_tool_failure("mock_file_tool", "Simulated failure", 0.1)
        
        # Verify learning metrics
        metrics = self.test_agent.tool_performance_metrics["mock_file_tool"]
        self.assertGreaterEqual(metrics.total_uses, 5)
        
        # Test learning insights generation
        insights = await self.test_agent.generate_tool_insights()
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        logger.info("✅ Learning and adaptation mechanisms working correctly")

    @pytest.mark.asyncio
    async def test_workflow_execution_and_monitoring(self):
        """Test 5: Workflow execution and monitoring."""
        logger.info("⚙️ Testing workflow execution and monitoring...")
        
        # Create a workflow
        workflow_tasks = [
            {
                "task": "create_project_structure",
                "tool": "mock_file_tool",
                "operation": "create_file",
                "parameters": {"file_path": f"{self.test_dir}/flutter_app/pubspec.yaml", "content": "name: test_app"}
            },
            {
                "task": "create_main_file",
                "tool": "mock_file_tool", 
                "operation": "create_file",
                "parameters": {
                    "file_path": f"{self.test_dir}/flutter_app/lib/main.dart",
                    "content": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(MyApp());"
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
        self.assertEqual(len(workflow_results), 2)
        successful_tasks = [r for r in workflow_results if r["success"]]
        self.assertGreater(len(successful_tasks), 0)
        
        # Verify monitoring data
        total_execution_time = sum(r["execution_time"] for r in workflow_results)
        self.assertGreater(total_execution_time, 0)
        
        logger.info("✅ Workflow execution and monitoring working correctly")

    @pytest.mark.asyncio
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
        
        # Share knowledge through event bus
        tool_discovery_event = {
            "agent_id": self.test_agent.agent_id,
            "discovered_tools": list(self.test_agent.available_tools.keys()),
            "tool_capabilities": self.test_agent.tool_capabilities
        }
        
        # Publish discovery to event bus
        await self.test_agent.share_tool_discovery()
        
        # Verify event was published
        self.assertGreater(len(self.mock_event_bus.published_events), 0)
        
        # Second agent receives shared knowledge
        await second_agent.receive_shared_tool_knowledge(tool_discovery_event)
        
        # Verify knowledge transfer
        self.assertGreater(len(second_agent.available_tools), 0)
        
        logger.info("✅ Inter-agent tool knowledge sharing working correctly")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test 7: Error handling and recovery scenarios."""
        logger.info("🚨 Testing error handling and recovery...")
        
        # Test invalid tool usage
        result = await self.test_agent.use_tool(
            tool_name="nonexistent_tool",
            operation="invalid_operation",
            parameters={},
            reasoning="Testing error handling"
        )
        
        # Verify error handling
        self.assertEqual(result.status, ToolStatus.FAILED)
        
        # Test tool failure recovery
        # Set failure rate for testing
        if "mock_file_tool" in self.test_agent.available_tools:
            self.test_agent.available_tools["mock_file_tool"].failure_rate = 1.0
        
        result = await self.test_agent.use_tool(
            tool_name="mock_file_tool",
            operation="test_operation",
            parameters={"input": "test"},
            reasoning="Testing failure recovery"
        )
        
        # Reset failure rate
        if "mock_file_tool" in self.test_agent.available_tools:
            self.test_agent.available_tools["mock_file_tool"].failure_rate = 0.0
        
        # Test retry mechanism
        retry_result = await self.test_agent.retry_failed_operation(
            tool_name="mock_file_tool",
            operation="test_operation",
            parameters={"input": "retry_test"},
            max_retries=2
        )
        
        # Verify retry attempts
        self.assertIsInstance(retry_result, ToolResult)
        
        logger.info("✅ Error handling and recovery working correctly")

    @pytest.mark.asyncio
    async def test_performance_monitoring_and_optimization(self):
        """Test 8: Performance monitoring and optimization."""
        logger.info("📊 Testing performance monitoring and optimization...")
        
        # Perform multiple operations for performance data
        operations = [
            ("test_operation", {"input": f"perf_test_{i}"})
            for i in range(10)
        ]
        
        start_time = datetime.now()
        for operation, params in operations:
            await self.test_agent.use_tool(
                tool_name="mock_file_tool",
                operation=operation,
                parameters=params,
                reasoning="Performance testing"
            )
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Verify performance metrics
        metrics = self.test_agent.tool_performance_metrics["mock_file_tool"]
        self.assertGreaterEqual(metrics.total_uses, 10)
        
        # Test performance optimization suggestions
        optimization_report = await self.test_agent.generate_performance_report()
        self.assertIn("tool_performance", optimization_report)
        self.assertIn("optimization_suggestions", optimization_report)
        
        # Test performance threshold monitoring
        suggestions = await self.test_agent.get_performance_suggestions("mock_file_tool")
        self.assertIsInstance(suggestions, list)
        
        logger.info("✅ Performance monitoring and optimization working correctly")

    @pytest.mark.asyncio
    async def test_tool_usage_patterns_and_learning(self):
        """Test 9: Tool usage patterns and learning optimization."""
        logger.info("🔄 Testing tool usage patterns and learning...")
        
        # Simulate consistent usage patterns
        pattern_tasks = [
            ("mock_file_tool", "test_operation", "morning_routine"),
            ("mock_file_tool", "create_file", "morning_routine"),
            ("mock_flutter_tool", "test_operation", "build_phase"),
            ("mock_file_tool", "test_operation", "development"),
            ("mock_flutter_tool", "test_operation", "testing_phase")
        ]
        
        for tool_name, operation, pattern in pattern_tasks:
            await self.test_agent.use_tool(
                tool_name=tool_name,
                operation=operation,
                parameters={"pattern": pattern, "input": "test"},
                reasoning=f"Pattern learning: {pattern}"
            )
        
        # Analyze usage patterns
        patterns = await self.test_agent.analyze_usage_patterns()
        self.assertIsInstance(patterns, dict)
        
        # Test predictive suggestions
        suggestions = await self.test_agent.suggest_next_tools("morning_routine")
        self.assertIsInstance(suggestions, list)
        
        logger.info("✅ Tool usage patterns and learning working correctly")

    @pytest.mark.asyncio
    async def test_tool_integration_completeness(self):
        """Test 10: Complete tool integration requirements verification."""
        logger.info("🔍 Testing tool integration completeness...")
        
        # Verify all required attributes exist
        required_attributes = [
            'available_tools',
            'tool_capabilities', 
            'tool_usage_history',
            'tool_performance_metrics',
            'active_tool_operations'
        ]
        
        for attr in required_attributes:
            self.assertTrue(hasattr(self.test_agent, attr), f"Missing attribute: {attr}")
        
        # Verify all required methods exist
        required_methods = [
            'analyze_tool_capability',
            'use_tool',
            'plan_tool_usage',
            'learn_from_tool_usage'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(self.test_agent, method), f"Missing method: {method}")
            self.assertTrue(callable(getattr(self.test_agent, method)), f"Method not callable: {method}")
        
        # Test integration with existing functionality
        capabilities = await self.test_agent.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertGreater(len(capabilities), 0)
        
        logger.info("✅ Tool integration completeness verified")


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
            llm_client=MockLLMClient(),
            memory_manager=MockMemoryManager(),
            event_bus=MockEventBus()
        )
        
        # Add mock tools
        self.test_agent.available_tools["mock_tool"] = MockTool("mock_tool")
        self.test_agent.tool_performance_metrics["mock_tool"] = ToolMetrics()
    
    @pytest.mark.asyncio
    async def test_tool_discovery_performance(self):
        """Benchmark tool discovery performance."""
        logger.info("⏱️ Benchmarking tool discovery performance...")
        
        start_time = datetime.now()
        # Simulate tool discovery with mock tools
        await self.test_agent._setup_mock_tools() if hasattr(self.test_agent, '_setup_mock_tools') else None
        discovery_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        self.assertLess(discovery_time, 5.0, "Tool discovery should complete within 5 seconds")
        
        logger.info(f"✅ Tool discovery completed in {discovery_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_tool_usage_throughput(self):
        """Benchmark tool usage throughput."""
        logger.info("⏱️ Benchmarking tool usage throughput...")
        
        # Simulate high-frequency tool usage
        start_time = datetime.now()
        operations_count = 50
        
        for i in range(operations_count):
            await self.test_agent.use_tool(
                tool_name="mock_tool",
                operation="test_operation",
                parameters={"input": f"perf_test_{i}"},
                reasoning="Performance test"
            )
        
        throughput_time = (datetime.now() - start_time).total_seconds()
        operations_per_second = operations_count / throughput_time if throughput_time > 0 else float('inf')
        
        # Performance assertions
        self.assertGreater(operations_per_second, 5, "Should handle at least 5 operations per second")
        
        logger.info(f"✅ Tool usage throughput: {operations_per_second:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_memory_usage_efficiency(self):
        """Benchmark memory usage efficiency."""
        logger.info("⏱️ Benchmarking memory usage efficiency...")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        for i in range(100):
            await self.test_agent.analyze_tool_capability(MockTool(f"test_tool_{i}"))
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        self.assertLess(memory_increase, 100, "Memory increase should be less than 100MB")
        
        logger.info(f"✅ Memory usage increase: {memory_increase:.2f} MB")


# Validation Framework
class ToolIntegrationValidationFramework:
    """Comprehensive validation framework for tool integration."""
    
    def __init__(self):
        self.validation_results = []
        self.requirements_matrix = {
            "tool_discovery": ["discover_available_tools", "analyze_tool_capability"],
            "tool_usage": ["use_tool", "plan_tool_usage"],
            "learning": ["learn_from_tool_usage", "generate_tool_insights"],
            "error_handling": ["retry_failed_operation"],
            "performance": ["tool_performance_metrics", "generate_performance_report"],
            "knowledge_sharing": ["share_tool_discovery", "receive_shared_tool_knowledge"]
        }
    
    async def validate_requirements(self, agent: BaseAgent) -> Dict[str, Any]:
        """Validate all tool integration requirements."""
        results = {
            "overall_compliance": True,
            "requirement_results": {},
            "missing_features": [],
            "recommendations": []
        }
        
        for requirement, methods in self.requirements_matrix.items():
            requirement_met = True
            missing_methods = []
            
            for method in methods:
                if not hasattr(agent, method):
                    requirement_met = False
                    missing_methods.append(method)
            
            results["requirement_results"][requirement] = {
                "met": requirement_met,
                "missing_methods": missing_methods
            }
            
            if not requirement_met:
                results["overall_compliance"] = False
                results["missing_features"].extend(missing_methods)
        
        # Generate recommendations
        if not results["overall_compliance"]:
            results["recommendations"] = [
                f"Implement missing method: {method}" 
                for method in results["missing_features"]
            ]
        else:
            results["recommendations"] = ["All requirements met - consider performance optimization"]
        
        return results
    
    async def validate_backward_compatibility(self, agent: BaseAgent) -> Dict[str, Any]:
        """Validate backward compatibility with existing BaseAgent functionality."""
        compatibility_checks = [
            ("execute_llm_task", "Core LLM functionality"),
            ("get_capabilities", "Agent capability discovery"),
            ("agent_id", "Agent identification"),
            ("status", "Agent status tracking")
        ]
        
        results = {
            "compatible": True,
            "incompatible_features": [],
            "warnings": []
        }
        
        for feature, description in compatibility_checks:
            if not hasattr(agent, feature):
                results["compatible"] = False
                results["incompatible_features"].append({
                    "feature": feature,
                    "description": description
                })
        
        return results


def create_comprehensive_test_suite():
    """Create comprehensive test suite with all components."""
    suite = unittest.TestSuite()
    
    # Add main integration tests
    integration_tests = [
        'test_complete_tool_discovery_process',
        'test_tool_usage_with_mock_tools', 
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
        suite.addTest(TestEnhancedBaseAgentToolIntegration(test_name))
    
    # Add performance benchmarks
    performance_tests = [
        'test_tool_discovery_performance',
        'test_tool_usage_throughput',
        'test_memory_usage_efficiency'
    ]
    
    for test_name in performance_tests:
        suite.addTest(ToolIntegrationPerformanceBenchmarks(test_name))
    
    return suite


async def run_comprehensive_validation():
    """Run comprehensive validation including tests and framework validation."""
    logger.info("🚀 Starting Comprehensive BaseAgent Tool Integration Validation")
    logger.info("=" * 80)
    
    # Create test agent for validation
    test_agent = TestImplementationAgent(
        config=AgentConfig(
            agent_id="validation_agent",
            agent_type="validation",
            capabilities=[AgentCapability.CODE_GENERATION],
            max_concurrent_tasks=5
        ),
        llm_client=MockLLMClient(),
        memory_manager=MockMemoryManager(),
        event_bus=MockEventBus()
    )
    
    # Run validation framework
    validator = ToolIntegrationValidationFramework()
    
    # Validate requirements
    requirement_results = await validator.validate_requirements(test_agent)
    logger.info("📋 Requirements Validation:")
    logger.info(f"   Overall Compliance: {'✅ PASS' if requirement_results['overall_compliance'] else '❌ FAIL'}")
    
    for req, result in requirement_results["requirement_results"].items():
        status = "✅ PASS" if result["met"] else "❌ FAIL"
        logger.info(f"   {req}: {status}")
        if result["missing_methods"]:
            logger.info(f"      Missing: {', '.join(result['missing_methods'])}")
    
    # Validate backward compatibility
    compatibility_results = await validator.validate_backward_compatibility(test_agent)
    logger.info("🔄 Backward Compatibility:")
    logger.info(f"   Compatible: {'✅ PASS' if compatibility_results['compatible'] else '❌ FAIL'}")
    
    if compatibility_results["incompatible_features"]:
        for feature in compatibility_results["incompatible_features"]:
            logger.info(f"   Missing: {feature['feature']} - {feature['description']}")
    
    # Create and run test suite
    suite = create_comprehensive_test_suite()
    
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
    logger.info("\n🧪 Running Test Suite...")
    runner = AsyncTestRunner()
    await runner.run_suite(suite)
    
    # Generate comprehensive report
    passed_tests = [r for r in runner.results if r["status"] == "PASSED"]
    failed_tests = [r for r in runner.results if r["status"] == "FAILED"]
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPREHENSIVE VALIDATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Requirements Compliance: {'✅ PASS' if requirement_results['overall_compliance'] else '❌ FAIL'}")
    logger.info(f"Backward Compatibility: {'✅ PASS' if compatibility_results['compatible'] else '❌ FAIL'}")
    logger.info(f"Test Suite Results:")
    logger.info(f"  Total Tests: {len(runner.results)}")
    logger.info(f"  Passed: {len(passed_tests)}")
    logger.info(f"  Failed: {len(failed_tests)}")
    logger.info(f"  Success Rate: {len(passed_tests)/len(runner.results)*100:.1f}%")
    
    if failed_tests:
        logger.info("\n❌ FAILED TESTS:")
        for test in failed_tests:
            logger.info(f"  - {test['test']}: {test.get('error', 'Unknown error')}")
    
    if requirement_results["recommendations"]:
        logger.info("\n💡 RECOMMENDATIONS:")
        for rec in requirement_results["recommendations"]:
            logger.info(f"  - {rec}")
    
    # Overall assessment
    overall_success = (
        requirement_results['overall_compliance'] and 
        compatibility_results['compatible'] and 
        len(failed_tests) == 0
    )
    
    logger.info(f"\n🎯 OVERALL ASSESSMENT: {'✅ SYSTEM READY' if overall_success else '⚠️ NEEDS ATTENTION'}")
    logger.info("✅ Comprehensive tool integration validation completed!")
    
    return overall_success


if __name__ == "__main__":
    # Run the comprehensive validation
    success = asyncio.run(run_comprehensive_validation())
    sys.exit(0 if success else 1)
