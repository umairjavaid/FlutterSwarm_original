#!/usr/bin/env python3
"""
Verification script for BaseAgent tool-related attributes and supporting models.

This script verifies that:
1. All required tool-related attributes are present in BaseAgent
2. Supporting models have the correct fields and types
3. Models are properly initialized with defaults
4. Attributes are correctly typed and accessible
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, List

# Import the models and BaseAgent
try:
    from src.models.tool_models import (
        ToolUsageEntry, ToolMetrics, ToolUnderstanding, AsyncTask,
        ToolStatus, TaskStatus
    )
    from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
    
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class MockLLMClient:
    """Mock LLM client for testing."""
    async def generate(self, **kwargs):
        return {"response": "mock response"}


class MockMemoryManager:
    """Mock memory manager for testing."""
    pass


class MockEventBus:
    """Mock event bus for testing."""
    async def subscribe(self, topic, handler):
        pass
    
    async def publish(self, topic, data):
        pass


class MockBaseTool:
    """Simple mock tool for testing (not inheriting from BaseTool to avoid complex dependencies)."""
    
    def __init__(self, name="mock_tool"):
        self.name = name
        self.version = "1.0.0"
        self.is_available = True


class TestBaseAgent(BaseAgent):
    """Test implementation of BaseAgent."""
    
    async def _get_default_system_prompt(self):
        return "Test agent system prompt"
    
    async def get_capabilities(self):
        return ["testing"]


def verify_tool_usage_entry():
    """Verify ToolUsageEntry model structure and required fields."""
    print("\n📋 Verifying ToolUsageEntry model...")
    
    # Test default initialization
    entry = ToolUsageEntry()
    
    required_fields = [
        'timestamp', 'tool_name', 'operation', 'parameters', 
        'result', 'reasoning'
    ]
    
    for field in required_fields:
        if not hasattr(entry, field):
            print(f"❌ Missing required field: {field}")
            return False
        else:
            print(f"✅ Field present: {field}")
    
    # Test with data
    entry = ToolUsageEntry(
        agent_id="test_agent",
        tool_name="test_tool",
        operation="test_operation",
        parameters={"param1": "value1"},
        result={"status": "success"},
        reasoning="Test reasoning for tool usage"
    )
    
    assert entry.agent_id == "test_agent"
    assert entry.tool_name == "test_tool"
    assert entry.operation == "test_operation"
    assert entry.parameters == {"param1": "value1"}
    assert entry.result == {"status": "success"}
    assert entry.reasoning == "Test reasoning for tool usage"
    assert isinstance(entry.timestamp, datetime)
    
    print("✅ ToolUsageEntry model verification complete")
    return True


def verify_tool_metrics():
    """Verify ToolMetrics model structure and required fields."""
    print("\n📊 Verifying ToolMetrics model...")
    
    # Test default initialization
    metrics = ToolMetrics()
    
    required_fields = ['success_rate', 'avg_duration', 'error_count', 'last_used']
    
    for field in required_fields:
        if not hasattr(metrics, field):
            print(f"❌ Missing required field: {field}")
            return False
        else:
            print(f"✅ Field present: {field}")
    
    # Test backward compatibility
    if hasattr(metrics, 'average_execution_time'):
        print("✅ Backward compatibility property: average_execution_time")
    
    # Test with data
    metrics = ToolMetrics(
        success_rate=0.95,
        avg_duration=2.5,
        error_count=3,
        last_used=datetime.now()
    )
    
    assert metrics.success_rate == 0.95
    assert metrics.avg_duration == 2.5
    assert metrics.error_count == 3
    assert isinstance(metrics.last_used, datetime)
    
    # Test backward compatibility alias
    assert metrics.average_execution_time == metrics.avg_duration
    
    print("✅ ToolMetrics model verification complete")
    return True


def verify_tool_understanding():
    """Verify ToolUnderstanding model structure and required fields."""
    print("\n🧠 Verifying ToolUnderstanding model...")
    
    # Test initialization with required fields
    understanding = ToolUnderstanding(
        tool_name="test_tool",
        agent_id="test_agent", 
        capabilities_summary="Tool can perform various operations"
    )
    
    required_fields = ['capabilities_summary', 'usage_scenarios', 'parameter_patterns']
    
    for field in required_fields:
        if not hasattr(understanding, field):
            print(f"❌ Missing required field: {field}")
            return False
        else:
            print(f"✅ Field present: {field}")
    
    # Test backward compatibility
    if hasattr(understanding, 'capability_summary'):
        print("✅ Backward compatibility property: capability_summary")
        assert understanding.capability_summary == understanding.capabilities_summary
    
    assert understanding.tool_name == "test_tool"
    assert understanding.agent_id == "test_agent"
    assert understanding.capabilities_summary == "Tool can perform various operations"
    
    print("✅ ToolUnderstanding model verification complete")
    return True


def verify_async_task():
    """Verify AsyncTask model structure and functionality."""
    print("\n⏱️ Verifying AsyncTask model...")
    
    # Test default initialization
    task = AsyncTask()
    
    required_fields = ['task_id', 'name', 'description', 'operation', 'tool_name', 'agent_id']
    
    for field in required_fields:
        if not hasattr(task, field):
            print(f"❌ Missing required field: {field}")
            return False
        else:
            print(f"✅ Field present: {field}")
    
    # Test with data
    task = AsyncTask(
        name="test_task",
        description="Test long-running operation",
        operation="test_operation", 
        tool_name="test_tool",
        agent_id="test_agent"
    )
    
    assert task.name == "test_task"
    assert task.description == "Test long-running operation"
    assert task.operation == "test_operation"
    assert task.tool_name == "test_tool"
    assert task.agent_id == "test_agent"
    assert task.status == TaskStatus.PENDING
    
    # Test methods
    assert not task.is_running()
    assert not task.is_completed()
    
    print("✅ AsyncTask model verification complete")
    return True


async def verify_base_agent_attributes():
    """Verify BaseAgent has all required tool-related attributes."""
    print("\n🤖 Verifying BaseAgent tool-related attributes...")
    
    # Create test agent
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="test",
        capabilities=[AgentCapability.CODE_GENERATION]
    )
    
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    agent = TestBaseAgent(config, llm_client, memory_manager, event_bus)
    
    # Wait a bit for async initialization
    await asyncio.sleep(0.1)
    
    required_attributes = [
        'available_tools',
        'tool_capabilities', 
        'tool_usage_history',
        'tool_performance_metrics',
        'active_tool_operations'
    ]
    
    for attr_name in required_attributes:
        if not hasattr(agent, attr_name):
            print(f"❌ Missing required attribute: {attr_name}")
            return False
        else:
            attr_value = getattr(agent, attr_name)
            print(f"✅ Attribute present: {attr_name} = {type(attr_value)}")
    
    # Verify correct types
    assert isinstance(agent.available_tools, dict)
    assert isinstance(agent.tool_capabilities, dict)
    assert isinstance(agent.tool_usage_history, list)
    assert isinstance(agent.tool_performance_metrics, dict)
    assert isinstance(agent.active_tool_operations, dict)
    
    # Test adding tool usage entry
    usage_entry = ToolUsageEntry(
        agent_id=agent.agent_id,
        tool_name="test_tool",
        operation="test_op",
        parameters={"param": "value"},
        result={"status": "success"},
        reasoning="Testing tool usage tracking"
    )
    
    agent.tool_usage_history.append(usage_entry)
    assert len(agent.tool_usage_history) == 1
    assert agent.tool_usage_history[0].tool_name == "test_tool"
    
    # Test adding tool metrics
    metrics = ToolMetrics(
        success_rate=1.0,
        avg_duration=1.5,
        error_count=0,
        last_used=datetime.now()
    )
    
    agent.tool_performance_metrics["test_tool"] = metrics
    assert "test_tool" in agent.tool_performance_metrics
    assert agent.tool_performance_metrics["test_tool"].success_rate == 1.0
    
    # Test adding active task
    task = AsyncTask(
        name="test_async_task",
        tool_name="test_tool",
        agent_id=agent.agent_id,
        operation="long_running_op"
    )
    
    agent.active_tool_operations["task_1"] = task
    assert "task_1" in agent.active_tool_operations
    assert agent.active_tool_operations["task_1"].name == "test_async_task"
    
    print("✅ BaseAgent tool-related attributes verification complete")
    return True


async def main():
    """Run all verification tests."""
    print("🔍 Starting BaseAgent tool attributes verification...")
    print("=" * 60)
    
    success = True
    
    # Verify all models
    success &= verify_tool_usage_entry()
    success &= verify_tool_metrics()
    success &= verify_tool_understanding()
    success &= verify_async_task()
    
    # Verify BaseAgent integration
    success &= await verify_base_agent_attributes()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All verifications passed! BaseAgent tool integration is complete and working.")
        print("\n📋 Summary:")
        print("✅ All required attributes added to BaseAgent class:")
        print("   - available_tools: Dict[str, BaseTool]")
        print("   - tool_capabilities: Dict[str, List[str]]") 
        print("   - tool_usage_history: List[ToolUsageEntry]")
        print("   - tool_performance_metrics: Dict[str, ToolMetrics]")
        print("   - active_tool_operations: Dict[str, AsyncTask]")
        print("\n✅ All supporting models updated in src/models/tool_models.py:")
        print("   - ToolUsageEntry: Added result and reasoning fields")
        print("   - ToolMetrics: Updated field names (avg_duration, error_count, last_used)")
        print("   - ToolUnderstanding: Updated capabilities_summary field name")
        print("   - AsyncTask: Complete implementation for long-running operations")
        print("\n✅ Backward compatibility maintained with property aliases")
        print("✅ Proper typing and initialization with defaults")
        print("✅ Integration with existing BaseAgent structure")
        
        return 0
    else:
        print("❌ Some verifications failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
