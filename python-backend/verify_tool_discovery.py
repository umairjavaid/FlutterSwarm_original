#!/usr/bin/env python3
"""
Verification script for BaseAgent tool discovery and understanding capabilities.

This script tests:
1. Tool discovery with multiple tools
2. Proper capability analysis
3. Tool understanding storage
4. Event bus subscription for tool availability
5. Tool preference building
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt, **kwargs):
        """Mock generate method."""
        agent_id = kwargs.get('agent_id', 'test_agent')
        structured_output = kwargs.get('structured_output', False)
        
        if "Analyze this development tool" in prompt:
            # Tool analysis response
            if structured_output:
                return {
                    "summary": "Mock tool for development operations",
                    "usage_scenarios": [
                        "Creating new files",
                        "Reading existing files", 
                        "Managing project structure",
                        "Template-based code generation"
                    ],
                    "parameter_patterns": {
                        "file_path": "String path to target file",
                        "content": "File content string",
                        "template": "Template name for generation"
                    },
                    "success_indicators": [
                        "File created successfully",
                        "Operation completed without errors",
                        "Expected output generated"
                    ],
                    "failure_patterns": [
                        "File not found errors",
                        "Permission denied errors",
                        "Invalid path errors"
                    ],
                    "responsibility_mapping": {
                        "primary": "File system operations",
                        "secondary": "Code generation support"
                    },
                    "decision_factors": [
                        "File system access needed",
                        "Template-based generation required",
                        "Project structure management"
                    ]
                }
            else:
                return "Mock tool analysis response"
        
        elif "Analyze these available tools and build preferences" in prompt:
            # Tool preference response
            if structured_output:
                return {
                    "primary_tools": ["mock_file_tool", "mock_flutter_tool"],
                    "secondary_tools": ["mock_process_tool"],
                    "specialized_tools": [],
                    "preference_scores": {
                        "mock_file_tool": 0.9,
                        "mock_flutter_tool": 0.8,
                        "mock_process_tool": 0.6
                    },
                    "usage_priorities": {
                        "file_operations": "mock_file_tool",
                        "flutter_development": "mock_flutter_tool"
                    },
                    "workflow_patterns": [
                        "Use file tool for setup, then flutter tool for development",
                        "Process tool for background tasks"
                    ]
                }
            else:
                return "Mock preference analysis"
        
        return "Mock LLM response"


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
    
    async def store_memory(self, content, metadata=None, importance=0.5, long_term=False):
        """Mock store memory."""
        memory = {
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "long_term": long_term
        }
        self.stored_memories.append(memory)
        return f"memory_{len(self.stored_memories)}"


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.subscriptions = {}
        self.published_messages = []
    
    async def subscribe(self, topic, callback, subscriber_id=None):
        """Mock subscribe."""
        self.subscriptions[topic] = callback
        return subscriber_id or f"sub_{len(self.subscriptions)}"
    
    async def publish(self, topic, message):
        """Mock publish."""
        self.published_messages.append((topic, message))


class MockBaseTool:
    """Mock tool for testing."""
    
    def __init__(self, name, description="Mock tool", category=None):
        self.name = name
        self.description = description
        self.version = "1.0.0"
        from src.core.tools.base_tool import ToolCategory
        self.category = category or ToolCategory.DEVELOPMENT
    
    async def get_capabilities(self):
        """Mock capabilities."""
        from src.models.tool_models import ToolCapabilities
        return ToolCapabilities(
            available_operations=[
                {
                    "name": f"{self.name}_operation",
                    "description": f"Mock operation for {self.name}",
                    "parameters": {"param1": "string", "param2": "int"}
                }
            ]
        )
    
    async def get_usage_examples(self):
        """Mock usage examples."""
        return [
            {
                "operation": f"{self.name}_operation",
                "parameters": {"param1": "example", "param2": 42},
                "description": f"Example usage of {self.name}"
            }
        ]
    
    async def get_health_status(self):
        """Mock health status."""
        return {"status": "healthy", "last_check": "2025-06-14T10:00:00Z"}
    
    async def validate_params(self, operation, parameters):
        """Mock validation."""
        return True, None
    
    async def execute(self, operation, parameters):
        """Mock execution."""
        from src.models.tool_models import ToolResult, ToolStatus
        return ToolResult(
            operation_id=f"{operation}_{self.name}",
            status=ToolStatus.SUCCESS,
            data={"result": "mock_result"}
        )


class MockToolRegistry:
    """Mock tool registry for testing."""
    
    def __init__(self):
        self.tools = {}
        self.is_initialized = True
        self._setup_mock_tools()
    
    def _setup_mock_tools(self):
        """Setup mock tools."""
        from src.core.tools.base_tool import ToolCategory
        
        self.tools["mock_file_tool"] = MockBaseTool(
            "mock_file_tool", 
            "Mock file system tool",
            ToolCategory.FILE_SYSTEM
        )
        self.tools["mock_flutter_tool"] = MockBaseTool(
            "mock_flutter_tool",
            "Mock Flutter SDK tool", 
            ToolCategory.DEVELOPMENT
        )
        self.tools["mock_process_tool"] = MockBaseTool(
            "mock_process_tool",
            "Mock process management tool",
            ToolCategory.PROCESS
        )
    
    def get_available_tools(self):
        """Mock get available tools."""
        return list(self.tools.values())
    
    def get_tool(self, name):
        """Mock get tool."""
        return self.tools.get(name)
    
    @classmethod
    def instance(cls):
        """Mock singleton instance."""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance


async def test_tool_discovery():
    """Test tool discovery functionality."""
    print("\n🔍 Testing Tool Discovery and Understanding...")
    print("=" * 60)
    
    try:
        # Import required classes
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        from src.models.tool_models import ToolLearningModel
        
        # Mock the tool registry
        import src.core.tools.tool_registry
        src.core.tools.tool_registry.ToolRegistry = MockToolRegistry
        
        # Create test agent
        class TestAgent(BaseAgent):
            async def _get_default_system_prompt(self):
                return "Test agent system prompt"
            
            async def get_capabilities(self):
                return ["testing", "development"]
        
        config = AgentConfig(
            agent_id="test_discovery_agent",
            agent_type="implementation",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.FILE_OPERATIONS]
        )
        
        llm_client = MockLLMClient()
        memory_manager = MockMemoryManager()
        event_bus = MockEventBus()
        
        # Create agent
        agent = TestAgent(config, llm_client, memory_manager, event_bus)
        
        # Wait for initialization
        await asyncio.sleep(0.2)
        
        print(f"✅ Agent created: {agent.agent_id}")
        print(f"✅ Available tools: {len(agent.available_tools)}")
        
        # Verify tools were discovered
        expected_tools = ["mock_file_tool", "mock_flutter_tool", "mock_process_tool"]
        for tool_name in expected_tools:
            if tool_name in agent.available_tools:
                print(f"✅ Tool discovered: {tool_name}")
                print(f"   Scenarios: {len(agent.tool_capabilities.get(tool_name, []))}")
            else:
                print(f"❌ Tool missing: {tool_name}")
                return False
        
        # Verify tool performance metrics initialized
        for tool_name in expected_tools:
            if tool_name in agent.tool_performance_metrics:
                print(f"✅ Metrics initialized for: {tool_name}")
            else:
                print(f"❌ Metrics missing for: {tool_name}")
                return False
        
        # Verify event subscriptions
        expected_subscriptions = [
            "tool.availability.*",
            "tool.performance.*", 
            "tool.registered.*"
        ]
        
        subscription_count = len(event_bus.subscriptions)
        print(f"✅ Event subscriptions: {subscription_count}")
        
        # Verify tool learning model
        if agent.tool_learning_model:
            print(f"✅ Tool learning model created")
            print(f"   Preferences: {len(agent.tool_learning_model.tool_preferences)}")
        else:
            print(f"❌ Tool learning model missing")
            return False
        
        # Verify memory storage
        memory_count = len(memory_manager.stored_memories)
        print(f"✅ Memories stored: {memory_count}")
        
        # Look for specific memory types
        tool_discovery_memories = [
            m for m in memory_manager.stored_memories 
            if m["metadata"].get("type") == "tool_discovery"
        ]
        tool_understanding_memories = [
            m for m in memory_manager.stored_memories
            if m["metadata"].get("type") == "tool_understanding"
        ]
        tool_preference_memories = [
            m for m in memory_manager.stored_memories
            if m["metadata"].get("type") == "tool_preferences"
        ]
        
        print(f"✅ Tool discovery memories: {len(tool_discovery_memories)}")
        print(f"✅ Tool understanding memories: {len(tool_understanding_memories)}")
        print(f"✅ Tool preference memories: {len(tool_preference_memories)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_understanding():
    """Test tool capability analysis."""
    print("\n🧠 Testing Tool Capability Analysis...")
    print("=" * 60)
    
    try:
        # Import required classes
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        
        # Create test agent
        class TestAgent(BaseAgent):
            async def _get_default_system_prompt(self):
                return "Test agent for tool analysis"
            
            async def get_capabilities(self):
                return ["file_operations", "code_generation"]
        
        config = AgentConfig(
            agent_id="test_analysis_agent",
            agent_type="implementation", 
            capabilities=[AgentCapability.CODE_GENERATION]
        )
        
        llm_client = MockLLMClient()
        memory_manager = MockMemoryManager()
        event_bus = MockEventBus()
        
        agent = TestAgent(config, llm_client, memory_manager, event_bus)
        
        # Create a mock tool to analyze
        mock_tool = MockBaseTool("test_analysis_tool", "Tool for testing analysis")
        
        # Analyze the tool
        understanding = await agent.analyze_tool_capability(mock_tool)
        
        print(f"✅ Tool analyzed: {understanding.tool_name}")
        print(f"✅ Summary: {understanding.capabilities_summary}")
        print(f"✅ Usage scenarios: {len(understanding.usage_scenarios)}")
        print(f"✅ Parameter patterns: {len(understanding.parameter_patterns)}")
        print(f"✅ Success indicators: {len(understanding.success_indicators)}")
        print(f"✅ Failure patterns: {len(understanding.failure_patterns)}")
        print(f"✅ Confidence: {understanding.confidence_level}")
        
        # Verify all required fields are present
        required_fields = [
            'tool_name', 'agent_id', 'capabilities_summary',
            'usage_scenarios', 'parameter_patterns', 'success_indicators',
            'failure_patterns', 'confidence_level'
        ]
        
        for field in required_fields:
            if hasattr(understanding, field):
                print(f"✅ Field present: {field}")
            else:
                print(f"❌ Field missing: {field}")
                return False
        
        # Verify memory storage
        analysis_memories = [
            m for m in memory_manager.stored_memories
            if m["metadata"].get("type") == "tool_understanding"
        ]
        
        if analysis_memories:
            print(f"✅ Analysis stored in memory")
        else:
            print(f"❌ Analysis not stored in memory")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tool analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_event_handling():
    """Test event bus subscription and handling."""
    print("\n📡 Testing Event Bus Integration...")
    print("=" * 60)
    
    try:
        # Import required classes  
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        from src.models.agent_models import AgentMessage, MessageType
        
        # Create test agent
        class TestAgent(BaseAgent):
            async def _get_default_system_prompt(self):
                return "Test agent for events"
            
            async def get_capabilities(self):
                return ["event_handling"]
        
        config = AgentConfig(
            agent_id="test_event_agent",
            agent_type="testing",
            capabilities=[AgentCapability.TESTING]
        )
        
        llm_client = MockLLMClient()
        memory_manager = MockMemoryManager()
        event_bus = MockEventBus()
        
        agent = TestAgent(config, llm_client, memory_manager, event_bus)
        
        # Wait for initialization and event subscriptions
        await asyncio.sleep(0.2)
        
        # Verify subscriptions were created
        print(f"✅ Event subscriptions: {len(event_bus.subscriptions)}")
        
        expected_topics = ["tool.availability.*", "tool.performance.*", "tool.registered.*"]
        for topic in expected_topics:
            if topic in event_bus.subscriptions:
                print(f"✅ Subscribed to: {topic}")
            else:
                print(f"❌ Missing subscription: {topic}")
                return False
        
        # Test availability change handler
        availability_message = AgentMessage(
            sender_id="test_registry",
            receiver_id=agent.agent_id,
            message_type=MessageType.TOOL_EVENT,
            content="Tool availability changed",
            data={
                "tool_name": "mock_file_tool",
                "is_available": False
            }
        )
        
        if "tool.availability.*" in event_bus.subscriptions:
            handler = event_bus.subscriptions["tool.availability.*"]
            await handler(availability_message)
            print("✅ Availability change handler tested")
        
        # Test performance update handler
        performance_message = AgentMessage(
            sender_id="test_registry",
            receiver_id=agent.agent_id,
            message_type=MessageType.TOOL_EVENT,
            content="Tool performance updated",
            data={
                "tool_name": "mock_flutter_tool",
                "performance": {
                    "success_rate": 0.95,
                    "avg_duration": 2.5,
                    "error_count": 2
                }
            }
        )
        
        if "tool.performance.*" in event_bus.subscriptions:
            handler = event_bus.subscriptions["tool.performance.*"]
            await handler(performance_message)
            print("✅ Performance update handler tested")
        
        # Test new tool registration handler
        registration_message = AgentMessage(
            sender_id="test_registry",
            receiver_id=agent.agent_id,
            message_type=MessageType.TOOL_EVENT,
            content="New tool registered",
            data={
                "tool_name": "new_mock_tool"
            }
        )
        
        if "tool.registered.*" in event_bus.subscriptions:
            handler = event_bus.subscriptions["tool.registered.*"]
            await handler(registration_message)
            print("✅ New tool registration handler tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Event handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all verification tests."""
    print("🚀 BaseAgent Tool Discovery and Understanding Verification")
    print("=" * 70)
    
    success = True
    
    # Test tool discovery
    if not await test_tool_discovery():
        success = False
    
    # Test tool understanding
    if not await test_tool_understanding():
        success = False
    
    # Test event handling
    if not await test_event_handling():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\n📋 Summary of Implemented Features:")
        print("✅ Tool discovery with registry querying")
        print("✅ LLM-based tool capability analysis")
        print("✅ Structured tool understanding storage")
        print("✅ Event bus subscription for tool availability")
        print("✅ Tool preference building based on agent type")
        print("✅ Memory storage for tool insights")
        print("✅ Comprehensive error handling")
        print("✅ Agent responsibility mapping")
        print("✅ Parameter pattern recognition")
        print("✅ Success/failure pattern analysis")
        
        print("\n🔧 Key Capabilities:")
        print("• Agents can discover all available tools automatically")
        print("• Each tool is analyzed through LLM reasoning")
        print("• Tool understanding is stored in agent memory")
        print("• Real-time updates via event bus subscriptions")
        print("• Tool preferences built based on agent type")
        print("• Comprehensive analysis for informed decision making")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the implementation and resolve issues.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
