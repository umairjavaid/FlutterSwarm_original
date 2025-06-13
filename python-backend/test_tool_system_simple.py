#!/usr/bin/env python3
"""
Simple test script to verify the basic tool system functionality.
"""
import asyncio
import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, '.')

# Set up basic logging
logging.basicConfig(level=logging.INFO)

async def test_basic_functionality():
    """Test basic tool system functionality."""
    print("🚀 Starting Simple Tool System Test")
    
    try:
        # Test 1: Import all major components
        print("\n1. Testing imports...")
        from src.core.tools.tool_registry import ToolRegistry
        from src.core.tools.file_system_tool import FileSystemTool
        from src.core.tools.process_tool import ProcessTool
        from src.core.tools.flutter_sdk_tool import FlutterSDKTool
        from src.agents.base_agent import BaseAgent
        from src.agents.implementation_agent import ImplementationAgent
        print("✅ All imports successful")
        
        # Test 2: Initialize tool registry
        print("\n2. Testing tool registry initialization...")
        registry = ToolRegistry.get_instance()
        print("✅ Tool registry initialized")
        
        # Test 3: Register tools
        print("\n3. Testing tool registration...")
        file_tool = FileSystemTool()
        process_tool = ProcessTool()
        sdk_tool = FlutterSDKTool()
        
        await registry.register_tool(file_tool)
        await registry.register_tool(process_tool) 
        await registry.register_tool(sdk_tool)
        
        tools = registry.get_available_tools()
        print(f"✅ Registered {len(tools)} tools: {[tool.name for tool in tools]}")
        
        # Test 4: Test tool capabilities
        print("\n4. Testing tool capabilities...")
        for tool in tools:
            capabilities = await tool.get_capabilities()
            print(f"   - {tool.name}: {len(capabilities.available_operations)} operations")
        print("✅ Tool capabilities retrieved")
        
        # Test 5: Test basic agent initialization  
        print("\n5. Testing agent initialization...")
        # Create a mock config and dependencies for testing
        from src.agents.base_agent import AgentConfig
        from src.core.memory_manager import MemoryManager
        from src.core.event_bus import EventBus
        
        # Use a simple mock LLM client
        class MockLLMClient:
            async def generate_response(self, *args, **kwargs):
                return "Mock response"
        
        config = AgentConfig(
            agent_id="test_agent", 
            agent_type="implementation",
            capabilities=[],
            llm_model="gpt-4"
        )
        llm_client = MockLLMClient()
        memory_manager = MemoryManager(agent_id="test_agent")
        event_bus = EventBus()
        
        agent = ImplementationAgent(config, llm_client, memory_manager, event_bus)
        await agent._initialize_tool_system()
        print("✅ Agent tool system initialized")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)
