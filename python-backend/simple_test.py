#!/usr/bin/env python3
"""
Simple test to check execute_task method.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_simple():
    """Simple test."""
    try:
        from src.agents.architecture_agent import ArchitectureAgent
        from src.agents.base_agent import AgentConfig, AgentCapability
        from src.core.event_bus import EventBus
        from src.core.memory_manager import MemoryManager
        
        print("✅ All imports successful")
        
        # Create simple objects
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="architecture",
            capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS]
        )
        print("✅ Config created")
        
        # Create mock client
        class MockClient:
            async def generate(self, prompt, **kwargs):
                return {"status": "success"}
                
        memory = MemoryManager(agent_id="test")
        event_bus = EventBus()
        llm_client = MockClient()
        
        print("✅ Dependencies created")
        
        # Create agent
        agent = ArchitectureAgent(config, llm_client, memory, event_bus)
        print("✅ Agent created")
        
        # Test execute_task method exists
        if hasattr(agent, 'execute_task'):
            print("✅ execute_task method exists")
            
            # Try to call it
            result = await agent.execute_task({
                "description": "Test task",
                "task_type": "analysis"
            })
            print(f"✅ execute_task call successful: {type(result)}")
            return True
        else:
            print("❌ execute_task method missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple())
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
