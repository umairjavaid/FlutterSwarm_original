#!/usr/bin/env python3
"""
Quick test to verify execute_task method works on ArchitectureAgent.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.architecture_agent import ArchitectureAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager

class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, messages=None, **kwargs):
        return "Test response: Architecture analysis completed successfully."

async def test_execute_task():
    """Test the execute_task method on ArchitectureAgent."""
    
    print("🧪 Testing execute_task method on ArchitectureAgent...")
    
    # Create mock dependencies
    llm_client = MockLLMClient()
    memory_manager = MemoryManager(agent_id="test_arch_agent")
    event_bus = EventBus()
    
    # Create agent config
    config = AgentConfig(
        agent_id="test_arch_agent",
        agent_type="architecture",
        capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
        max_concurrent_tasks=1,
        llm_model="gpt-4",
        temperature=0.3,
        max_tokens=2000,
        timeout=60
    )
    
    # Create ArchitectureAgent
    agent = ArchitectureAgent(config, llm_client, memory_manager, event_bus)
    
    # Test task data (format used by LangGraph supervisor)
    task_data = {
        "description": "Analyze project architecture for Flutter app",
        "task_type": "analysis", 
        "priority": "normal"
    }
    
    try:
        # Call execute_task method
        result = await agent.execute_task(task_data)
        
        print("✅ execute_task method exists and callable!")
        print(f"📊 Result keys: {list(result.keys())}")
        print(f"📝 Status: {result.get('status')}")
        print(f"🎯 Task ID: {result.get('task_id')}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_execute_task())
        exit_code = 0 if success else 1
        print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
        exit(exit_code)
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        exit(1)
