#!/usr/bin/env python3
"""
Debug test to verify execute_task method on ArchitectureAgent when created like in supervisor.
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
    async def generate(self, prompt, **kwargs):
        return {"response": "Mock response", "status": "success"}
    
    async def generate_response(self, prompt, **kwargs):
        return "Mock response"

async def test_supervisor_agent_creation():
    """Test the agent creation exactly like in supervisor."""
    
    print("🧪 Testing ArchitectureAgent creation like in supervisor...")
    
    # Create dependencies exactly like in supervisor
    memory_manager = MemoryManager(agent_id="shared_memory")
    event_bus = EventBus()
    llm_client = MockLLMClient()
    
    # Create agent config exactly like in supervisor
    architecture_config = AgentConfig(
        agent_id="architecture_agent",
        agent_type="architecture",
        llm_model="gpt-4",
        capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
        max_concurrent_tasks=2
    )
    
    # Create ArchitectureAgent exactly like in supervisor
    architecture_agent = ArchitectureAgent(
        config=architecture_config,
        llm_client=llm_client,
        memory_manager=memory_manager,
        event_bus=event_bus
    )
    
    # Check if execute_task method exists
    print(f"✅ Agent created: {architecture_agent.agent_id}")
    print(f"📋 Agent type: {type(architecture_agent)}")
    print(f"🔍 Has execute_task: {hasattr(architecture_agent, 'execute_task')}")
    print(f"📋 Base classes: {[cls.__name__ for cls in type(architecture_agent).__mro__]}")
    
    # Try to call the method
    if hasattr(architecture_agent, 'execute_task'):
        try:
            result = await architecture_agent.execute_task({
                "description": "Test architecture analysis",
                "task_type": "analysis",
                "priority": "normal"
            })
            print(f"✅ execute_task called successfully!")
            print(f"📊 Result type: {type(result)}")
            print(f"📝 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return True
        except Exception as e:
            print(f"❌ execute_task failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ execute_task method not found!")
        
        # Inspect the object more thoroughly
        print(f"📋 Agent dir: {[attr for attr in dir(architecture_agent) if not attr.startswith('_')]}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_supervisor_agent_creation())
        exit_code = 0 if success else 1
        print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
        exit(exit_code)
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
