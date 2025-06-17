#!/usr/bin/env python3
"""
Simple test to verify LLM-generated content and file writing only.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_llm_and_file_writing():
    """Test LLM generation and file writing directly."""
    print("🧪 Testing LLM Generation and File Writing")
    
    try:
        # Import required modules
        from src.agents.implementation_agent_new import ImplementationAgent
        from src.agents.base_agent import AgentConfig
        from src.core.llm_client import LLMClient
        from src.core.memory_manager import MemoryManager
        from src.core.event_bus import EventBus
        
        # Initialize components
        config = AgentConfig(
            agent_id="test_implementation_agent",
            name="Test Implementation Agent",
            description="Test agent for personal finance tracker",
            llm_model="local_fallback",
            temperature=0.7,
            max_tokens=4000
        )
        
        llm_client = LLMClient()
        memory_manager = MemoryManager(agent_id="test_agent")
        event_bus = EventBus()
        
        # Create implementation agent
        agent = ImplementationAgent(
            config=config,
            llm_client=llm_client,
            memory_manager=memory_manager,
            event_bus=event_bus
        )
        
        # Initialize the agent
        await agent.initialize()
        
        # Test the LLM prompt generation
        print("🔄 Testing LLM prompt generation...")
        features = ["finance_tracker", "expense_tracking", "budget_management"]
        prompt = agent._create_feature_implementation_prompt("personal_finance_tracker", features)
        print(f"✅ Prompt generated, length: {len(prompt)} characters")
        
        # Test LLM execution
        print("🔄 Testing LLM execution...")
        context = {
            "project_name": "personal_finance_tracker",
            "features": features,
        }
        
        llm_result = await agent.execute_llm_task(
            user_prompt=prompt,
            context=context,
            structured_output=True
        )
        
        print(f"✅ LLM result received: {type(llm_result)}")
        print(f"📋 LLM result keys: {list(llm_result.keys())}")
        
        # Process the response
        print("🔄 Processing LLM response...")
        processed_result = await agent._process_implementation_response(llm_result)
        
        print(f"✅ Processed result: {type(processed_result)}")
        print(f"📋 Processed result keys: {list(processed_result.keys())}")
        
        # Check if we have main_dart content
        if "main_dart" in processed_result:
            main_dart = processed_result["main_dart"]
            print(f"✅ main_dart found, length: {len(main_dart)} characters")
            
            # Check content
            if "PersonalFinanceApp" in main_dart or "FinanceDashboard" in main_dart:
                print("✅ main_dart contains personal finance content")
            elif "MyHomePage" in main_dart:
                print("❌ main_dart contains demo content")
            else:
                print("⚠️  main_dart content unclear")
                print("First 200 characters:")
                print(main_dart[:200])
        else:
            print("❌ No main_dart found in processed result")
        
        # Test file writing directly
        print("🔄 Testing direct file writing...")
        
        # Create test directory
        test_dir = Path(__file__).parent.parent / "flutter_projects" / "test_personal_finance"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Write files using the agent's method
        generated_files = await agent._write_feature_code(str(test_dir), processed_result)
        
        print(f"✅ Files written: {len(generated_files)}")
        for file_path in generated_files:
            print(f"  📄 {file_path}")
        
        # Verify main.dart content
        main_dart_file = test_dir / "lib" / "main.dart"
        if main_dart_file.exists():
            content = main_dart_file.read_text()
            if "PersonalFinanceApp" in content:
                print("✅ Written main.dart contains personal finance content")
            else:
                print("❌ Written main.dart does not contain personal finance content")
                print("First 200 characters:")
                print(content[:200])
        else:
            print("❌ main.dart file was not written")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_llm_and_file_writing())
    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
