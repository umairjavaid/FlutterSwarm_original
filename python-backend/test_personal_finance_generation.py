#!/usr/bin/env python3
"""
Test script to directly test personal finance tracker generation
without going through the full CLI workflow.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_personal_finance_generation():
    """Test the personal finance tracker generation directly."""
    print("🧪 Testing Personal Finance Tracker Generation")
    
    try:
        # Import required modules
        from src.agents.implementation_agent_new import ImplementationAgent
        from src.agents.base_agent import AgentConfig
        from src.models.task_models import TaskContext
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
        
        # Create task context for personal finance tracker
        task_context = TaskContext(
            task_id="test_personal_finance",
            description="Create a personal finance tracker app with expense tracking, budget management, and financial reports",
            metadata={
                "project_name": "personal_finance_tracker",
                "features": [
                    "finance_tracker",
                    "expense_tracking", 
                    "budget_management",
                    "financial_reports",
                    "data_management",
                    "authentication",
                    "ui_components"
                ]
            }
        )
        
        # Process the task
        print("🔄 Processing personal finance tracker task...")
        result = await agent.process_task(task_context)
        
        print(f"✅ Task completed with status: {result.status}")
        print(f"📁 Deliverables: {len(result.deliverables.get('generated_files', []))} files generated")
        
        # Check if files were created in the correct location
        flutter_projects_dir = Path(__file__).parent.parent / "flutter_projects"
        personal_finance_dir = flutter_projects_dir / "personal_finance_tracker"
        
        if personal_finance_dir.exists():
            print(f"✅ Project created in correct location: {personal_finance_dir}")
            
            # Check main.dart content
            main_dart_file = personal_finance_dir / "lib" / "main.dart"
            if main_dart_file.exists():
                content = main_dart_file.read_text()
                if "PersonalFinanceApp" in content or "FinanceDashboard" in content:
                    print("✅ main.dart contains personal finance tracker code (not demo app)")
                elif "MyHomePage" in content and "_counter" in content:
                    print("❌ main.dart contains demo counter app - LLM generation failed")
                    print("📋 Using fallback implementation would be better")
                else:
                    print("⚠️  main.dart content unclear - checking first 200 characters:")
                    print(content[:200])
            else:
                print("❌ main.dart file not found")
        else:
            print(f"❌ Project not found in correct location: {personal_finance_dir}")
            
            # Check if it was created in the wrong location
            wrong_location = Path(__file__).parent / "personal_finance_tracker"
            if wrong_location.exists():
                print(f"❌ Project found in wrong location: {wrong_location}")
            else:
                print("❌ Project not found anywhere")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_personal_finance_generation())
    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
