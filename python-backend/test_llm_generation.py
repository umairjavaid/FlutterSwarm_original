#!/usr/bin/env python3
"""
Simple test to verify LLM-only code generation without templates
"""

import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, '/home/umair/Desktop/FlutterSwarm/python-backend/src')

from agents.implementation_agent_new import ImplementationAgent
from models.task_models import TaskContext, TaskType, TaskStatus
from models.project_models import ArchitecturePattern

async def test_llm_generation():
    """Test that the implementation agent can generate code without templates."""
    
    print("🔍 Testing LLM-only code generation...")
    
    # Create a simple task context
    task_context = TaskContext(
        task_id="test_task",
        task_type=TaskType.IMPLEMENTATION,
        description="Create a simple Flutter expense tracker app with categories and budget tracking",
        project_name="expense_tracker",
        features=["expense_tracking", "budget_management", "categories"]
    )
    
    # Initialize the implementation agent
    agent = ImplementationAgent()
    
    try:
        # Generate code dynamically
        result = await agent._generate_code_dynamically(task_context, {})
        
        print(f"✅ Code generation successful!")
        print(f"📁 Files generated: {len(result.get('code_files', {}))}")
        
        files = result.get('code_files', {})
        for file_path, content in files.items():
            print(f"  📄 {file_path}: {len(content)} characters")
        
        if result.get('dependencies'):
            print(f"📦 Dependencies: {result.get('dependencies')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_generation())
    sys.exit(0 if success else 1)
