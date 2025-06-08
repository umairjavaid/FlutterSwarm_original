#!/usr/bin/env python3
"""
Debug test to figure out what's wrong.
"""

import sys
import traceback

try:
    print("Testing core imports...")
    from core.agent_types import AgentType, WorkflowState
    print("✅ Core types imported")
    
    print("Testing base agent import...")
    from agents.base_agent import BaseAgent
    print("✅ Base agent imported")
    
    print("Testing DevOps agent import...")
    from agents.devops_agent import DevOpsAgent
    print("✅ DevOps agent imported")
    
    print("Testing agent creation...")
    config = {'deployment_targets': ['web']}
    agent = DevOpsAgent(config)
    print(f"✅ Agent created: {agent.agent_id}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
