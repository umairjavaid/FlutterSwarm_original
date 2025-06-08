#!/usr/bin/env python3
"""
Simple test to verify DevOps Agent imports work.
"""

import sys
import traceback

try:
    print("Testing imports...")
    
    from core.agent_types import AgentType
    print("✅ AgentType imported")
    
    from core.langgraph_orchestrator import WorkflowState
    print("✅ WorkflowState imported")
    
    from agents.devops_agent import DevOpsAgent
    print("✅ DevOpsAgent imported")
    
    # Test basic functionality
    config = {
        'deployment_targets': ['android', 'ios', 'web'],
        'ci_cd_provider': 'github_actions'
    }
    
    devops_agent = DevOpsAgent(config)
    print(f"✅ DevOps Agent created: {devops_agent.agent_id}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
