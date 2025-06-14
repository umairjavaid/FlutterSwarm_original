#!/usr/bin/env python3
"""
Verification script for BaseAgent tool integration attributes.

This script verifies that all required tool-related attributes are properly
added to the BaseAgent class with correct types and initialization.
"""

import sys
import inspect
from typing import get_type_hints

# Add the src directory to the path
sys.path.insert(0, '/workspaces/FlutterSwarm/python-backend/src')

def verify_base_agent_tool_integration():
    """Verify BaseAgent tool integration attributes."""
    print("Verifying BaseAgent tool integration...")
    
    try:
        # Import the required classes
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        from src.models.tool_models import (
            ToolUsageEntry, ToolMetrics, ToolUnderstanding, AsyncTask,
            ToolLearningModel, ToolResult, ToolStatus
        )
        from src.core.tools.base_tool import BaseTool
        
        print("✓ All imports successful")
        
        # Check BaseAgent type annotations
        hints = get_type_hints(BaseAgent.__init__)
        print(f"✓ BaseAgent type hints: {len(hints)} parameters")
        
        # Verify AgentConfig has all required tool attributes
        config_hints = get_type_hints(AgentConfig)
        
        required_config_attrs = [
            'available_tools',
            'tool_capabilities', 
            'tool_usage_history',
            'tool_performance_metrics',
            'active_tool_operations'
        ]
        
        for attr in required_config_attrs:
            if attr in config_hints:
                print(f"✓ AgentConfig.{attr}: {config_hints[attr]}")
            else:
                print(f"✗ Missing AgentConfig.{attr}")
                return False
        
        # Verify that models exist and are properly structured
        models_to_check = [
            (ToolUsageEntry, ['agent_id', 'tool_name', 'operation', 'parameters', 'timestamp']),
            (ToolMetrics, ['total_uses', 'success_rate', 'average_execution_time']),
            (AsyncTask, ['task_id', 'name', 'status', 'result']),
        ]
        
        for model_class, required_fields in models_to_check:
            print(f"\n✓ Checking {model_class.__name__}:")
            if hasattr(model_class, '__dataclass_fields__'):
                fields = model_class.__dataclass_fields__
                for field in required_fields:
                    if field in fields:
                        print(f"  ✓ {field}: {fields[field].type}")
                    else:
                        print(f"  ✗ Missing field: {field}")
                        return False
            else:
                print(f"  ✗ {model_class.__name__} is not a dataclass")
                return False
        
        print("\n✓ All verifications passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def verify_async_task_functionality():
    """Verify AsyncTask class functionality."""
    print("\nVerifying AsyncTask functionality...")
    
    try:
        from src.models.tool_models import AsyncTask, TaskStatus
        import asyncio
        
        # Create an AsyncTask instance
        task = AsyncTask(
            name="test_task",
            description="Test async task",
            operation="test_operation",
            tool_name="test_tool",
            agent_id="test_agent"
        )
        
        print(f"✓ AsyncTask created: {task.task_id}")
        print(f"✓ Initial status: {task.status}")
        print(f"✓ Is running: {task.is_running()}")
        print(f"✓ Is completed: {task.is_completed()}")
        
        # Test basic functionality
        assert task.status == TaskStatus.PENDING
        assert not task.is_running()
        assert not task.is_completed()
        assert task.progress == 0.0
        
        print("✓ AsyncTask basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"✗ AsyncTask verification failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("BaseAgent Tool Integration Verification")
    print("=" * 60)
    
    success = True
    
    # Verify base agent tool integration
    if not verify_base_agent_tool_integration():
        success = False
    
    # Verify AsyncTask functionality
    if not verify_async_task_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VERIFICATIONS PASSED! 🎉")
        print("BaseAgent tool integration is properly implemented.")
    else:
        print("❌ VERIFICATION FAILED")
        print("Some issues need to be addressed.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()
