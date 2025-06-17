#!/usr/bin/env python3
"""
Test script to verify that the import conflicts have been resolved.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly without conflicts."""
    
    try:
        # Test importing from models (canonical location)
        from src.models.tool_models import ToolStatus, ToolResult, ToolOperation
        print("✓ Successfully imported from src.models.tool_models")
        
        # Test importing from base_tool (should re-export from models)
        from src.core.tools.base_tool import BaseTool, ToolStatus as BaseToolStatus, ToolResult as BaseToolResult
        print("✓ Successfully imported from src.core.tools.base_tool")
        
        # Test that they are the same classes
        assert ToolStatus == BaseToolStatus, "ToolStatus should be the same class"
        assert ToolResult == BaseToolResult, "ToolResult should be the same class"
        print("✓ Verified that imports are consistent")
        
        # Test creating a ToolResult with new structure
        result = ToolResult(
            operation_id="test-123",
            status=ToolStatus.SUCCESS,
            data={"message": "test successful"}
        )
        
        # Test that new methods exist
        assert hasattr(result, 'mark_completed'), "ToolResult should have mark_completed method"
        assert hasattr(result, 'add_error'), "ToolResult should have add_error method"
        assert hasattr(result, 'is_successful'), "ToolResult should have is_successful method"
        assert hasattr(result, 'error_code'), "ToolResult should have error_code attribute"
        print("✓ Verified that ToolResult has expected methods and attributes")
        
        # Test the methods work
        assert result.is_successful(), "Result should be successful"
        result.mark_completed()
        assert result.execution_time >= 0, "Execution time should be set"
        print("✓ Verified that ToolResult methods work correctly")
        
        # Test importing flutter_sdk_tool (should work without conflicts)
        from src.core.tools.flutter_sdk_tool import FlutterSDKTool
        print("✓ Successfully imported FlutterSDKTool")
        
        print("\n🎉 All import conflict tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
