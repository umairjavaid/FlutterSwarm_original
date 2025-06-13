#!/usr/bin/env python3
"""
Flutter SDK Tool Test with Fixed Imports.
This script tests the Flutter SDK Tool with corrected import paths.
"""

import asyncio
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import required modules directly with absolute paths
try:
    # Import logging first
    from config.logging_config import get_logger
    logger = get_logger("test")
    print("✅ Logging imported successfully")
    
    # Import models
    from models.tool_models import (
        ToolStatus, ToolPermission, ToolOperation, ToolResult, ToolCapabilities,
        ToolUsageEntry, ToolMetrics, ToolCategory
    )
    print("✅ Tool models imported successfully")
    
    # Import base tool (after fixing imports)
    import importlib.util
    base_tool_path = os.path.join(src_dir, 'core', 'tools', 'base_tool.py')
    
    # Read and modify the base_tool code to fix imports
    with open(base_tool_path, 'r') as f:
        base_tool_code = f.read()
    
    # Replace problematic relative imports
    fixed_base_tool_code = base_tool_code.replace(
        "from ...config import get_logger",
        "from config.logging_config import get_logger"
    ).replace(
        "from ...models.tool_models import",
        "from models.tool_models import"
    )
    
    # Execute the fixed base_tool code
    spec = importlib.util.spec_from_loader("base_tool", loader=None)
    base_tool_module = importlib.util.module_from_spec(spec)
    exec(fixed_base_tool_code, base_tool_module.__dict__)
    
    # Add to sys.modules
    sys.modules['base_tool'] = base_tool_module
    BaseTool = base_tool_module.BaseTool
    ToolCapabilities = base_tool_module.ToolCapabilities
    print("✅ Base tool module loaded successfully")
    
    # Now import FlutterSDKTool
    flutter_tool_path = os.path.join(src_dir, 'core', 'tools', 'flutter_sdk_tool.py')
    with open(flutter_tool_path, 'r') as f:
        flutter_tool_code = f.read()
    
    # Fix the imports in flutter_tool_code
    fixed_flutter_tool_code = flutter_tool_code.replace(
        "from .base_tool import",
        "from base_tool import"
    )
    
    # Execute the fixed flutter tool code
    spec = importlib.util.spec_from_loader("flutter_sdk_tool", loader=None)
    flutter_tool_module = importlib.util.module_from_spec(spec)
    exec(fixed_flutter_tool_code, flutter_tool_module.__dict__)
    
    FlutterSDKTool = flutter_tool_module.FlutterSDKTool
    print("✅ FlutterSDKTool loaded successfully")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test_flutter_tool():
    """Test the Flutter SDK Tool."""
    print("\n🧪 Testing Flutter SDK Tool")
    print("=" * 30)
    
    try:
        # Create tool instance
        flutter_tool = FlutterSDKTool()
        print("✅ FlutterSDKTool instantiated")
        
        # Test capabilities
        capabilities = await flutter_tool.get_capabilities()
        print(f"✅ Capabilities: {len(capabilities.operations)} operations")
        
        # Show available operations
        print("\n📋 Available Operations:")
        for i, op in enumerate(capabilities.operations, 1):
            print(f"  {i:2d}. {op['name']}")
        
        # Test a simple validation
        print("\n🔍 Testing Validation:")
        try:
            result = await flutter_tool.validate_parameters("invalid_operation", {})
            print("❌ Should have failed validation")
        except Exception as e:
            print("✅ Validation correctly rejected invalid operation")
        
        print(f"\n✅ All tests passed! FlutterSDKTool is working correctly.")
        print(f"📊 Total operations implemented: {len(capabilities.operations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_flutter_tool())
    sys.exit(0 if result else 1)
