#!/usr/bin/env python3
"""
Quick test for Flutter SDK Tool to verify current implementation status.
"""

import asyncio
import os
import sys
import tempfile

# Ensure we can import from the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

async def main():
    """Test basic Flutter SDK Tool functionality."""
    print("🧪 Quick Flutter SDK Tool Test")
    print("=" * 35)
    
    try:
        # Import with absolute import to avoid relative import issues
        from core.tools.base_tool import BaseTool, ToolStatus
        print("✅ base_tool imported successfully")
        
        from core.tools.flutter_sdk_tool import FlutterSDKTool
        print("✅ FlutterSDKTool imported successfully")
        
        # Create tool instance
        flutter_tool = FlutterSDKTool()
        print("✅ FlutterSDKTool instantiated successfully")
        
        # Test capabilities
        capabilities = await flutter_tool.get_capabilities()
        print(f"✅ Capabilities retrieved: {len(capabilities.operations)} operations")
        
        # List operations
        print("\n📋 Available Operations:")
        for i, op in enumerate(capabilities.operations[:5], 1):  # Show first 5
            print(f"  {i}. {op['name']}: {op['description'][:60]}...")
        
        if len(capabilities.operations) > 5:
            print(f"  ... and {len(capabilities.operations) - 5} more operations")
        
        print(f"\n✅ All tests passed! Tool has {len(capabilities.operations)} operations implemented.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the async test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
