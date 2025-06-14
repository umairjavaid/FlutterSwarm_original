#!/usr/bin/env python3
"""
Debug script to isolate the JSON serialization issue.
"""
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from src.core.tools.file_system_tool import FileSystemTool
from src.models.tool_models import ToolPermission

async def debug_serialization():
    """Debug the JSON serialization issue."""
    print("🔍 Debugging JSON Serialization Issue")
    
    try:
        # Create tool
        tool = FileSystemTool()
        
        # Get capabilities
        print("1. Getting capabilities...")
        capabilities = await tool.get_capabilities()
        print(f"   Capabilities type: {type(capabilities)}")
        
        # Try to serialize the whole capabilities object
        print("2. Testing capabilities.__dict__ serialization...")
        try:
            caps_dict = capabilities.__dict__
            print(f"   Capabilities dict keys: {list(caps_dict.keys())}")
            json_str = json.dumps(caps_dict)
            print("   ✅ Successfully serialized capabilities.__dict__")
        except Exception as e:
            print(f"   ❌ Failed to serialize capabilities.__dict__: {e}")
            
            # Let's examine the available_operations
            print("3. Examining available_operations...")
            ops = caps_dict.get('available_operations', [])
            print(f"   Operations count: {len(ops)}")
            
            if ops:
                first_op = ops[0]
                print(f"   First operation keys: {list(first_op.keys())}")
                
                # Check if the issue is in required_permissions
                perms = first_op.get('required_permissions', [])
                print(f"   Required permissions: {perms}")
                print(f"   Permission types: {[type(p) for p in perms]}")
                
                # Try to serialize permission values
                perm_values = [p.value if hasattr(p, 'value') else p for p in perms]
                print(f"   Permission values: {perm_values}")
                
                # Try serializing individual parts
                try:
                    json.dumps(first_op['name'])
                    print("   ✅ name serializes")
                except Exception as e:
                    print(f"   ❌ name fails: {e}")
                    
                try:
                    json.dumps(first_op['required_permissions'])
                    print("   ✅ required_permissions serializes")
                except Exception as e:
                    print(f"   ❌ required_permissions fails: {e}")
        
        print("\n4. Testing manual serialization fix...")
        
        # Try the serialization fix I implemented
        def serialize_capabilities(caps):
            """Convert capabilities to JSON-serializable format."""
            serializable = {}
            for key, value in caps.__dict__.items():
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle lists/tuples that might contain enums
                    serializable[key] = []
                    for item in value:
                        if hasattr(item, 'value'):  # Enum
                            serializable[key].append(item.value)
                        elif isinstance(item, dict):
                            # Handle nested dicts
                            serialized_item = {}
                            for k, v in item.items():
                                if k == 'required_permissions' and hasattr(v, '__iter__'):
                                    # Handle permissions list
                                    serialized_item[k] = [p.value if hasattr(p, 'value') else p for p in v]
                                elif hasattr(v, 'value'):
                                    serialized_item[k] = v.value
                                else:
                                    serialized_item[k] = v
                            serializable[key].append(serialized_item)
                        else:
                            serializable[key].append(item)
                elif hasattr(value, 'value'):  # Enum
                    serializable[key] = value.value
                else:
                    serializable[key] = value
            return serializable

        serializable_capabilities = serialize_capabilities(capabilities)
        json_str = json.dumps(serializable_capabilities)
        print("   ✅ Manual serialization works!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_serialization())
