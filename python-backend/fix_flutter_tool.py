#!/usr/bin/env python3
"""
Quick script to fix FlutterSDKTool get_capabilities method.
"""

import re

# Read the file
with open('/workspaces/FlutterSwarm/python-backend/src/core/tools/flutter_sdk_tool.py', 'r') as f:
    content = f.read()

# Find the get_capabilities method and replace all ToolOperation instances
start_marker = 'async def get_capabilities(self) -> ToolCapabilities:'
end_marker = 'return ToolCapabilities('

start_pos = content.find(start_marker)
end_pos = content.find(end_marker, start_pos)

if start_pos != -1 and end_pos != -1:
    # Extract the method content
    method_content = content[start_pos:end_pos]
    
    # Replace ToolOperation with dictionary
    fixed_content = method_content.replace('ToolOperation(', '{')
    fixed_content = fixed_content.replace('name=', '"name":')
    fixed_content = fixed_content.replace('description=', '"description":')
    fixed_content = fixed_content.replace('parameters_schema=', '"parameters_schema":')
    fixed_content = fixed_content.replace('output_schema=', '"output_schema":')
    fixed_content = fixed_content.replace('required_permissions=', '"required_permissions":')
    fixed_content = fixed_content.replace('examples=', '"examples":')
    fixed_content = fixed_content.replace('error_codes=', '"error_codes":')
    fixed_content = fixed_content.replace('estimated_duration=', '"estimated_duration":')
    fixed_content = fixed_content.replace('supports_cancellation=', '"supports_cancellation":')
    
    # Replace closing parentheses with closing braces for ToolOperation instances
    # This is a bit tricky - we need to find the correct closing parens
    lines = fixed_content.split('\n')
    result_lines = []
    paren_count = 0
    brace_count = 0
    in_tool_operation = False
    
    for line in lines:
        if '"name":' in line and not in_tool_operation:
            in_tool_operation = True
            # Replace the opening { if it was a ToolOperation
            if line.strip().startswith('{'):
                line = line.replace('{', '{', 1)
            result_lines.append(line)
        elif in_tool_operation:
            # Count braces and parens to find the end
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')
            
            # If we have a line that ends the ToolOperation
            if (paren_count <= 0 and brace_count <= 0 and 
                (line.strip().endswith('),') or line.strip().endswith(')'))):
                # Replace the closing paren with closing brace
                if line.strip().endswith('),'):
                    line = line.replace('),', '},')
                elif line.strip().endswith(')'):
                    line = line.replace(')', '}')
                in_tool_operation = False
                paren_count = 0
                brace_count = 0
            
            result_lines.append(line)
        else:
            result_lines.append(line)
    
    # Rebuild the file content
    new_content = content[:start_pos] + '\n'.join(result_lines) + content[end_pos:]
    
    # Write back
    with open('/workspaces/FlutterSwarm/python-backend/src/core/tools/flutter_sdk_tool.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed FlutterSDKTool get_capabilities method")
else:
    print("Could not find get_capabilities method boundaries")
