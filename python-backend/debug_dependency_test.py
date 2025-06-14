#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.learning_models import ToolOperation, WorkflowResult, AdvancedToolWorkflowMixin
from src.core.tools.base_tool import BaseTool, ToolCategory

class MockTool(BaseTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description=f"Mock tool {name}",
            version="1.0.0",
            required_permissions=[],
            category=ToolCategory.DEVELOPMENT
        )

    async def get_capabilities(self):
        return {"available_operations": [{"name": "test_operation", "description": "Test operation"}]}

    async def validate_params(self, operation: str, params):
        return True, None

    async def execute(self, operation: str, params, operation_id=None):
        return {"status": "success", "data": f"Result from {operation}"}
    
    async def get_usage_examples(self):
        return [{"operation": "test_operation", "params": {"test": "value"}}]

class MockAgent(AdvancedToolWorkflowMixin):
    def __init__(self):
        self.tools = {}
        self.logger = None

    def add_tool(self, tool):
        self.tools[tool.name] = tool

    def get_tool(self, name):
        return self.tools.get(name)

async def test_missing_dependency():
    print("Testing missing dependency detection...")
    
    agent = MockAgent()
    agent.add_tool(MockTool("tool_a"))
    
    # Create operation with missing dependency
    op = ToolOperation(tool_name="tool_a", operation="test", dependencies=["missing_op_id"])
    
    result = await agent.execute_tool_workflow([op])
    
    print(f"Result errors: {result.errors}")
    print(f"Workflow error: {result.errors.get('workflow', 'NO WORKFLOW ERROR')}")
    
    # Check if missing dependency is detected
    workflow_error = result.errors.get("workflow", "")
    has_missing_dep = "Missing dependency" in workflow_error
    
    print(f"Has missing dependency error: {has_missing_dep}")
    print(f"Error message: '{workflow_error}'")
    
    return has_missing_dep

if __name__ == "__main__":
    success = asyncio.run(test_missing_dependency())
    print(f"Test passed: {success}")
