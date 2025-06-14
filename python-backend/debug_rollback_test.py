#!/usr/bin/env python3
import asyncio
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.learning_models import ToolOperation, WorkflowResult, AdvancedToolWorkflowMixin
from src.core.tools.base_tool import BaseTool, ToolCategory
from src.models.tool_models import ToolResult, ToolStatus

class MockTool(BaseTool):
    def __init__(self, name: str, should_fail: bool = False):
        super().__init__(
            name=name,
            description=f"Mock tool {name}",
            version="1.0.0",
            required_permissions=[],
            category=ToolCategory.DEVELOPMENT
        )
        self.should_fail = should_fail
        self.rollback_calls = []

    async def get_capabilities(self):
        return {"available_operations": [{"name": "test_operation", "description": "Test operation"}]}

    async def validate_params(self, operation: str, params):
        return True, None

    async def execute(self, operation: str, params, operation_id=None):
        await asyncio.sleep(0.1)
        
        if self.should_fail:
            return ToolResult(
                operation_id=operation_id or str(uuid.uuid4()),
                status=ToolStatus.FAILURE,
                error_message="Mock tool failure"
            )
        
        return ToolResult(
            operation_id=operation_id or str(uuid.uuid4()),
            status=ToolStatus.SUCCESS,
            data=f"Result from {self.name}.{operation}"
        )

    async def rollback(self, operation: str, params):
        """Mock rollback for testing."""
        self.rollback_calls.append((operation, params))
        await asyncio.sleep(0.05)
    
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

async def test_rollback_mechanism():
    print("Testing rollback mechanism...")
    
    agent = MockAgent()
    tool_a = MockTool("tool_a", should_fail=False)
    tool_b = MockTool("tool_b", should_fail=False)
    tool_c = MockTool("tool_c", should_fail=True)  # This one will fail
    
    agent.add_tool(tool_a)
    agent.add_tool(tool_b)
    agent.add_tool(tool_c)
    
    # Create workflow where the last operation fails
    op1 = ToolOperation(tool_name="tool_a", operation="step1", dependencies=[])
    op2 = ToolOperation(tool_name="tool_b", operation="step2", dependencies=[op1.operation_id])
    op3 = ToolOperation(tool_name="tool_c", operation="step3", dependencies=[op2.operation_id])  # Will fail
    
    workflow = [op1, op2, op3]
    result = await agent.execute_tool_workflow(workflow)
    
    print(f"Operations completed: {result.operations_completed}")
    print(f"Errors: {result.errors}")
    print(f"tool_a rollback calls: {len(tool_a.rollback_calls)}")
    print(f"tool_b rollback calls: {len(tool_b.rollback_calls)}")
    print(f"tool_c rollback calls: {len(tool_c.rollback_calls)}")
    
    # Should have completed op1 and op2, but failed on op3
    has_errors = len(result.errors) > 0
    has_op3_error = op3.operation_id in result.errors
    has_rollbacks = len(tool_a.rollback_calls) > 0 or len(tool_b.rollback_calls) > 0
    
    print(f"Has errors: {has_errors}")
    print(f"Has op3 error: {has_op3_error}")
    print(f"Has rollbacks: {has_rollbacks}")
    
    return has_errors and has_op3_error and has_rollbacks

if __name__ == "__main__":
    success = asyncio.run(test_rollback_mechanism())
    print(f"Test passed: {success}")
