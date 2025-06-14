#!/usr/bin/env python3
"""
Debug script to test workflow individually
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

# Setup path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.learning_models import ToolOperation, WorkflowResult, AdvancedToolWorkflowMixin
from src.models.tool_models import ToolStatus, TaskStatus, AsyncTask
from src.core.tools.base_tool import BaseTool, ToolCategory, ToolPermission
from src.models.tool_models import ToolResult


class MockTool(BaseTool):
    """Mock tool for testing workflow operations."""
    
    def __init__(self, name: str, delay: float = 0.1, should_fail: bool = False):
        super().__init__(
            name=name,
            description=f"Mock tool {name} for testing",
            version="1.0.0",
            required_permissions=[],
            category=ToolCategory.DEVELOPMENT
        )
        self.delay = delay
        self.should_fail = should_fail
        self.operations = {}
        self.rollback_calls = []
    
    async def get_capabilities(self):
        return {
            "available_operations": [
                {"name": "test_operation", "description": "Test operation"}
            ]
        }
    
    async def validate_params(self, operation: str, params: Dict[str, Any]):
        return True, None
    
    async def execute(self, operation: str, params: Dict[str, Any], operation_id: Optional[str] = None):
        await asyncio.sleep(self.delay)
        
        if self.should_fail:
            return ToolResult(
                operation_id=operation_id or str(uuid.uuid4()),
                status=ToolStatus.FAILURE,
                error_message="Mock tool failure",
                execution_time=self.delay
            )
        
        return ToolResult(
            operation_id=operation_id or str(uuid.uuid4()),
            status=ToolStatus.SUCCESS,
            data=f"Result from {self.name}.{operation}({params})",
            execution_time=self.delay
        )
    
    async def rollback(self, operation: str, params: Dict[str, Any]):
        """Mock rollback for testing."""
        self.rollback_calls.append((operation, params))
        await asyncio.sleep(0.05)  # Quick rollback
    
    async def get_usage_examples(self):
        return [{"operation": "test_operation", "params": {"test": "value"}}]


class MockAgent(AdvancedToolWorkflowMixin):
    """Mock agent with workflow capabilities for testing."""
    
    def __init__(self):
        self.agent_id = "test_agent_" + str(uuid.uuid4())[:8]
        self.agent_type = "test"
        self.tools = {}
        self.active_operations = {}
        self.operation_status = {}
        self._monitoring_active = False
        self.logger = None
    
    def get_tool(self, tool_name: str):
        """Get tool by name."""
        return self.tools.get(tool_name)
    
    def add_tool(self, tool: MockTool):
        """Add tool for testing."""
        self.tools[tool.name] = tool
    
    async def cleanup_operation(self, op_id: str):
        """Mock cleanup operation."""
        print(f"🧹 Cleaning up operation: {op_id}")
    
    async def notify_agents(self, op_id: str, status: str):
        """Mock agent notification."""
        print(f"📢 Notifying agents: {op_id} -> {status}")


async def test_dependency_validation():
    """Test dependency validation specifically."""
    print("🔍 Testing Dependency Validation")
    
    agent = MockAgent()
    agent.add_tool(MockTool("tool_a"))
    agent.add_tool(MockTool("tool_b"))
    agent.add_tool(MockTool("tool_c"))
    
    # Test valid workflow (no cycles)
    op1 = ToolOperation(tool_name="tool_a", operation="step1", dependencies=[])
    op2 = ToolOperation(tool_name="tool_b", operation="step2", dependencies=[op1.operation_id])
    op3 = ToolOperation(tool_name="tool_c", operation="step3", dependencies=[op2.operation_id])
    
    print(f"Created operations:")
    print(f"  - {op1.operation_id}: tool_a.step1 (deps: {op1.dependencies})")
    print(f"  - {op2.operation_id}: tool_b.step2 (deps: {op2.dependencies})")
    print(f"  - {op3.operation_id}: tool_c.step3 (deps: {op3.dependencies})")
    
    valid_workflow = [op1, op2, op3]
    result = await agent.execute_tool_workflow(valid_workflow)
    
    print(f"Valid workflow result:")
    print(f"  - Completed: {result.operations_completed}")
    print(f"  - Errors: {result.errors}")
    print(f"  - Duration: {result.total_duration:.3f}s")
    
    has_cycle_error = "Dependency cycle detected" in result.errors.get("workflow", "")
    print(f"  - Has cycle error: {has_cycle_error}")
    
    if not has_cycle_error:
        print("✅ Valid workflow accepted")
    else:
        print("❌ Valid workflow incorrectly rejected")
        return False
    
    # Test cyclic dependency (should be rejected)
    print("\nTesting cyclic dependency...")
    op4 = ToolOperation(tool_name="tool_a", operation="step4", dependencies=[])
    op5 = ToolOperation(tool_name="tool_b", operation="step5", dependencies=[op4.operation_id])
    op6 = ToolOperation(tool_name="tool_c", operation="step6", dependencies=[op5.operation_id])
    # Create cycle: op4 depends on op6
    op4.dependencies = [op6.operation_id]
    
    print(f"Created cyclic operations:")
    print(f"  - {op4.operation_id}: tool_a.step4 (deps: {op4.dependencies})")
    print(f"  - {op5.operation_id}: tool_b.step5 (deps: {op5.dependencies})")
    print(f"  - {op6.operation_id}: tool_c.step6 (deps: {op6.dependencies})")
    
    cyclic_workflow = [op4, op5, op6]
    result = await agent.execute_tool_workflow(cyclic_workflow)
    
    print(f"Cyclic workflow result:")
    print(f"  - Completed: {result.operations_completed}")
    print(f"  - Errors: {result.errors}")
    print(f"  - Duration: {result.total_duration:.3f}s")
    
    has_cycle_error = "Dependency cycle detected" in result.errors.get("workflow", "")
    print(f"  - Has cycle error: {has_cycle_error}")
    
    if has_cycle_error:
        print("✅ Cyclic workflow correctly rejected")
    else:
        print("❌ Cyclic workflow incorrectly accepted")
        return False
    
    return True


async def main():
    success = await test_dependency_validation()
    print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
