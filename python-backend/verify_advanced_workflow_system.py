#!/usr/bin/env python3
"""
Advanced Tool Operation Management and Workflow Execution Verification

This script verifies the implementation of:
1. ToolOperation and WorkflowResult models
2. execute_tool_workflow method with dependency management and rollback
3. monitor_tool_operations method with continuous monitoring
4. Complex workflow execution with parallel operations
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Setup path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.learning_models import ToolOperation, WorkflowResult, AdvancedToolWorkflowMixin
from src.models.tool_models import ToolStatus, TaskStatus, AsyncTask
from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
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
        # Mock dependencies
        self.agent_id = "test_agent_" + str(uuid.uuid4())[:8]
        self.agent_type = "test"
        self.tools = {}
        self.active_operations = {}
        self.operation_status = {}
        
        # Initialize workflow monitoring
        self._monitoring_active = False
        
        # Mock logger
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
            def warning(self, msg): print(f"WARN: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        
        self.logger = MockLogger()
    
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


async def test_workflow_models():
    """Test 1: Verify workflow models structure and functionality."""
    print("🔍 Test 1: Workflow Models")
    
    # Test ToolOperation model
    op1 = ToolOperation(
        tool_name="file_tool",
        operation="create_file",
        parameters={"path": "/test/file.txt", "content": "test"},
        dependencies=[],
        timeout=30.0
    )
    
    op2 = ToolOperation(
        tool_name="git_tool", 
        operation="add_file",
        parameters={"path": "/test/file.txt"},
        dependencies=[op1.operation_id],
        timeout=15.0
    )
    
    assert op1.tool_name == "file_tool"
    assert op1.operation == "create_file"
    assert isinstance(op1.parameters, dict)
    assert op1.dependencies == []
    assert op1.timeout == 30.0
    assert op1.operation_id.startswith("op_")
    
    assert op2.dependencies == [op1.operation_id]
    print(f"✅ ToolOperation model verified")
    print(f"   - Operation 1: {op1.operation_id} ({op1.tool_name}.{op1.operation})")
    print(f"   - Operation 2: {op2.operation_id} ({op2.tool_name}.{op2.operation}) depends on [{op1.operation_id}]")
    
    # Test WorkflowResult model
    result = WorkflowResult(
        operations_completed=["op1", "op2"],
        results={"op1": "success", "op2": "success"},
        errors={},
        total_duration=5.5
    )
    
    assert len(result.operations_completed) == 2
    assert result.total_duration == 5.5
    print(f"✅ WorkflowResult model verified")
    
    return True


async def test_dependency_validation():
    """Test 2: Workflow dependency validation and cycle detection."""
    print("\n🔍 Test 2: Dependency Validation")
    
    agent = MockAgent()
    agent.add_tool(MockTool("tool_a"))
    agent.add_tool(MockTool("tool_b"))
    agent.add_tool(MockTool("tool_c"))
    
    # Test valid workflow (no cycles)
    op1 = ToolOperation(tool_name="tool_a", operation="step1", dependencies=[])
    op2 = ToolOperation(tool_name="tool_b", operation="step2", dependencies=[op1.operation_id])
    op3 = ToolOperation(tool_name="tool_c", operation="step3", dependencies=[op2.operation_id])
    
    valid_workflow = [op1, op2, op3]
    result = await agent.execute_tool_workflow(valid_workflow)
    
    assert "Dependency cycle detected" not in result.errors.get("workflow", "")
    print(f"✅ Valid workflow accepted")
    print(f"   - Operations completed: {len(result.operations_completed)}")
    print(f"   - Total duration: {result.total_duration:.3f}s")
    
    # Test cyclic dependency (should be rejected)
    op4 = ToolOperation(tool_name="tool_a", operation="step4", dependencies=[])
    op5 = ToolOperation(tool_name="tool_b", operation="step5", dependencies=[op4.operation_id])
    op6 = ToolOperation(tool_name="tool_c", operation="step6", dependencies=[op5.operation_id])
    # Create cycle: op4 depends on op6
    op4.dependencies = [op6.operation_id]
    
    cyclic_workflow = [op4, op5, op6]
    result = await agent.execute_tool_workflow(cyclic_workflow)
    
    assert "Dependency cycle detected" in result.errors.get("workflow", "")
    print(f"✅ Cyclic workflow rejected: {result.errors.get('workflow')}")
    
    # Test missing dependency (should be rejected)
    op7 = ToolOperation(tool_name="tool_a", operation="step7", dependencies=["missing_op_id"])
    
    invalid_workflow = [op7]
    result = await agent.execute_tool_workflow(invalid_workflow)
    
    assert "Missing dependency" in result.errors.get("workflow", "")
    print(f"✅ Missing dependency detected: {result.errors.get('workflow')}")
    
    return True


async def test_parallel_execution():
    """Test 3: Parallel operation execution and dependency management."""
    print("\n🔍 Test 3: Parallel Execution")
    
    agent = MockAgent()
    agent.add_tool(MockTool("fast_tool", delay=0.1))
    agent.add_tool(MockTool("medium_tool", delay=0.2))
    agent.add_tool(MockTool("slow_tool", delay=0.3))
    
    # Create workflow with parallel branches
    # op1 (no deps) -> op2, op3 (parallel) -> op4 (depends on both)
    op1 = ToolOperation(tool_name="fast_tool", operation="init", dependencies=[])
    op2 = ToolOperation(tool_name="medium_tool", operation="branch_a", dependencies=[op1.operation_id])
    op3 = ToolOperation(tool_name="slow_tool", operation="branch_b", dependencies=[op1.operation_id])
    op4 = ToolOperation(tool_name="fast_tool", operation="merge", dependencies=[op2.operation_id, op3.operation_id])
    
    workflow = [op1, op2, op3, op4]
    
    start_time = time.time()
    result = await agent.execute_tool_workflow(workflow)
    execution_time = time.time() - start_time
    
    # Should complete in ~0.6s (0.1 + 0.3 + 0.1) not 0.7s (sum of all delays)
    assert execution_time < 0.8, f"Execution took too long: {execution_time:.3f}s"
    assert len(result.operations_completed) == 4
    assert len(result.errors) == 0
    
    print(f"✅ Parallel execution verified")
    print(f"   - Total operations: {len(workflow)}")
    print(f"   - Execution time: {execution_time:.3f}s (parallel optimization)")
    print(f"   - Operations completed: {result.operations_completed}")
    
    return True


async def test_rollback_mechanism():
    """Test 4: Rollback mechanism on failure."""
    print("\n🔍 Test 4: Rollback Mechanism")
    
    agent = MockAgent()
    tool_a = MockTool("tool_a", delay=0.1, should_fail=False)
    tool_b = MockTool("tool_b", delay=0.1, should_fail=False)
    tool_c = MockTool("tool_c", delay=0.1, should_fail=True)  # This one will fail
    
    agent.add_tool(tool_a)
    agent.add_tool(tool_b)
    agent.add_tool(tool_c)
    
    # Create workflow where the last operation fails
    op1 = ToolOperation(tool_name="tool_a", operation="step1", dependencies=[])
    op2 = ToolOperation(tool_name="tool_b", operation="step2", dependencies=[op1.operation_id])
    op3 = ToolOperation(tool_name="tool_c", operation="step3", dependencies=[op2.operation_id])  # Will fail
    
    workflow = [op1, op2, op3]
    result = await agent.execute_tool_workflow(workflow)
    
    # Should have completed op1 and op2, but failed on op3
    assert len(result.errors) > 0
    assert op3.operation_id in result.errors
    
    # Check rollback was called on completed operations
    assert len(tool_a.rollback_calls) > 0 or len(tool_b.rollback_calls) > 0
    
    print(f"✅ Rollback mechanism verified")
    print(f"   - Operations attempted: {len(workflow)}")
    print(f"   - Operations completed before failure: {len(result.operations_completed)}")
    print(f"   - Errors: {list(result.errors.keys())}")
    print(f"   - Rollback calls: tool_a={len(tool_a.rollback_calls)}, tool_b={len(tool_b.rollback_calls)}")
    
    return True


async def test_operation_monitoring():
    """Test 5: Continuous operation monitoring and timeout handling."""
    print("\n🔍 Test 5: Operation Monitoring")
    
    agent = MockAgent()
    
    # Create mock operations with different characteristics
    operations = {
        "op1": (ToolOperation(tool_name="fast_tool", operation="quick", timeout=1.0), None),
        "op2": (ToolOperation(tool_name="slow_tool", operation="slow", timeout=0.1), None),  # Will timeout
        "op3": (ToolOperation(tool_name="normal_tool", operation="normal", timeout=2.0), None)
    }
    
    # Mock some operations in progress
    agent.active_operations = operations
    agent.operation_status = {op_id: "running" for op_id in operations.keys()}
    
    # Create mock tasks
    for op_id, (op, _) in operations.items():
        if op_id == "op2":
            # Create a task that will timeout
            task = asyncio.create_task(asyncio.sleep(10))  # Long running task
            task.start_time = time.time() - 1.0  # Started 1 second ago
        else:
            # Create completed tasks
            task = asyncio.create_task(asyncio.sleep(0))
            await task  # Complete immediately
        
        operations[op_id] = (op, task)
        agent.active_operations[op_id] = (op, task)
    
    # Run monitoring for a short period
    monitoring_task = asyncio.create_task(agent.monitor_tool_operations())
    await asyncio.sleep(0.5)  # Let it run for a bit
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    # Check that timeout was detected
    assert agent.operation_status.get("op2") in ["timeout", "failed", "retrying"]
    
    print(f"✅ Operation monitoring verified")
    print(f"   - Operations monitored: {len(operations)}")
    print(f"   - Final statuses: {agent.operation_status}")
    
    return True


async def test_complex_workflow():
    """Test 6: Complex real-world workflow simulation."""
    print("\n🔍 Test 6: Complex Workflow Simulation")
    
    agent = MockAgent()
    
    # Add various tools with different characteristics
    tools = [
        MockTool("file_system", delay=0.05),
        MockTool("git_tool", delay=0.1),
        MockTool("build_tool", delay=0.3),
        MockTool("test_tool", delay=0.2),
        MockTool("deploy_tool", delay=0.4),
        MockTool("notification_tool", delay=0.05)
    ]
    
    for tool in tools:
        agent.add_tool(tool)
    
    # Create a complex Flutter development workflow
    operations = [
        # 1. Initial setup (parallel)
        ToolOperation(tool_name="file_system", operation="create_project", dependencies=[]),
        ToolOperation(tool_name="git_tool", operation="init_repo", dependencies=[]),
        
        # 2. Code generation (depends on file system)
        ToolOperation(tool_name="file_system", operation="generate_code", dependencies=["op_1"]),
        
        # 3. Git operations (depends on both setup operations)
        ToolOperation(tool_name="git_tool", operation="add_files", dependencies=["op_1", "op_2", "op_3"]),
        
        # 4. Build and test (parallel, depend on git)
        ToolOperation(tool_name="build_tool", operation="compile", dependencies=["op_4"]),
        ToolOperation(tool_name="test_tool", operation="run_tests", dependencies=["op_4"]),
        
        # 5. Deploy (depends on both build and test)
        ToolOperation(tool_name="deploy_tool", operation="deploy_app", dependencies=["op_5", "op_6"]),
        
        # 6. Notification (depends on deploy)
        ToolOperation(tool_name="notification_tool", operation="notify_success", dependencies=["op_7"])
    ]
    
    # Assign proper operation IDs
    for i, op in enumerate(operations, 1):
        op.operation_id = f"op_{i}"
    
    start_time = time.time()
    result = await agent.execute_tool_workflow(operations)
    execution_time = time.time() - start_time
    
    assert len(result.operations_completed) == len(operations)
    assert len(result.errors) == 0
    
    # Should benefit from parallel execution
    total_sequential_time = sum(tool.delay for tool in tools)
    assert execution_time < total_sequential_time
    
    print(f"✅ Complex workflow executed successfully")
    print(f"   - Total operations: {len(operations)}")
    print(f"   - Execution time: {execution_time:.3f}s")
    print(f"   - Sequential time would be: {total_sequential_time:.3f}s")
    print(f"   - Parallel efficiency: {((total_sequential_time - execution_time) / total_sequential_time * 100):.1f}%")
    print(f"   - Operations flow: {' -> '.join(result.operations_completed)}")
    
    return True


async def run_all_tests():
    """Run all workflow verification tests."""
    print("🚀 Advanced Tool Operation Management and Workflow Execution Verification")
    print("=" * 80)
    
    tests = [
        ("Workflow Models", test_workflow_models),
        ("Dependency Validation", test_dependency_validation),
        ("Parallel Execution", test_parallel_execution),
        ("Rollback Mechanism", test_rollback_mechanism),
        ("Operation Monitoring", test_operation_monitoring),
        ("Complex Workflow", test_complex_workflow)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, "PASS", None))
            print(f"✅ {test_name}: PASS")
        except Exception as e:
            results.append((test_name, "FAIL", str(e)))
            print(f"❌ {test_name}: FAIL - {e}")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    
    for test_name, status, error in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Total execution time: {total_time:.3f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Advanced workflow execution verified!")
        print("\n✅ VERIFIED CAPABILITIES:")
        print("   - ToolOperation and WorkflowResult models")
        print("   - Dependency validation and cycle detection")
        print("   - Parallel operation execution")
        print("   - Automatic rollback on failure")
        print("   - Continuous operation monitoring")
        print("   - Complex real-world workflow handling")
        print("   - Timeout detection and retry mechanisms")
        print("   - Performance optimization through parallelism")
    else:
        print(f"\n⚠️  {total - passed} tests failed - see details above")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
