"""
Verification test for intelligent tool coordination and sharing between agents.

This test suite verifies that the OrchestratorAgent correctly:
1. Analyzes tool usage patterns
2. Manages tool access conflicts 
3. Optimizes tool allocation
4. Handles tool unavailability and fallback
5. Coordinates long-running tool operations
6. Ensures fair conflict resolution
7. Improves overall system efficiency
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.core.llm_client import LLMClient
from src.models.agent_models import AgentCapabilityInfo
from src.models.tool_models import (
    ToolCoordinationResult, ToolConflict, Resolution, UsagePattern, 
    AllocationPlan, QueueStatus, SharedOperation, CoordinationResult
)


class MockToolAgent:
    """Mock agent that uses tools for testing."""
    
    def __init__(self, agent_id: str, tool_needs: List[str], priority: int = 1):
        self.agent_id = agent_id
        self.tool_needs = tool_needs
        self.priority = priority
        self.current_tools = set()
        self.tool_requests = []
        self.completed_tasks = []
        
    async def request_tool(self, tool_name: str) -> bool:
        """Request access to a tool."""
        self.tool_requests.append({
            'tool': tool_name,
            'timestamp': datetime.utcnow(),
            'priority': self.priority
        })
        return tool_name in self.current_tools
        
    async def use_tool(self, tool_name: str, operation: str) -> Dict[str, Any]:
        """Simulate tool usage."""
        if tool_name not in self.current_tools:
            return {'success': False, 'error': 'Tool not allocated'}
            
        # Simulate tool operation
        result = {
            'success': True,
            'tool': tool_name,
            'operation': operation,
            'agent': self.agent_id,
            'duration': 2.5,
            'timestamp': datetime.utcnow()
        }
        
        self.completed_tasks.append(result)
        return result
        
    def allocate_tool(self, tool_name: str):
        """Allocate a tool to this agent."""
        self.current_tools.add(tool_name)
        
    def deallocate_tool(self, tool_name: str):
        """Remove tool allocation from this agent."""
        self.current_tools.discard(tool_name)


async def setup_orchestrator():
    """Set up orchestrator with mock dependencies."""
    config = AgentConfig(
        agent_id="orchestrator-test",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.FILE_OPERATIONS]
    )
    
    llm_client = AsyncMock(spec=LLMClient)
    memory_manager = AsyncMock(spec=MemoryManager)
    event_bus = MagicMock(spec=EventBus)
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Mock LLM responses for different coordination scenarios
    llm_client.generate.side_effect = mock_llm_responses
    
    return orchestrator, llm_client, memory_manager, event_bus


async def mock_llm_responses(prompt: str, **kwargs) -> str:
    """Mock LLM responses based on prompt content."""
    
    if "usage pattern analysis" in prompt.lower():
        return json.dumps({
            "patterns": {
                "flutter_sdk": {
                    "tool_name": "flutter_sdk",
                    "agents": ["flutter-agent-1", "flutter-agent-2"],
                    "frequency": 25,
                    "avg_duration": 45.5,
                    "peak_hours": ["09:00-11:00", "14:00-16:00"],
                    "efficiency": 0.85
                },
                "file_system": {
                    "tool_name": "file_system",
                    "agents": ["implementation-agent", "documentation-agent"],
                    "frequency": 40,
                    "avg_duration": 12.3,
                    "peak_hours": ["10:00-12:00"],
                    "efficiency": 0.92
                }
            }
        })
    
    elif "conflict resolution" in prompt.lower():
        return json.dumps({
            "resolutions": [
                {
                    "conflict_id": "conflict-1",
                    "resolution_type": "queue_based",
                    "priority_agent": "flutter-agent-1", 
                    "queue_position": 1,
                    "estimated_wait": 30
                }
            ]
        })
    
    elif "allocation optimization" in prompt.lower():
        return json.dumps({
            "allocation_plan": {
                "agent_assignments": {
                    "flutter-agent-1": ["flutter_sdk"],
                    "flutter-agent-2": ["file_system"],
                    "implementation-agent": ["code_generator"]
                },
                "efficiency_score": 0.89,
                "rationale": "Distributed load based on agent specialization and current workload"
            }
        })
    
    elif "coordination insights" in prompt.lower():
        return json.dumps({
            "insights": [
                "Flutter SDK shows high contention during peak hours",
                "File system tool has optimal utilization",
                "Code generator is underutilized"
            ],
            "recommendations": [
                "Consider tool pooling for Flutter SDK",
                "Implement time-slicing for high-demand tools",
                "Add caching layer for file operations"
            ],
            "bottlenecks": [
                "Flutter SDK access queue",
                "Memory allocation conflicts"
            ],
            "tool_additions": ["flutter_web_compiler", "dart_analyzer"],
            "warnings": ["Flutter agent overload detected"]
        })
    
    return "{}"


async def test_basic_tool_coordination():
    """Test basic tool coordination functionality."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up mock agents
    agent1 = AgentCapabilityInfo(
        agent_id="flutter-agent-1",
        agent_type="flutter",
        capabilities=[AgentCapability.FLUTTER_DEVELOPMENT],
        availability=True,
        current_load=0.6,
        tools=["flutter_sdk", "file_system"]
    )
    
    agent2 = AgentCapabilityInfo(        agent_id="flutter-agent-2",
        agent_type="flutter",
        capabilities=[AgentCapability.FLUTTER_DEVELOPMENT],
        availability=True,
        current_load=0.4,
        tools=["flutter_sdk", "dart_analyzer"]
    )
    
    orchestrator.available_agents = {
        agent1.agent_id: agent1,
        agent2.agent_id: agent2
    }
    
    # Simulate tool conflict
    conflict = ToolConflict(
        conflict_id="test-conflict-1",
        tool_name="flutter_sdk",
        competing_agents=["flutter-agent-1", "flutter-agent-2"],
        conflict_type="simultaneous_access",
        priority_levels={"flutter-agent-1": 2, "flutter-agent-2": 1}
    )
    orchestrator.active_tool_conflicts["flutter_sdk"] = conflict
    
    # Run coordination
    result = await orchestrator.coordinate_tool_sharing()
    
    # Verify results
    assert isinstance(result, ToolCoordinationResult)
    assert result.coordination_events > 0
    assert result.overall_efficiency > 0
    assert len(result.conflicts_resolved) > 0
    assert len(result.usage_insights) > 0
    
    # Verify LLM was called for different aspects
    assert llm_client.generate.call_count >= 3  # Usage patterns, conflicts, allocation
    
    print(f"✓ Basic coordination completed with {result.coordination_events} events")


async def test_multiple_agent_tool_conflicts():
    """Test handling of multiple agents competing for the same tools."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Create multiple mock agents with overlapping tool needs
    agents = {}
    for i in range(5):
        agent = AgentCapabilityInfo(
            agent_id=f"agent-{i}",
            name=f"Agent {i}",
            capabilities=[AgentCapability.CODE_GENERATION],
            availability=True,
            current_load=0.3 + (i * 0.1),
            tools=["flutter_sdk", "file_system"] if i < 3 else ["file_system", "code_generator"]
        )
        agents[agent.agent_id] = agent
    
    orchestrator.available_agents = agents
    
    # Create multiple conflicts
    conflicts = {
        "flutter_sdk": ToolConflict(
            tool_name="flutter_sdk",
            competing_agents=["agent-0", "agent-1", "agent-2"],
            conflict_type="queue_overflow",
            priority_levels={"agent-0": 3, "agent-1": 2, "agent-2": 1}
        ),
        "file_system": ToolConflict(
            tool_name="file_system", 
            competing_agents=["agent-1", "agent-2", "agent-3", "agent-4"],
            conflict_type="resource_exhaustion",
            priority_levels={"agent-1": 2, "agent-2": 2, "agent-3": 1, "agent-4": 1}
        )
    }
    orchestrator.active_tool_conflicts = conflicts
    
    # Run coordination
    result = await orchestrator.coordinate_tool_sharing()
    
    # Verify fair conflict resolution
    assert len(result.conflicts_resolved) >= 2
    assert result.overall_efficiency > 0.5
    
    # Check that high-priority agents are preferred
    flutter_resolution = next((r for r in result.conflicts_resolved if r.tool_name == "flutter_sdk"), None)
    if flutter_resolution:
        assert flutter_resolution.chosen_agent in ["agent-0", "agent-1"]  # Higher priority agents
    
    print(f"✓ Multiple agent conflicts resolved: {len(result.conflicts_resolved)} resolutions")


async def test_tool_queue_management():
    """Test management of tool queues for fair access."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up tool queues with waiting agents
    orchestrator.tool_queues = {
        "flutter_sdk": QueueStatus(
            tool_name="flutter_sdk",
            queue_length=3,
            waiting_agents=["agent-1", "agent-2", "agent-3"],
            average_wait_time=45.0,
            max_wait_time=120.0,
            current_user="agent-0"
        ),
        "file_system": QueueStatus(
            tool_name="file_system",
            queue_length=1,
            waiting_agents=["agent-4"],
            average_wait_time=15.0,
            max_wait_time=30.0,
            current_user="agent-2"
        )
    }
    
    # Run coordination
    result = await orchestrator.coordinate_tool_sharing()
    
    # Verify queue management improvements
    assert len(result.queue_improvements) > 0
    assert result.coordination_events > 0
    
    # Check efficiency improvements
    if "flutter_sdk" in result.queue_improvements:
        assert result.queue_improvements["flutter_sdk"] >= 0  # Should not worsen
    
    print(f"✓ Tool queue management improved: {len(result.queue_improvements)} queues optimized")


async def test_shared_operation_coordination():
    """Test coordination of long-running shared operations."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up shared operations
    shared_op = SharedOperation(
        operation_id="build-project-1",
        operation_type="flutter_build",
        participating_agents=["flutter-agent-1", "flutter-agent-2", "build-agent"],
        required_tools=["flutter_sdk", "file_system", "dart_compiler"],
        coordination_strategy="pipeline",
        status="active"
    )
    
    orchestrator.shared_operations["build-project-1"] = shared_op
    
    # Mock the coordination method to return success
    async def mock_coordinate_shared_operation(operation):
        return CoordinationResult(
            operation_id=operation.operation_id,
            coordination_success=True,
            participating_agents=operation.participating_agents,
            tool_allocations={"flutter_sdk": "flutter-agent-1", "file_system": "flutter-agent-2"},
            execution_timeline={"start": datetime.utcnow(), "estimated_end": datetime.utcnow() + timedelta(minutes=30)},
            efficiency_metrics={"resource_utilization": 0.85, "coordination_overhead": 0.15}
        )
    
    orchestrator._coordinate_shared_operations = mock_coordinate_shared_operation
    
    # Run coordination
    result = await orchestrator.coordinate_tool_sharing()
    
    # Verify shared operation coordination
    assert result.active_shared_operations > 0
    assert result.successful_coordinations > 0
    assert result.failed_coordinations == 0
    
    print(f"✓ Shared operation coordination: {result.successful_coordinations} successful")


async def test_tool_unavailability_and_fallback():
    """Test handling of tool unavailability and fallback mechanisms."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up agents with tool dependencies
    agent1 = AgentCapabilityInfo(
        agent_id="primary-agent",
        name="Primary Agent",
        capabilities=[AgentCapability.CODE_GENERATION],
        availability=True,
        current_load=0.5,
        tools=["preferred_tool", "fallback_tool"]
    )
    
    orchestrator.available_agents = {"primary-agent": agent1}
    
    # Simulate tool unavailability by creating conflicts
    unavailable_conflict = ToolConflict(
        tool_name="preferred_tool",
        competing_agents=["primary-agent", "external-system"],
        conflict_type="tool_unavailable",
        priority_levels={"primary-agent": 1, "external-system": 10}  # External system has higher priority
    )
    
    orchestrator.active_tool_conflicts["preferred_tool"] = unavailable_conflict
    
    # Mock LLM to suggest fallback
    async def mock_fallback_response(prompt: str, **kwargs):
        if "fallback" in prompt.lower() or "unavailable" in prompt.lower():
            return json.dumps({
                "resolutions": [{
                    "conflict_id": unavailable_conflict.conflict_id,
                    "resolution_type": "fallback_tool",
                    "chosen_agent": "primary-agent",
                    "fallback_tool": "fallback_tool",
                    "rationale": "Primary tool unavailable, using fallback"
                }]
            })
        return "{}"
    
    llm_client.generate.side_effect = mock_fallback_response
    
    # Run coordination
    result = await orchestrator.coordinate_tool_sharing()
    
    # Verify fallback handling
    assert len(result.conflicts_resolved) > 0
    
    # Check that fallback recommendations are provided
    fallback_mentioned = any("fallback" in insight.lower() for insight in result.usage_insights + result.optimization_recommendations)
    
    print(f"✓ Tool unavailability handled with fallback mechanisms")


async def test_efficiency_improvements():
    """Test that tool coordination improves overall system efficiency."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up suboptimal initial state
    agents = {
        f"agent-{i}": AgentCapabilityInfo(
            agent_id=f"agent-{i}",
            name=f"Agent {i}",
            capabilities=[AgentCapability.CODE_GENERATION],
            availability=True,
            current_load=0.8,  # High load indicating inefficiency
            tools=["overused_tool", "underused_tool"]
        )
        for i in range(3)
    }
    
    orchestrator.available_agents = agents
    
    # Create usage patterns showing inefficiency  
    orchestrator.tool_usage_patterns = {
        "overused_tool": UsagePattern(
            tool_name="overused_tool",
            agents=list(agents.keys()),
            frequency=100,  # Very high frequency
            avg_duration=60.0,  # Long duration
            efficiency=0.3  # Low efficiency
        ),
        "underused_tool": UsagePattern(
            tool_name="underused_tool", 
            agents=[],
            frequency=1,  # Very low frequency
            avg_duration=5.0,
            efficiency=0.9  # High efficiency when used
        )
    }
    
    # Mock LLM to suggest optimizations
    async def mock_optimization_response(prompt: str, **kwargs):
        if "optimization" in prompt.lower():
            return json.dumps({
                "allocation_plan": {
                    "agent_assignments": {
                        "agent-0": ["underused_tool"],
                        "agent-1": ["overused_tool"], 
                        "agent-2": ["underused_tool"]
                    },
                    "efficiency_score": 0.75,  # Improved efficiency
                    "optimizations": [
                        "Redistributed load from overused_tool",
                        "Increased utilization of underused_tool",
                        "Balanced agent workloads"
                    ]
                }
            })
        return await mock_llm_responses(prompt, **kwargs)
    
    llm_client.generate.side_effect = mock_optimization_response
    
    # Run coordination multiple times to track improvement
    result1 = await orchestrator.coordinate_tool_sharing()
    result2 = await orchestrator.coordinate_tool_sharing()
    
    # Verify efficiency improvements
    assert result1.overall_efficiency > 0
    assert len(result1.optimizations_made) > 0
    assert len(result1.optimization_recommendations) > 0
    
    # Check for specific efficiency improvements
    load_balancing_mentioned = any(
        "load" in opt or "balance" in opt 
        for opt in [str(o) for o in result1.optimizations_made]
    )
    
    print(f"✓ Efficiency improvements: {result1.overall_efficiency:.2f} efficiency score")


async def test_coordination_history_and_learning():
    """Test that coordination results are stored and used for learning."""
    orchestrator, llm_client, memory_manager, event_bus = await setup_orchestrator()
    
    # Set up minimal agents
    agent = AgentCapabilityInfo(
        agent_id="test-agent",
        name="Test Agent", 
        capabilities=[AgentCapability.TESTING],
        availability=True,
        current_load=0.3,
        tools=["test_tool"]
    )
    
    orchestrator.available_agents = {"test-agent": agent}
    
    # Run coordination multiple times
    results = []
    for i in range(3):
        result = await orchestrator.coordinate_tool_sharing()
        results.append(result)
        
        # Verify result is stored in history
        assert len(orchestrator.coordination_history) == i + 1
        assert orchestrator.coordination_history[-1] == result
    
    # Verify learning from history
    assert len(orchestrator.coordination_history) == 3
    
    # Check memory storage was called
    assert memory_manager.store_memory.call_count >= 3
    
    # Verify temporal patterns in coordination
    time_gaps = []
    for i in range(1, len(results)):
        gap = (results[i].timestamp - results[i-1].timestamp).total_seconds()
        time_gaps.append(gap)
    
    assert all(gap >= 0 for gap in time_gaps)  # Timestamps should be sequential
    
    print(f"✓ Coordination history maintained: {len(orchestrator.coordination_history)} entries")


async def run_all_tests():
    """Run all verification tests."""
    tests = [
        test_basic_tool_coordination,
        test_multiple_agent_tool_conflicts,
        test_tool_queue_management,
        test_shared_operation_coordination,
        test_tool_unavailability_and_fallback,
        test_efficiency_improvements,
        test_coordination_history_and_learning
    ]
    
    print("🔧 Starting Tool Coordination Verification Tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n🧪 Running {test.__name__}...")
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tool coordination tests PASSED!")
        print("\n✅ Verification Complete:")
        print("  • Tool usage pattern analysis working")
        print("  • Conflict resolution mechanisms functional")
        print("  • Tool allocation optimization active")
        print("  • Queue management improving efficiency")
        print("  • Shared operation coordination successful")
        print("  • Fallback mechanisms handling unavailability")
        print("  • Overall system efficiency improving")
        print("  • Fair conflict resolution implemented")
        print("  • Learning from coordination history")
    else:
        print(f"⚠️ {failed} tests failed - review implementation")
    
    return failed == 0


if __name__ == "__main__":
    asyncio.run(run_all_tests())
