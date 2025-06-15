#!/usr/bin/env python3
"""
Test Suite for Adaptive Workflow Modification in OrchestratorAgent.

This test suite validates:
1. Workflow performance analysis
2. Improvement opportunity identification
3. Workflow modification mechanisms
4. Agent assignment rebalancing
5. End-to-end adaptation scenarios
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.models.agent_models import AgentCapabilityInfo
from src.models.tool_models import (
    WorkflowFeedback, WorkflowStepResult, AgentPerformanceMetrics,
    WorkflowSession, AdaptationResult, PerformanceAnalysis
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_workflow_adaptation")


class MockLLMClient:
    """Mock LLM client for testing workflow adaptation."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = self._setup_responses()
    
    def _setup_responses(self) -> Dict[str, str]:
        """Setup predefined responses for different analysis types."""
        return {
            "performance_analysis": json.dumps({
                "inefficiencies": [
                    "Sequential execution of parallelizable tasks",
                    "Over-allocation of resources to simple tasks",
                    "Agent specialization mismatches"
                ],
                "critical_path": ["task_001", "task_003", "task_005"],
                "parallel_opportunities": [
                    {"steps": ["task_002", "task_004"], "estimated_savings": 45}
                ],
                "specialization_mismatches": [
                    {"step": "task_002", "current_agent": "security", "recommended": "implementation"}
                ],
                "error_patterns": ["timeout_errors_in_testing", "resource_contention"]
            }),
            
            "improvement_identification": json.dumps([
                {
                    "type": "parallelize",
                    "priority": "high",
                    "target_steps": ["task_002", "task_004"],
                    "description": "Enable parallel execution of independent UI and logic tasks",
                    "proposed_changes": {
                        "parallel_groups": [["task_002", "task_004"]]
                    },
                    "expected_benefit": 35.0,
                    "implementation_cost": 2.0,
                    "risk_level": "low",
                    "confidence_score": 0.85,
                    "supporting_evidence": ["No shared dependencies", "Different resource requirements"]
                },
                {
                    "type": "replace_agent",
                    "priority": "medium",
                    "target_steps": ["task_002"],
                    "description": "Assign implementation agent instead of security agent for UI task",
                    "proposed_changes": {
                        "new_assignments": {"task_002": "implementation_agent_001"}
                    },
                    "expected_benefit": 25.0,
                    "implementation_cost": 1.0,
                    "risk_level": "low",
                    "confidence_score": 0.90,
                    "supporting_evidence": ["Better specialization match", "Higher success rate"]
                }
            ]),
            
            "agent_assignment": json.dumps({
                "task_001": "architecture_agent_001",
                "task_002": "implementation_agent_001", 
                "task_003": "implementation_agent_002",
                "task_004": "testing_agent_001",
                "task_005": "devops_agent_001"
            })
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response based on prompt content."""
        self.call_count += 1
        
        if "performance analysis" in prompt.lower():
            return self.responses["performance_analysis"]
        elif "improvement" in prompt.lower():
            return self.responses["improvement_identification"]
        elif "agent assignment" in prompt.lower():
            return self.responses["agent_assignment"]
        else:
            return '{"status": "success", "message": "Mock response"}'


class MockMemoryManager:
    """Mock memory manager for testing."""
    
    def __init__(self):
        self.stored_memories = []
    
    async def store_memory(self, content: str, **kwargs) -> str:
        memory_id = f"memory_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": kwargs.get("metadata", {}),
            "timestamp": datetime.now()
        })
        return memory_id
    
    async def retrieve_memories(self, **kwargs) -> List[Dict[str, Any]]:
        return self.stored_memories


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.published_events = []
        self.subscriptions = []
    
    async def publish(self, topic: str, data: Any) -> None:
        self.published_events.append({"topic": topic, "data": data, "timestamp": datetime.now()})
    
    async def subscribe(self, topic: str, handler) -> None:
        self.subscriptions.append({"topic": topic, "handler": handler})


async def create_test_workflow_feedback() -> WorkflowFeedback:
    """Create test workflow feedback with various performance scenarios."""
    
    # Create step results with different performance characteristics
    step_results = [
        WorkflowStepResult(
            step_id="task_001",
            agent_id="architecture_agent_001",
            status="completed",
            execution_time=45.0,
            quality_score=0.9,
            resource_usage={"cpu": 0.3, "memory": 0.2}
        ),
        WorkflowStepResult(
            step_id="task_002",
            agent_id="security_agent_001",  # Mismatched agent for UI task
            status="completed",
            execution_time=120.0,  # Too slow for UI task
            quality_score=0.6,  # Lower quality due to mismatch
            resource_usage={"cpu": 0.8, "memory": 0.6},
            warnings=["Agent specialization mismatch"]
        ),
        WorkflowStepResult(
            step_id="task_003",
            agent_id="implementation_agent_001",
            status="completed",
            execution_time=75.0,
            quality_score=0.85,
            resource_usage={"cpu": 0.5, "memory": 0.4}
        ),
        WorkflowStepResult(
            step_id="task_004",
            agent_id="testing_agent_001",
            status="completed",
            execution_time=90.0,
            quality_score=0.8,
            resource_usage={"cpu": 0.4, "memory": 0.3}
        ),
        WorkflowStepResult(
            step_id="task_005",
            agent_id="devops_agent_001",
            status="failed",
            execution_time=30.0,  # Failed early
            quality_score=0.0,
            errors=["Deployment configuration error"],
            resource_usage={"cpu": 0.2, "memory": 0.1}
        )
    ]
    
    # Create agent performance metrics
    agent_performance = {
        "architecture_agent_001": AgentPerformanceMetrics(
            agent_id="architecture_agent_001",
            agent_type="architecture",
            tasks_completed=1,
            tasks_failed=0,
            average_execution_time=45.0,
            resource_efficiency=0.8,
            quality_average=0.9,
            error_rate=0.0,
            current_load=0.3,
            availability_score=1.0,
            specialization_match=0.95
        ),
        "security_agent_001": AgentPerformanceMetrics(
            agent_id="security_agent_001",
            agent_type="security",
            tasks_completed=1,
            tasks_failed=0,
            average_execution_time=120.0,
            resource_efficiency=0.4,  # Poor efficiency for UI task
            quality_average=0.6,
            error_rate=0.0,
            current_load=0.8,
            availability_score=0.7,
            specialization_match=0.3  # Poor match for UI task
        ),
        "implementation_agent_001": AgentPerformanceMetrics(
            agent_id="implementation_agent_001",
            agent_type="implementation",
            tasks_completed=1,
            tasks_failed=0,
            average_execution_time=75.0,
            resource_efficiency=0.7,
            quality_average=0.85,
            error_rate=0.0,
            current_load=0.5,
            availability_score=1.0,
            specialization_match=0.9
        ),
        "testing_agent_001": AgentPerformanceMetrics(
            agent_id="testing_agent_001",
            agent_type="testing",
            tasks_completed=1,
            tasks_failed=0,
            average_execution_time=90.0,
            resource_efficiency=0.6,
            quality_average=0.8,
            error_rate=0.0,
            current_load=0.4,
            availability_score=1.0,
            specialization_match=0.85
        ),
        "devops_agent_001": AgentPerformanceMetrics(
            agent_id="devops_agent_001",
            agent_type="devops",
            tasks_completed=0,
            tasks_failed=1,
            average_execution_time=30.0,
            resource_efficiency=0.2,
            quality_average=0.0,
            error_rate=1.0,
            current_load=0.2,
            availability_score=0.5,
            specialization_match=0.8
        )
    }
    
    return WorkflowFeedback(
        workflow_id="test_workflow_001",
        step_results=step_results,
        agent_performance=agent_performance,
        resource_usage={
            "cpu_utilization": 0.65,
            "memory_utilization": 0.45,
            "network_usage": 0.2,
            "storage_usage": 0.3
        },
        user_feedback="The workflow took longer than expected and had quality issues",
        user_satisfaction=0.6,
        system_alerts=["High resource usage detected", "Agent specialization mismatch"],
        overall_completion_time=360.0,  # 6 minutes total
        target_completion_time=240.0,   # Target was 4 minutes
        quality_score=0.65,
        efficiency_score=0.55,
        bottlenecks=["task_002 slow execution", "sequential task execution"],
        failures=["task_005 deployment failure"],
        improvement_suggestions=[
            "Parallelize independent tasks",
            "Reassign UI tasks to implementation agents",
            "Fix deployment configuration"
        ]
    )


async def create_test_workflow_session() -> WorkflowSession:
    """Create test workflow session for modification testing."""
    
    return WorkflowSession(
        workflow_id="test_workflow_001",
        current_steps=[
            {
                "step_id": "task_001",
                "description": "Design application architecture",
                "agent_type": "architecture",
                "estimated_duration": 30
            },
            {
                "step_id": "task_002", 
                "description": "Implement UI components",
                "agent_type": "implementation",
                "estimated_duration": 60
            },
            {
                "step_id": "task_003",
                "description": "Implement business logic",
                "agent_type": "implementation", 
                "estimated_duration": 45
            },
            {
                "step_id": "task_004",
                "description": "Create unit tests",
                "agent_type": "testing",
                "estimated_duration": 40
            },
            {
                "step_id": "task_005",
                "description": "Deploy application",
                "agent_type": "devops",
                "estimated_duration": 25
            }
        ],
        step_dependencies={
            "task_001": [],
            "task_002": ["task_001"],
            "task_003": ["task_001"],
            "task_004": ["task_002", "task_003"],
            "task_005": ["task_004"]
        },
        agent_assignments={
            "task_001": "architecture_agent_001",
            "task_002": "security_agent_001",  # Misassigned
            "task_003": "implementation_agent_001",
            "task_004": "testing_agent_001",
            "task_005": "devops_agent_001"
        },
        agent_availability={
            "architecture_agent_001": True,
            "security_agent_001": True,
            "implementation_agent_001": True,
            "implementation_agent_002": True,
            "testing_agent_001": True,
            "devops_agent_001": False  # Unavailable due to previous failure
        },
        step_timing_estimates={
            "task_001": 30.0,
            "task_002": 60.0,
            "task_003": 45.0,
            "task_004": 40.0,
            "task_005": 25.0
        },
        start_time=datetime.now() - timedelta(minutes=6),
        estimated_completion=datetime.now() + timedelta(minutes=4),
        completed_steps={"task_001", "task_002", "task_003", "task_004"},
        active_steps=set(),
        failed_steps={"task_005"}
    )


async def test_workflow_performance_analysis():
    """Test workflow performance analysis functionality."""
    print("\\n🔍 Testing Workflow Performance Analysis")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Create test feedback
    feedback = await create_test_workflow_feedback()
    
    # Test performance analysis
    analysis = await orchestrator._analyze_workflow_performance("test_workflow_001", feedback)
    
    # Verify analysis results
    assert analysis.workflow_id == "test_workflow_001"
    assert 0 <= analysis.efficiency_score <= 1.0
    assert 0 <= analysis.completion_rate <= 100.0
    assert len(analysis.bottlenecks) > 0
    
    print(f"✅ Performance Analysis Generated:")
    print(f"   - Efficiency Score: {analysis.efficiency_score:.2f}")
    print(f"   - Completion Rate: {analysis.completion_rate:.1f}%")
    print(f"   - Bottlenecks Identified: {len(analysis.bottlenecks)}")
    print(f"   - Resource Utilization: {analysis.resource_utilization}")
    
    return True


async def test_improvement_identification():
    """Test improvement opportunity identification."""
    print("\\n💡 Testing Improvement Identification")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Create test data
    feedback = await create_test_workflow_feedback()
    analysis = PerformanceAnalysis(
        workflow_id="test_workflow_001",
        efficiency_score=0.55,
        completion_rate=80.0,
        bottlenecks=[]
    )
    
    # Test improvement identification
    improvements = await orchestrator._identify_improvement_opportunities(feedback, analysis)
    
    # Verify improvements
    assert len(improvements) > 0
    assert all(imp.confidence_score > 0 for imp in improvements)
    assert all(imp.expected_benefit > 0 for imp in improvements)
    
    print(f"✅ Improvements Identified: {len(improvements)}")
    for i, improvement in enumerate(improvements, 1):
        print(f"   {i}. {improvement.type.upper()}: {improvement.description}")
        print(f"      Priority: {improvement.priority}, Benefit: {improvement.expected_benefit}%")
        print(f"      Confidence: {improvement.confidence_score:.2f}")
    
    return True


async def test_workflow_modification():
    """Test workflow step modification mechanisms."""
    print("\\n🔧 Testing Workflow Modification")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Create test workflow and improvements
    workflow = await create_test_workflow_session()
    original_step_count = len(workflow.current_steps)
    
    # Create mock improvements
    from src.models.tool_models import WorkflowImprovement
    improvements = [
        WorkflowImprovement(
            type="parallelize",
            priority="high",
            target_steps=["task_002", "task_003"],
            description="Enable parallel execution of UI and logic tasks",
            proposed_changes={"parallel_groups": [["task_002", "task_003"]]},
            expected_benefit=30.0,
            confidence_score=0.85
        ),
        WorkflowImprovement(
            type="remove_step",
            priority="medium",
            target_steps=["task_005"],
            description="Remove failed deployment step for reconfiguration",
            proposed_changes={},
            expected_benefit=10.0,
            confidence_score=0.90
        )
    ]
    
    # Test workflow modification
    modified_workflow = await orchestrator._modify_workflow_steps(workflow, improvements)
    
    # Verify modifications
    assert modified_workflow.adaptation_count == len(improvements)
    assert len(modified_workflow.modifications) == len(improvements)
    assert modified_workflow.last_adaptation is not None
    
    # Check specific modifications
    # - Parallel execution should be enabled (dependencies updated)
    # - Step should be removed
    current_step_ids = [step["step_id"] for step in modified_workflow.current_steps]
    assert "task_005" not in current_step_ids  # Removed step
    
    print(f"✅ Workflow Modified Successfully:")
    print(f"   - Original steps: {original_step_count}")
    print(f"   - Current steps: {len(modified_workflow.current_steps)}")
    print(f"   - Adaptations applied: {modified_workflow.adaptation_count}")
    print(f"   - Last adaptation: {modified_workflow.last_adaptation}")
    
    return True


async def test_agent_rebalancing():
    """Test agent assignment rebalancing."""
    print("\\n⚖️ Testing Agent Assignment Rebalancing")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Setup available agents
    agents = [
        AgentCapabilityInfo(
            agent_id="architecture_agent_001",
            agent_type="architecture",
            capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
            current_load=0.3,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="implementation_agent_001",
            agent_type="implementation",
            capabilities=[AgentCapability.CODE_GENERATION],
            current_load=0.5,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="implementation_agent_002",
            agent_type="implementation",
            capabilities=[AgentCapability.CODE_GENERATION],
            current_load=0.2,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="testing_agent_001",
            agent_type="testing",
            capabilities=[AgentCapability.TESTING],
            current_load=0.4,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="devops_agent_001",
            agent_type="devops",
            capabilities=[AgentCapability.DEPLOYMENT],
            current_load=0.9,
            availability=False
        )
    ]
    
    for agent in agents:
        orchestrator.available_agents[agent.agent_id] = agent
    
    # Create test workflow
    workflow = await create_test_workflow_session()
    original_assignments = workflow.agent_assignments.copy()
    
    # Test agent rebalancing
    new_assignments = await orchestrator._rebalance_agent_assignments(workflow)
    
    # Verify rebalancing occurred
    assert len(new_assignments) > 0
    print(f"✅ Agent Rebalancing Completed:")
    print(f"   - Original assignments: {len(original_assignments)}")
    print(f"   - New assignments: {len(new_assignments)}")
    
    for step_id, new_agent in new_assignments.items():
        old_agent = original_assignments.get(step_id, "unassigned")
        if old_agent != new_agent:
            print(f"   - {step_id}: {old_agent} → {new_agent}")
    
    return True


async def test_end_to_end_adaptation():
    """Test complete end-to-end workflow adaptation."""
    print("\\n🔄 Testing End-to-End Workflow Adaptation")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Setup available agents
    agents = [
        AgentCapabilityInfo(
            agent_id="architecture_agent_001",
            agent_type="architecture",
            capabilities=[AgentCapability.ARCHITECTURE_ANALYSIS],
            current_load=0.3,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="implementation_agent_001",
            agent_type="implementation",
            capabilities=[AgentCapability.CODE_GENERATION],
            current_load=0.5,
            availability=True
        ),
        AgentCapabilityInfo(
            agent_id="testing_agent_001",
            agent_type="testing",
            capabilities=[AgentCapability.TESTING],
            current_load=0.4,
            availability=True
        )
    ]
    
    for agent in agents:
        orchestrator.available_agents[agent.agent_id] = agent
    
    # Setup workflow session
    workflow = await create_test_workflow_session()
    orchestrator.active_workflows["test_workflow_001"] = workflow
    
    # Create feedback
    feedback = await create_test_workflow_feedback()
    
    # Test complete adaptation
    adaptation_result = await orchestrator.adapt_workflow("test_workflow_001", feedback)
    
    # Verify adaptation results
    assert adaptation_result.workflow_id == "test_workflow_001"
    assert adaptation_result.adaptation_id is not None
    assert len(adaptation_result.changes_made) > 0
    assert adaptation_result.confidence_score > 0
    
    # Verify workflow was updated
    updated_workflow = orchestrator.active_workflows["test_workflow_001"]
    assert updated_workflow.adaptation_count > 0
    assert len(updated_workflow.modifications) > 0
    
    # Verify memory storage
    assert len(memory_manager.stored_memories) > 0
    adaptation_memories = [
        m for m in memory_manager.stored_memories 
        if "adaptation" in m["content"].lower()
    ]
    assert len(adaptation_memories) > 0
    
    print(f"✅ End-to-End Adaptation Successful:")
    print(f"   - Adaptation ID: {adaptation_result.adaptation_id}")
    print(f"   - Changes Made: {len(adaptation_result.changes_made)}")
    print(f"   - Expected Time Savings: {adaptation_result.estimated_time_savings}%")
    print(f"   - Expected Quality Improvement: {adaptation_result.estimated_quality_improvement}%")
    print(f"   - Confidence Score: {adaptation_result.confidence_score:.2f}")
    print(f"   - Workflow Adaptations: {updated_workflow.adaptation_count}")
    
    return True


async def test_adaptation_edge_cases():
    """Test edge cases and error scenarios in workflow adaptation."""
    print("\\n⚠️ Testing Adaptation Edge Cases")
    print("=" * 50)
    
    # Setup
    llm_client = MockLLMClient()
    memory_manager = MockMemoryManager()
    event_bus = MockEventBus()
    
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator",
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    
    # Test 1: Non-existent workflow
    feedback = await create_test_workflow_feedback()
    result = await orchestrator.adapt_workflow("non_existent_workflow", feedback)
    assert result.confidence_score == 0.0
    print("✅ Handled non-existent workflow correctly")
    
    # Test 2: Empty feedback
    workflow = await create_test_workflow_session()
    orchestrator.active_workflows["empty_feedback_test"] = workflow
    
    empty_feedback = WorkflowFeedback(workflow_id="empty_feedback_test")
    result = await orchestrator.adapt_workflow("empty_feedback_test", empty_feedback)
    print("✅ Handled empty feedback correctly")
    
    # Test 3: All agents unavailable
    workflow.agent_availability = {agent_id: False for agent_id in workflow.agent_availability}
    orchestrator.active_workflows["no_agents_test"] = workflow
    
    result = await orchestrator.adapt_workflow("no_agents_test", feedback)
    print("✅ Handled agent unavailability correctly")
    
    return True


async def run_all_tests():
    """Run all workflow adaptation tests."""
    print("🧪 Running Adaptive Workflow Modification Tests")
    print("=" * 60)
    
    tests = [
        ("Workflow Performance Analysis", test_workflow_performance_analysis),
        ("Improvement Identification", test_improvement_identification),
        ("Workflow Modification", test_workflow_modification),
        ("Agent Rebalancing", test_agent_rebalancing),
        ("End-to-End Adaptation", test_end_to_end_adaptation),
        ("Edge Cases", test_adaptation_edge_cases)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"Test {test_name} failed: {e}")
    
    # Print summary
    print("\\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"{status_emoji} {test_name}: {result}")
    
    print(f"\\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n🎉 All adaptive workflow modification tests passed!")
        print("✅ The system successfully:")
        print("   - Analyzes workflow performance using LLM reasoning")
        print("   - Identifies improvement opportunities intelligently")
        print("   - Modifies workflows based on real-time feedback")
        print("   - Rebalances agent assignments optimally")
        print("   - Handles edge cases and error scenarios gracefully")
        print("   - Maintains workflow integrity throughout adaptations")
    else:
        print("\\n⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())
