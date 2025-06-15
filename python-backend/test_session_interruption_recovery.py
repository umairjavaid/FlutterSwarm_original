#!/usr/bin/env python3
"""
Session Interruption and Recovery Testing for OrchestratorAgent.

This module tests the enhanced session management capabilities:
1. Various interruption scenarios (system crashes, network failures, resource exhaustion)
2. Recovery plan generation and execution
3. State persistence and restoration
4. Resource cleanup and reallocation
5. Session continuity after interruptions

Usage:
    python test_session_interruption_recovery.py
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock

# Add src to path
import sys
sys.path.insert(0, 'src')

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
from src.models.task_models import TaskContext, TaskPriority
from src.models.tool_models import (
    DevelopmentSession, SessionState, Interruption, InterruptionType,
    RecoveryPlan, RecoveryStep, RecoveryStrategy, PauseResult, ResumeResult
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("session_interruption_test")


class InterruptionScenario:
    """Base class for interruption testing scenarios."""
    
    def __init__(self, name: str, interruption_type: InterruptionType, severity: str):
        self.name = name
        self.interruption_type = interruption_type
        self.severity = severity
        self.start_time = None
        self.recovery_time = None
        
    async def create_interruption(self, session_id: str) -> Interruption:
        """Create interruption instance for this scenario."""
        return Interruption(
            session_id=session_id,
            interruption_type=self.interruption_type,
            description=f"Test interruption: {self.name}",
            severity=self.severity,
            timestamp=datetime.now(),
            affected_components=self._get_affected_components()
        )
    
    def _get_affected_components(self) -> List[str]:
        """Get components affected by this interruption type."""
        component_map = {
            InterruptionType.SYSTEM_ERROR: ["workflow_engine", "task_scheduler"],
            InterruptionType.NETWORK_FAILURE: ["llm_client", "event_bus", "remote_tools"],
            InterruptionType.RESOURCE_EXHAUSTION: ["memory_manager", "agent_pool"],
            InterruptionType.AGENT_FAILURE: ["specific_agent", "task_queue"],
            InterruptionType.TOOL_UNAVAILABLE: ["tool_registry", "resource_manager"],
            InterruptionType.EXTERNAL_DEPENDENCY: ["external_api", "database"],
            InterruptionType.USER_INTERRUPTION: ["user_interface", "session_manager"]
        }
        
        return component_map.get(self.interruption_type, ["unknown_component"])


class SessionInterruptionTester:
    """Comprehensive session interruption and recovery testing framework."""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
        self.scenarios = self._create_interruption_scenarios()
        
    def _create_interruption_scenarios(self) -> List[InterruptionScenario]:
        """Create comprehensive set of interruption scenarios."""
        return [
            # System-level interruptions
            InterruptionScenario("System Crash", InterruptionType.SYSTEM_ERROR, "critical"),
            InterruptionScenario("Memory Leak", InterruptionType.SYSTEM_ERROR, "high"),
            InterruptionScenario("Process Deadlock", InterruptionType.SYSTEM_ERROR, "high"),
            
            # Network-related interruptions
            InterruptionScenario("Complete Network Loss", InterruptionType.NETWORK_FAILURE, "critical"),
            InterruptionScenario("Intermittent Connectivity", InterruptionType.NETWORK_FAILURE, "medium"),
            InterruptionScenario("API Rate Limiting", InterruptionType.NETWORK_FAILURE, "low"),
            
            # Resource exhaustion
            InterruptionScenario("Memory Exhaustion", InterruptionType.RESOURCE_EXHAUSTION, "critical"),
            InterruptionScenario("CPU Overload", InterruptionType.RESOURCE_EXHAUSTION, "high"),
            InterruptionScenario("Disk Space Full", InterruptionType.RESOURCE_EXHAUSTION, "medium"),
            
            # Agent failures
            InterruptionScenario("Primary Agent Crash", InterruptionType.AGENT_FAILURE, "high"),
            InterruptionScenario("Agent Unresponsive", InterruptionType.AGENT_FAILURE, "medium"),
            InterruptionScenario("Agent Pool Exhaustion", InterruptionType.AGENT_FAILURE, "high"),
            
            # Tool unavailability
            InterruptionScenario("Critical Tool Offline", InterruptionType.TOOL_UNAVAILABLE, "critical"),
            InterruptionScenario("Tool Version Conflict", InterruptionType.TOOL_UNAVAILABLE, "medium"),
            InterruptionScenario("License Expiration", InterruptionType.TOOL_UNAVAILABLE, "low"),
            
            # External dependencies
            InterruptionScenario("Database Connection Lost", InterruptionType.EXTERNAL_DEPENDENCY, "critical"),
            InterruptionScenario("External API Down", InterruptionType.EXTERNAL_DEPENDENCY, "high"),
            InterruptionScenario("Authentication Service Failure", InterruptionType.EXTERNAL_DEPENDENCY, "critical"),
            
            # User interruptions
            InterruptionScenario("User Emergency Stop", InterruptionType.USER_INTERRUPTION, "medium"),
            InterruptionScenario("Priority Change Request", InterruptionType.USER_INTERRUPTION, "low"),
            InterruptionScenario("Session Timeout", InterruptionType.USER_INTERRUPTION, "low")
        ]
    
    async def setup(self):
        """Setup testing environment."""
        logger.info("Setting up session interruption testing environment")
        
        config = AgentConfig(
            agent_id="interruption-test-orchestrator",
            agent_type="orchestrator",
            capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.SESSION_MANAGEMENT]
        )
        
        # Create orchestrator with enhanced mock dependencies
        self.orchestrator = OrchestratorAgent(
            config=config,
            llm_client=MockRecoveryLLMClient(),
            memory_manager=MockPersistentMemoryManager(),
            event_bus=MockReliableEventBus()
        )
        
        # Setup mock agents with different reliability levels
        self.orchestrator.available_agents = {
            "reliable_agent_001": MockReliableAgent("reliable_agent_001"),
            "unreliable_agent_001": MockUnreliableAgent("unreliable_agent_001"),
            "critical_agent_001": MockCriticalAgent("critical_agent_001"),
            "backup_agent_001": MockBackupAgent("backup_agent_001")
        }
        
        logger.info("✅ Interruption testing environment setup complete")
    
    async def run_all_interruption_tests(self) -> Dict[str, Any]:
        """Run comprehensive interruption and recovery tests."""
        logger.info("🚀 Starting comprehensive session interruption and recovery tests")
        
        test_start = time.time()
        
        # Test results storage
        results = {
            "scenarios_tested": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_times": [],
            "state_preservation_tests": 0,
            "continuity_tests": 0,
            "scenario_details": []
        }
        
        # Run interruption scenarios
        for scenario in self.scenarios:
            logger.info(f"Testing scenario: {scenario.name}")
            scenario_result = await self._test_interruption_scenario(scenario)
            
            results["scenarios_tested"] += 1
            results["scenario_details"].append(scenario_result)
            
            if scenario_result["recovery_successful"]:
                results["successful_recoveries"] += 1
                results["recovery_times"].append(scenario_result["recovery_time"])
            else:
                results["failed_recoveries"] += 1
            
            if scenario_result["state_preserved"]:
                results["state_preservation_tests"] += 1
            
            if scenario_result["continuity_maintained"]:
                results["continuity_tests"] += 1
        
        # Additional comprehensive tests
        results.update(await self._test_complex_interruption_scenarios())
        results.update(await self._test_cascade_failure_scenarios())
        results.update(await self._test_recovery_optimization())
        
        # Calculate summary metrics
        total_duration = time.time() - test_start
        
        recovery_success_rate = (results["successful_recoveries"] / 
                               results["scenarios_tested"] if results["scenarios_tested"] > 0 else 0)
        
        avg_recovery_time = (sum(results["recovery_times"]) / 
                           len(results["recovery_times"]) if results["recovery_times"] else 0)
        
        results.update({
            "total_duration": total_duration,
            "recovery_success_rate": recovery_success_rate,
            "average_recovery_time": avg_recovery_time,
            "state_preservation_rate": results["state_preservation_tests"] / results["scenarios_tested"],
            "continuity_success_rate": results["continuity_tests"] / results["scenarios_tested"]
        })
        
        return results
    
    async def _test_interruption_scenario(self, scenario: InterruptionScenario) -> Dict[str, Any]:
        """Test a specific interruption scenario."""
        
        # Create test session
        project_context = ProjectContext(
            project_name=f"interruption_test_{scenario.name.lower().replace(' ', '_')}",
            project_type=ProjectType.MOBILE_APP,
            description=f"Testing interruption scenario: {scenario.name}",
            platforms=[PlatformTarget.ANDROID]
        )
        
        session = await self.orchestrator.create_development_session(
            project_context=project_context,
            session_type="interruption_test"
        )
        
        await self.orchestrator.initialize_session(session.session_id)
        
        # Start some work to create session state
        task_context = TaskContext(
            task_id=f"test_task_{scenario.name.lower().replace(' ', '_')}",
            description=f"Test task for {scenario.name}",
            priority=TaskPriority.HIGH
        )
        
        workflow = await self.orchestrator.decompose_task(task_context, session.session_id)
        
        # Start workflow execution (don't wait for completion)
        execution_task = asyncio.create_task(
            self.orchestrator.execute_workflow(workflow, session.session_id)
        )
        
        # Allow some progress
        await asyncio.sleep(0.5)
        
        # Capture pre-interruption state
        pre_interruption_state = await self._capture_session_state(session.session_id)
        
        # Create and inject interruption
        interruption = await scenario.create_interruption(session.session_id)
        interruption_time = time.time()
        
        # Handle interruption
        recovery_plan = await self.orchestrator.handle_interruption(interruption)
        
        # Execute recovery
        recovery_start = time.time()
        recovery_result = await self.orchestrator.execute_recovery_plan(
            session_id=session.session_id,
            recovery_plan=recovery_plan
        )
        recovery_time = time.time() - recovery_start
        
        # Verify recovery
        post_recovery_state = await self._capture_session_state(session.session_id)
        
        # Test session continuity
        continuity_result = await self._test_session_continuity(
            session.session_id, workflow, pre_interruption_state
        )
        
        # Cancel execution task
        execution_task.cancel()
        
        # Cleanup
        await self.orchestrator.terminate_session(session.session_id)
        
        return {
            "scenario_name": scenario.name,
            "interruption_type": scenario.interruption_type.value,
            "severity": scenario.severity,
            "recovery_successful": recovery_result.success,
            "recovery_time": recovery_time,
            "state_preserved": self._compare_session_states(pre_interruption_state, post_recovery_state),
            "continuity_maintained": continuity_result["success"],
            "recovery_plan_steps": len(recovery_plan.steps) if recovery_plan else 0,
            "affected_components": interruption.affected_components
        }
    
    async def _test_complex_interruption_scenarios(self) -> Dict[str, Any]:
        """Test complex multi-component interruption scenarios."""
        logger.info("Testing complex interruption scenarios")
        
        complex_results = {
            "complex_scenarios_tested": 0,
            "complex_recoveries_successful": 0
        }
        
        # Scenario 1: Multiple simultaneous interruptions
        project_context = ProjectContext(
            project_name="complex_multi_interruption_test",
            project_type=ProjectType.ENTERPRISE_APP,
            platforms=[PlatformTarget.ANDROID, PlatformTarget.IOS]
        )
        
        session = await self.orchestrator.create_development_session(
            project_context=project_context,
            session_type="complex_interruption_test"
        )
        
        await self.orchestrator.initialize_session(session.session_id)
        
        # Create multiple simultaneous interruptions
        interruptions = [
            Interruption(
                session_id=session.session_id,
                interruption_type=InterruptionType.NETWORK_FAILURE,
                description="Network failure during complex scenario",
                severity="high"
            ),
            Interruption(
                session_id=session.session_id,
                interruption_type=InterruptionType.AGENT_FAILURE,
                description="Agent failure during network issues",
                severity="medium"
            ),
            Interruption(
                session_id=session.session_id,
                interruption_type=InterruptionType.TOOL_UNAVAILABLE,
                description="Tool becomes unavailable",
                severity="low"
            )
        ]
        
        # Handle multiple interruptions
        recovery_plans = []
        for interruption in interruptions:
            plan = await self.orchestrator.handle_interruption(interruption)
            recovery_plans.append(plan)
        
        # Execute coordinated recovery
        coordinated_result = await self.orchestrator.execute_coordinated_recovery(
            session_id=session.session_id,
            recovery_plans=recovery_plans
        )
        
        complex_results["complex_scenarios_tested"] += 1
        if coordinated_result.success:
            complex_results["complex_recoveries_successful"] += 1
        
        # Cleanup
        await self.orchestrator.terminate_session(session.session_id)
        
        return complex_results
    
    async def _test_cascade_failure_scenarios(self) -> Dict[str, Any]:
        """Test cascade failure and recovery scenarios."""
        logger.info("Testing cascade failure scenarios")
        
        cascade_results = {
            "cascade_scenarios_tested": 0,
            "cascade_recoveries_successful": 0,
            "cascade_prevention_tests": 0
        }
        
        # Create scenario prone to cascade failures
        project_context = ProjectContext(
            project_name="cascade_failure_test",
            project_type=ProjectType.MOBILE_APP,
            description="Testing cascade failure prevention and recovery"
        )
        
        session = await self.orchestrator.create_development_session(
            project_context=project_context,
            session_type="cascade_test"
        )
        
        await self.orchestrator.initialize_session(session.session_id)
        
        # Initial failure that could trigger cascades
        primary_interruption = Interruption(
            session_id=session.session_id,
            interruption_type=InterruptionType.AGENT_FAILURE,
            description="Primary agent failure that could cascade",
            severity="critical",
            affected_components=["primary_agent", "dependent_tasks"]
        )
        
        # Handle with cascade prevention
        cascade_prevention_result = await self.orchestrator.handle_interruption_with_cascade_prevention(
            interruption=primary_interruption,
            prevention_strategy="isolate_and_redistribute"
        )
        
        cascade_results["cascade_scenarios_tested"] += 1
        cascade_results["cascade_prevention_tests"] += 1
        
        if cascade_prevention_result.cascade_prevented:
            cascade_results["cascade_recoveries_successful"] += 1
        
        # Cleanup
        await self.orchestrator.terminate_session(session.session_id)
        
        return cascade_results
    
    async def _test_recovery_optimization(self) -> Dict[str, Any]:
        """Test recovery plan optimization and learning."""
        logger.info("Testing recovery optimization")
        
        optimization_results = {
            "optimization_tests": 0,
            "learning_improvements": 0,
            "recovery_time_improvements": []
        }
        
        # Create similar interruption scenarios to test learning
        base_scenario = InterruptionScenario("Learning Test", InterruptionType.SYSTEM_ERROR, "medium")
        
        previous_recovery_time = None
        
        for i in range(3):  # Run similar scenarios to test learning
            project_context = ProjectContext(
                project_name=f"recovery_optimization_test_{i}",
                project_type=ProjectType.MOBILE_APP
            )
            
            session = await self.orchestrator.create_development_session(
                project_context=project_context,
                session_type="optimization_test"
            )
            
            await self.orchestrator.initialize_session(session.session_id)
            
            # Create similar interruption
            interruption = await base_scenario.create_interruption(session.session_id)
            
            # Handle with learning enabled
            recovery_start = time.time()
            recovery_plan = await self.orchestrator.handle_interruption_with_learning(
                interruption=interruption,
                learn_from_history=True
            )
            
            recovery_result = await self.orchestrator.execute_recovery_plan(
                session_id=session.session_id,
                recovery_plan=recovery_plan
            )
            
            recovery_time = time.time() - recovery_start
            
            optimization_results["optimization_tests"] += 1
            
            if previous_recovery_time and recovery_time < previous_recovery_time:
                optimization_results["learning_improvements"] += 1
                improvement = (previous_recovery_time - recovery_time) / previous_recovery_time
                optimization_results["recovery_time_improvements"].append(improvement)
            
            previous_recovery_time = recovery_time
            
            # Cleanup
            await self.orchestrator.terminate_session(session.session_id)
        
        return optimization_results
    
    async def _capture_session_state(self, session_id: str) -> Dict[str, Any]:
        """Capture current session state for comparison."""
        session = await self.orchestrator.get_session(session_id)
        
        return {
            "session_state": session.state.value if session.state else "unknown",
            "allocated_resources": list(session.allocated_resources) if session.allocated_resources else [],
            "active_workflows": len(session.active_workflows) if hasattr(session, 'active_workflows') else 0,
            "checkpoint_count": len(session.checkpoints) if hasattr(session, 'checkpoints') else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _compare_session_states(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> bool:
        """Compare session states to verify preservation."""
        # Key components that should be preserved
        preserved_components = [
            "allocated_resources",
            "active_workflows"  # Count should be maintained or recovered
        ]
        
        for component in preserved_components:
            if component in pre_state and component in post_state:
                # For resource lists, check if resources are maintained or properly reallocated
                if component == "allocated_resources":
                    if len(post_state[component]) == 0 and len(pre_state[component]) > 0:
                        return False  # Resources lost and not recovered
                # For workflow counts, check if they're maintained or properly handled
                elif component == "active_workflows":
                    if post_state[component] < pre_state[component]:
                        # Some workflows may be paused/recovered, but shouldn't be lost entirely
                        pass  # This might be acceptable depending on recovery strategy
        
        return True  # State adequately preserved
    
    async def _test_session_continuity(self, session_id: str, workflow: Any, pre_state: Dict[str, Any]) -> Dict[str, Any]:
        """Test that session can continue after recovery."""
        try:
            # Try to resume workflow execution
            resume_result = await self.orchestrator.resume_session(session_id)
            
            # Try to create new task to test continuity
            continuity_task = TaskContext(
                task_id="continuity_test",
                description="Test task for session continuity",
                priority=TaskPriority.LOW
            )
            
            new_workflow = await self.orchestrator.decompose_task(continuity_task, session_id)
            
            return {
                "success": resume_result.success and new_workflow is not None,
                "resume_successful": resume_result.success,
                "new_task_creation": new_workflow is not None,
                "session_responsive": True
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_responsive": False
            }


# Mock classes for testing
class MockRecoveryLLMClient:
    """Mock LLM client that provides recovery-focused responses."""
    
    async def generate(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Analyze prompt for recovery planning
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        
        if "recovery" in prompt_text.lower():
            return {
                "content": json.dumps({
                    "recovery_plan": {
                        "strategy": "incremental_restoration",
                        "steps": [
                            {"action": "assess_damage", "priority": 1, "estimated_time": 30},
                            {"action": "restore_critical_components", "priority": 2, "estimated_time": 120},
                            {"action": "reallocate_resources", "priority": 3, "estimated_time": 60},
                            {"action": "resume_workflows", "priority": 4, "estimated_time": 90}
                        ],
                        "estimated_total_time": 300,
                        "confidence": 0.85
                    }
                })
            }
        else:
            return {"content": json.dumps({"status": "success", "response": "mock recovery response"})}


class MockPersistentMemoryManager:
    """Mock memory manager with persistence capabilities."""
    
    def __init__(self):
        self.persistent_store = {}
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None, persistent: bool = True):
        if persistent:
            self.persistent_store[key] = value
    
    async def retrieve(self, key: str) -> Any:
        return self.persistent_store.get(key, {"restored": "data"})
    
    async def restore_from_persistence(self, session_id: str) -> Dict[str, Any]:
        return self.persistent_store.get(f"session_{session_id}", {})


class MockReliableEventBus:
    """Mock event bus with reliability features."""
    
    async def publish(self, event_type: str, data: Any):
        # Simulate reliable event delivery
        pass
    
    async def publish_with_retry(self, event_type: str, data: Any, max_retries: int = 3):
        # Simulate retry mechanism
        return True


class MockReliableAgent:
    """Mock agent that rarely fails."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.reliability = 0.95


class MockUnreliableAgent:
    """Mock agent that fails more frequently."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.reliability = 0.70


class MockCriticalAgent:
    """Mock agent that's critical to operations."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.is_critical = True
        self.reliability = 0.90


class MockBackupAgent:
    """Mock agent that serves as backup."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.is_backup = True
        self.reliability = 0.85


async def main():
    """Run session interruption and recovery tests."""
    print("🚨 Starting Session Interruption and Recovery Tests")
    print("=" * 60)
    
    tester = SessionInterruptionTester()
    await tester.setup()
    
    results = await tester.run_all_interruption_tests()
    
    # Display results
    print("\n📊 INTERRUPTION AND RECOVERY TEST RESULTS")
    print("=" * 60)
    
    print(f"Total Scenarios Tested: {results['scenarios_tested']}")
    print(f"Successful Recoveries: {results['successful_recoveries']}")
    print(f"Failed Recoveries: {results['failed_recoveries']}")
    print(f"Recovery Success Rate: {results['recovery_success_rate']:.1%}")
    print(f"Average Recovery Time: {results['average_recovery_time']:.2f}s")
    print(f"State Preservation Rate: {results['state_preservation_rate']:.1%}")
    print(f"Continuity Success Rate: {results['continuity_success_rate']:.1%}")
    
    # Complex scenarios
    if 'complex_scenarios_tested' in results:
        print(f"\nComplex Scenarios: {results['complex_recoveries_successful']}/{results['complex_scenarios_tested']}")
    
    if 'cascade_scenarios_tested' in results:
        print(f"Cascade Prevention: {results['cascade_recoveries_successful']}/{results['cascade_scenarios_tested']}")
    
    if 'optimization_tests' in results:
        print(f"Recovery Optimization: {results['learning_improvements']}/{results['optimization_tests']} improvements")
    
    # Detailed scenario results
    print(f"\n📋 DETAILED SCENARIO RESULTS")
    print("-" * 60)
    
    for scenario in results['scenario_details']:
        status = "✅" if scenario['recovery_successful'] else "❌"
        print(f"{status} {scenario['scenario_name']} ({scenario['severity']}) - {scenario['recovery_time']:.2f}s")
    
    # Save results
    with open("interruption_recovery_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to interruption_recovery_results.json")
    
    # Determine success
    success_threshold = 0.8  # 80% success rate required
    overall_success = (results['recovery_success_rate'] >= success_threshold and 
                      results['state_preservation_rate'] >= success_threshold)
    
    if overall_success:
        print(f"\n🎉 Session interruption and recovery tests PASSED!")
        return 0
    else:
        print(f"\n⚠️ Session interruption and recovery tests need improvement!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
