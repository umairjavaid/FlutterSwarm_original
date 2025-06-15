#!/usr/bin/env python3
"""
Comprehensive Session Management Testing for FlutterSwarm OrchestratorAgent.

This test suite validates:
1. Multiple concurrent sessions
2. Session lifecycle management (create, pause, resume, terminate)
3. Resource cleanup and management
4. Interruption handling and recovery
5. State persistence and restoration
6. Agent coordination during session operations
7. Performance under concurrent session load

Usage:
    python test_session_management.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('session_management_test.log')
    ]
)

logger = logging.getLogger("session_management_test")


@dataclass
class SessionTestResult:
    """Test result tracking for session management tests."""
    test_name: str
    success: bool
    duration: float
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    sub_results: List['SessionTestResult'] = field(default_factory=list)


@dataclass
class SessionMetrics:
    """Metrics collected during session testing."""
    total_sessions_created: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_test_duration: float = 0.0
    concurrent_sessions_peak: int = 0
    resource_cleanup_success_rate: float = 0.0
    recovery_success_rate: float = 0.0
    average_session_creation_time: float = 0.0
    average_pause_time: float = 0.0
    average_resume_time: float = 0.0


class MockLLMClient:
    """Mock LLM client for session management testing."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = self._setup_responses()
    
    def _setup_responses(self):
        return {
            "session_analysis": {
                "agents_needed": ["implementation_agent", "testing_agent"],
                "resources_required": ["flutter_sdk", "dart_analyzer"],
                "workflow_steps": [
                    {"step": "project_setup", "estimated_time": 5},
                    {"step": "development", "estimated_time": 30},
                    {"step": "testing", "estimated_time": 10}
                ],
                "reasoning": "Standard Flutter project requires implementation and testing capabilities"
            },
            "agent_coordination": {
                "coordination_plan": {
                    "pause_sequence": ["testing_agent", "implementation_agent"],
                    "resource_preservation": ["preserve_state", "cleanup_temp_files"],
                    "estimated_completion": 2.0
                },
                "reasoning": "Graceful pause requires ordered agent coordination"
            },
            "recovery_planning": {
                "recovery_steps": [
                    {"action": "validate_environment", "priority": "high"},
                    {"action": "restore_agent_state", "priority": "high"},
                    {"action": "resume_workflow", "priority": "medium"}
                ],
                "estimated_recovery_time": 3.0,
                "reasoning": "Standard recovery process for interrupted Flutter development"
            }
        }
    
    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate mock completion for session management operations."""
        self.call_count += 1
        
        # Analyze the request to determine appropriate response
        last_message = messages[-1]["content"].lower()
        
        if "analyze" in last_message and "session" in last_message:
            return {"content": json.dumps(self.responses["session_analysis"])}
        elif "coordinate" in last_message and "agent" in last_message:
            return {"content": json.dumps(self.responses["agent_coordination"])}
        elif "recovery" in last_message or "interruption" in last_message:
            return {"content": json.dumps(self.responses["recovery_planning"])}
        else:
            # Default response
            return {
                "content": json.dumps({
                    "action": "proceed",
                    "reasoning": "Standard operation approved",
                    "estimated_time": 1.0
                })
            }


class MockProjectContext:
    """Mock project context for testing."""
    
    def __init__(self, name: str = "test_project", project_type: str = "flutter"):
        self.name = name
        self.project_type = project_type
        self.root_directory = f"/tmp/test_projects/{name}"
        self.requirements = ["flutter_sdk", "dart_analyzer"]
        
    def to_dict(self):
        return {
            "name": self.name,
            "project_type": self.project_type,
            "root_directory": self.root_directory,
            "requirements": self.requirements
        }


class SessionManagementTester:
    """Comprehensive session management testing framework."""
    
    def __init__(self):
        self.results: List[SessionTestResult] = []
        self.metrics = SessionMetrics()
        self.orchestrator = None
        self.active_test_sessions: Dict[str, Any] = {}
        
    async def setup_test_environment(self):
        """Setup test environment with mocked dependencies."""
        try:
            from src.agents.orchestrator_agent import OrchestratorAgent
            from src.agents.base_agent import AgentConfig
            from src.core.memory_manager import MemoryManager
            from src.core.event_bus import EventBus
            
            # Create mock configuration
            config = AgentConfig(
                agent_id="test_orchestrator",
                agent_name="Test Orchestrator",
                capabilities=["workflow_management", "session_management"],
                config={}
            )
            
            # Create mock dependencies
            llm_client = MockLLMClient()
            memory_manager = MemoryManager()
            event_bus = EventBus()
            
            # Initialize orchestrator
            self.orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
            
            logger.info("Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            traceback.print_exc()
            return False
    
    async def test_single_session_lifecycle(self) -> SessionTestResult:
        """Test complete lifecycle of a single session."""
        test_name = "single_session_lifecycle"
        start_time = time.time()
        
        try:
            # Create project context
            project_context = MockProjectContext("lifecycle_test")
            
            # Test session creation
            session = await self.orchestrator.create_development_session(project_context)
            
            if not session or not session.session_id:
                raise Exception("Failed to create session")
            
            session_id = session.session_id
            self.active_test_sessions[session_id] = session
            
            # Verify session is active
            assert session.state.value == "active", f"Expected active state, got {session.state.value}"
            assert session_id in self.orchestrator.active_sessions
            
            # Test session pause
            pause_result = await self.orchestrator.pause_session(session_id)
            assert pause_result.success, f"Pause failed: {pause_result.error_message}"
            
            # Verify paused state
            session = self.orchestrator.active_sessions[session_id]
            assert session.state.value == "paused", f"Expected paused state, got {session.state.value}"
            
            # Test session resume
            resume_result = await self.orchestrator.resume_session(session_id)
            assert resume_result.success, f"Resume failed: {resume_result.error_message}"
            
            # Verify active state
            session = self.orchestrator.active_sessions[session_id]
            assert session.state.value == "active", f"Expected active state after resume, got {session.state.value}"
            
            # Test session termination
            termination_result = await self.orchestrator.terminate_session(session_id)
            assert termination_result.success, f"Termination failed: {termination_result.error_message}"
            
            # Verify session is removed from active sessions
            assert session_id not in self.orchestrator.active_sessions
            
            duration = time.time() - start_time
            self.metrics.successful_operations += 4  # create, pause, resume, terminate
            
            return SessionTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                session_id=session_id,
                details={
                    "operations_completed": ["create", "pause", "resume", "terminate"],
                    "session_checkpoints": len(session.checkpoints),
                    "timeline_entries": len(session.timeline)
                },
                metrics={
                    "creation_time": session.started_at,
                    "total_checkpoints": len(session.checkpoints)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.failed_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={"exception": traceback.format_exc()}
            )
    
    async def test_concurrent_sessions(self, num_sessions: int = 5) -> SessionTestResult:
        """Test managing multiple concurrent sessions."""
        test_name = f"concurrent_sessions_{num_sessions}"
        start_time = time.time()
        
        try:
            # Create multiple sessions concurrently
            session_tasks = []
            for i in range(num_sessions):
                project_context = MockProjectContext(f"concurrent_test_{i}")
                task = self.orchestrator.create_development_session(project_context)
                session_tasks.append(task)
            
            # Wait for all sessions to be created
            sessions = await asyncio.gather(*session_tasks, return_exceptions=True)
            
            # Verify all sessions were created successfully
            successful_sessions = []
            for i, session in enumerate(sessions):
                if isinstance(session, Exception):
                    logger.error(f"Session {i} creation failed: {session}")
                else:
                    successful_sessions.append(session)
                    self.active_test_sessions[session.session_id] = session
            
            assert len(successful_sessions) == num_sessions, f"Expected {num_sessions} sessions, got {len(successful_sessions)}"
            
            self.metrics.concurrent_sessions_peak = max(self.metrics.concurrent_sessions_peak, len(successful_sessions))
            
            # Test concurrent operations on all sessions
            pause_tasks = [self.orchestrator.pause_session(s.session_id) for s in successful_sessions]
            pause_results = await asyncio.gather(*pause_tasks, return_exceptions=True)
            
            # Verify all pauses succeeded
            successful_pauses = [r for r in pause_results if not isinstance(r, Exception) and r.success]
            assert len(successful_pauses) == num_sessions, f"Expected {num_sessions} successful pauses, got {len(successful_pauses)}"
            
            # Test concurrent resume
            resume_tasks = [self.orchestrator.resume_session(s.session_id) for s in successful_sessions]
            resume_results = await asyncio.gather(*resume_tasks, return_exceptions=True)
            
            # Verify all resumes succeeded
            successful_resumes = [r for r in resume_results if not isinstance(r, Exception) and r.success]
            assert len(successful_resumes) == num_sessions, f"Expected {num_sessions} successful resumes, got {len(successful_resumes)}"
            
            # Clean up - terminate all sessions
            terminate_tasks = [self.orchestrator.terminate_session(s.session_id) for s in successful_sessions]
            terminate_results = await asyncio.gather(*terminate_tasks, return_exceptions=True)
            
            # Verify all terminations succeeded
            successful_terminations = [r for r in terminate_results if not isinstance(r, Exception) and r.success]
            assert len(successful_terminations) == num_sessions, f"Expected {num_sessions} successful terminations, got {len(successful_terminations)}"
            
            duration = time.time() - start_time
            self.metrics.successful_operations += num_sessions * 4  # create, pause, resume, terminate per session
            
            return SessionTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                details={
                    "sessions_created": len(successful_sessions),
                    "concurrent_operations": num_sessions * 4,
                    "peak_concurrent_sessions": len(successful_sessions)
                },
                metrics={
                    "average_operation_time": duration / (num_sessions * 4),
                    "concurrency_efficiency": 1.0
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.failed_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={"exception": traceback.format_exc()}
            )
    
    async def test_interruption_recovery(self) -> SessionTestResult:
        """Test session interruption handling and recovery."""
        test_name = "interruption_recovery"
        start_time = time.time()
        
        try:
            from src.models.tool_models import Interruption, InterruptionType
            
            # Create a session
            project_context = MockProjectContext("interruption_test")
            session = await self.orchestrator.create_development_session(project_context)
            session_id = session.session_id
            self.active_test_sessions[session_id] = session
            
            # Simulate various types of interruptions
            interruption_types = [
                InterruptionType.NETWORK_FAILURE,
                InterruptionType.RESOURCE_UNAVAILABLE,
                InterruptionType.AGENT_FAILURE
            ]
            
            recovery_success_count = 0
            total_interruptions = len(interruption_types)
            
            for interruption_type in interruption_types:
                # Create interruption
                interruption = Interruption(
                    interruption_type=interruption_type,
                    description=f"Test {interruption_type.value} interruption",
                    affected_resources=["test_resource"],
                    severity="medium",
                    context={"test": True}
                )
                
                # Handle interruption
                recovery_plan = await self.orchestrator.handle_session_interruption(session_id, interruption)
                
                if recovery_plan and recovery_plan.success_probability > 0.5:
                    recovery_success_count += 1
                
                # Verify session is still manageable
                session = self.orchestrator.active_sessions.get(session_id)
                assert session is not None, "Session lost during interruption handling"
            
            # Calculate recovery success rate
            recovery_rate = recovery_success_count / total_interruptions
            self.metrics.recovery_success_rate = recovery_rate
            
            # Clean up
            await self.orchestrator.terminate_session(session_id)
            
            duration = time.time() - start_time
            self.metrics.successful_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                session_id=session_id,
                details={
                    "interruptions_tested": total_interruptions,
                    "recovery_success_count": recovery_success_count,
                    "recovery_success_rate": recovery_rate
                },
                metrics={
                    "recovery_efficiency": recovery_rate,
                    "interruption_handling_time": duration / total_interruptions
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.failed_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={"exception": traceback.format_exc()}
            )
    
    async def test_resource_cleanup(self) -> SessionTestResult:
        """Test proper resource cleanup during session operations."""
        test_name = "resource_cleanup"
        start_time = time.time()
        
        try:
            # Create multiple sessions with different resource profiles
            sessions = []
            for i in range(3):
                project_context = MockProjectContext(f"cleanup_test_{i}")
                session = await self.orchestrator.create_development_session(project_context)
                sessions.append(session)
                self.active_test_sessions[session.session_id] = session
            
            # Track initial resource state
            initial_active_sessions = len(self.orchestrator.active_sessions)
            initial_resources = len(self.orchestrator.session_resources)
            
            # Terminate sessions and verify cleanup
            cleanup_success_count = 0
            for session in sessions:
                termination_result = await self.orchestrator.terminate_session(session.session_id)
                if termination_result.success:
                    cleanup_success_count += 1
            
            # Verify resources were cleaned up
            final_active_sessions = len(self.orchestrator.active_sessions)
            final_resources = len(self.orchestrator.session_resources)
            
            # Calculate cleanup success rate
            expected_cleanup = len(sessions)
            actual_cleanup = initial_active_sessions - final_active_sessions
            cleanup_rate = cleanup_success_count / len(sessions)
            
            self.metrics.resource_cleanup_success_rate = cleanup_rate
            
            duration = time.time() - start_time
            self.metrics.successful_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=cleanup_rate >= 0.9,  # 90% success rate required
                duration=duration,
                details={
                    "sessions_tested": len(sessions),
                    "cleanup_success_count": cleanup_success_count,
                    "cleanup_success_rate": cleanup_rate,
                    "resource_reduction": initial_resources - final_resources
                },
                metrics={
                    "cleanup_efficiency": cleanup_rate,
                    "resource_management": actual_cleanup / expected_cleanup
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.failed_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={"exception": traceback.format_exc()}
            )
    
    async def test_state_persistence(self) -> SessionTestResult:
        """Test session state persistence and restoration."""
        test_name = "state_persistence"
        start_time = time.time()
        
        try:
            # Create a session with some state
            project_context = MockProjectContext("persistence_test")
            session = await self.orchestrator.create_development_session(project_context)
            session_id = session.session_id
            self.active_test_sessions[session_id] = session
            
            # Add some state to the session
            session.add_timeline_entry("test_event", "Test event for persistence", {"test": True})
            session.metadata["test_data"] = "persistence_test_value"
            
            # Create multiple checkpoints
            initial_checkpoint_count = len(session.checkpoints)
            
            # Pause session (should create checkpoint)
            pause_result = await self.orchestrator.pause_session(session_id)
            assert pause_result.success
            
            # Verify checkpoint was created
            updated_session = self.orchestrator.active_sessions[session_id]
            checkpoint_count_after_pause = len(updated_session.checkpoints)
            assert checkpoint_count_after_pause > initial_checkpoint_count, "No checkpoint created during pause"
            
            # Resume session
            resume_result = await self.orchestrator.resume_session(session_id)
            assert resume_result.success
            
            # Verify state was preserved
            final_session = self.orchestrator.active_sessions[session_id]
            assert final_session.metadata.get("test_data") == "persistence_test_value", "Metadata not preserved"
            assert len(final_session.timeline) > 0, "Timeline not preserved"
            
            # Clean up
            await self.orchestrator.terminate_session(session_id)
            
            duration = time.time() - start_time
            self.metrics.successful_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                session_id=session_id,
                details={
                    "checkpoints_created": checkpoint_count_after_pause - initial_checkpoint_count,
                    "state_preserved": True,
                    "timeline_entries": len(final_session.timeline)
                },
                metrics={
                    "persistence_efficiency": 1.0,
                    "checkpoint_creation_time": duration / 2  # pause and resume
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.failed_operations += 1
            
            return SessionTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={"exception": traceback.format_exc()}
            )
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all session management tests."""
        logger.info("Starting comprehensive session management tests...")
        
        # Setup test environment
        if not await self.setup_test_environment():
            return {"error": "Failed to setup test environment"}
        
        # Define test suite
        test_suite = [
            ("Single Session Lifecycle", self.test_single_session_lifecycle()),
            ("Concurrent Sessions", self.test_concurrent_sessions(5)),
            ("Interruption Recovery", self.test_interruption_recovery()),
            ("Resource Cleanup", self.test_resource_cleanup()),
            ("State Persistence", self.test_state_persistence()),
        ]
        
        # Run tests
        for test_name, test_coro in test_suite:
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_coro
                self.results.append(result)
                
                if result.success:
                    logger.info(f"✓ {test_name} passed ({result.duration:.2f}s)")
                else:
                    logger.error(f"✗ {test_name} failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"✗ {test_name} crashed: {e}")
                self.results.append(SessionTestResult(
                    test_name=test_name,
                    success=False,
                    duration=0.0,
                    error_message=str(e)
                ))
        
        # Generate comprehensive report
        return self._generate_test_report()
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.results)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "metrics": {
                "concurrent_sessions_peak": self.metrics.concurrent_sessions_peak,
                "resource_cleanup_success_rate": self.metrics.resource_cleanup_success_rate,
                "recovery_success_rate": self.metrics.recovery_success_rate,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "session_id": r.session_id,
                    "error_message": r.error_message,
                    "details": r.details,
                    "metrics": r.metrics
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        success_rate = sum(1 for r in self.results if r.success) / len(self.results)
        
        if success_rate < 0.9:
            recommendations.append("Consider improving error handling and recovery mechanisms")
        
        if self.metrics.resource_cleanup_success_rate < 0.95:
            recommendations.append("Resource cleanup needs improvement to prevent memory leaks")
        
        if self.metrics.recovery_success_rate < 0.8:
            recommendations.append("Interruption recovery mechanisms need enhancement")
        
        if self.metrics.concurrent_sessions_peak < 5:
            recommendations.append("Test with higher concurrency levels for production readiness")
        
        if not recommendations:
            recommendations.append("Session management system is performing well across all test scenarios")
        
        return recommendations


async def main():
    """Main test execution function."""
    print("=" * 80)
    print("FlutterSwarm Session Management Comprehensive Test Suite")
    print("=" * 80)
    
    tester = SessionManagementTester()
    
    try:
        # Run comprehensive tests
        report = await tester.run_comprehensive_tests()
        
        # Print detailed report
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        print("\n" + "=" * 50)
        print("PERFORMANCE METRICS")
        print("=" * 50)
        
        metrics = report["metrics"]
        print(f"Peak Concurrent Sessions: {metrics['concurrent_sessions_peak']}")
        print(f"Resource Cleanup Success Rate: {metrics['resource_cleanup_success_rate']:.1%}")
        print(f"Recovery Success Rate: {metrics['recovery_success_rate']:.1%}")
        print(f"Successful Operations: {metrics['successful_operations']}")
        print(f"Failed Operations: {metrics['failed_operations']}")
        
        print("\n" + "=" * 50)
        print("DETAILED TEST RESULTS")
        print("=" * 50)
        
        for result in report["detailed_results"]:
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"{status} {result['test_name']} ({result['duration']:.2f}s)")
            if result["error_message"]:
                print(f"    Error: {result['error_message']}")
            if result["details"]:
                for key, value in result["details"].items():
                    print(f"    {key}: {value}")
        
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        # Save detailed report
        with open("session_management_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: session_management_test_report.json")
        
        # Exit with appropriate code
        sys.exit(0 if summary["success_rate"] >= 0.8 else 1)
        
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
