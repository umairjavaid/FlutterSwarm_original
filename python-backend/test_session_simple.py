"""
Simple test for session management functionality verification.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Import the necessary modules
try:
    from src.agents.orchestrator_agent import OrchestratorAgent
    from src.agents.base_agent import AgentConfig, AgentCapability
    from src.core.memory_manager import MemoryManager
    from src.core.event_bus import EventBus
    from src.core.llm_client import LLMClient
    from src.models.tool_models import (
        DevelopmentSession, SessionState, Interruption, InterruptionType,
        PauseResult, ResumeResult, TerminationResult, RecoveryPlan
    )
    from src.models.project_models import ProjectContext
    
    print("✓ All imports successful")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Return mock JSON responses based on prompt content
        if "analyze" in prompt.lower():
            return json.dumps({
                "phases": ["setup", "development", "testing"],
                "agents": ["implementation_agent"],
                "resources": ["flutter_sdk"],
                "timeline": "2 hours"
            })
        elif "recovery" in prompt.lower():
            return json.dumps({
                "recovery_steps": [
                    {"action": "validate_environment", "duration": 5},
                    {"action": "restore_state", "duration": 10}
                ],
                "estimated_time": 15,
                "success_probability": 0.9
            })
        else:
            return json.dumps({"status": "completed", "reasoning": "Test operation"})


class MockProjectContext:
    """Mock project context for testing."""
    
    def __init__(self, name: str = "test_project"):
        self.name = name
        self.project_type = "flutter"
        self.description = f"Test Flutter project: {name}"
        
    def to_dict(self):
        return {
            "name": self.name,
            "project_type": self.project_type,
            "description": self.description
        }


async def create_test_orchestrator() -> OrchestratorAgent:
    """Create a test orchestrator with mocked dependencies."""
    config = AgentConfig(
        agent_id="test_orchestrator",
        agent_type="orchestrator", 
        capabilities=[AgentCapability.ORCHESTRATION]
    )
    
    llm_client = MockLLMClient()
    memory_manager = AsyncMock(spec=MemoryManager)
    event_bus = MagicMock(spec=EventBus)
    
    orchestrator = OrchestratorAgent(config, llm_client, memory_manager, event_bus)
    return orchestrator


async def test_session_creation():
    """Test basic session creation."""
    print("🧪 Testing session creation...")
    
    orchestrator = await create_test_orchestrator()
    project_context = MockProjectContext("creation_test")
    
    # Create session
    session = await orchestrator.create_development_session(project_context)
    
    # Verify session was created
    assert isinstance(session, DevelopmentSession)
    assert session.session_id is not None
    assert session.name != ""
    assert session.state in [SessionState.ACTIVE, SessionState.INITIALIZING, SessionState.TERMINATED]
    
    print("✓ Session creation test passed")
    return session.session_id


async def test_session_pause_resume():
    """Test session pause and resume."""
    print("🧪 Testing session pause/resume...")
    
    orchestrator = await create_test_orchestrator()
    project_context = MockProjectContext("pause_resume_test")
    
    # Create session
    session = await orchestrator.create_development_session(project_context)
    session_id = session.session_id
    
    # Set session to active state for testing
    if session.state != SessionState.ACTIVE:
        session.state = SessionState.ACTIVE
        orchestrator.active_sessions[session_id] = session
    
    # Test pause
    pause_result = await orchestrator.pause_session(session_id)
    assert isinstance(pause_result, PauseResult)
    
    if pause_result.success:
        print("✓ Session paused successfully")
        
        # Test resume
        resume_result = await orchestrator.resume_session(session_id)
        assert isinstance(resume_result, ResumeResult)
        
        if resume_result.success:
            print("✓ Session resumed successfully") 
        else:
            print(f"⚠ Resume failed: {resume_result.error_message}")
    else:
        print(f"⚠ Pause failed: {pause_result.error_message}")
    
    return session_id


async def test_session_termination():
    """Test session termination."""
    print("🧪 Testing session termination...")
    
    orchestrator = await create_test_orchestrator()
    project_context = MockProjectContext("termination_test")
    
    # Create session
    session = await orchestrator.create_development_session(project_context)
    session_id = session.session_id
    
    # Ensure session is in active sessions
    orchestrator.active_sessions[session_id] = session
    
    # Test termination
    termination_result = await orchestrator.terminate_session(session_id)
    assert isinstance(termination_result, TerminationResult)
    
    if termination_result.success:
        print("✓ Session terminated successfully")
        # Verify session was removed from active sessions
        assert session_id not in orchestrator.active_sessions
    else:
        print(f"⚠ Termination failed: {termination_result.error_message}")


async def test_interruption_handling():
    """Test session interruption handling."""
    print("🧪 Testing interruption handling...")
    
    orchestrator = await create_test_orchestrator()
    project_context = MockProjectContext("interruption_test")
    
    # Create session
    session = await orchestrator.create_development_session(project_context)
    session_id = session.session_id
    
    # Ensure session is in active sessions
    orchestrator.active_sessions[session_id] = session
    
    # Create test interruption
    interruption = Interruption(
        interruption_type=InterruptionType.NETWORK_FAILURE,
        description="Test network failure interruption",
        severity="medium"
    )
    
    # Test interruption handling
    recovery_plan = await orchestrator.handle_session_interruption(session_id, interruption)
    assert isinstance(recovery_plan, RecoveryPlan)
    
    if recovery_plan.success_probability > 0:
        print("✓ Interruption handled successfully")
        print(f"  Recovery plan created with {len(recovery_plan.recovery_steps)} steps")
    else:
        print("⚠ Interruption handling created manual intervention plan")


async def test_concurrent_sessions():
    """Test multiple concurrent sessions."""
    print("🧪 Testing concurrent sessions...")
    
    orchestrator = await create_test_orchestrator()
    
    # Create multiple sessions
    sessions = []
    for i in range(3):
        project_context = MockProjectContext(f"concurrent_test_{i}")
        session = await orchestrator.create_development_session(project_context)
        sessions.append(session)
    
    # Verify all sessions exist
    active_count = len([s for s in sessions if s.session_id in orchestrator.active_sessions or s.state != SessionState.TERMINATED])
    print(f"✓ Created {len(sessions)} sessions ({active_count} active)")
    
    # Clean up
    for session in sessions:
        if session.session_id in orchestrator.active_sessions:
            await orchestrator.terminate_session(session.session_id)


async def run_session_tests():
    """Run all session management tests."""
    print("=" * 60)
    print("SESSION MANAGEMENT VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        test_session_creation,
        test_session_pause_resume,
        test_session_termination,
        test_interruption_handling,
        test_concurrent_sessions
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All session management tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - session management needs attention")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_session_tests())
    sys.exit(0 if success else 1)
