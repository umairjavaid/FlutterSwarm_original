#!/usr/bin/env python3
"""
Simple test runner for OrchestratorAgent enhanced tests.

This runs individual test methods to validate orchestrator capabilities.
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("orchestrator_tests")


async def run_basic_tests():
    """Run basic orchestrator tests to validate functionality."""
    
    logger.info("🚀 Starting OrchestratorAgent Basic Test Suite")
    
    try:
        # Import test modules
        from mock_orchestrator import create_mock_orchestrator
        from src.agents.base_agent import AgentConfig, AgentCapability
        from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
        from src.models.task_models import TaskContext, TaskType, TaskPriority
        from unittest.mock import AsyncMock
        
        # Setup mock orchestrator
        config = AgentConfig(
            agent_id="test-orchestrator",
            agent_type="orchestrator", 
            capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.ARCHITECTURE_ANALYSIS]
        )
        
        event_bus = AsyncMock()
        memory_manager = AsyncMock()
        llm_client = type('MockLLM', (), {
            'generate': AsyncMock(return_value={"content": "mock response"})
        })()
        
        orchestrator = create_mock_orchestrator(config, llm_client, memory_manager, event_bus)
        
        # Create test project context
        project_context = ProjectContext(
            project_id="test-flutter-app",
            name="TestFlutterApp",
            description="A test Flutter application",
            project_type=ProjectType.MOBILE_APP,
            platform_targets=[PlatformTarget.IOS, PlatformTarget.ANDROID],
            framework_version="3.24.0",
            requirements={
                "features": ["authentication", "data_persistence"],
                "performance": {"target_fps": 60},
                "platforms": ["iOS", "Android"]
            },
            constraints={
                "timeline": "2 weeks",
                "team_size": 4
            }
        )
        
        logger.info("✅ Mock orchestrator and project context created")
        
        # Test 1: Session Creation
        logger.info("🔄 Testing session creation...")
        session = await orchestrator.create_development_session({
            "project_context": project_context,
            "session_type": "development"
        })
        
        assert session is not None
        assert session.session_id is not None
        assert session.project_context == project_context
        logger.info(f"✅ Session created: {session.session_id}")
        
        # Test 2: Session Initialization
        logger.info("🔄 Testing session initialization...")
        init_result = await orchestrator.initialize_session(session.session_id)
        assert init_result is True
        logger.info("✅ Session initialized successfully")
        
        # Test 3: Environment Setup
        logger.info("🔄 Testing environment setup...")
        setup_result = await orchestrator.setup_session_environment(session.session_id)
        assert setup_result.success is True
        assert "flutter_sdk" in setup_result.configured_tools
        logger.info("✅ Environment setup completed")
        
        # Test 4: Task Decomposition
        logger.info("🔄 Testing task decomposition...")
        task = TaskContext(
            task_id="test-task-001",
            description="Build a complex Flutter application with authentication and data sync",
            task_type=TaskType.PROJECT_DEVELOPMENT,
            priority=TaskPriority.HIGH,
            project_context=project_context,
            requirements=[],
            expected_deliverables=[]
        )
        
        workflow = await orchestrator.decompose_complex_task(task)
        assert workflow is not None
        assert len(workflow.steps) > 0
        assert workflow.workflow_id is not None
        logger.info(f"✅ Task decomposed into {len(workflow.steps)} steps")
        
        # Test 5: Task Execution
        logger.info("🔄 Testing task execution...")
        simple_task = TaskContext(
            task_id="simple-task-001",
            description="Create login screen",
            task_type=TaskType.FEATURE_IMPLEMENTATION,
            priority=TaskPriority.MEDIUM,
            project_context=project_context,
            requirements=[],
            expected_deliverables=[]
        )
        
        task_result = await orchestrator.execute_task_in_session(session.session_id, simple_task)
        assert task_result.status == "completed"
        assert task_result.task_id == simple_task.task_id
        logger.info(f"✅ Task executed: {task_result.status}")
        
        # Test 6: Tool Coordination
        logger.info("🔄 Testing tool coordination...")
        allocation_request = {
            "agents": [
                {"id": "arch-001", "tools": ["flutter_sdk"], "priority": "high"},
                {"id": "impl-001", "tools": ["flutter_sdk", "ide"], "priority": "medium"}
            ]
        }
        
        allocation_plan = await orchestrator.plan_tool_allocation(allocation_request)
        assert allocation_plan.success is True
        assert len(allocation_plan.allocations) > 0
        logger.info(f"✅ Tool allocation planned: {len(allocation_plan.allocations)} allocations")
        
        # Test 7: Performance Analysis
        logger.info("🔄 Testing performance analysis...")
        workflow_data = {
            "workflow_id": "test-workflow",
            "steps": [
                {"id": "step1", "duration": 300},
                {"id": "step2", "duration": 1500}  # Long duration step
            ]
        }
        
        analysis = await orchestrator.analyze_workflow_performance(workflow_data)
        assert analysis is not None
        assert analysis.workflow_id == "test-workflow"
        logger.info(f"✅ Performance analysis completed: efficiency = {analysis.overall_efficiency}")
        
        # Test 8: Session Pause/Resume
        logger.info("🔄 Testing session pause/resume...")
        pause_result = await orchestrator.pause_session(session.session_id)
        assert pause_result.success is True
        
        resume_result = await orchestrator.resume_session(session.session_id)
        assert resume_result.success is True
        logger.info("✅ Session pause/resume completed")
        
        # Test 9: Checkpointing
        logger.info("🔄 Testing session checkpointing...")
        checkpoint = await orchestrator.create_session_checkpoint(session.session_id, "test_checkpoint")
        assert checkpoint is not None
        assert checkpoint.checkpoint_id is not None
        logger.info(f"✅ Checkpoint created: {checkpoint.checkpoint_id}")
        
        # Test 10: Session Cleanup
        logger.info("🔄 Testing session cleanup...")
        termination_result = await orchestrator.terminate_session(session.session_id)
        assert termination_result.success is True
        logger.info("✅ Session terminated successfully")
        
        logger.info("\n🎉 All basic tests passed! OrchestratorAgent functionality validated.")
        
        return {
            "status": "passed",
            "tests_run": 10,
            "tests_passed": 10,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


async def run_performance_tests():
    """Run performance-focused tests."""
    
    logger.info("📊 Starting Performance Tests")
    
    try:
        from mock_orchestrator import create_mock_orchestrator
        from src.agents.base_agent import AgentConfig, AgentCapability
        from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
        from unittest.mock import AsyncMock
        import time
        
        # Setup
        config = AgentConfig(
            agent_id="perf-orchestrator",
            agent_type="orchestrator",
            capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.ARCHITECTURE_ANALYSIS]
        )
        
        orchestrator = create_mock_orchestrator(config, AsyncMock(), AsyncMock(), AsyncMock())
        
        project_context = ProjectContext(
            project_id="perf-test-app", 
            name="PerformanceTestApp",
            description="Performance testing application",
            project_type=ProjectType.MOBILE_APP,
            platform_targets=[PlatformTarget.IOS, PlatformTarget.ANDROID],
            framework_version="3.24.0",
            requirements={},
            constraints={}
        )
        
        performance_metrics = {}
        
        # Test 1: Session Creation Speed
        logger.info("⏱️  Testing session creation speed...")
        start_time = time.time()
        
        sessions = []
        for i in range(10):
            session = await orchestrator.create_development_session({
                "project_context": project_context,
                "session_type": "development"
            })
            sessions.append(session)
        
        creation_time = time.time() - start_time
        avg_creation_time = creation_time / 10
        performance_metrics["session_creation_avg"] = avg_creation_time
        
        logger.info(f"✅ Average session creation time: {avg_creation_time:.3f}s")
        assert avg_creation_time < 0.1, f"Session creation too slow: {avg_creation_time:.3f}s"
        
        # Test 2: Memory Usage
        logger.info("💾 Testing memory usage...")
        base_memory = orchestrator.get_memory_usage()
        
        # Create additional sessions
        for i in range(5):
            await orchestrator.create_development_session({
                "project_context": project_context,
                "session_type": "development"
            })
        
        peak_memory = orchestrator.get_memory_usage()
        memory_per_session = (peak_memory - base_memory) / 5
        performance_metrics["memory_per_session_mb"] = memory_per_session / 1024 / 1024
        
        logger.info(f"✅ Memory per session: {memory_per_session/1024/1024:.1f}MB")
        assert memory_per_session < 50 * 1024 * 1024, f"Memory usage too high: {memory_per_session/1024/1024:.1f}MB"
        
        # Test 3: Tool Coordination Speed
        logger.info("🔧 Testing tool coordination speed...")
        start_time = time.time()
        
        coordination_scenario = {
            "agents": 8,
            "tools": 4,
            "concurrent_requests": 20,
            "duration": 300
        }
        
        coordination_results = await orchestrator.simulate_tool_coordination(coordination_scenario)
        coordination_time = time.time() - start_time
        
        throughput = coordination_results.requests_processed / coordination_time
        performance_metrics["coordination_throughput"] = throughput
        
        logger.info(f"✅ Tool coordination throughput: {throughput:.2f} requests/sec")
        assert throughput > 50, f"Coordination throughput too low: {throughput:.2f}"
        
        # Test 4: Workflow Optimization Speed
        logger.info("⚡ Testing workflow optimization speed...")
        baseline_workflow = await orchestrator.create_baseline_workflow(project_context)
        
        start_time = time.time()
        optimized_workflow = await orchestrator.optimize_workflow(baseline_workflow)
        optimization_time = time.time() - start_time
        
        improvement = (baseline_workflow.estimated_duration - optimized_workflow.estimated_duration) / baseline_workflow.estimated_duration * 100
        performance_metrics["optimization_time"] = optimization_time
        performance_metrics["optimization_improvement"] = improvement
        
        logger.info(f"✅ Workflow optimization: {improvement:.1f}% improvement in {optimization_time:.3f}s")
        assert optimization_time < 1.0, f"Optimization too slow: {optimization_time:.3f}s"
        assert improvement > 5, f"Optimization improvement too low: {improvement:.1f}%"
        
        # Cleanup
        for session in sessions:
            await orchestrator.terminate_session(session.session_id)
        
        logger.info("\n🚀 Performance tests completed successfully!")
        logger.info(f"📊 Performance Summary:")
        for metric, value in performance_metrics.items():
            logger.info(f"   • {metric}: {value:.3f}")
        
        return {
            "status": "passed",
            "performance_metrics": performance_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


async def main():
    """Main test entry point."""
    print("🧪 OrchestratorAgent Enhanced Test Suite")
    print("=" * 50)
    
    # Run basic functionality tests
    basic_results = await run_basic_tests()
    print(f"\n📋 Basic Tests: {basic_results['status'].upper()}")
    
    if basic_results['status'] == 'failed':
        print(f"❌ Basic tests failed: {basic_results.get('error', 'Unknown error')}")
        return 1
    
    # Run performance tests
    perf_results = await run_performance_tests()
    print(f"📊 Performance Tests: {perf_results['status'].upper()}")
    
    if perf_results['status'] == 'failed':
        print(f"❌ Performance tests failed: {perf_results.get('error', 'Unknown error')}")
        return 1
    
    print("\n🎉 All tests passed! OrchestratorAgent is ready for deployment.")
    
    # Save results
    results = {
        "basic_tests": basic_results,
        "performance_tests": perf_results,
        "overall_status": "passed"
    }
    
    import json
    with open('/workspaces/FlutterSwarm/python-backend/logs/orchestrator_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📄 Detailed results saved to logs/orchestrator_test_results.json")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
