#!/usr/bin/env python3
"""
Simple test for adaptive workflow modification.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.tool_models import (
    WorkflowFeedback, WorkflowStepResult, AgentPerformanceMetrics,
    WorkflowSession, AdaptationResult, PerformanceAnalysis
)

def test_model_imports():
    """Test that all adaptive workflow models can be imported and instantiated."""
    print("Testing model imports...")
    
    # Test WorkflowStepResult
    step_result = WorkflowStepResult(
        step_id="test_step",
        agent_id="test_agent",
        status="completed",
        execution_time=30.5,
        quality_score=0.85
    )
    print(f"✓ WorkflowStepResult: {step_result.step_id}")
    
    # Test AgentPerformanceMetrics
    agent_metrics = AgentPerformanceMetrics(
        agent_id="test_agent",
        agent_type="implementation",
        tasks_completed=5,
        tasks_failed=1,
        average_execution_time=25.0,
        resource_efficiency=0.8,
        quality_average=0.85,
        error_rate=0.2
    )
    print(f"✓ AgentPerformanceMetrics: {agent_metrics.agent_id}")
    
    # Test WorkflowFeedback
    feedback = WorkflowFeedback(
        workflow_id="test_workflow",
        step_results=[step_result],
        agent_performance={"test_agent": agent_metrics},
        overall_completion_time=120.0,
        target_completion_time=90.0,
        quality_score=0.85,
        efficiency_score=0.75,
        user_satisfaction=0.9
    )
    print(f"✓ WorkflowFeedback: {feedback.workflow_id}")
    
    # Test WorkflowSession
    session = WorkflowSession(
        workflow_id="test_workflow",
        current_steps=[
            {"step_id": "step_1", "description": "Test step 1"},
            {"step_id": "step_2", "description": "Test step 2"}
        ],
        agent_assignments={"step_1": "agent_1", "step_2": "agent_2"}
    )
    print(f"✓ WorkflowSession: {session.workflow_id}")
    
    # Test AdaptationResult
    adaptation = AdaptationResult(
        workflow_id="test_workflow",
        changes_made=[{"type": "parallelize", "description": "Enable parallel execution"}],
        confidence_score=0.8,
        estimated_time_savings=30.0
    )
    print(f"✓ AdaptationResult: {adaptation.workflow_id}")
    
    print("All model imports successful!")

def test_orchestrator_import():
    """Test importing orchestrator agent."""
    try:
        from src.agents.orchestrator_agent import OrchestratorAgent
        print("✓ OrchestratorAgent import successful")
        return True
    except Exception as e:
        print(f"✗ OrchestratorAgent import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple Adaptive Workflow Test ===")
    
    try:
        test_model_imports()
        print()
        
        if test_orchestrator_import():
            print("✓ All basic imports successful - adaptive workflow framework is ready!")
        else:
            print("✗ Some imports failed")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
