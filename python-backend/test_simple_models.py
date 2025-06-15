#!/usr/bin/env python3
"""
Simple test for workflow adaptation models.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing basic model imports...")
    
    from src.models.tool_models import (
        WorkflowFeedback, WorkflowStepResult, AgentPerformanceMetrics,
        PerformanceAnalysis, WorkflowImprovement, WorkflowSession, AdaptationResult
    )
    print("✅ Tool models imported successfully")
    
    # Test creating basic instances
    feedback = WorkflowFeedback(workflow_id="test_001")
    print(f"✅ WorkflowFeedback created: {feedback.workflow_id}")
    
    step_result = WorkflowStepResult(
        step_id="step_001",
        agent_id="agent_001", 
        status="completed",
        execution_time=45.0
    )
    print(f"✅ WorkflowStepResult created: {step_result.step_id}")
    
    session = WorkflowSession(workflow_id="test_workflow")
    print(f"✅ WorkflowSession created: {session.session_id}")
    
    result = AdaptationResult(workflow_id="test_workflow")
    print(f"✅ AdaptationResult created: {result.adaptation_id}")
    
    print("\n🎉 All models created successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
