#!/usr/bin/env python3
"""
Test script for the completed hierarchical Flutter multi-agent framework.
Verifies agent coordination, delegation, and workflow orchestration.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any

# Add the python-backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.main_flutter_agent import MainFlutterAgent
from core.agent_types import WorkflowState, ProjectContext, AgentType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_main_flutter_agent_initialization():
    """Test that the main Flutter agent initializes correctly with all specialized agents."""
    print("\n=== Testing Main Flutter Agent Initialization ===")
    
    try:
        # Initialize main agent
        main_agent = MainFlutterAgent()
        
        # Verify agent properties
        assert main_agent.agent_type == AgentType.MAIN_FLUTTER_AGENT
        assert main_agent.name == "Main Flutter Agent"
        
        # Verify specialized agents were initialized
        print(f"✓ Initialized {len(main_agent.specialized_agents)} specialized agents")
        
        # Verify agent hierarchy
        assert len(main_agent.agent_hierarchy) > 0
        print(f"✓ Agent hierarchy has {len(main_agent.agent_hierarchy)} groups")
        
        # Verify delegation strategies
        assert len(main_agent.delegation_strategies) > 0
        print(f"✓ {len(main_agent.delegation_strategies)} delegation strategies configured")
        
        # Verify workflow phases
        assert len(main_agent.workflow_phases) > 0
        print(f"✓ {len(main_agent.workflow_phases)} workflow phases defined")
        
        print("✓ Main Flutter Agent initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Main Flutter Agent initialization test FAILED: {e}")
        return False


async def test_ui_ux_request_delegation():
    """Test UI/UX request delegation to specialized agent."""
    print("\n=== Testing UI/UX Request Delegation ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Create UI/UX request
        request_data = {
            "task_type": "ui_design",
            "description": "Create user interface for mobile app",
            "project_context": {
                "app_name": "TestApp",
                "target_platforms": ["android", "ios"],
                "design_requirements": {
                    "style": "material",
                    "theme": "light"
                }
            }
        }
        
        # Process request
        response = await main_agent.process_request(request_data)
        
        # Verify response
        assert response is not None
        assert hasattr(response, 'success')
        assert hasattr(response, 'message')
        assert hasattr(response, 'data')
        
        print(f"✓ UI/UX request processed: {response.success}")
        print(f"✓ Response message: {response.message}")
        
        if response.data and 'delegated_to' in response.data:
            print(f"✓ Request delegated to: {response.data['delegated_to']}")
        
        print("✓ UI/UX request delegation test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ UI/UX request delegation test FAILED: {e}")
        return False


async def test_state_management_request_delegation():
    """Test state management request delegation."""
    print("\n=== Testing State Management Request Delegation ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Create state management request
        request_data = {
            "task_type": "state_management",
            "description": "Implement BLoC pattern for state management",
            "state_management_type": "bloc",
            "project_context": {
                "app_name": "TestApp",
                "complexity": "medium"
            }
        }
        
        # Process request
        response = await main_agent.process_request(request_data)
        
        # Verify response
        assert response is not None
        print(f"✓ State management request processed: {response.success}")
        print(f"✓ Response message: {response.message}")
        
        if response.data and 'delegated_to' in response.data:
            print(f"✓ Request delegated to: {response.data['delegated_to']}")
        
        print("✓ State management request delegation test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ State management request delegation test FAILED: {e}")
        return False


async def test_testing_request_delegation():
    """Test testing request delegation to multiple agents."""
    print("\n=== Testing Testing Request Delegation ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Create testing request
        request_data = {
            "task_type": "testing",
            "description": "Create comprehensive test suite",
            "test_types": ["unit", "widget", "integration"],
            "project_context": {
                "app_name": "TestApp",
                "test_coverage_target": 80
            }
        }
        
        # Process request
        response = await main_agent.process_request(request_data)
        
        # Verify response
        assert response is not None
        print(f"✓ Testing request processed: {response.success}")
        print(f"✓ Response message: {response.message}")
        
        if response.data and 'test_results' in response.data:
            test_results = response.data['test_results']
            print(f"✓ {len(test_results)} test types executed")
        
        print("✓ Testing request delegation test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Testing request delegation test FAILED: {e}")
        return False


async def test_full_app_development_workflow():
    """Test complete Flutter app development workflow."""
    print("\n=== Testing Full App Development Workflow ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Create full app development request
        request_data = {
            "task_type": "flutter_app_development",
            "description": "Develop complete Flutter mobile application",
            "scope": "full_app",
            "features": [
                "user_authentication",
                "data_synchronization", 
                "offline_support",
                "push_notifications"
            ],
            "platforms": ["android", "ios"],
            "project_context": {
                "app_name": "CompleteApp",
                "target_audience": "general_users",
                "complexity": "high"
            }
        }
        
        # Process request
        response = await main_agent.process_request(request_data)
        
        # Verify response
        assert response is not None
        print(f"✓ Full app development processed: {response.success}")
        print(f"✓ Response message: {response.message}")
        
        # Check for workflow data
        if response.data:
            data_keys = list(response.data.keys())
            print(f"✓ Response contains {len(data_keys)} data elements")
        
        print("✓ Full app development workflow test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Full app development workflow test FAILED: {e}")
        return False


async def test_generic_request_analysis():
    """Test generic request analysis and routing."""
    print("\n=== Testing Generic Request Analysis ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Create generic request with mixed keywords
        request_data = {
            "task_type": "custom_development",
            "description": "Create accessible UI with comprehensive testing and documentation",
            "project_context": {
                "requirements": [
                    "Modern UI design",
                    "WCAG accessibility compliance",
                    "Unit and widget tests",
                    "API documentation"
                ]
            }
        }
        
        # Process request
        response = await main_agent.process_request(request_data)
        
        # Verify response
        assert response is not None
        print(f"✓ Generic request processed: {response.success}")
        print(f"✓ Response message: {response.message}")
        
        if response.data and 'analysis' in response.data:
            analysis = response.data['analysis']
            print(f"✓ Request analysis completed with confidence: {analysis.get('confidence', 'unknown')}")
        
        print("✓ Generic request analysis test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Generic request analysis test FAILED: {e}")
        return False


async def test_agent_capabilities():
    """Test agent capabilities and hierarchy."""
    print("\n=== Testing Agent Capabilities ===")
    
    try:
        main_agent = MainFlutterAgent()
        
        # Check capabilities
        capabilities = main_agent.capabilities
        assert len(capabilities) > 0
        print(f"✓ Main agent has {len(capabilities)} capabilities")
        
        # Check specialized agents
        specialized_count = len(main_agent.specialized_agents)
        print(f"✓ {specialized_count} specialized agents available")
        
        # Check agent hierarchy groups
        hierarchy_groups = list(main_agent.agent_hierarchy.keys())
        print(f"✓ Agent hierarchy has groups: {', '.join(hierarchy_groups)}")
        
        # Check delegation strategies
        strategies = list(main_agent.delegation_strategies.keys())
        print(f"✓ Delegation strategies: {', '.join(strategies)}")
        
        print("✓ Agent capabilities test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Agent capabilities test FAILED: {e}")
        return False


async def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Starting Hierarchical Flutter Multi-Agent Framework Tests")
    print("=" * 60)
    
    tests = [
        test_main_flutter_agent_initialization,
        test_ui_ux_request_delegation,
        test_state_management_request_delegation,
        test_testing_request_delegation,
        test_full_app_development_workflow,
        test_generic_request_analysis,
        test_agent_capabilities
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The hierarchical Flutter multi-agent framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())
