#!/usr/bin/env python3
"""
Comprehensive Test Runner for BaseAgent Tool Integration
Executes and validates all integration requirements
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path

# Add project to path  
sys.path.append(str(Path(__file__).parent.parent))

def run_comprehensive_integration_tests():
    """Run all BaseAgent tool integration tests"""
    
    print("🚀 Running Comprehensive BaseAgent Tool Integration Tests")
    print("="*80)
    
    # Test files to run
    test_files = [
        "tests/agents/test_base_agent_tools.py",
        "test_tool_discovery_verification.py", 
        "test_comprehensive_integration.py",
        "test_tool_learning_and_sharing.py",
        "test_tool_system_integration.py"
    ]
    
    # Test arguments
    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--strict-markers",  # Strict marker checking
    ]
    
    # Add test files that exist
    existing_test_files = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_test_files.append(test_file)
            print(f"✅ Found test file: {test_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not existing_test_files:
        print("❌ No test files found to execute")
        return False
    
    # Run tests
    pytest_args.extend(existing_test_files)
    
    print(f"\n🧪 Running pytest with args: {' '.join(pytest_args)}")
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ Tests failed with exit code: {result}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_tests()
    sys.exit(0 if success else 1)
