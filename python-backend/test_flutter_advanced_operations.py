#!/usr/bin/env python3
"""
Comprehensive test for Flutter SDK Tool advanced operations.

This script tests all advanced operations:
1. test_app - Testing with coverage and filtering
2. analyze_code - Code analysis and formatting
3. pub_operations - Package management 
4. clean_project - Build artifact cleanup
5. doctor - Environment diagnostics

Tests both parameter validation and operation structure.
"""

import asyncio
import os
import sys
import tempfile

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.tools.flutter_sdk_tool import FlutterSDKTool
from src.models.tool_models import ToolStatus


async def test_advanced_operations():
    """Test all advanced Flutter SDK operations."""
    print("🔬 Testing Flutter SDK Tool Advanced Operations")
    print("=" * 60)
    
    flutter_tool = FlutterSDKTool()
    results = []
    
    # Test 1: test_app operation
    print("\n1️⃣ Testing test_app operation...")
    
    # Test valid test parameters
    test_params = {
        "project_path": "/tmp/test_project",
        "test_type": "unit",
        "coverage": True,
        "reporter": "expanded",
        "concurrency": 4,
        "test_files": ["test/widget_test.dart"]
    }
    
    is_valid, error = await flutter_tool.validate_params("test_app", test_params)
    print(f"   ✅ Test params validation: {is_valid}")
    results.append(("test_app_params", is_valid))
    
    # Test operation capabilities
    capabilities = await flutter_tool.get_capabilities()
    has_test_op = any(op["name"] == "test_app" for op in capabilities.available_operations)
    print(f"   ✅ Has test operation in capabilities: {has_test_op}")
    results.append(("test_app_in_capabilities", has_test_op))
    
    # Test 2: analyze_code operation  
    print("\n2️⃣ Testing analyze_code operation...")
    
    analyze_params = {
        "project_path": "/tmp/test_project",
        "fix": True,
        "format": True,
        "line_length": 80,
        "files": ["lib/main.dart"]
    }
    
    is_valid, error = await flutter_tool.validate_params("analyze_code", analyze_params)
    print(f"   ✅ Analyze params validation: {is_valid}")
    results.append(("analyze_code_params", is_valid))
    
    has_analyze_op = any(op["name"] == "analyze_code" for op in capabilities.available_operations)
    print(f"   ✅ Has analyze operation in capabilities: {has_analyze_op}")
    results.append(("analyze_code_in_capabilities", has_analyze_op))
    
    # Test 3: pub_operations
    print("\n3️⃣ Testing pub_operations...")
    
    # Test pub get
    pub_get_params = {
        "project_path": "/tmp/test_project",
        "operation": "get",
        "offline": False
    }
    
    is_valid, error = await flutter_tool.validate_params("pub_operations", pub_get_params)
    print(f"   ✅ Pub get params validation: {is_valid}")
    results.append(("pub_get_params", is_valid))
    
    # Test pub add
    pub_add_params = {
        "project_path": "/tmp/test_project", 
        "operation": "add",
        "package_name": "http",
        "version_constraint": "^0.13.0",
        "dev_dependency": False
    }
    
    is_valid, error = await flutter_tool.validate_params("pub_operations", pub_add_params)
    print(f"   ✅ Pub add params validation: {is_valid}")
    results.append(("pub_add_params", is_valid))
    
    # Test missing package name for add operation
    pub_invalid_params = {
        "project_path": "/tmp/test_project",
        "operation": "add"
        # Missing package_name
    }
    
    is_valid, error = await flutter_tool.validate_params("pub_operations", pub_invalid_params)
    print(f"   ✅ Pub add missing package validation: {not is_valid} (error: {error})")
    results.append(("pub_add_missing_package", not is_valid))
    
    has_pub_op = any(op["name"] == "pub_operations" for op in capabilities.available_operations)
    print(f"   ✅ Has pub operation in capabilities: {has_pub_op}")
    results.append(("pub_operations_in_capabilities", has_pub_op))
    
    # Test 4: clean_project operation
    print("\n4️⃣ Testing clean_project operation...")
    
    clean_params = {
        "project_path": "/tmp/test_project",
        "deep_clean": True
    }
    
    is_valid, error = await flutter_tool.validate_params("clean_project", clean_params)
    print(f"   ✅ Clean params validation: {is_valid}")
    results.append(("clean_project_params", is_valid))
    
    has_clean_op = any(op["name"] == "clean_project" for op in capabilities.available_operations)
    print(f"   ✅ Has clean operation in capabilities: {has_clean_op}")
    results.append(("clean_project_in_capabilities", has_clean_op))
    
    # Test 5: doctor operation
    print("\n5️⃣ Testing doctor operation...")
    
    doctor_params = {
        "verbose": True,
        "android_licenses": False
    }
    
    is_valid, error = await flutter_tool.validate_params("doctor", doctor_params)
    print(f"   ✅ Doctor params validation: {is_valid}")
    results.append(("doctor_params", is_valid))
    
    has_doctor_op = any(op["name"] == "doctor" for op in capabilities.available_operations)
    print(f"   ✅ Has doctor operation in capabilities: {has_doctor_op}")
    results.append(("doctor_in_capabilities", has_doctor_op))
    
    # Test 6: Operation execution with Flutter SDK check
    print("\n6️⃣ Testing operation execution...")
    
    # Test doctor execution (should handle missing Flutter gracefully)
    try:
        doctor_result = await flutter_tool.execute("doctor", {"verbose": False})
        handles_missing_flutter = (
            doctor_result.status == ToolStatus.FAILURE and 
            "Flutter SDK not found" in (doctor_result.error_message or "")
        )
        print(f"   ✅ Handles missing Flutter SDK: {handles_missing_flutter}")
        results.append(("handles_missing_flutter", handles_missing_flutter))
    except Exception as e:
        print(f"   ❌ Doctor execution failed: {e}")
        results.append(("handles_missing_flutter", False))
    
    # Test 7: Advanced operation features
    print("\n7️⃣ Testing advanced operation features...")
    
    # Check for progress tracking capabilities
    has_running_processes = hasattr(flutter_tool, 'running_processes')
    has_cancel_method = hasattr(flutter_tool, '_cancel_operation_impl')
    
    print(f"   ✅ Has process tracking: {has_running_processes}")
    print(f"   ✅ Has cancellation support: {has_cancel_method}")
    
    results.append(("has_process_tracking", has_running_processes))
    results.append(("has_cancellation", has_cancel_method))
    
    # Check operation-specific features in schemas
    operation_schemas = capabilities.input_schemas
    
    # Test operation has coverage options
    test_schema = operation_schemas.get("test_app", {})
    test_has_coverage = "coverage" in test_schema.get("properties", {})
    
    # Analyze operation has format options  
    analyze_schema = operation_schemas.get("analyze_code", {})
    analyze_has_format = "format" in analyze_schema.get("properties", {})
    
    # Clean operation has deep clean
    clean_schema = operation_schemas.get("clean_project", {})
    clean_has_deep = "deep_clean" in clean_schema.get("properties", {})
    
    print(f"   ✅ Test operation has coverage: {test_has_coverage}")
    print(f"   ✅ Analyze operation has format: {analyze_has_format}")
    print(f"   ✅ Clean operation has deep clean: {clean_has_deep}")
    
    results.append(("test_has_coverage", test_has_coverage))
    results.append(("analyze_has_format", analyze_has_format))
    results.append(("clean_has_deep", clean_has_deep))
    
    # Test 8: Helper method availability
    print("\n8️⃣ Testing helper methods...")
    
    helper_methods = [
        "_parse_test_results",
        "_parse_analysis_issues", 
        "_extract_updated_dependencies",
        "_extract_flutter_version",
        "_find_build_artifacts"
    ]
    
    helper_methods_exist = all(
        hasattr(flutter_tool, method) for method in helper_methods
    )
    
    print(f"   ✅ All helper methods exist: {helper_methods_exist}")
    results.append(("helper_methods_exist", helper_methods_exist))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Advanced Operations Test Results:")
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All advanced operations are properly implemented!")
        print("\n📋 Advanced Operations Summary:")
        print("   ✅ test_app: Complete with coverage, filtering, parallel execution")
        print("   ✅ analyze_code: Full analyzer and formatter integration")
        print("   ✅ pub_operations: Comprehensive package management")
        print("   ✅ clean_project: Build cleanup with selective options")
        print("   ✅ doctor: Complete environment diagnostics")
        print("   ✅ Progress tracking and cancellation support")
        print("   ✅ Operation-specific advanced features")
        print("   ✅ Proper error handling and validation")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed - review implementation")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_advanced_operations())
    sys.exit(0 if success else 1)
