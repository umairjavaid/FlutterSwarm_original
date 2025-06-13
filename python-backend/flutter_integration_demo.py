#!/usr/bin/env python3
"""
Integration example for Flutter SDK Tool.

This demonstrates how to use each of the four core operations
and shows the expected workflow for Flutter development.
"""

import asyncio
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.tools.flutter_sdk_tool import FlutterSDKTool
from src.models.tool_models import ToolStatus


async def demonstrate_flutter_workflow():
    """Demonstrate a complete Flutter development workflow."""
    print("🎬 Flutter SDK Tool Integration Example")
    print("=" * 50)
    
    flutter_tool = FlutterSDKTool()
    
    print("This example shows how the Flutter SDK tool would be used")
    print("in a real development workflow when Flutter SDK is available.\n")
    
    # Step 1: Create a new Flutter project
    print("1️⃣ Creating a new Flutter project...")
    create_params = {
        "project_name": "awesome_weather_app",
        "output_directory": "./projects",
        "template": "app",
        "org": "com.mycompany",
        "description": "An awesome weather application",
        "platforms": ["android", "ios", "web"],
        "pub": True
    }
    
    print("   Parameters:")
    for key, value in create_params.items():
        print(f"     {key}: {value}")
    
    result = await flutter_tool.execute("create_project", create_params)
    print(f"   Status: {result.status}")
    print(f"   Message: {result.error_message or 'Would create project successfully'}")
    
    if result.status == ToolStatus.SUCCESS:
        print("   ✅ Project created with:")
        print("     • Multi-platform support (Android, iOS, Web)")
        print("     • Custom organization identifier")
        print("     • Comprehensive project structure")
    
    print()
    
    # Step 2: Add additional platform support
    print("2️⃣ Adding desktop platform support...")
    platform_params = {
        "project_path": "./projects/awesome_weather_app",
        "platforms": ["windows", "macos", "linux"]
    }
    
    print("   Parameters:")
    for key, value in platform_params.items():
        print(f"     {key}: {value}")
    
    result = await flutter_tool.execute("add_platform", platform_params)
    print(f"   Status: {result.status}")
    print(f"   Message: {result.error_message or 'Would add platforms successfully'}")
    
    if result.status == ToolStatus.SUCCESS:
        print("   ✅ Desktop platforms added:")
        print("     • Windows desktop support")
        print("     • macOS desktop support") 
        print("     • Linux desktop support")
    
    print()
    
    # Step 3: Build the application
    print("3️⃣ Building release version for Android...")
    build_params = {
        "project_path": "./projects/awesome_weather_app",
        "platform": "android",
        "build_mode": "release",
        "build_name": "1.0.0",
        "build_number": "1",
        "obfuscate": True,
        "dart_defines": {
            "API_KEY": "your_weather_api_key",
            "ENVIRONMENT": "production"
        }
    }
    
    print("   Parameters:")
    for key, value in build_params.items():
        print(f"     {key}: {value}")
    
    result = await flutter_tool.execute("build_app", build_params)
    print(f"   Status: {result.status}")
    print(f"   Message: {result.error_message or 'Would build successfully'}")
    
    if result.status == ToolStatus.SUCCESS:
        print("   ✅ Android APK built with:")
        print("     • Release optimization")
        print("     • Code obfuscation")
        print("     • Custom build version")
        print("     • Environment-specific configuration")
    
    print()
    
    # Step 4: Run the application
    print("4️⃣ Running application on emulator...")
    run_params = {
        "project_path": "./projects/awesome_weather_app",
        "device_id": "emulator-5554",
        "platform": "android",
        "build_mode": "debug",
        "dart_defines": {
            "API_KEY": "dev_weather_api_key",
            "ENVIRONMENT": "development"
        },
        "enable_software_rendering": False
    }
    
    print("   Parameters:")
    for key, value in run_params.items():
        print(f"     {key}: {value}")
    
    result = await flutter_tool.execute("run_app", run_params)
    print(f"   Status: {result.status}")
    print(f"   Message: {result.error_message or 'Would start app successfully'}")
    
    if result.status == ToolStatus.SUCCESS:
        print("   ✅ Application running with:")
        print("     • Hot reload enabled")
        print("     • Debug mode for development")
        print("     • Development API configuration")
        print("     • Real-time code updates")
    
    print()
    
    # Show capabilities summary
    print("🔧 Tool Capabilities Summary:")
    capabilities = await flutter_tool.get_capabilities()
    
    for op in capabilities.available_operations:
        print(f"\n   📋 {op['name']}:")
        print(f"      Description: {op['description']}")
        print(f"      Duration: ~{op['estimated_duration']}s")
        print(f"      Cancellable: {op['supports_cancellation']}")
        print(f"      Error Codes: {', '.join(op['error_codes'])}")
    
    print("\n" + "=" * 50)
    print("✨ Flutter SDK Tool Integration Complete!")
    print("\nThe tool provides:")
    print("  🏗️  Complete project lifecycle management")
    print("  🔧  Multi-platform development support")
    print("  ⚡  Hot reload and development server integration")
    print("  🛡️  Robust error handling and validation")
    print("  📊  Structured output for agent integration")
    print("  🎯  Production-ready build configurations")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(demonstrate_flutter_workflow())
    print(f"\nDemo completed: {'Successfully' if success else 'With errors'}")
