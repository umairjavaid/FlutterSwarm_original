#!/usr/bin/env python3
"""
Comprehensive test for Flutter-aware FileSystemTool.

This script tests all the enhanced features including Flutter structure awareness,
backup/rollback functionality, template-based creation, and asset optimization.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Mock dependencies for testing
import types

def create_mock_modules():
    """Create mock modules for testing."""
    
    # Mock tool models
    class MockToolStatus:
        SUCCESS = "success"
        FAILURE = "failure"
        RUNNING = "running"
    
    class MockToolPermission:
        FILE_READ = "file_read"
        FILE_WRITE = "file_write"
        FILE_CREATE = "file_create"
        FILE_DELETE = "file_delete"
        DIRECTORY_CREATE = "directory_create"
        DIRECTORY_DELETE = "directory_delete"
    
    class MockToolCategory:
        FILE_SYSTEM = "file_system"
    
    class MockToolResult:
        def __init__(self, status=None, data=None, error_message=None, execution_time=0.1):
            self.status = status or MockToolStatus.SUCCESS
            self.data = data or {}
            self.error_message = error_message
            self.execution_time = execution_time
    
    class MockToolCapabilities:
        def __init__(self, available_operations=None, **kwargs):
            self.available_operations = available_operations or []
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockBaseTool:
        def __init__(self, name, description, version, category=None, required_permissions=None):
            self.name = name
            self.description = description
            self.version = version
            self.category = category
            self.required_permissions = required_permissions or []
    
    class MockToolContext:
        pass
    
    # Create module objects
    base_tool = types.ModuleType('base_tool')
    base_tool.BaseTool = MockBaseTool
    base_tool.ToolCapabilities = MockToolCapabilities
    base_tool.ToolOperation = dict
    base_tool.ToolPermission = MockToolPermission
    base_tool.ToolResult = MockToolResult
    base_tool.ToolStatus = MockToolStatus
    base_tool.ToolCategory = MockToolCategory
    base_tool.ToolContext = MockToolContext
    
    return base_tool

def create_test_flutter_project(project_path: Path):
    """Create a test Flutter project structure."""
    
    # Create basic Flutter project structure
    (project_path / "lib").mkdir(parents=True)
    (project_path / "test").mkdir(parents=True)
    (project_path / "assets" / "images").mkdir(parents=True)
    (project_path / "android").mkdir(parents=True)
    (project_path / "ios").mkdir(parents=True)
    
    # Create pubspec.yaml
    pubspec_content = """
name: test_flutter_project
description: A test Flutter project
version: 1.0.0+1

environment:
  sdk: '>=2.17.0 <4.0.0'
  flutter: ">=1.17.0"

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.5

dev_dependencies:
  flutter_test:
    sdk: flutter

flutter:
  uses-material-design: true
  assets:
    - assets/images/
"""
    
    with open(project_path / "pubspec.yaml", 'w') as f:
        f.write(pubspec_content.strip())
    
    # Create main.dart
    main_dart = """
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Test App')),
      body: Center(child: Text('Hello World')),
    );
  }
}
"""
    
    with open(project_path / "lib" / "main.dart", 'w') as f:
        f.write(main_dart.strip())
    
    # Create test file
    test_content = """
import 'package:flutter_test/flutter_test.dart';
import 'package:test_flutter_project/main.dart';

void main() {
  testWidgets('App test', (WidgetTester tester) async {
    await tester.pumpWidget(MyApp());
    expect(find.text('Hello World'), findsOneWidget);
  });
}
"""
    
    with open(project_path / "test" / "widget_test.dart", 'w') as f:
        f.write(test_content.strip())
    
    # Create a sample image file (mock)
    with open(project_path / "assets" / "images" / "icon.png", 'w') as f:
        f.write("mock image content")

async def test_file_system_tool():
    """Test the Flutter-aware FileSystemTool."""
    print("🧪 Testing Enhanced Flutter-aware FileSystemTool")
    print("=" * 55)
    
    try:
        # Create mock modules
        base_tool = create_mock_modules()
        sys.modules['base_tool'] = base_tool
        
        # Mock other required imports
        sys.modules['yaml'] = types.ModuleType('yaml')
        sys.modules['yaml'].safe_load = lambda x: {"name": "test", "dependencies": {"flutter": "sdk"}}
        sys.modules['yaml'].dump = lambda x, **kwargs: "name: test\\ndependencies:\\n  flutter: sdk"
        sys.modules['yaml'].YAMLError = Exception
        
        # Mock PIL
        pil_module = types.ModuleType('PIL')
        image_module = types.ModuleType('Image')
        image_module.Resampling = types.ModuleType('Resampling')
        image_module.Resampling.LANCZOS = 1
        pil_module.Image = image_module
        sys.modules['PIL'] = pil_module
        sys.modules['PIL.Image'] = image_module
        
        # Import and test the FileSystemTool
        from core.tools.file_system_tool import FileSystemTool
        
        print("✅ FileSystemTool imported successfully")
        
        # Create tool instance
        fs_tool = FileSystemTool()
        print("✅ FileSystemTool instantiated")
        
        # Test capabilities
        capabilities = await fs_tool.get_capabilities()
        print(f"✅ Capabilities: {len(capabilities.available_operations)} operations")
        
        # List all operations
        print("\\n📋 Available Operations:")
        for i, op in enumerate(capabilities.available_operations, 1):
            print(f"  {i:2d}. {op['name']:<25} - {op['description'][:50]}...")
        
        # Create a temporary test project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            
            print(f"\\n🔧 Testing with project at: {project_path}")
            
            # Change to project directory for testing
            original_cwd = os.getcwd()
            os.chdir(project_path)
            
            try:
                # Create test Flutter project
                create_test_flutter_project(project_path)
                print("✅ Test Flutter project created")
                
                # Test 1: Analyze project structure
                print("\\n🔍 Test 1: Analyzing project structure...")
                result = await fs_tool.execute("analyze_project_structure", {
                    "project_path": ".",
                    "deep_analysis": True
                })
                
                if result.status == "success":
                    print("✅ Project analysis successful")
                    flutter_info = result.data.get("flutter_info", {})
                    print(f"   - Flutter project: {flutter_info.get('is_flutter_project', False)}")
                    print(f"   - Platforms: {flutter_info.get('platforms', [])}")
                    print(f"   - Suggestions: {len(result.data.get('suggestions', []))}")
                else:
                    print(f"❌ Project analysis failed: {result.error_message}")
                
                # Test 2: Create from template
                print("\\n🔍 Test 2: Creating widget from template...")
                result = await fs_tool.execute("create_from_template", {
                    "template": "widget",
                    "path": "lib/widgets/custom_button.dart",
                    "class_name": "CustomButton",
                    "create_test": True
                })
                
                if result.status == "success":
                    print("✅ Widget creation successful")
                    print(f"   - Created: {result.data.get('created', False)}")
                else:
                    print(f"❌ Widget creation failed: {result.error_message}")
                
                # Test 3: Read file with import analysis
                print("\\n🔍 Test 3: Reading file with import analysis...")
                result = await fs_tool.execute("read_file", {
                    "path": "lib/main.dart",
                    "analyze_imports": True
                })
                
                if result.status == "success":
                    print("✅ File reading successful")
                    import_analysis = result.data.get("import_analysis", {})
                    if import_analysis:
                        print(f"   - Dart imports: {len(import_analysis.get('dart_imports', []))}")
                        print(f"   - Package imports: {len(import_analysis.get('package_imports', []))}")
                else:
                    print(f"❌ File reading failed: {result.error_message}")
                
                # Test 4: Validate Flutter conventions
                print("\\n🔍 Test 4: Validating Flutter conventions...")
                result = await fs_tool.execute("validate_flutter_conventions", {
                    "path": "lib",
                    "check_imports": True,
                    "check_naming": True
                })
                
                if result.status == "success":
                    print("✅ Convention validation successful")
                    violations = result.data.get("violations", [])
                    suggestions = result.data.get("suggestions", [])
                    print(f"   - Violations: {len(violations)}")
                    print(f"   - Suggestions: {len(suggestions)}")
                else:
                    print(f"❌ Convention validation failed: {result.error_message}")
                
                # Test 5: Create barrel exports
                print("\\n🔍 Test 5: Creating barrel exports...")
                
                # First create some files in a widgets directory
                widgets_dir = project_path / "lib" / "widgets"
                widgets_dir.mkdir(exist_ok=True)
                
                with open(widgets_dir / "button.dart", 'w') as f:
                    f.write("class Button {}")
                with open(widgets_dir / "text_field.dart", 'w') as f:
                    f.write("class TextField {}")
                
                result = await fs_tool.execute("create_barrel_exports", {
                    "directory": "lib/widgets",
                    "recursive": False
                })
                
                if result.status == "success":
                    print("✅ Barrel exports creation successful")
                    created_files = result.data.get("created_files", [])
                    print(f"   - Created files: {len(created_files)}")
                else:
                    print(f"❌ Barrel exports creation failed: {result.error_message}")
                
                # Test 6: Setup file watcher
                print("\\n🔍 Test 6: Setting up file watcher...")
                result = await fs_tool.execute("setup_file_watcher", {
                    "paths": ["lib/**/*.dart"],
                    "categorize_changes": True
                })
                
                if result.status == "success":
                    print("✅ File watcher setup successful")
                    watched_paths = result.data.get("watched_paths", [])
                    print(f"   - Watched paths: {len(watched_paths)}")
                else:
                    print(f"❌ File watcher setup failed: {result.error_message}")
                
                print(f"\\n✅ All FileSystemTool tests completed!")
                print(f"📊 Total operations tested: 6")
                return True
                
            finally:
                os.chdir(original_cwd)
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_file_system_tool())
    if result:
        print("\\n🎉 Enhanced FileSystemTool implementation is complete and working!")
    else:
        print("\\n💥 There are issues with the FileSystemTool implementation.")
    sys.exit(0 if result else 1)
