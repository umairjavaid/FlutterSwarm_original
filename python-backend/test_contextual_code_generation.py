#!/usr/bin/env python3
"""
Test suite for contextual code generation capabilities in ImplementationAgent.

This test verifies that the agent can intelligently analyze existing Flutter projects,
understand patterns and conventions, and generate code that seamlessly integrates
with the existing codebase.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import json  # Add missing import

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.implementation_agent import ImplementationAgent
from src.agents.base_agent import AgentConfig, AgentCapability
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.models.project_models import ProjectContext, ProjectType, PlatformTarget
from src.models.code_models import (
    ArchitectureStyle, CodeType, CodeConvention, ProjectStructure,
    CodeExample, IntegrationPlan, CodeGeneration
)
from src.models.tool_models import ToolResult, ToolStatus


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, **kwargs):
        """Mock LLM generation based on the prompt."""
        messages = kwargs.get("messages", [])
        if not messages:
            return {"content": "{}"}
        
        user_message = messages[-1].get("content", "").lower()
        
        # Mock architectural analysis
        if "identify" in user_message and "architectural" in user_message:
            return {
                "content": json.dumps({
                    "primary_style": "bloc_pattern",
                    "confidence": 0.85,
                    "evidence": ["bloc dependency found", "cubit files present"],
                    "secondary_patterns": ["provider"],
                    "recommendations": ["Consider consistent state management"]
                })
            }
        
        # Mock integration planning
        elif "integration plan" in user_message:
            return {
                "content": json.dumps({
                    "affected_files": ["lib/main.dart"],
                    "new_files": [
                        {"path": "lib/features/user_profile/user_profile_screen.dart", "purpose": "User profile UI"},
                        {"path": "lib/features/user_profile/user_profile_bloc.dart", "purpose": "Business logic"}
                    ],
                    "dependencies": ["cached_network_image"],
                    "integration_points": [
                        {"type": "navigation", "location": "lib/app_router.dart", "description": "Add route"}
                    ],
                    "modifications": [
                        {"file": "lib/main.dart", "change": "Add route registration", "breaking": False}
                    ],
                    "testing": ["Unit tests for bloc", "Widget tests for screen"],
                    "configuration": [],
                    "architectural_impact": {"layer": "presentation", "complexity": "low"},
                    "complexity": "medium",
                    "risks": {"breaking_changes": False, "performance_impact": "minimal"},
                    "implementation_order": ["bloc", "screen", "navigation"]
                })
            }
        
        # Mock code generation
        elif "generate flutter code" in user_message:
            return {
                "content": json.dumps({
                    "files": {
                        "lib/features/user_profile/user_profile_screen.dart": """
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'user_profile_bloc.dart';

class UserProfileScreen extends StatelessWidget {
  const UserProfileScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => UserProfileBloc(),
      child: Scaffold(
        appBar: AppBar(title: const Text('User Profile')),
        body: BlocBuilder<UserProfileBloc, UserProfileState>(
          builder: (context, state) {
            return const Center(
              child: Text('User Profile Content'),
            );
          },
        ),
      ),
    );
  }
}
""",
                        "lib/features/user_profile/user_profile_bloc.dart": """
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

// Events
abstract class UserProfileEvent extends Equatable {
  @override
  List<Object> get props => [];
}

class LoadUserProfile extends UserProfileEvent {}

// States
abstract class UserProfileState extends Equatable {
  @override
  List<Object> get props => [];
}

class UserProfileInitial extends UserProfileState {}
class UserProfileLoading extends UserProfileState {}
class UserProfileLoaded extends UserProfileState {}

// Bloc
class UserProfileBloc extends Bloc<UserProfileEvent, UserProfileState> {
  UserProfileBloc() : super(UserProfileInitial()) {
    on<LoadUserProfile>(_onLoadUserProfile);
  }

  void _onLoadUserProfile(LoadUserProfile event, Emitter<UserProfileState> emit) {
    emit(UserProfileLoading());
    // Implementation here
    emit(UserProfileLoaded());
  }
}
"""
                    },
                    "imports": ["package:flutter_bloc/flutter_bloc.dart", "package:equatable/equatable.dart"],
                    "documentation": "User profile feature with BLoC pattern",
                    "integration_notes": ["Register route in app router", "Add navigation from main menu"]
                })
            }
        
        # Default response
        return {"content": "{}"}


class MockFileSystemTool:
    """Mock file system tool."""
    
    async def execute(self, operation: str, params: dict):
        """Mock file system operations."""
        if operation == "list_directory":
            return ToolResult(
                tool_name="file_system",
                operation=operation,
                status=ToolStatus.SUCCESS,
                data={
                    "files": [
                        {"path": "lib/main.dart", "type": "file"},
                        {"path": "lib/app.dart", "type": "file"},
                        {"path": "lib/features/auth/auth_bloc.dart", "type": "file"},
                        {"path": "lib/features/home/home_screen.dart", "type": "file"},
                        {"path": "pubspec.yaml", "type": "file"},
                        {"path": "test/widget_test.dart", "type": "file"}
                    ]
                }
            )
        elif operation == "read_file":
            file_path = params.get("file_path", "")
            if "auth_bloc.dart" in file_path:
                return ToolResult(
                    tool_name="file_system",
                    operation=operation,
                    status=ToolStatus.SUCCESS,
                    data={
                        "content": """
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

abstract class AuthEvent extends Equatable {}
class LoginRequested extends AuthEvent {
  @override
  List<Object> get props => [];
}

abstract class AuthState extends Equatable {}
class AuthInitial extends AuthState {
  @override
  List<Object> get props => [];
}

class AuthBloc extends Bloc<AuthEvent, AuthState> {
  AuthBloc() : super(AuthInitial());
}
"""
                    }
                )
            elif "home_screen.dart" in file_path:
                return ToolResult(
                    tool_name="file_system",
                    operation=operation,
                    status=ToolStatus.SUCCESS,
                    data={
                        "content": """
import 'package:flutter/material.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Home')),
      body: const Center(child: Text('Welcome')),
    );
  }
}
"""
                    }
                )
            else:
                return ToolResult(
                    tool_name="file_system",
                    operation=operation,
                    status=ToolStatus.SUCCESS,
                    data={"content": "// Mock file content"}
                )
        elif operation == "find_files":
            return ToolResult(
                tool_name="file_system",
                operation=operation,
                status=ToolStatus.SUCCESS,
                data={
                    "files": [
                        "lib/main.dart",
                        "lib/features/auth/auth_bloc.dart",
                        "lib/features/home/home_screen.dart"
                    ]
                }
            )
        
        return ToolResult(
            tool_name="file_system",
            operation=operation,
            status=ToolStatus.SUCCESS,
            data={}
        )


async def test_contextual_code_generation():
    """Test the complete contextual code generation workflow."""
    print("🧪 Testing Contextual Code Generation")
    
    # Setup mock components
    mock_llm = MockLLMClient()
    mock_memory = MagicMock()
    mock_event_bus = MagicMock()
    
    # Create agent config
    config = AgentConfig(
        agent_id="test_implementation_agent",
        agent_type="implementation",
        capabilities=[AgentCapability.CODE_GENERATION],
        llm_model="gpt-4"
    )
    
    # Create implementation agent
    agent = ImplementationAgent(config, mock_llm, mock_memory, mock_event_bus)
    
    # Mock tool registry and tools
    mock_file_tool = MockFileSystemTool()
    agent.file_tool = mock_file_tool
    agent.tool_registry = MagicMock()
    
    # Mock the use_tool method
    async def mock_use_tool(tool_name, operation, params, description):
        return await mock_file_tool.execute(operation, params)
    
    agent.use_tool = mock_use_tool
    
    # Create test project context
    with tempfile.TemporaryDirectory() as temp_dir:
        project_context = ProjectContext(
            id="test_project",
            name="TestFlutterApp",
            path=temp_dir,
            project_type=ProjectType.APP,
            target_platforms={PlatformTarget.ANDROID, PlatformTarget.IOS},
            dependencies={
                "flutter_bloc": {"version": "^8.0.0"},
                "equatable": {"version": "^2.0.0"}
            }
        )
        
        # Test the main contextual code generation
        print("  📝 Testing contextual code generation...")
        
        feature_request = "Create a user profile screen with avatar, name, and settings"
        
        try:
            result = await agent.generate_contextual_code(feature_request, project_context)
            
            # Verify result structure
            assert isinstance(result, CodeGeneration), "Should return CodeGeneration object"
            assert result.request_description == feature_request, "Should preserve request description"
            assert len(result.generated_code) > 0, "Should generate code files"
            
            print("  ✅ Contextual code generation completed successfully")
            print(f"     Generated {len(result.generated_code)} files")
            print(f"     Total lines: {result.get_total_lines()}")
            
            # Verify generated code contains expected elements
            generated_files = list(result.generated_code.keys())
            print(f"     Generated files: {generated_files}")
            
            # Check that architectural style was detected
            arch_style = result.metadata.get("architectural_style")
            print(f"     Detected architecture: {arch_style}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Contextual code generation failed: {e}")
            return False


async def test_project_structure_analysis():
    """Test project structure analysis capabilities."""
    print("🏗️  Testing Project Structure Analysis")
    
    # Setup
    mock_llm = MockLLMClient()
    mock_memory = MagicMock()
    mock_event_bus = MagicMock()
    
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="implementation",
        capabilities=[AgentCapability.CODE_GENERATION]
    )
    
    agent = ImplementationAgent(config, mock_llm, mock_memory, mock_event_bus)
    
    # Mock tools
    mock_file_tool = MockFileSystemTool()
    agent.file_tool = mock_file_tool
    
    async def mock_use_tool(tool_name, operation, params, description):
        return await mock_file_tool.execute(operation, params)
    
    agent.use_tool = mock_use_tool
    
    try:
        # Test structure analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            structure = await agent._analyze_project_structure(temp_dir)
            
            assert isinstance(structure, ProjectStructure), "Should return ProjectStructure"
            assert structure.root_path == temp_dir, "Should set correct root path"
            
            print("  ✅ Project structure analysis completed")
            print(f"     Key directories: {list(structure.key_directories.keys())}")
            print(f"     Architecture layers: {list(structure.architecture_layers.keys())}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Project structure analysis failed: {e}")
        return False


async def test_architectural_style_detection():
    """Test architectural style detection."""
    print("🏛️  Testing Architectural Style Detection")
    
    # Setup
    mock_llm = MockLLMClient()
    mock_memory = MagicMock()
    mock_event_bus = MagicMock()
    
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="implementation",
        capabilities=[AgentCapability.CODE_GENERATION]
    )
    
    agent = ImplementationAgent(config, mock_llm, mock_memory, mock_event_bus)
    
    try:
        # Create test project context with BLoC dependencies
        with tempfile.TemporaryDirectory() as temp_dir:
            project_context = ProjectContext(
                id="test_project",
                name="TestApp", 
                path=temp_dir,
                project_type=ProjectType.APP,
                dependencies={
                    "flutter_bloc": {"version": "^8.0.0"},
                    "equatable": {"version": "^2.0.0"}
                }
            )
            
            # Test architectural style detection
            style = await agent._identify_architectural_style(project_context)
            
            assert style is not None, "Should detect architectural style"
            assert isinstance(style, ArchitectureStyle), "Should return ArchitectureStyle enum"
            
            print(f"  ✅ Detected architectural style: {style.value}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Architectural style detection failed: {e}")
        return False


async def test_similar_code_finding():
    """Test finding similar code in project."""
    print("🔍 Testing Similar Code Finding")
    
    # Setup
    mock_llm = MockLLMClient()
    mock_memory = MagicMock()
    mock_event_bus = MagicMock()
    
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="implementation",
        capabilities=[AgentCapability.CODE_GENERATION]
    )
    
    agent = ImplementationAgent(config, mock_llm, mock_memory, mock_event_bus)
    
    # Mock tools
    mock_file_tool = MockFileSystemTool()
    agent.file_tool = mock_file_tool
    
    async def mock_use_tool(tool_name, operation, params, description):
        return await mock_file_tool.execute(operation, params)
    
    agent.use_tool = mock_use_tool
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_context = ProjectContext(
                id="test_project",
                name="TestApp",
                path=temp_dir,
                project_type=ProjectType.APP
            )
            
            # Test finding similar code
            feature_request = "authentication screen"
            similar_code = await agent._find_similar_code(feature_request, project_context)
            
            assert isinstance(similar_code, list), "Should return list of CodeExample"
            
            print(f"  ✅ Found {len(similar_code)} similar code examples")
            
            for example in similar_code:
                assert isinstance(example, CodeExample), "Should be CodeExample objects"
                print(f"     - {example.file_path} (score: {example.similarity_score})")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Similar code finding failed: {e}")
        return False


async def test_code_understanding():
    """Test existing code understanding capabilities."""
    print("🧠 Testing Code Understanding")
    
    # Setup
    mock_llm = MockLLMClient()
    mock_memory = MagicMock()
    mock_event_bus = MagicMock()
    
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="implementation",
        capabilities=[AgentCapability.CODE_GENERATION]
    )
    
    agent = ImplementationAgent(config, mock_llm, mock_memory, mock_event_bus)
    
    # Mock tools
    mock_file_tool = MockFileSystemTool()
    agent.file_tool = mock_file_tool
    
    async def mock_use_tool(tool_name, operation, params, description):
        return await mock_file_tool.execute(operation, params)
    
    agent.use_tool = mock_use_tool
    
    try:
        # Test understanding existing code
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_bloc.dart"
            
            understanding = await agent.understand_existing_code(str(test_file))
            
            assert understanding is not None, "Should return CodeUnderstanding"
            print(f"  ✅ Code understanding completed for {test_file.name}")
            print(f"     Code type: {understanding.code_type}")
            print(f"     Patterns found: {len(understanding.patterns)}")
            print(f"     Dependencies: {len(understanding.dependencies)}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Code understanding failed: {e}")
        return False


async def main():
    """Run all contextual code generation tests."""
    print("🚀 Starting Contextual Code Generation Tests\n")
    
    tests = [
        ("Project Structure Analysis", test_project_structure_analysis),
        ("Architectural Style Detection", test_architectural_style_detection), 
        ("Similar Code Finding", test_similar_code_finding),
        ("Code Understanding", test_code_understanding),
        ("Contextual Code Generation", test_contextual_code_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All contextual code generation tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    asyncio.run(main())
