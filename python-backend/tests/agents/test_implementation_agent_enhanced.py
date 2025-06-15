"""
Test suite for project-aware code generation features in ImplementationAgent.

This test verifies the new capabilities for understanding existing code patterns,
extracting project conventions, and generating context-aware Flutter code.
"""
import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

import sys
import os
sys.path.append('/workspaces/FlutterSwarm/python-backend')
sys.path.append('/workspaces/FlutterSwarm/python-backend/src')

from src.agents.implementation_agent import ImplementationAgent
from src.models.code_models import (
    CodeGeneration, CodeUnderstanding, ProjectContext as CodeProjectContext,
    GenerationEntry, CodePattern, ProjectStructure, CodeType, ArchitectureStyle,
    CodeConvention, CodeAnalysisResult
)
from src.models.agent_models import AgentMessage, TaskResult
from src.models.task_models import TaskContext, TaskType
from src.models.tool_models import ToolResult, ToolStatus
from src.core.event_bus import EventBus
from src.core.memory_manager import MemoryManager
from src.agents.base_agent import AgentConfig


class TestImplementationAgentCodeUnderstanding:
    """Test project-aware code generation capabilities."""

    @pytest.fixture
    async def implementation_agent(self):
        """Create a test implementation agent."""
        # Mock dependencies
        mock_llm_client = AsyncMock()
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_event_bus = Mock(spec=EventBus)
        
        # Create agent config
        config = AgentConfig(
            agent_id="test_implementation_agent",
            agent_type="implementation",
            capabilities=[],
            llm_model="gpt-4",
            temperature=0.2,
            max_tokens=4000
        )
        
        # Create agent
        agent = ImplementationAgent(
            config=config,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            event_bus=mock_event_bus
        )
        
        # Mock file system tool
        agent.file_tool = AsyncMock()
        
        return agent

    @pytest.fixture
    def sample_flutter_widget(self):
        """Sample Flutter widget code for testing."""
        return '''
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

/// A sample login screen widget following Flutter best practices
class LoginScreen extends StatefulWidget {
  const LoginScreen({Key? key}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Login'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: BlocListener<AuthBloc, AuthState>(
        listener: (context, state) {
          if (state is AuthSuccess) {
            Navigator.pushReplacementNamed(context, '/home');
          } else if (state is AuthFailure) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(state.error)),
            );
          }
        },
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextFormField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                    labelText: 'Email',
                    prefixIcon: Icon(Icons.email),
                  ),
                  validator: (value) {
                    if (value?.isEmpty ?? true) {
                      return 'Please enter your email';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _passwordController,
                  decoration: const InputDecoration(
                    labelText: 'Password',
                    prefixIcon: Icon(Icons.lock),
                  ),
                  obscureText: true,
                  validator: (value) {
                    if (value?.isEmpty ?? true) {
                      return 'Please enter your password';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 24),
                BlocBuilder<AuthBloc, AuthState>(
                  builder: (context, state) {
                    return ElevatedButton(
                      onPressed: state is AuthLoading ? null : _handleLogin,
                      child: state is AuthLoading
                          ? const CircularProgressIndicator()
                          : const Text('Login'),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _handleLogin() {
    if (_formKey.currentState?.validate() ?? false) {
      final email = _emailController.text;
      final password = _passwordController.text;
      
      context.read<AuthBloc>().add(
        LoginEvent(email: email, password: password),
      );
    }
  }
}
'''

    @pytest.fixture
    def sample_bloc_code(self):
        """Sample BLoC code for testing."""
        return '''
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

part 'auth_event.dart';
part 'auth_state.dart';

/// Authentication Business Logic Component
/// 
/// Manages authentication state and processes login/logout events
/// following the BLoC pattern for predictable state management.
class AuthBloc extends Bloc<AuthEvent, AuthState> {
  final AuthRepository _authRepository;
  
  AuthBloc({required AuthRepository authRepository})
      : _authRepository = authRepository,
        super(AuthInitial()) {
    on<LoginEvent>(_onLogin);
    on<LogoutEvent>(_onLogout);
    on<CheckAuthStatusEvent>(_onCheckAuthStatus);
  }

  Future<void> _onLogin(LoginEvent event, Emitter<AuthState> emit) async {
    emit(AuthLoading());
    
    try {
      final user = await _authRepository.signIn(
        email: event.email,
        password: event.password,
      );
      
      if (user != null) {
        emit(AuthSuccess(user: user));
      } else {
        emit(const AuthFailure(error: 'Invalid credentials'));
      }
    } catch (e) {
      emit(AuthFailure(error: e.toString()));
    }
  }

  Future<void> _onLogout(LogoutEvent event, Emitter<AuthState> emit) async {
    emit(AuthLoading());
    
    try {
      await _authRepository.signOut();
      emit(AuthInitial());
    } catch (e) {
      emit(AuthFailure(error: e.toString()));
    }
  }

  Future<void> _onCheckAuthStatus(
    CheckAuthStatusEvent event,
    Emitter<AuthState> emit,
  ) async {
    final user = await _authRepository.getCurrentUser();
    
    if (user != null) {
      emit(AuthSuccess(user: user));
    } else {
      emit(AuthInitial());
    }
  }
}
'''

    async def test_detect_code_type_widget(self, implementation_agent, sample_flutter_widget):
        """Test detection of Flutter widget code type."""
        code_type = implementation_agent._detect_code_type(
            "lib/screens/login_screen.dart", 
            sample_flutter_widget
        )
        assert code_type == CodeType.WIDGET

    async def test_detect_code_type_bloc(self, implementation_agent, sample_bloc_code):
        """Test detection of BLoC code type."""
        code_type = implementation_agent._detect_code_type(
            "lib/blocs/auth_bloc.dart",
            sample_bloc_code
        )
        assert code_type == CodeType.BLOC

    async def test_extract_code_structure(self, implementation_agent, sample_flutter_widget):
        """Test extraction of code structure from Flutter widget."""
        structure = implementation_agent._extract_code_structure(
            sample_flutter_widget, 
            {"structure": {"main_widget": "LoginScreen"}}
        )
        
        assert "classes" in structure
        assert any(cls["name"] == "LoginScreen" for cls in structure["classes"])
        assert any(cls["name"] == "_LoginScreenState" for cls in structure["classes"])
        assert "imports" in structure
        assert "package:flutter/material.dart" in structure["imports"]

    async def test_extract_code_patterns(self, implementation_agent, sample_flutter_widget):
        """Test extraction of code patterns from Flutter widget."""
        patterns = implementation_agent._extract_code_patterns(
            sample_flutter_widget,
            {
                "patterns": [
                    {
                        "type": "state_management",
                        "description": "Uses BLoC pattern for state management",
                        "examples": ["BlocListener", "BlocBuilder"],
                        "frequency": 2,
                        "confidence": 0.9
                    }
                ]
            }
        )
        
        # Should detect StatefulWidget pattern
        widget_patterns = [p for p in patterns if p.pattern_type == "widget_pattern"]
        assert len(widget_patterns) > 0
        
        # Should include LLM-detected patterns
        state_patterns = [p for p in patterns if p.pattern_type == "state_management"]
        assert len(state_patterns) > 0

    async def test_extract_conventions(self, implementation_agent, sample_flutter_widget):
        """Test extraction of coding conventions."""
        conventions = implementation_agent._extract_conventions(
            sample_flutter_widget,
            {
                "conventions": {
                    "naming_convention": "camelCase",
                    "import_style": "package_imports"
                }
            }
        )
        
        assert CodeConvention.NAMING_CONVENTION in conventions
        assert conventions[CodeConvention.NAMING_CONVENTION] == "camelCase"
        assert CodeConvention.IMPORT_STYLE in conventions

    async def test_extract_dependencies(self, implementation_agent, sample_flutter_widget):
        """Test extraction of dependencies from imports."""
        dependencies = implementation_agent._extract_dependencies(
            sample_flutter_widget,
            {}
        )
        
        assert "flutter" in dependencies
        assert "flutter_bloc" in dependencies

    async def test_calculate_complexity_metrics(self, implementation_agent, sample_flutter_widget):
        """Test calculation of code complexity metrics."""
        metrics = implementation_agent._calculate_complexity_metrics(sample_flutter_widget)
        
        assert "lines_of_code" in metrics
        assert "comment_ratio" in metrics
        assert "cyclomatic_complexity" in metrics
        assert "max_nesting_depth" in metrics
        
        assert metrics["lines_of_code"] > 0
        assert metrics["cyclomatic_complexity"] > 0

    async def test_understand_existing_code_success(self, implementation_agent, sample_flutter_widget):
        """Test successful code understanding with mocked LLM."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dart', delete=False) as f:
            f.write(sample_flutter_widget)
            temp_path = f.name

        try:
            # Mock file tool response
            implementation_agent.file_tool.execute = AsyncMock(return_value=ToolResult(
                operation_id="test_op",
                status=ToolStatus.SUCCESS,
                data={
                    "content": sample_flutter_widget,
                    "file_info": {"size": len(sample_flutter_widget)}
                }
            ))
            
            # Mock LLM response
            implementation_agent._llm_call = AsyncMock(return_value={
                "structure": {
                    "main_widget": "LoginScreen",
                    "architecture": "stateful_widget_with_bloc"
                },
                "patterns": [
                    {
                        "type": "bloc_pattern",
                        "description": "Uses BLoC for state management",
                        "examples": ["BlocListener", "BlocBuilder"],
                        "frequency": 2,
                        "confidence": 0.9
                    }
                ],
                "conventions": {
                    "naming_convention": "camelCase",
                    "import_style": "package_imports"
                },
                "relationships": {
                    "state_management": ["AuthBloc"]
                },
                "quality": {
                    "maintainability": "high",
                    "documentation_quality": "good"
                },
                "suggestions": [
                    "Consider extracting form validation logic",
                    "Add more comprehensive error handling"
                ]
            })

            # Mock use_tool method
            implementation_agent.use_tool = AsyncMock(return_value=ToolResult(
                operation_id="test_op",
                status=ToolStatus.SUCCESS,
                data={
                    "content": sample_flutter_widget,
                    "file_info": {"size": len(sample_flutter_widget)}
                }
            ))
            
            # Test the method
            understanding = await implementation_agent.understand_existing_code(temp_path)
            
            # Verify results
            assert isinstance(understanding, CodeUnderstanding)
            assert understanding.file_path == temp_path
            assert understanding.code_type == CodeType.WIDGET
            assert len(understanding.patterns) > 0
            assert len(understanding.conventions) > 0
            assert len(understanding.dependencies) > 0
            assert "lines_of_code" in understanding.complexity_metrics
            assert len(understanding.suggestions) > 0

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    async def test_understand_existing_code_file_not_found(self, implementation_agent):
        """Test code understanding with non-existent file."""
        understanding = await implementation_agent.understand_existing_code("/non/existent/file.dart")
        
        # Should return error understanding
        assert isinstance(understanding, CodeUnderstanding)
        assert "error" in understanding.structure
        assert len(understanding.suggestions) > 0
        assert "Could not analyze code" in understanding.suggestions[0]

    async def test_update_project_patterns(self, implementation_agent):
        """Test updating project-level pattern knowledge."""
        # Create test patterns
        pattern1 = CodePattern(
            pattern_id="test_pattern_1",
            pattern_type="widget_pattern",
            description="Test pattern 1",
            frequency=1,
            confidence=0.8
        )
        
        pattern2 = CodePattern(
            pattern_id="test_pattern_1",  # Same ID to test update
            pattern_type="widget_pattern",
            description="Test pattern 1 updated",
            frequency=2,
            confidence=0.9
        )
        
        # Update patterns
        await implementation_agent._update_project_patterns([pattern1])
        assert "test_pattern_1" in implementation_agent.code_patterns
        assert implementation_agent.code_patterns["test_pattern_1"].frequency == 1
        
        # Update existing pattern
        await implementation_agent._update_project_patterns([pattern2])
        updated_pattern = implementation_agent.code_patterns["test_pattern_1"]
        assert updated_pattern.frequency == 3  # 1 + 2
        assert updated_pattern.confidence == 0.85  # (0.8 + 0.9) / 2

    async def test_project_context_attributes(self, implementation_agent):
        """Test that new project context attributes are properly initialized."""
        # Check new attributes exist
        assert hasattr(implementation_agent, 'project_context')
        assert hasattr(implementation_agent, 'code_patterns')
        assert hasattr(implementation_agent, 'project_structure')
        assert hasattr(implementation_agent, 'generation_history')
        
        # Check initial values
        assert implementation_agent.project_context is None
        assert isinstance(implementation_agent.code_patterns, dict)
        assert implementation_agent.project_structure is None
        assert isinstance(implementation_agent.generation_history, list)

    async def test_code_type_detection_comprehensive(self, implementation_agent):
        """Test comprehensive code type detection scenarios."""
        test_cases = [
            ("test/widget_test.dart", "class TestWidget {}", CodeType.TEST),
            ("lib/models/user.dart", "class User {}", CodeType.MODEL),
            ("lib/services/api_service.dart", "class ApiService {}", CodeType.SERVICE),
            ("lib/repositories/user_repository.dart", "class UserRepository {}", CodeType.REPOSITORY),
            ("lib/controllers/home_controller.dart", "class HomeController {}", CodeType.CONTROLLER),
            ("lib/main.dart", "void main() { runApp(MyApp()); }", CodeType.CONFIGURATION),
            ("lib/utils/helpers.dart", "String formatDate() {}", CodeType.UTILITY),
        ]
        
        for file_path, content, expected_type in test_cases:
            detected_type = implementation_agent._detect_code_type(file_path, content)
            assert detected_type == expected_type, f"Failed for {file_path}: expected {expected_type}, got {detected_type}"


async def run_implementation_agent_tests():
    """Run all implementation agent tests."""
    print("🧪 Running Implementation Agent Project-Aware Code Generation Tests")
    print("=" * 80)
    
    # Create test instance
    test_instance = TestImplementationAgentCodeUnderstanding()
    
    # Setup agent
    agent = await test_instance.implementation_agent()
    
    # Test code type detection
    widget_code = test_instance.sample_flutter_widget()
    bloc_code = test_instance.sample_bloc_code()
    
    print("✅ Testing code type detection...")
    await test_instance.test_detect_code_type_widget(agent, widget_code)
    await test_instance.test_detect_code_type_bloc(agent, bloc_code)
    print("   ✓ Widget detection: PASSED")
    print("   ✓ BLoC detection: PASSED")
    
    print("✅ Testing code structure extraction...")
    await test_instance.test_extract_code_structure(agent, widget_code)
    print("   ✓ Structure extraction: PASSED")
    
    print("✅ Testing pattern extraction...")
    await test_instance.test_extract_code_patterns(agent, widget_code)
    print("   ✓ Pattern extraction: PASSED")
    
    print("✅ Testing convention extraction...")
    await test_instance.test_extract_conventions(agent, widget_code)
    print("   ✓ Convention extraction: PASSED")
    
    print("✅ Testing dependency extraction...")
    await test_instance.test_extract_dependencies(agent, widget_code)
    print("   ✓ Dependency extraction: PASSED")
    
    print("✅ Testing complexity metrics...")
    await test_instance.test_calculate_complexity_metrics(agent, widget_code)
    print("   ✓ Complexity calculation: PASSED")
    
    print("✅ Testing comprehensive code understanding...")
    await test_instance.test_understand_existing_code_success(agent, widget_code)
    print("   ✓ Full code understanding: PASSED")
    
    print("✅ Testing error handling...")
    await test_instance.test_understand_existing_code_file_not_found(agent)
    print("   ✓ Error handling: PASSED")
    
    print("✅ Testing pattern updates...")
    await test_instance.test_update_project_patterns(agent)
    print("   ✓ Pattern updates: PASSED")
    
    print("✅ Testing attribute initialization...")
    await test_instance.test_project_context_attributes(agent)
    print("   ✓ Attribute initialization: PASSED")
    
    print("✅ Testing comprehensive code type detection...")
    await test_instance.test_code_type_detection_comprehensive(agent)
    print("   ✓ Comprehensive detection: PASSED")
    
    print("\n" + "=" * 80)
    print("🎉 All Implementation Agent tests completed successfully!")
    print("✅ Project-aware code generation features are working correctly")
    print("✅ Code understanding and pattern extraction verified")
    print("✅ Enhanced ImplementationAgent ready for intelligent Flutter development")


if __name__ == "__main__":
    asyncio.run(run_implementation_agent_tests())
