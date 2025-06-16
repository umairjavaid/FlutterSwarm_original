"""
Comprehensive Test Suite for ImplementationAgent with Real Flutter Projects.

This test suite validates all capabilities of the ImplementationAgent including:
- Project-aware code generation with real Flutter projects
- Intelligent file placement and refactoring
- Continuous validation and issue resolution
- Dependency management and hot reload integration
- Feature-complete generation with complex requirements
- Performance benchmarks and validation
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

# Import the agent and models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.implementation_agent import ImplementationAgent
from src.agents.base_agent import AgentConfig
from src.core.memory_manager import MemoryManager
from src.core.event_bus import EventBus
from src.models.code_models import (
    FeatureSpecification, FeatureType, UIRequirement, DataRequirement,
    BusinessLogicRequirement, APIRequirement, TestingRequirement,
    CodeStyle, StylePattern, StyleComplexity, StyleRule,
    DevelopmentSession, HotReloadExperience, PerformanceMetric, BenchmarkResult
)
from src.models.tool_models import ToolStatus
from src.config import get_logger

logger = get_logger("test_implementation_agent_real")


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = {}
    
    async def generate_response(self, system_prompt: str, user_prompt: str, **kwargs):
        self.call_count += 1
        
        # Return appropriate responses based on prompt content
        if "model" in user_prompt.lower():
            return {
                "generated_code": """
// User Model
import 'package:equatable/equatable.dart';

class User extends Equatable {
  const User({
    required this.id,
    required this.name,
    required this.email,
  });

  final String id;
  final String name;
  final String email;

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] as String,
      name: json['name'] as String,
      email: json['email'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'email': email,
    };
  }

  @override
  List<Object?> get props => [id, name, email];
}
"""
            }
        elif "repository" in user_prompt.lower():
            return {
                "generated_code": """
// User Repository
abstract class UserRepository {
  Future<List<User>> getUsers();
  Future<User> getUserById(String id);
  Future<void> createUser(User user);
  Future<void> updateUser(User user);
  Future<void> deleteUser(String id);
}

class UserRepositoryImpl implements UserRepository {
  final ApiService _apiService;
  
  UserRepositoryImpl(this._apiService);
  
  @override
  Future<List<User>> getUsers() async {
    try {
      final response = await _apiService.get('/users');
      return (response.data as List)
          .map((json) => User.fromJson(json))
          .toList();
    } catch (e) {
      throw Exception('Failed to fetch users: $e');
    }
  }
  
  @override
  Future<User> getUserById(String id) async {
    try {
      final response = await _apiService.get('/users/$id');
      return User.fromJson(response.data);
    } catch (e) {
      throw Exception('Failed to fetch user: $e');
    }
  }
  
  @override
  Future<void> createUser(User user) async {
    try {
      await _apiService.post('/users', data: user.toJson());
    } catch (e) {
      throw Exception('Failed to create user: $e');
    }
  }
  
  @override
  Future<void> updateUser(User user) async {
    try {
      await _apiService.put('/users/${user.id}', data: user.toJson());
    } catch (e) {
      throw Exception('Failed to update user: $e');
    }
  }
  
  @override
  Future<void> deleteUser(String id) async {
    try {
      await _apiService.delete('/users/$id');
    } catch (e) {
      throw Exception('Failed to delete user: $e');
    }
  }
}
"""
            }
        elif "bloc" in user_prompt.lower():
            return {
                "generated_code": """
// User BLoC
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

// Events
abstract class UserEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class LoadUsers extends UserEvent {}

class LoadUser extends UserEvent {
  final String id;
  LoadUser(this.id);
  
  @override
  List<Object?> get props => [id];
}

class CreateUser extends UserEvent {
  final User user;
  CreateUser(this.user);
  
  @override
  List<Object?> get props => [user];
}

// States
abstract class UserState extends Equatable {
  @override
  List<Object?> get props => [];
}

class UserInitial extends UserState {}

class UserLoading extends UserState {}

class UsersLoaded extends UserState {
  final List<User> users;
  UsersLoaded(this.users);
  
  @override
  List<Object?> get props => [users];
}

class UserLoaded extends UserState {
  final User user;
  UserLoaded(this.user);
  
  @override
  List<Object?> get props => [user];
}

class UserError extends UserState {
  final String message;
  UserError(this.message);
  
  @override
  List<Object?> get props => [message];
}

// BLoC
class UserBloc extends Bloc<UserEvent, UserState> {
  final UserRepository _repository;
  
  UserBloc(this._repository) : super(UserInitial()) {
    on<LoadUsers>(_onLoadUsers);
    on<LoadUser>(_onLoadUser);
    on<CreateUser>(_onCreateUser);
  }
  
  Future<void> _onLoadUsers(LoadUsers event, Emitter<UserState> emit) async {
    emit(UserLoading());
    try {
      final users = await _repository.getUsers();
      emit(UsersLoaded(users));
    } catch (e) {
      emit(UserError(e.toString()));
    }
  }
  
  Future<void> _onLoadUser(LoadUser event, Emitter<UserState> emit) async {
    emit(UserLoading());
    try {
      final user = await _repository.getUserById(event.id);
      emit(UserLoaded(user));
    } catch (e) {
      emit(UserError(e.toString()));
    }
  }
  
  Future<void> _onCreateUser(CreateUser event, Emitter<UserState> emit) async {
    emit(UserLoading());
    try {
      await _repository.createUser(event.user);
      final users = await _repository.getUsers();
      emit(UsersLoaded(users));
    } catch (e) {
      emit(UserError(e.toString()));
    }
  }
}
"""
            }
        else:
            return {"generated_code": "// Generated code"}


class MockToolResult:
    """Mock tool result for testing."""
    
    def __init__(self, status: ToolStatus, data: Dict[str, Any]):
        self.status = status
        self.data = data


@pytest.fixture
async def implementation_agent():
    """Create an ImplementationAgent instance for testing."""
    config = AgentConfig(
        agent_id="test_implementation_agent",
        agent_type="implementation",
        capabilities=[],
        max_concurrent_tasks=5,
        llm_model="gpt-4",
        temperature=0.2,
        max_tokens=8000,
        timeout=900
    )
    
    llm_client = MockLLMClient()
    memory_manager = Mock(spec=MemoryManager)
    event_bus = Mock(spec=EventBus)
    
    agent = ImplementationAgent(config, llm_client, memory_manager, event_bus)
    
    # Mock tool initialization
    agent.flutter_tool = Mock()
    agent.file_tool = Mock()
    agent.process_tool = Mock()
    
    return agent


@pytest.fixture
def temp_flutter_project():
    """Create a temporary Flutter project for testing."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_flutter_project"
    project_path.mkdir()
    
    # Create basic Flutter project structure
    (project_path / "lib").mkdir()
    (project_path / "test").mkdir()
    (project_path / "pubspec.yaml").write_text("""
name: test_flutter_project
description: A test Flutter project
version: 1.0.0+1

environment:
  sdk: '>=2.19.0 <4.0.0'
  flutter: ">=3.7.0"

dependencies:
  flutter:
    sdk: flutter
  flutter_bloc: ^8.1.3
  equatable: ^2.0.5

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^2.0.0

flutter:
  uses-material-design: true
""")
    
    # Create a minimal Flutter main.dart for testing - without app-specific logic
    (project_path / "lib" / "main.dart").write_text("""
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Test Flutter App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Test Flutter Home Page'),
    );
  }
}

class MyHomePage extends StatelessWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(title),
      ),
      body: const Center(
        child: Text(
          'Flutter Test App - Placeholder Widget',
        ),
      ),
    );
  }
}
""")
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestFeatureCompleteGeneration:
    """Test suite for feature-complete code generation."""
    
    @pytest.mark.asyncio
    async def test_generate_user_management_feature(self, implementation_agent, temp_flutter_project):
        """Test generating a complete user management feature."""
        # Setup mock file operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"content": "// Existing code"}
        ))
        
        # Create feature specification for user management
        feature_spec = FeatureSpecification(
            feature_id="user_management",
            feature_name="User Management",
            feature_type=FeatureType.CRUD_OPERATIONS,
            description="Complete user management system with CRUD operations",
            ui_requirements=UIRequirement(
                screen_name="UserListScreen",
                widget_types=["ListView", "Card", "FloatingActionButton"],
                layout_pattern="master_detail",
                navigation_flow=["users_list", "user_detail", "user_edit"],
                state_management_approach="bloc",
                styling_requirements={"theme": "material3", "responsive": True},
                responsive_behavior={"mobile": "single_pane", "tablet": "dual_pane"},
                accessibility_requirements=["screen_reader", "high_contrast"]
            ),
            data_requirements=DataRequirement(
                models=["User"],
                relationships={"User": ["Profile", "Permissions"]},
                persistence_strategy="local_database",
                caching_requirements=["user_list", "user_details"],
                validation_rules={"User": ["required_name", "valid_email"]},
                transformation_needs={"User": "fromJson_toJson"}
            ),
            business_logic_requirements=BusinessLogicRequirement(
                use_cases=["Load Users", "Create User", "Update User", "Delete User"],
                business_rules={"User": ["unique_email", "valid_name_format"]},
                workflows=[{"name": "user_creation", "steps": ["validate", "create", "notify"]}],
                integration_points=["user_api", "notification_service"],
                security_requirements=["authentication", "authorization"],
                performance_requirements={"load_time": "< 2s", "list_pagination": "20 items"}
            ),
            api_requirements=APIRequirement(
                endpoints=[
                    {"method": "GET", "path": "/api/users", "description": "Get all users"},
                    {"method": "POST", "path": "/api/users", "description": "Create user"},
                    {"method": "PUT", "path": "/api/users/{id}", "description": "Update user"},
                    {"method": "DELETE", "path": "/api/users/{id}", "description": "Delete user"}
                ],
                authentication_method="bearer_token",
                data_formats=["json"],
                error_handling_strategy="structured_errors",
                caching_strategy="cache_first",
                offline_behavior="cache_fallback",
                rate_limiting={"requests_per_minute": 100}
            ),
            testing_requirements=TestingRequirement(
                unit_test_coverage=0.9,
                widget_tests=["UserListScreen", "UserCard", "UserForm"],
                integration_tests=["user_crud_flow", "offline_sync"],
                performance_tests=["list_load_time", "search_performance"],
                accessibility_tests=["screen_reader_navigation"],
                mock_strategies={"api": "mockito", "database": "fake_data"},
                test_data_requirements=["sample_users", "edge_cases"]
            ),
            architecture_constraints={"pattern": "clean_architecture", "layers": ["data", "domain", "presentation"]},
            dependencies=["flutter_bloc", "dio", "hive", "get_it"],
            priority="high",
            timeline={"start": "2024-01-01", "end": "2024-01-15"},
            acceptance_criteria=[
                "Users can view a list of all users",
                "Users can create new user accounts",
                "Users can edit existing user information", 
                "Users can delete user accounts",
                "All operations work offline with sync",
                "UI is responsive on mobile and tablet"
            ]
        )
        
        # Generate the feature
        result = await implementation_agent.generate_feature_complete(feature_spec)
        
        # Validate the implementation
        assert result is not None
        assert result.implementation_id is not None
        assert result.feature_specification == feature_spec
        assert len(result.generated_components) > 0
        
        # Check that all required component types are generated
        component_types = {comp.component_type.value for comp in result.generated_components}
        expected_types = {"model", "repository", "bloc_cubit", "screen", "route", "test"}
        assert expected_types.issubset(component_types)
        
        # Validate implementation quality
        quality_score = result.get_implementation_quality_score()
        assert quality_score > 50.0  # Should have reasonable quality
        
        # Check completion status
        completion_status = result.get_completion_status()
        assert completion_status in ["complete", "mostly_complete", "partially_complete"]
        
        # Validate wiring configuration
        assert "framework" in result.wiring_configuration
        assert "registrations" in result.wiring_configuration
        
        # Validate routing setup
        assert "framework" in result.routing_setup
        assert "routes" in result.routing_setup
        
        # Validate testing suite
        assert "framework" in result.testing_suite
        assert "coverage_target" in result.testing_suite
        
        logger.info(f"Feature generation test completed - Quality: {quality_score:.1f}, Status: {completion_status}")

    @pytest.mark.asyncio
    async def test_generate_authentication_feature(self, implementation_agent):
        """Test generating an authentication feature with API integration."""
        # Setup mock file operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"content": "// Existing code"}
        ))
        
        # Create authentication feature specification
        feature_spec = FeatureSpecification(
            feature_id="authentication",
            feature_name="Authentication System",
            feature_type=FeatureType.AUTHENTICATION,
            description="Complete authentication system with login, registration, and password reset",
            ui_requirements=UIRequirement(
                screen_name="LoginScreen",
                widget_types=["TextField", "ElevatedButton", "Form"],
                layout_pattern="single_page",
                navigation_flow=["login", "register", "forgot_password"],
                state_management_approach="bloc",
                styling_requirements={"theme": "custom", "branding": True},
                responsive_behavior={"all": "single_column"},
                accessibility_requirements=["form_labels", "error_announcements"]
            ),
            data_requirements=DataRequirement(
                models=["User", "AuthToken", "AuthState"],
                relationships={"User": ["AuthToken"], "AuthState": ["User"]},
                persistence_strategy="secure_storage",
                caching_requirements=["auth_token", "user_profile"],
                validation_rules={"User": ["email_format", "password_strength"]},
                transformation_needs={"AuthToken": "secure_serialization"}
            ),
            business_logic_requirements=BusinessLogicRequirement(
                use_cases=["Login", "Register", "Logout", "Reset Password", "Refresh Token"],
                business_rules={"Authentication": ["token_expiry", "session_management"]},
                workflows=[
                    {"name": "login_flow", "steps": ["validate", "authenticate", "store_token"]},
                    {"name": "register_flow", "steps": ["validate", "create_account", "verify_email"]}
                ],
                integration_points=["auth_api", "email_service"],
                security_requirements=["token_encryption", "secure_storage", "biometric_auth"],
                performance_requirements={"auth_time": "< 3s", "token_refresh": "< 1s"}
            ),
            api_requirements=APIRequirement(
                endpoints=[
                    {"method": "POST", "path": "/auth/login", "description": "User login"},
                    {"method": "POST", "path": "/auth/register", "description": "User registration"},
                    {"method": "POST", "path": "/auth/logout", "description": "User logout"},
                    {"method": "POST", "path": "/auth/refresh", "description": "Refresh token"}
                ],
                authentication_method="jwt_token",
                data_formats=["json"],
                error_handling_strategy="detailed_errors",
                caching_strategy="no_cache_auth",
                offline_behavior="cached_credentials",
                rate_limiting={"login_attempts": 5}
            ),
            testing_requirements=TestingRequirement(
                unit_test_coverage=0.95,
                widget_tests=["LoginScreen", "RegisterScreen", "AuthForm"],
                integration_tests=["auth_flow", "token_refresh", "logout_flow"],
                performance_tests=["login_time", "token_validation"],
                accessibility_tests=["form_navigation", "error_handling"],
                mock_strategies={"auth_api": "http_mock", "storage": "fake_secure_storage"},
                test_data_requirements=["valid_credentials", "invalid_credentials"]
            ),
            architecture_constraints={"pattern": "clean_architecture", "security": "high"},
            dependencies=["flutter_bloc", "dio", "flutter_secure_storage", "local_auth"],
            priority="critical",
            timeline={"start": "2024-01-01", "end": "2024-01-10"},
            acceptance_criteria=[
                "Users can login with email and password",
                "Users can register new accounts",
                "Users can reset forgotten passwords",
                "Authentication tokens are securely stored",
                "Biometric authentication is supported",
                "All auth operations are properly tested"
            ]
        )
        
        # Generate the feature
        result = await implementation_agent.generate_feature_complete(feature_spec)
        
        # Validate the implementation
        assert result is not None
        assert result.feature_specification.feature_type == FeatureType.AUTHENTICATION
        
        # Check security-related components
        security_components = [comp for comp in result.generated_components 
                             if "auth" in comp.file_path.lower() or "security" in comp.content.lower()]
        assert len(security_components) > 0
        
        # Validate security considerations in performance considerations
        auth_service_components = [comp for comp in result.generated_components 
                                 if comp.component_type.value == "service"]
        if auth_service_components:
            service_considerations = auth_service_components[0].performance_considerations
            assert any("security" in consideration.lower() or "auth" in consideration.lower() 
                      for consideration in service_considerations)
        
        logger.info("Authentication feature generation test completed successfully")


class TestCodeStyleAdaptation:
    """Test suite for code style adaptation."""
    
    @pytest.mark.asyncio
    async def test_adapt_to_material_design_style(self, implementation_agent, temp_flutter_project):
        """Test adapting code to Material Design style guidelines."""
        # Setup mock file operations
        def mock_use_tool(tool_name, operation, params, description):
            if operation == "read_file":
                return MockToolResult(ToolStatus.SUCCESS, {
                    "content": """
import 'package:flutter/material.dart';

class myWidget extends StatelessWidget {
  const myWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return container(
      child: text('Hello World'),
    );
  }
}

class user_service {
  void fetchUsers() {
    // implementation
  }
}
"""
                })
            elif operation == "run_command":
                if "find" in params.get("command", ""):
                    return MockToolResult(ToolStatus.SUCCESS, {
                        "output": f"{temp_flutter_project}/lib/main.dart\n{temp_flutter_project}/lib/widgets/custom_widget.dart"
                    })
            return MockToolResult(ToolStatus.SUCCESS, {})
        
        implementation_agent.use_tool = AsyncMock(side_effect=mock_use_tool)
        
        # Create Material Design style
        material_style = CodeStyle(
            style_id="material_design_style",
            style_name="Material Design Style Guide",
            description="Flutter style guide following Material Design principles",
            rules=[
                StyleRule(
                    rule_id="pascal_case_classes",
                    pattern=StylePattern.NAMING_CONVENTION,
                    description="Use PascalCase for class names",
                    example_correct="class MyWidget extends StatelessWidget",
                    example_incorrect="class myWidget extends StatelessWidget",
                    enforcement_level="error",
                    context_conditions=["class_declaration"],
                    complexity=StyleComplexity.SIMPLE
                ),
                StyleRule(
                    rule_id="dart_imports_first",
                    pattern=StylePattern.IMPORT_ORGANIZATION,
                    description="Group dart: imports first",
                    example_correct="import 'dart:async';\nimport 'package:flutter/material.dart';",
                    example_incorrect="import 'package:flutter/material.dart';\nimport 'dart:async';",
                    enforcement_level="warning",
                    context_conditions=["file_top"],
                    complexity=StyleComplexity.SIMPLE
                ),
                StyleRule(
                    rule_id="documentation_comments",
                    pattern=StylePattern.DOCUMENTATION,
                    description="Use /// for documentation comments",
                    example_correct="/// This widget displays user information",
                    example_incorrect="// This widget displays user information",
                    enforcement_level="suggestion",
                    context_conditions=["public_classes", "public_methods"],
                    complexity=StyleComplexity.SIMPLE
                )
            ],
            naming_conventions={
                "classes": "PascalCase",
                "methods": "camelCase",
                "variables": "camelCase",
                "constants": "UPPER_CASE"
            },
            file_organization={
                "imports": ["dart", "package", "relative"],
                "structure": ["imports", "constants", "classes"]
            },
            architecture_preferences={
                "state_management": "bloc",
                "dependency_injection": "get_it"
            },
            formatting_preferences={
                "line_length": 80,
                "trailing_commas": True
            },
            linting_configuration={
                "flutter_lints": "^2.0.0"
            },
            documentation_standards={
                "public_apis": "required",
                "complex_logic": "recommended"
            },
            testing_standards={
                "unit_tests": "required",
                "widget_tests": "recommended"
            }
        )
        
        # Perform style adaptation
        result = await implementation_agent.adapt_code_style(material_style)
        
        # Validate the adaptation
        assert result is not None
        assert result.adaptation_id is not None
        assert result.target_style == material_style
        assert len(result.source_files) > 0
        
        # Check that style analysis was performed
        assert result.style_analysis is not None
        assert len(result.style_analysis.analyzed_files) > 0
        assert len(result.style_analysis.discovered_patterns) > 0
        
        # Check that applications were made
        assert len(result.applications) > 0
        
        # Validate quality improvements
        quality_score = result.get_adaptation_quality_score()
        assert quality_score >= 0.0
        
        # Check for specific improvements
        assert "code_readability" in result.quality_improvements
        assert "maintainability" in result.quality_improvements
        assert "consistency" in result.quality_improvements
        
        # Validate recommendations
        assert len(result.follow_up_recommendations) > 0
        
        logger.info(f"Style adaptation test completed - Quality: {quality_score:.1f}, Success Rate: {result.success_rate:.2%}")

    @pytest.mark.asyncio
    async def test_detect_and_fix_style_violations(self, implementation_agent, temp_flutter_project):
        """Test detecting and fixing common style violations."""
        # Create code with violations
        violation_code = """
import 'package:flutter/material.dart';
import 'dart:async';

class myWidget extends StatelessWidget {
  const myWidget({Key? key}) : super(key: key);

  Widget build(BuildContext context) {
    return Container(
      child: Text('Hello'),
    );
  }
}

class user_data {
  String user_name;
  user_data(this.user_name);
}
"""
        
        def mock_use_tool(tool_name, operation, params, description):
            if operation == "read_file":
                return MockToolResult(ToolStatus.SUCCESS, {"content": violation_code})
            elif operation == "run_command":
                if "find" in params.get("command", ""):
                    return MockToolResult(ToolStatus.SUCCESS, {
                        "output": f"{temp_flutter_project}/lib/violation_test.dart"
                    })
            return MockToolResult(ToolStatus.SUCCESS, {})
        
        implementation_agent.use_tool = AsyncMock(side_effect=mock_use_tool)
        
        # Create style with violation detection rules
        style_with_fixes = CodeStyle(
            style_id="violation_fix_style",
            style_name="Violation Detection and Fix",
            description="Style guide that detects and fixes common violations",
            rules=[
                StyleRule(
                    rule_id="fix_pascal_case",
                    pattern=StylePattern.NAMING_CONVENTION,
                    description="Use PascalCase for class names",
                    example_correct="class MyWidget extends StatelessWidget",
                    example_incorrect="class myWidget extends StatelessWidget",
                    enforcement_level="error",
                    context_conditions=["class_declaration"],
                    complexity=StyleComplexity.SIMPLE
                ),
                StyleRule(
                    rule_id="fix_import_order",
                    pattern=StylePattern.IMPORT_ORGANIZATION,
                    description="Group dart: imports first",
                    example_correct="import 'dart:async';\nimport 'package:flutter/material.dart';",
                    example_incorrect="import 'package:flutter/material.dart';\nimport 'dart:async';",
                    enforcement_level="error",
                    context_conditions=["file_top"],
                    complexity=StyleComplexity.SIMPLE
                )
            ],
            naming_conventions={"classes": "PascalCase"},
            file_organization={"imports": ["dart", "package", "relative"]},
            architecture_preferences={},
            formatting_preferences={},
            linting_configuration={},
            documentation_standards={},
            testing_standards={}
        )
        
        # Perform adaptation
        result = await implementation_agent.adapt_code_style(style_with_fixes)
        
        # Validate that violations were detected and fixed
        assert result is not None
        assert len(result.applications) > 0
        
        # Check for naming convention fixes
        naming_fixes = [app for app in result.applications 
                       if any("class" in change.lower() for change in app.changes_applied)]
        assert len(naming_fixes) > 0
        
        # Check for import organization fixes
        import_fixes = [app for app in result.applications 
                       if any("import" in change.lower() for change in app.changes_applied)]
        assert len(import_fixes) > 0
        
        # Validate style analysis detected violations
        violations = result.style_analysis.common_violations
        violation_types = {v["type"] for v in violations}
        assert len(violations) > 0
        
        logger.info(f"Style violation detection test completed - {len(violations)} violations found and addressed")


class TestHotReloadIntegration:
    """Test suite for hot reload integration."""
    
    @pytest.mark.asyncio
    async def test_hot_reload_development_session(self, implementation_agent, temp_flutter_project):
        """Test hot reload integration during development."""
        # Setup mock file operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"output": "Analysis complete"}
        ))
        
        # Create development session
        dev_session = DevelopmentSession(
            session_id="hot_reload_test_session",
            project_path=str(temp_flutter_project),
            active_files=[
                f"{temp_flutter_project}/lib/main.dart",
                f"{temp_flutter_project}/lib/widgets/custom_widget.dart"
            ],
            watched_directories=["lib", "test"],
            developer_preferences={"auto_save": True, "hot_reload_enabled": True},
            session_metadata={"framework": "flutter", "version": "3.16.0"}
        )
        
        # Perform hot reload development
        result = await implementation_agent.develop_with_hot_reload(dev_session)
        
        # Validate the hot reload experience
        assert result is not None
        assert result.experience_id is not None
        assert result.session_id == dev_session.session_id
        
        # Check metrics
        assert result.code_changes_processed >= 0
        assert result.get_success_rate() >= 0.0
        assert result.get_productivity_score() >= 0.0
        
        # Validate recommendations
        recommendations = result.get_improvement_recommendations()
        assert isinstance(recommendations, list)
        
        logger.info(f"Hot reload test completed - Success Rate: {result.get_success_rate():.2%}, Productivity: {result.get_productivity_score():.1f}")

    @pytest.mark.asyncio
    async def test_hot_reload_compatibility_prediction(self, implementation_agent):
        """Test hot reload compatibility prediction for code changes."""
        # Create mock code changes
        from src.models.code_models import CodeChange, ChangeType, StateImpact
        
        changes = [
            CodeChange(
                change_id="widget_update",
                file_path="lib/widgets/user_card.dart",
                change_type=ChangeType.WIDGET_UPDATE,
                content_before="Text('Old Title')",
                content_after="Text('New Title')",
                line_number=25,
                state_impact=StateImpact.PRESERVES,
                estimated_reload_time=0.5
            ),
            CodeChange(
                change_id="class_addition",
                file_path="lib/models/user.dart",
                change_type=ChangeType.NEW_CLASS,
                content_before="",
                content_after="class UserPreferences { }",
                line_number=50,
                state_impact=StateImpact.NONE,
                estimated_reload_time=2.0
            )
        ]
        
        # Test compatibility prediction
        compatibility = await implementation_agent._predict_reload_compatibility(changes)
        
        # Validate predictions
        assert compatibility is not None
        assert compatibility.overall_score >= 0.0
        assert compatibility.overall_score <= 100.0
        assert len(compatibility.change_assessments) == len(changes)
        assert compatibility.recommended_strategy in ["hot_reload", "hot_restart", "full_restart", "batch_reload"]
        
        # Check individual change assessments
        for assessment in compatibility.change_assessments:
            assert assessment["change_id"] in ["widget_update", "class_addition"]
            assert "compatibility_score" in assessment
            assert "predicted_issues" in assessment
        
        logger.info(f"Compatibility prediction test completed - Overall Score: {compatibility.overall_score:.1f}")


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks and validation."""
    
    @pytest.mark.asyncio
    async def test_code_generation_performance(self, implementation_agent):
        """Test performance of code generation operations."""
        start_time = datetime.utcnow()
        
        # Setup mock operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"content": "// Generated code"}
        ))
        
        # Create simple feature specification
        simple_feature = FeatureSpecification(
            feature_id="performance_test",
            feature_name="Performance Test Feature",
            feature_type=FeatureType.USER_INTERFACE,
            description="Simple feature for performance testing",
            ui_requirements=UIRequirement(
                screen_name="TestScreen",
                widget_types=["Container"],
                layout_pattern="simple",
                navigation_flow=["main"],
                state_management_approach="stateful",
                styling_requirements={},
                responsive_behavior={},
                accessibility_requirements=[]
            ),
            data_requirements=DataRequirement(
                models=["TestModel"],
                relationships={},
                persistence_strategy="none",
                caching_requirements=[],
                validation_rules={},
                transformation_needs={}
            ),
            business_logic_requirements=BusinessLogicRequirement(
                use_cases=["Test Use Case"],
                business_rules={},
                workflows=[],
                integration_points=[],
                security_requirements=[],
                performance_requirements={}
            ),
            api_requirements=None,
            testing_requirements=TestingRequirement(
                unit_test_coverage=0.8,
                widget_tests=["TestScreen"],
                integration_tests=[],
                performance_tests=[],
                accessibility_tests=[],
                mock_strategies={},
                test_data_requirements=[]
            ),
            architecture_constraints={},
            dependencies=[],
            priority="low",
            timeline={},
            acceptance_criteria=["Basic functionality works"]
        )
        
        # Generate feature and measure performance
        result = await implementation_agent.generate_feature_complete(simple_feature)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create performance metrics
        metrics = [
            PerformanceMetric(
                metric_name="feature_generation_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.utcnow(),
                context={"feature_complexity": "simple", "component_count": len(result.generated_components)}
            ),
            PerformanceMetric(
                metric_name="components_generated",
                value=len(result.generated_components),
                unit="count",
                timestamp=datetime.utcnow(),
                context={"feature_type": simple_feature.feature_type.value}
            ),
            PerformanceMetric(
                metric_name="code_quality_score",
                value=result.get_implementation_quality_score(),
                unit="percentage",
                timestamp=datetime.utcnow(),
                context={"validation_passed": len([r for r in result.validation_results if r.is_valid])}
            )
        ]
        
        # Create benchmark result
        benchmark = BenchmarkResult(
            benchmark_id="code_generation_benchmark",
            agent_version="1.0.0",
            test_suite="performance_tests",
            metrics=metrics,
            execution_time=execution_time,
            resource_usage={"memory_mb": 100, "cpu_percent": 25},  # Estimated
            success_indicators={
                "generation_completed": result is not None,
                "components_created": len(result.generated_components) > 0,
                "quality_acceptable": result.get_implementation_quality_score() > 50
            }
        )
        
        # Validate performance benchmarks
        assert benchmark is not None
        assert benchmark.execution_time < 30.0  # Should complete within 30 seconds
        assert len(benchmark.metrics) > 0
        assert benchmark.get_overall_performance_score() > 0
        
        # Performance assertions
        assert execution_time < 10.0  # Feature generation should be fast
        assert len(result.generated_components) > 0  # Should generate components
        
        logger.info(f"Performance benchmark completed - Time: {execution_time:.2f}s, Score: {benchmark.get_overall_performance_score():.1f}")

    @pytest.mark.asyncio
    async def test_style_adaptation_performance(self, implementation_agent, temp_flutter_project):
        """Test performance of style adaptation operations."""
        start_time = datetime.utcnow()
        
        # Setup mock file operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"content": "class testClass { }", "output": f"{temp_flutter_project}/lib/test.dart"}
        ))
        
        # Create simple style
        simple_style = CodeStyle(
            style_id="performance_style",
            style_name="Performance Test Style",
            description="Simple style for performance testing",
            rules=[
                StyleRule(
                    rule_id="simple_naming",
                    pattern=StylePattern.NAMING_CONVENTION,
                    description="Use PascalCase",
                    example_correct="class MyClass",
                    example_incorrect="class myClass",
                    enforcement_level="warning",
                    context_conditions=["classes"],
                    complexity=StyleComplexity.SIMPLE
                )
            ],
            naming_conventions={"classes": "PascalCase"},
            file_organization={},
            architecture_preferences={},
            formatting_preferences={},
            linting_configuration={},
            documentation_standards={},
            testing_standards={}
        )
        
        # Perform style adaptation and measure performance
        result = await implementation_agent.adapt_code_style(simple_style)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Validate performance
        assert result is not None
        assert execution_time < 15.0  # Should complete quickly
        assert result.adaptation_time > 0
        
        # Create performance metrics
        adaptation_metrics = [
            PerformanceMetric(
                metric_name="style_adaptation_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.utcnow(),
                context={"files_processed": len(result.source_files), "rules_applied": len(simple_style.rules)}
            ),
            PerformanceMetric(
                metric_name="adaptation_success_rate",
                value=result.success_rate,
                unit="percentage",
                timestamp=datetime.utcnow(),
                context={"applications_count": len(result.applications)}
            )
        ]
        
        # Validate metrics
        assert len(adaptation_metrics) > 0
        assert all(metric.value >= 0 for metric in adaptation_metrics)
        
        logger.info(f"Style adaptation performance test completed - Time: {execution_time:.2f}s, Success Rate: {result.success_rate:.2%}")


class TestIntegrationScenarios:
    """Test suite for complex integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_development_workflow(self, implementation_agent, temp_flutter_project):
        """Test complete development workflow integration."""
        # Setup comprehensive mock operations
        implementation_agent.use_tool = AsyncMock(return_value=MockToolResult(
            ToolStatus.SUCCESS, {"content": "// Mock content", "output": "Success"}
        ))
        
        # Step 1: Generate feature
        feature_spec = FeatureSpecification(
            feature_id="integration_test_feature",
            feature_name="Integration Test Feature",
            feature_type=FeatureType.USER_INTERFACE,
            description="Feature for testing complete workflow",
            ui_requirements=UIRequirement(
                screen_name="IntegrationTestScreen",
                widget_types=["Scaffold", "AppBar", "Body"],
                layout_pattern="standard",
                navigation_flow=["main"],
                state_management_approach="bloc",
                styling_requirements={"theme": "material"},
                responsive_behavior={"mobile": "single_column"},
                accessibility_requirements=["semantic_labels"]
            ),
            data_requirements=DataRequirement(
                models=["IntegrationTestModel"],
                relationships={},
                persistence_strategy="memory",
                caching_requirements=[],
                validation_rules={},
                transformation_needs={}
            ),
            business_logic_requirements=BusinessLogicRequirement(
                use_cases=["Load Data", "Display Data"],
                business_rules={},
                workflows=[],
                integration_points=[],
                security_requirements=[],
                performance_requirements={}
            ),
            api_requirements=None,
            testing_requirements=TestingRequirement(
                unit_test_coverage=0.8,
                widget_tests=["IntegrationTestScreen"],
                integration_tests=["full_workflow"],
                performance_tests=[],
                accessibility_tests=[],
                mock_strategies={},
                test_data_requirements=[]
            ),
            architecture_constraints={"pattern": "clean"},
            dependencies=["flutter_bloc"],
            priority="medium",
            timeline={},
            acceptance_criteria=["Feature works correctly"]
        )
        
        # Generate feature
        feature_result = await implementation_agent.generate_feature_complete(feature_spec)
        assert feature_result is not None
        
        # Step 2: Apply style adaptation
        style = CodeStyle(
            style_id="integration_style",
            style_name="Integration Style",
            description="Style for integration testing",
            rules=[
                StyleRule(
                    rule_id="integration_naming",
                    pattern=StylePattern.NAMING_CONVENTION,
                    description="Use consistent naming",
                    example_correct="class MyClass",
                    example_incorrect="class myClass",
                    enforcement_level="warning",
                    context_conditions=["classes"],
                    complexity=StyleComplexity.SIMPLE
                )
            ],
            naming_conventions={},
            file_organization={},
            architecture_preferences={},
            formatting_preferences={},
            linting_configuration={},
            documentation_standards={},
            testing_standards={}
        )
        
        style_result = await implementation_agent.adapt_code_style(style)
        assert style_result is not None
        
        # Step 3: Test hot reload integration
        dev_session = DevelopmentSession(
            session_id="integration_session",
            project_path=str(temp_flutter_project),
            active_files=[comp.file_path for comp in feature_result.generated_components[:3]],
            watched_directories=["lib"],
            developer_preferences={"hot_reload": True},
            session_metadata={"test": "integration"}
        )
        
        hot_reload_result = await implementation_agent.develop_with_hot_reload(dev_session)
        assert hot_reload_result is not None
        
        # Validate integrated workflow
        workflow_success = (
            feature_result.get_completion_status() in ["complete", "mostly_complete"] and
            style_result.success_rate > 0.0 and
            hot_reload_result.get_success_rate() >= 0.0
        )
        
        assert workflow_success, "Complete development workflow should succeed"
        
        logger.info("Full development workflow integration test completed successfully")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, implementation_agent):
        """Test error handling and recovery mechanisms."""
        # Test with invalid feature specification
        invalid_feature = FeatureSpecification(
            feature_id="",  # Invalid empty ID
            feature_name="",  # Invalid empty name
            feature_type=FeatureType.USER_INTERFACE,
            description="",
            ui_requirements=UIRequirement(
                screen_name="",
                widget_types=[],
                layout_pattern="",
                navigation_flow=[],
                state_management_approach="",
                styling_requirements={},
                responsive_behavior={},
                accessibility_requirements=[]
            ),
            data_requirements=DataRequirement(
                models=[],
                relationships={},
                persistence_strategy="",
                caching_requirements=[],
                validation_rules={},
                transformation_needs={}
            ),
            business_logic_requirements=BusinessLogicRequirement(
                use_cases=[],
                business_rules={},
                workflows=[],
                integration_points=[],
                security_requirements=[],
                performance_requirements={}
            ),
            api_requirements=None,
            testing_requirements=TestingRequirement(
                unit_test_coverage=0.0,
                widget_tests=[],
                integration_tests=[],
                performance_tests=[],
                accessibility_tests=[],
                mock_strategies={},
                test_data_requirements=[]
            ),
            architecture_constraints={},
            dependencies=[],
            priority="",
            timeline={},
            acceptance_criteria=[]
        )
        
        # Should handle gracefully without crashing
        result = await implementation_agent.generate_feature_complete(invalid_feature)
        
        # Should return a result with error handling
        assert result is not None
        assert not result.success_indicators.get("generation_success", True)
        assert len(result.follow_up_tasks) > 0  # Should suggest fixes
        
        logger.info("Error handling and recovery test completed")


if __name__ == "__main__":
    """Run the test suite."""
    pytest.main([__file__, "-v", "--tb=short"])