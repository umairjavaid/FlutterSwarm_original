"""
Implementation Agent for FlutterSwarm Multi-Agent System.

This agent specializes in code generation, feature development,
and implementation of Flutter applications based on specifications.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..config import get_logger

logger = get_logger("implementation_agent")


class ImplementationAgent(BaseAgent):
    """
    Specialized agent for Flutter application implementation and code generation.
    
    This agent handles:
    - Feature implementation based on specifications
    - Code generation for UI components and business logic
    - Integration of third-party packages and APIs
    - Refactoring and code optimization
    - Custom widget development
    - State management implementation
    - Navigation setup and routing
    - API integration and data management
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        # Override config for implementation-specific settings
        impl_config = AgentConfig(
            agent_id=config.agent_id or f"implementation_agent_{str(uuid.uuid4())[:8]}",
            agent_type="implementation",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS
            ],
            max_concurrent_tasks=5,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.2,  # Lower temperature for more consistent code generation
            max_tokens=8000,
            timeout=900,  # Longer timeout for complex implementations
            metadata=config.metadata
        )
        
        super().__init__(impl_config, llm_client, memory_manager, event_bus)
        
        # Implementation-specific state
        self.supported_features = [
            "ui_components", "business_logic", "state_management", 
            "navigation", "api_integration", "database_integration",
            "authentication", "custom_widgets", "animations", 
            "platform_specific_code", "third_party_integrations"
        ]
        
        self.flutter_patterns = {
            "widgets": ["stateless", "stateful", "inherited", "provider"],
            "state_management": ["bloc", "cubit", "provider", "riverpod", "getx"],
            "navigation": ["navigator", "go_router", "auto_route"],
            "data_persistence": ["hive", "sqflite", "shared_preferences"],
            "networking": ["dio", "http", "graphql"],
            "architecture": ["clean", "mvvm", "mvc", "bloc_pattern"]
        }
        
        self.code_templates = {
            "widget_template": self._get_widget_template(),
            "bloc_template": self._get_bloc_template(),
            "repository_template": self._get_repository_template(),
            "model_template": self._get_model_template()
        }
        
        logger.info(f"Implementation Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the implementation agent."""
        return """
You are the Implementation Agent in the FlutterSwarm multi-agent system, specializing in Flutter application development and code generation.

CORE EXPERTISE:
- Flutter/Dart application development and best practices
- UI/UX implementation with Flutter widgets and layouts
- State management implementation (BLoC, Provider, Riverpod, GetX)
- API integration and data management
- Custom widget development and animations
- Platform-specific implementations (iOS, Android, Web, Desktop)
- Third-party package integration and configuration
- Code optimization and refactoring techniques

IMPLEMENTATION RESPONSIBILITIES:
1. Feature Development: Transform specifications into working Flutter code
2. UI Implementation: Create responsive and accessible user interfaces
3. Business Logic: Implement application logic and data processing
4. State Management: Set up and configure state management solutions
5. API Integration: Connect applications with backend services
6. Database Integration: Implement local and remote data storage
7. Navigation Setup: Configure routing and navigation flows
8. Testing Implementation: Create unit and widget tests for code

CODING STANDARDS:
- Follow Flutter and Dart style guidelines and conventions
- Write clean, maintainable, and well-documented code
- Implement proper error handling and edge case management
- Use appropriate design patterns and architectural principles
- Ensure responsive design for multiple screen sizes
- Implement accessibility features and internationalization
- Optimize performance and memory usage
- Follow security best practices for data handling

CODE GENERATION APPROACH:
1. Analyze requirements and specifications thoroughly
2. Plan component structure and data flow
3. Generate clean, production-ready code
4. Include comprehensive error handling
5. Add meaningful comments and documentation
6. Implement proper testing strategies
7. Consider performance and scalability implications
8. Ensure platform-specific optimizations

Always generate complete, working code solutions with proper imports, error handling, and documentation.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of implementation-specific capabilities."""
        return [
            "feature_implementation",
            "ui_component_development",
            "business_logic_implementation",
            "state_management_setup",
            "api_integration",
            "database_integration",
            "navigation_implementation",
            "custom_widget_development",
            "animation_implementation",
            "platform_specific_development",
            "third_party_integration",
            "code_refactoring",
            "performance_optimization"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute implementation-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "feature_development":
                return await self._implement_feature(task_context, llm_analysis)
            elif task_type == "ui_implementation":
                return await self._implement_ui(task_context, llm_analysis)
            elif task_type == "code_generation":
                return await self._generate_code(task_context, llm_analysis)
            elif task_type == "refactoring":
                return await self._refactor_code(task_context, llm_analysis)
            elif task_type == "integration":
                return await self._implement_integration(task_context, llm_analysis)
            else:
                # Generic implementation processing
                return await self._process_implementation_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"Implementation processing failed: {e}")
            return {
                "error": str(e),
                "code_files": {},
                "implementation_notes": []
            }

    async def _implement_feature(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement a complete feature based on specifications."""
        logger.info(f"Implementing feature for task: {task_context.task_id}")
        
        feature_prompt = self._create_feature_implementation_prompt(task_context, llm_analysis)
        
        implementation_result = await self.execute_llm_task(
            user_prompt=feature_prompt,
            context={
                "task": task_context.to_dict(),
                "patterns": self.flutter_patterns,
                "templates": self.code_templates
            },
            structured_output=True
        )
        
        # Store implementation details in memory
        await self.memory_manager.store_memory(
            content=f"Feature implementation: {json.dumps(implementation_result)}",
            metadata={
                "type": "feature_implementation",
                "feature_name": task_context.metadata.get('feature_name'),
                "files_generated": len(implementation_result.get('files', {}))
            },
            correlation_id=task_context.correlation_id,
            importance=0.9,
            long_term=True
        )
        
        return {
            "implementation_result": implementation_result,
            "code_files": implementation_result.get("files", {}),
            "dependencies": implementation_result.get("dependencies", []),
            "setup_instructions": implementation_result.get("setup_instructions", []),
            "testing_files": implementation_result.get("test_files", {}),
            "implementation_notes": implementation_result.get("notes", [])
        }

    async def _implement_ui(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement UI components and screens."""
        logger.info(f"Implementing UI for task: {task_context.task_id}")
        
        ui_prompt = self._create_ui_implementation_prompt(task_context, llm_analysis)
        
        ui_implementation = await self.execute_llm_task(
            user_prompt=ui_prompt,
            context={
                "task": task_context.to_dict(),
                "ui_guidelines": self._get_ui_guidelines(),
                "responsive_patterns": self._get_responsive_patterns()
            },
            structured_output=True
        )
        
        # Store UI implementation
        await self.memory_manager.store_memory(
            content=f"UI implementation: {json.dumps(ui_implementation)}",
            metadata={
                "type": "ui_implementation",
                "components": ui_implementation.get('components', []),
                "screens": ui_implementation.get('screens', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        
        return {
            "ui_implementation": ui_implementation,
            "widget_files": ui_implementation.get("widgets", {}),
            "screen_files": ui_implementation.get("screens", {}),
            "style_files": ui_implementation.get("styles", {}),
            "assets": ui_implementation.get("assets", []),
            "responsive_breakpoints": ui_implementation.get("breakpoints", {})
        }

    # Template methods for code generation
    def _get_widget_template(self) -> str:
        """Get widget template for code generation."""
        return """
import 'package:flutter/material.dart';

class {widget_name} extends {widget_type} {{
  {constructor}
  
  {build_method}
}}
"""

    def _get_bloc_template(self) -> str:
        """Get BLoC template for state management."""
        return """
import 'package:bloc/bloc.dart';
import 'package:equatable/equatable.dart';

// Events
abstract class {bloc_name}Event extends Equatable {{
  @override
  List<Object> get props => [];
}}

// States
abstract class {bloc_name}State extends Equatable {{
  @override
  List<Object> get props => [];
}}

// BLoC
class {bloc_name}Bloc extends Bloc<{bloc_name}Event, {bloc_name}State> {{
  {bloc_name}Bloc() : super({initial_state}()) {{
    {event_handlers}
  }}
}}
"""

    def _get_repository_template(self) -> str:
        """Get repository template for data layer."""
        return """
abstract class {repository_name}Repository {{
  {methods}
}}

class {repository_name}RepositoryImpl implements {repository_name}Repository {{
  {dependencies}
  
  {repository_name}RepositoryImpl({{
    {constructor_params}
  }});
  
  {method_implementations}
}}
"""

    def _get_model_template(self) -> str:
        """Get model template for data models."""
        return """
import 'package:equatable/equatable.dart';

class {model_name} extends Equatable {{
  {properties}
  
  const {model_name}({{
    {constructor_params}
  }});
  
  {methods}
  
  @override
  List<Object?> get props => {props_list};
}}
"""

    # Prompt creation methods
    def _create_feature_implementation_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for feature implementation."""
        return f"""
Implement a complete Flutter feature based on the following specifications:

FEATURE REQUIREMENTS:
{task_context.description}

TECHNICAL ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please provide a comprehensive implementation including:

1. FEATURE ARCHITECTURE:
   - Component structure and organization
   - State management approach
   - Data flow and dependencies
   - Integration points with existing code

2. CODE IMPLEMENTATION:
   - All necessary Dart files with complete implementations
   - Proper Flutter widget structures
   - State management setup (BLoC, Provider, etc.)
   - Error handling and validation
   - Performance optimizations

3. UI IMPLEMENTATION:
   - Responsive widget layouts
   - Proper Material Design or Cupertino styling
   - Accessibility considerations
   - Animation and interaction details

4. DATA LAYER:
   - Model classes and data structures
   - Repository implementations
   - API service integrations
   - Local storage setup

5. TESTING STRATEGY:
   - Unit tests for business logic
   - Widget tests for UI components
   - Integration test considerations
   - Mock implementations for dependencies

6. DEPENDENCIES AND SETUP:
   - Required pubspec.yaml dependencies
   - Configuration files and setup
   - Platform-specific configurations
   - Environment setup instructions

Respond with a structured implementation containing all files and setup instructions.
"""

    def _create_ui_implementation_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for UI implementation."""
        return f"""
Implement Flutter UI components and screens based on the following requirements:

UI SPECIFICATIONS:
{task_context.description}

DESIGN ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

Please provide a complete UI implementation including:

1. SCREEN IMPLEMENTATIONS:
   - Complete screen widget implementations
   - Proper navigation setup and routing
   - State management integration
   - Error state and loading state handling

2. CUSTOM WIDGETS:
   - Reusable widget components
   - Proper widget composition
   - Parameter passing and customization
   - Widget lifecycle management

3. RESPONSIVE DESIGN:
   - Adaptive layouts for different screen sizes
   - Breakpoint-based responsive behavior
   - Platform-specific adaptations
   - Orientation handling

4. STYLING AND THEMING:
   - Consistent color schemes and typography
   - Theme data configuration
   - Custom widget styling
   - Dark/light theme support

5. ANIMATIONS AND INTERACTIONS:
   - Smooth transition animations
   - User interaction feedback
   - Custom animation implementations
   - Performance-optimized animations

6. ACCESSIBILITY:
   - Semantic labels and descriptions
   - Screen reader compatibility
   - Focus management
   - High contrast support

Provide complete, production-ready Flutter UI code with proper documentation.
"""

    # Helper methods for guidelines and patterns
    def _get_ui_guidelines(self) -> Dict[str, Any]:
        """Get UI development guidelines."""
        return {
            "material_design": True,
            "responsive_breakpoints": {"mobile": 600, "tablet": 1024, "desktop": 1440},
            "accessibility": ["semantic_labels", "contrast_ratios", "touch_targets"],
            "performance": ["widget_rebuilds", "image_optimization", "lazy_loading"]
        }

    def _get_responsive_patterns(self) -> Dict[str, Any]:
        """Get responsive design patterns."""
        return {
            "adaptive_layouts": ["LayoutBuilder", "MediaQuery", "OrientationBuilder"],
            "responsive_widgets": ["Flexible", "Expanded", "Wrap", "FittedBox"],
            "breakpoint_system": {"xs": 0, "sm": 576, "md": 768, "lg": 992, "xl": 1200}
        }
