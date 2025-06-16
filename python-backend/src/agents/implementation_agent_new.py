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
        
        # Remove hardcoded templates - use dynamic template system
        from ..core.template_engine import get_template_engine
        self.template_engine = get_template_engine()
        
        logger.info(f"Implementation Agent {self.agent_id} initialized with dynamic templates")

    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the implementation agent."""
        return """
You are the Implementation Agent in the FlutterSwarm multi-agent system, specializing in Flutter application development and code generation.

CORE EXPERTISE:
- Flutter/Dart application development and best practices
- UI/UX implementation with Flutter widgets and layouts
- State management implementation (BLoC, Provider, Riverpod, GetX)
- API integration and data management

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

    # Remove hardcoded template methods and replace with dynamic ones
    def _get_dynamic_template(self, template_type: str, architectural_pattern: str = "basic") -> str:
        """Get template dynamically from template engine."""
        from ..core.template_engine import TemplateType, ArchitecturalPattern, TemplateContext
        
        # Map string types to enums
        type_map = {
            "widget": TemplateType.WIDGET,
            "bloc": TemplateType.BLOC, 
            "repository": TemplateType.REPOSITORY,
            "model": TemplateType.MODEL,
            "service": TemplateType.SERVICE,
            "screen": TemplateType.SCREEN
        }
        
        pattern_map = {
            "basic": ArchitecturalPattern.BASIC_PATTERN,
            "clean": ArchitecturalPattern.CLEAN_ARCHITECTURE,
            "bloc": ArchitecturalPattern.BLOC_PATTERN,
            "provider": ArchitecturalPattern.PROVIDER_PATTERN
        }
        
        template_type_enum = type_map.get(template_type)
        pattern_enum = pattern_map.get(architectural_pattern, ArchitecturalPattern.BASIC_PATTERN)
        
        if template_type_enum:
            template = self.template_engine.get_template(template_type_enum, pattern_enum)
            return template.source if template else ""
        
        return ""

    def _create_dynamic_implementation_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for implementation using dynamic templates."""
        return f"""
Implement a complete Flutter feature based on the following specifications:

FEATURE REQUIREMENTS:
{task_context.description}

TECHNICAL ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please provide a comprehensive implementation using appropriate architectural patterns:

1. DYNAMIC TEMPLATE SELECTION:
   - Analyze requirements to determine optimal architectural pattern
   - Select appropriate templates (basic, clean architecture, BLoC, etc.)
   - Customize templates based on specific feature needs

2. FEATURE ARCHITECTURE:
   - Component structure based on selected pattern
   - State management approach matching user preferences
   - Data flow and dependencies optimized for requirements
   - Integration points with existing code

3. CODE IMPLEMENTATION:
   - Generate all necessary Dart files using dynamic templates
   - Implement proper Flutter widget structures
   - Setup state management according to selected pattern
   - Include error handling and validation
   - Apply performance optimizations

4. RESPONSIVE UI IMPLEMENTATION:
   - Create adaptive widget layouts for all screen sizes
   - Follow Material Design or Cupertino guidelines
   - Include accessibility considerations
   - Add smooth animations and interactions

5. DATA LAYER (if needed):
   - Generate model classes with proper serialization
   - Implement repository pattern for data access
   - Setup API service integrations
   - Configure local storage solutions

6. TESTING STRATEGY:
   - Generate unit tests for business logic
   - Create widget tests for UI components
   - Setup integration test framework
   - Provide mock implementations

7. DEPENDENCIES AND CONFIGURATION:
   - Determine required pubspec.yaml dependencies
   - Generate configuration files
   - Setup platform-specific configurations
   - Provide setup and deployment instructions

Respond with a structured implementation using dynamic templates that match the requirements.
"""

    async def _generate_code_dynamically(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code files dynamically based on task requirements."""
        logger.info(f"Generating code dynamically for task: {task_context.task_id}")
        
        try:
            # Analyze requirements to determine template strategy
            requirements = self._analyze_implementation_requirements(task_context, llm_analysis)
            
            # Create prompt for dynamic code generation
            generation_prompt = self._create_dynamic_implementation_prompt(task_context, llm_analysis)
            
            # Get LLM response for code structure
            code_plan = await self.execute_llm_task(
                user_prompt=generation_prompt,
                context={
                    "task": task_context.to_dict(),
                    "requirements": requirements,
                    "available_patterns": ["basic", "clean", "bloc", "provider", "riverpod"]
                },
                structured_output=True
            )
            
            if not code_plan:
                raise Exception("No code plan generated by LLM")
            
            # Generate files using template engine
            generated_files = {}
            
            if code_plan.get("files"):
                for file_info in code_plan["files"]:
                    file_path = file_info.get("path")
                    template_type = file_info.get("template_type", "widget")
                    pattern = file_info.get("architectural_pattern", "basic")
                    variables = file_info.get("variables", {})
                    
                    # Use template engine to generate content
                    content = self._generate_file_from_template(
                        template_type, pattern, variables, file_info.get("custom_content")
                    )
                    
                    if content:
                        generated_files[file_path] = content
            
            return {
                "code_files": generated_files,
                "implementation_notes": code_plan.get("instructions", []),
                "dependencies": code_plan.get("dependencies", []),
                "architectural_pattern": code_plan.get("pattern", "basic"),
                "total_files": len(generated_files)
            }
            
        except Exception as e:
            logger.error(f"Dynamic code generation failed: {e}")
            return {
                "error": str(e),
                "code_files": {},
                "implementation_notes": [f"Dynamic code generation failed: {str(e)}"]
            }
    
    def _analyze_implementation_requirements(
        self, 
        task_context: TaskContext, 
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze implementation requirements to determine optimal approach."""
        requirements = {
            "complexity": "simple",
            "architectural_pattern": "basic",
            "state_management": "stateful",
            "features": [],
            "ui_complexity": "basic",
            "data_requirements": None
        }
        
        description = task_context.description.lower()
        
        # Determine complexity
        if any(term in description for term in ["complex", "advanced", "enterprise", "large-scale"]):
            requirements["complexity"] = "complex"
        elif any(term in description for term in ["medium", "moderate", "standard"]):
            requirements["complexity"] = "medium"
        
        # Determine architectural pattern
        if "clean architecture" in description or "layered" in description:
            requirements["architectural_pattern"] = "clean"
        elif "bloc" in description or "cubit" in description:
            requirements["architectural_pattern"] = "bloc"
        elif "provider" in description:
            requirements["architectural_pattern"] = "provider"
        elif "riverpod" in description:
            requirements["architectural_pattern"] = "riverpod"
        
        # Extract features
        common_features = [
            "authentication", "login", "user management",
            "navigation", "routing", "drawer", "tabs",
            "forms", "validation", "input",
            "lists", "grid", "cards",
            "api", "network", "http",
            "database", "storage", "persistence",
            "notifications", "push", "alerts",
            "camera", "photos", "gallery",
            "maps", "location", "gps",
            "payments", "purchase", "billing"
        ]
        
        for feature in common_features:
            if feature in description:
                requirements["features"].append(feature)
        
        return requirements
    
    def _generate_file_from_template(
        self, 
        template_type: str, 
        pattern: str, 
        variables: Dict[str, Any],
        custom_content: Optional[str] = None
    ) -> Optional[str]:
        """Generate file content from template with variables."""
        from ..core.template_engine import TemplateType, ArchitecturalPattern, TemplateContext
        
        if custom_content:
            # Use custom content if provided
            return custom_content
        
        # Map template type and pattern
        type_map = {
            "widget": TemplateType.WIDGET,
            "screen": TemplateType.SCREEN,
            "model": TemplateType.MODEL,
            "bloc": TemplateType.BLOC,
            "repository": TemplateType.REPOSITORY,
            "service": TemplateType.SERVICE,
            "test": TemplateType.TEST
        }
        
        pattern_map = {
            "basic": ArchitecturalPattern.BASIC_PATTERN,
            "clean": ArchitecturalPattern.CLEAN_ARCHITECTURE,
            "bloc": ArchitecturalPattern.BLOC_PATTERN,
            "provider": ArchitecturalPattern.PROVIDER_PATTERN,
            "riverpod": ArchitecturalPattern.RIVERPOD_PATTERN
        }
        
        template_type_enum = type_map.get(template_type)
        pattern_enum = pattern_map.get(pattern, ArchitecturalPattern.BASIC_PATTERN)
        
        if not template_type_enum:
            logger.warning(f"Unknown template type: {template_type}")
            return None
        
        # Create template context
        context = TemplateContext(
            app_name=variables.get("app_name", "MyApp"),
            app_description=variables.get("app_description", "Flutter application"),
            architectural_pattern=pattern_enum,
            custom_variables=variables
        )
        
        # Render template
        return self.template_engine.render_template(template_type_enum, context)

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
