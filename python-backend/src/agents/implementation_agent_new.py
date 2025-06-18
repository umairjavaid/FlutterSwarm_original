"""
Implementation Agent for FlutterSwarm Multi-Agent System.

This agent specializes in code generation, feature development,
and implementation of Flutter applications based on specifications.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult, TaskStatus
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..models.tool_models import ToolStatus
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
        memory_manager: 'MemoryManager',
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
        
        logger.info(f"Implementation Agent {self.agent_id} initialized with LLM-based code generation")

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
                "patterns": self.flutter_patterns
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

    def _create_dynamic_implementation_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for implementation using pure LLM generation."""
        return f"""
You are implementing a complete Flutter feature. Generate ALL code files for the entire application.

FEATURE REQUIREMENTS:
{task_context.description}

TECHNICAL ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

CRITICAL: You must respond with ONLY a valid JSON object in this exact format:

{{
  "files": [
    {{
      "path": "lib/main.dart",
      "content": "COMPLETE FLUTTER CODE HERE - NOT JUST IMPORTS"
    }},
    {{
      "path": "lib/screens/home_screen.dart", 
      "content": "COMPLETE DART CODE FOR HOME SCREEN"
    }},
    {{
      "path": "lib/models/expense.dart",
      "content": "COMPLETE MODEL CLASS CODE"
    }}
  ],
  "main_dart": "COMPLETE main.dart CONTENT",
  "pubspec_yaml": "COMPLETE pubspec.yaml CONTENT",
  "dependencies": ["http", "provider", "shared_preferences"],
  "pattern": "clean",
  "instructions": ["Setup instructions"]
}}

CRITICAL REQUIREMENTS:
1. Respond with ONLY valid JSON - no markdown, no explanations, no code blocks
2. Start your response with {{ and end with }}
3. Escape all quotes inside strings with \\"
4. Generate a COMPLETE, FUNCTIONAL Flutter app with ALL requested features
5. Include proper state management (Provider/BLoC/Riverpod)
6. Create real UI screens with Material Design widgets
7. Add navigation between screens
8. Implement data models and services
9. Include proper error handling
10. Generate AT LEAST 8-10 files for a complete app structure

For a personal finance tracker, include:
- Expense tracking with categories
- Budget management
- Income tracking  
- Charts/graphs for financial analysis
- Goal setting functionality
- Data persistence

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT:"""

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
            
            # All code is now generated dynamically by LLM - no templates
            generated_files = {}
            
            # The code plan already contains the actual file content from LLM
            if code_plan.get("files"):
                for file_info in code_plan["files"]:
                    file_path = file_info.get("path")
                    content = file_info.get("content") or file_info.get("custom_content", "")
                    
                    if content and file_path:
                        generated_files[file_path] = content
            
            return {
                "code_files": generated_files,
                "implementation_notes": code_plan.get("instructions", []),
                "dependencies": code_plan.get("dependencies", []),
                "architectural_pattern": code_plan.get("pattern", "clean"),
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
    
    def _generate_file_with_llm(
        self, 
        template_type: str, 
        pattern: str, 
        variables: Dict[str, Any],
        custom_content: Optional[str] = None
    ) -> Optional[str]:
        """Generate file content using LLM instead of templates."""
        if custom_content:
            # Use custom content if provided
            return custom_content
        
        # For now, return None to rely on LLM generation in the main flow
        # This could be enhanced to ask LLM to generate specific file types
        logger.info(f"Using LLM generation for {template_type} with {pattern} pattern")
        return None

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

    async def process_task(self, task_context: TaskContext) -> TaskResult:
        """Process implementation tasks and generate real Flutter features."""
        logger.info(f"Processing implementation task: {task_context.task_id}")
        
        try:
            # Ensure tools are available
            if not self.available_tools:
                await self.discover_available_tools()
            
            # Extract project details from task context
            metadata = task_context.metadata
            project_name = metadata.get("project_name", "flutter_app")
            features = metadata.get("features", [])
            
            # Ensure projects are created in the flutter_projects directory
            flutter_projects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "flutter_projects"))
            # Always use flutter_projects directory - ignore any output_dir from metadata
            output_dir = flutter_projects_dir
            
            # Create flutter_projects directory if it doesn't exist
            os.makedirs(flutter_projects_dir, exist_ok=True)
            
            task_type = task_context.description.lower()
            
            # Normalize project name for Flutter SDK compliance
            normalized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
            # Remove any non-alphanumeric characters except underscores
            import re
            normalized_project_name = re.sub(r'[^a-z0-9_]', '', normalized_project_name)
            
            logger.info(f"Implementing task: {task_context.description}")
            logger.info(f"Project: {project_name} (normalized: {normalized_project_name}), Features: {features}")
            
            # Determine task type and route to appropriate handler
            if "project_setup" in task_type and "implement features" not in task_type:
                # Only project setup
                result = await self._handle_project_setup(task_context, normalized_project_name, output_dir)
            elif "feature_implementation" in task_type and "create flutter project" not in task_type:
                # Only feature implementation
                result = await self._handle_feature_implementation(task_context, normalized_project_name, features, output_dir)
            else:
                # Default: Create project AND implement features (most common case)
                logger.info(f"Creating project and implementing features for: {normalized_project_name}")
                
                # Step 1: Create the Flutter project
                project_result = await self._handle_project_setup(task_context, normalized_project_name, output_dir)
                
                # Step 2: Implement features if any are specified
                if features:
                    logger.info(f"Implementing features: {features}")
                    feature_result = await self._handle_feature_implementation(task_context, normalized_project_name, features, output_dir)
                    
                    # Combine results
                    result = {
                        "status": "completed",
                        "project_creation": project_result,
                        "feature_implementation": feature_result,
                        "deliverables": {
                            **project_result.get("deliverables", {}),
                            **feature_result.get("deliverables", {})
                        },
                        "metadata": {
                            **project_result.get("metadata", {}),
                            **feature_result.get("metadata", {}),
                            "features_implemented": features
                        }
                    }
                else:
                    # Just project creation
                    result = project_result
            
            # Create task result
            task_result = TaskResult(
                task_id=task_context.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result=result,
                deliverables=result.get("deliverables", {}),
                metrics=result.get("metrics", {}),
                metadata=result.get("metadata", {}),
                errors=result.get("errors", [])
            )
            
            logger.info(f"Successfully completed implementation task: {task_context.task_id}")
            return task_result
            
        except Exception as e:
            logger.error(f"Implementation task failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TaskResult(
                task_id=task_context.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result={"error": str(e)},
                deliverables={},
                metrics={},
                metadata={},
                errors=[str(e)]
            )

    async def _handle_project_setup(self, task_context: TaskContext, project_name: str, output_dir: str) -> Dict[str, Any]:
        """Handle Flutter project creation and setup."""
        logger.info(f"Setting up Flutter project: {project_name}")
        
        try:
            # Use Flutter SDK tool to create the project
            flutter_tool = None
            for tool_name, tool_instance in self.available_tools.items():
                if tool_name in ["flutter_sdk", "flutter_sdk_tool"]:
                    flutter_tool = tool_instance
                    break
            
            if not flutter_tool:
                logger.warning("Flutter SDK tool not found, attempting to discover tools again")
                await self.discover_available_tools()
                for tool_name, tool_instance in self.available_tools.items():
                    if tool_name in ["flutter_sdk", "flutter_sdk_tool"]:
                        flutter_tool = tool_instance
                        break
            
            if not flutter_tool:
                raise ValueError("Flutter SDK tool not available")
            
            # Create the Flutter project
            create_result = await flutter_tool.execute(
                operation="create_project",
                params={
                    "project_name": project_name,
                    "description": f"Flutter app: {project_name}",
                    "platforms": ["android", "ios", "web"],
                    "project_dir": output_dir,
                    "overwrite": True  # Allow overwriting existing projects
                }
            )
            
            if create_result.status != ToolStatus.SUCCESS:
                # If creation failed due to existing project, try to proceed with existing one
                project_path = f"{output_dir}/{project_name}"
                if "already exists" in str(create_result.error_message) and os.path.exists(project_path):
                    logger.info(f"Using existing Flutter project at: {project_path}")
                    return {
                        "status": "completed",
                        "project_path": project_path,
                        "deliverables": {
                            "project_structure": project_path,
                            "pubspec_yaml": f"{project_path}/pubspec.yaml",
                            "main_dart": f"{project_path}/lib/main.dart"
                        },
                        "metadata": {
                            "project_name": project_name,
                            "platforms": ["android", "ios", "web"],
                            "note": "Used existing project"
                        }
                    }
                else:
                    raise ValueError(f"Failed to create Flutter project: {create_result.error_message}")
            
            project_path = create_result.data.get("project_path")
            logger.info(f"Flutter project created at: {project_path}")
            
            return {
                "status": "completed",
                "project_path": project_path,
                "deliverables": {
                    "project_structure": project_path,
                    "pubspec_yaml": f"{project_path}/pubspec.yaml",
                    "main_dart": f"{project_path}/lib/main.dart"
                },
                "metadata": {
                    "project_name": project_name,
                    "platforms": ["android", "ios", "web"]
                }
            }
            
        except Exception as e:
            logger.error(f"Project setup failed: {e}")
            raise

    async def _handle_feature_implementation(self, task_context: TaskContext, project_name: str, features: List[str], output_dir: str) -> Dict[str, Any]:
        """Handle implementation of specific Flutter features."""
        logger.info(f"Implementing features {features} for project {project_name}")
        
        try:
            # Create feature implementation prompt
            feature_prompt = self._create_feature_implementation_prompt(project_name, features)
            
            # Generate Flutter code using LLM
            implementation_result = await self.execute_llm_task(
                user_prompt=feature_prompt,
                context={
                    "project_name": project_name,
                    "features": features,
                    "patterns": self.flutter_patterns
                },
                structured_output=True
            )
            
            # Handle both successful JSON and error responses
            processed_result = await self._process_implementation_response(implementation_result)
            
            # Write generated code to files
            project_path = f"{output_dir}/{project_name}"
            generated_files = await self._write_feature_code(project_path, processed_result)
            
            return {
                "status": "completed",
                "generated_files": generated_files,
                "deliverables": {
                    "main_dart": f"{project_path}/lib/main.dart",
                    "feature_widgets": f"{project_path}/lib/widgets/",
                    "state_management": f"{project_path}/lib/state/"
                },
                "metadata": {
                    "features_implemented": features,
                    "files_count": len(generated_files)
                }
            }
            
        except Exception as e:
            logger.error(f"Feature implementation failed: {e}")
            raise

    def _create_feature_implementation_prompt(self, project_name: str, features: List[str]) -> str:
        """Create a comprehensive prompt for implementing Flutter features."""
        features_text = ", ".join(features) if features else "basic mobile app functionality"
        
        return f"""
You are an expert Flutter developer. Generate a complete, feature-rich Flutter application for "{project_name}" with the following features: {features_text}.

CRITICAL INSTRUCTION: RESPOND WITH ONLY A VALID JSON OBJECT. NO MARKDOWN, NO EXPLANATION, NO TEXT BEFORE OR AFTER THE JSON.

REQUIREMENTS:
1. Create a fully functional Flutter app with real features, not just a counter template
2. Implement all requested features with proper UI and functionality
3. Use modern Flutter best practices and patterns
4. Include proper state management (BLoC, Provider, or similar)
5. Add navigation between different screens/features
6. Include proper error handling and loading states
7. Use appropriate Flutter packages for enhanced functionality

FEATURES TO IMPLEMENT:
{self._generate_feature_specifications(features)}

OUTPUT FORMAT - RESPOND WITH ONLY THIS JSON STRUCTURE:
{{
    "main_dart": "complete main.dart file content here",
    "pubspec_yaml": "complete pubspec.yaml with dependencies here",
    "files": {{
        "lib/screens/home_screen.dart": "complete home screen dart code here",
        "lib/screens/[feature]_screen.dart": "complete feature screen dart code here",
        "lib/widgets/[widget_name].dart": "complete custom widget dart code here",
        "lib/models/[model_name].dart": "complete data model dart code here",
        "lib/services/[service_name].dart": "complete service dart code here",
        "lib/state/[state_name].dart": "complete state management dart code here"
    }},
    "dependencies": ["list", "of", "flutter", "packages"],
    "setup_instructions": ["step by step setup instructions"],
    "features_implemented": ["list of implemented features"]
}}

IMPORTANT: Your response must be a valid JSON object that starts with {{ and ends with }}. Do not include any text before or after the JSON."""

    def _generate_feature_specifications(self, features: List[str]) -> str:
        """Generate detailed specifications for each feature."""
        if not features:
            return "- Basic mobile app with navigation and UI components"
        
        feature_specs = []
        
        for feature in features:
            feature_lower = feature.lower()
            if "photo" in feature_lower or "image" in feature_lower:
                feature_specs.append("- Photo sharing: Camera integration, image picker, photo gallery, upload functionality")
            elif "auth" in feature_lower or "login" in feature_lower:
                feature_specs.append("- User authentication: Login/register screens, form validation, secure authentication")
            elif "social" in feature_lower or "chat" in feature_lower or "messag" in feature_lower:
                feature_specs.append("- Social features: User profiles, messaging interface, social interactions")
            elif "music" in feature_lower or "audio" in feature_lower:
                feature_specs.append("- Music player: Audio playback, playlist management, music controls")
            elif "ecommerce" in feature_lower or "shop" in feature_lower:
                feature_specs.append("- E-commerce: Product catalog, shopping cart, checkout process")
            elif "weather" in feature_lower:
                feature_specs.append("- Weather app: Location-based weather, forecasts, weather data display")
            elif "finance" in feature_lower or "expense" in feature_lower or "budget" in feature_lower or "money" in feature_lower:
                feature_specs.append("- Personal finance tracker: Expense tracking, income management, budget planning, financial reports, transaction history")
            elif "portfolio" in feature_lower or "investment" in feature_lower or "stock" in feature_lower:
                feature_specs.append("- Investment portfolio: Portfolio tracking, stock monitoring, investment analytics, market data")
            elif "spending" in feature_lower or "expenditure" in feature_lower:
                feature_specs.append("- Spending tracker: Expense categorization, spending analytics, receipt management, cost tracking")
            elif "savings" in feature_lower or "save" in feature_lower:
                feature_specs.append("- Savings tracker: Savings goals, progress monitoring, target achievement, financial planning")
            else:
                feature_specs.append(f"- {feature}: Implement comprehensive functionality for {feature}")
        
        return "\n".join(feature_specs)

    async def _write_feature_code(self, project_path: str, implementation_result: Dict[str, Any]) -> List[str]:
        """Write generated code to Flutter project files."""
        generated_files = []
        
        try:
            # Ensure project directories exist
            await self._ensure_project_directories(project_path)
            
            # Use file system tool to write files
            file_tool = None
            for tool_name, tool_instance in self.available_tools.items():
                if tool_name == "file_system":
                    file_tool = tool_instance
                    break
            
            if not file_tool:
                logger.warning("File system tool not available, using fallback file writing")
                return await self._write_files_fallback(project_path, implementation_result)
            
            # Write main.dart
            main_dart_content = implementation_result.get("main_dart")
            if main_dart_content:
                main_path = f"{project_path}/lib/main.dart"
                operation_id = str(uuid.uuid4())
                result = await file_tool.execute(
                    operation="write_file",
                    params={
                        "path": main_path,
                        "content": main_dart_content
                    },
                    operation_id=operation_id
                )
                if result.status == ToolStatus.SUCCESS:
                    generated_files.append(main_path)
                    logger.info(f"Written main.dart to {main_path}")
            
            # Write pubspec.yaml
            pubspec_content = implementation_result.get("pubspec_yaml")
            if pubspec_content:
                pubspec_path = f"{project_path}/pubspec.yaml"
                operation_id = str(uuid.uuid4())
                result = await file_tool.execute(
                    operation="write_file",
                    params={
                        "path": pubspec_path,
                        "content": pubspec_content
                    },
                    operation_id=operation_id
                )
                if result.status == ToolStatus.SUCCESS:
                    generated_files.append(pubspec_path)
                    logger.info(f"Written pubspec.yaml to {pubspec_path}")
            
            # Write additional files
            files = implementation_result.get("files", {})
            for file_path, content in files.items():
                if content and content.strip():  # Only write non-empty files
                    full_path = f"{project_path}/{file_path}"
                    operation_id = str(uuid.uuid4())
                    result = await file_tool.execute(
                        operation="write_file",
                        params={
                            "path": full_path,
                            "content": content
                        },
                        operation_id=operation_id
                    )
                    if result.status == ToolStatus.SUCCESS:
                        generated_files.append(full_path)
                        logger.info(f"Written {file_path} to {full_path}")
            
            logger.info(f"Generated {len(generated_files)} files for Flutter project")
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to write feature code: {e}")
            # Try fallback method
            return await self._write_files_fallback(project_path, implementation_result)

    async def _ensure_project_directories(self, project_path: str) -> None:
        """Ensure all necessary project directories exist."""
        import os
        
        directories = [
            f"{project_path}/lib",
            f"{project_path}/lib/widgets",
            f"{project_path}/lib/screens",
            f"{project_path}/lib/models",
            f"{project_path}/lib/services",
            f"{project_path}/lib/state",
            f"{project_path}/test",
            f"{project_path}/android",
            f"{project_path}/ios"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    async def _write_files_fallback(self, project_path: str, implementation_result: Dict[str, Any]) -> List[str]:
        """Fallback method to write files directly using Python file operations."""
        import os
        generated_files = []
        
        try:
            # Ensure project directories exist
            await self._ensure_project_directories(project_path)
            
            # Write main.dart
            main_dart_content = implementation_result.get("main_dart")
            if main_dart_content:
                main_path = f"{project_path}/lib/main.dart"
                with open(main_path, 'w', encoding='utf-8') as f:
                    f.write(main_dart_content)
                generated_files.append(main_path)
                logger.info(f"Written main.dart to {main_path} (fallback)")
            
            # Write pubspec.yaml
            pubspec_content = implementation_result.get("pubspec_yaml")
            if pubspec_content:
                pubspec_path = f"{project_path}/pubspec.yaml"
                with open(pubspec_path, 'w', encoding='utf-8') as f:
                    f.write(pubspec_content)
                generated_files.append(pubspec_path)
                logger.info(f"Written pubspec.yaml to {pubspec_path} (fallback)")
            
            # Write additional files
            files = implementation_result.get("files", {})
            for file_path, content in files.items():
                if content and content.strip():  # Only write non-empty files
                    full_path = f"{project_path}/{file_path}"
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    generated_files.append(full_path)
                    logger.info(f"Written {file_path} to {full_path} (fallback)")
            
            logger.info(f"Generated {len(generated_files)} files using fallback method")
            return generated_files
            
        except Exception as e:
            logger.error(f"Fallback file writing also failed: {e}")
            return []

    async def _process_implementation_response(self, implementation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM implementation response, handling both JSON and raw text responses."""
        logger.info("Processing implementation response...")
        
        # Check if the response has an error (JSON parsing failed)
        if implementation_result.get("error") == "JSONDecodeError":
            logger.warning("LLM response JSON parsing failed, extracting code from raw response")
            raw_response = implementation_result.get("response", "")
            
            # Use robust JSON extraction
            from ..utils.json_utils import extract_json_from_llm_response
            extracted_json = extract_json_from_llm_response(raw_response)
            
            if extracted_json:
                logger.info("Successfully extracted JSON from raw response")
                return extracted_json
            else:
                # Fallback: extract code blocks manually
                logger.info("JSON extraction failed, using fallback code extraction")
                return self._extract_code_blocks_from_text(raw_response)
        
        # Check if we got a successful JSON response
        elif implementation_result.get("status") == "success" and isinstance(implementation_result.get("response"), dict):
            return implementation_result["response"]
        
        # Handle other successful cases
        elif "main_dart" in implementation_result or "files" in implementation_result:
            return implementation_result
        
        # Default fallback
        else:
            logger.warning("Unexpected implementation result format, using fallback")
            return self._create_fallback_implementation()

    def _extract_code_blocks_from_text(self, text: str) -> Dict[str, Any]:
        """Extract Dart code blocks and file information from raw text response."""
        import re
        import json
        
        result = {
            "files": {},
            "main_dart": "",
            "pubspec_yaml": "",
            "dependencies": []
        }
        
        # Strategy 1: Look for JSON structure in the text and try to extract it properly
        json_patterns = [
            r'\{[\s\S]*?"main_dart"[\s\S]*?\}(?=\s*```|\s*$)',  # Look for JSON ending before code blocks or end
            r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON within markdown code blocks
            r'\{[\s\S]*?"files"[\s\S]*?\}(?=\s*```|\s*[Ii]\'ll|\s*Let me|\s*$)',  # JSON ending before explanatory text
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, text, re.MULTILINE)
            for json_match in json_matches:
                try:
                    # Clean up the JSON string
                    json_str = json_match.strip()
                    if not json_str.startswith('{'):
                        continue
                    
                    # Try to fix truncated JSON by finding the last complete structure
                    if not json_str.endswith('}'):
                        # Find the last complete key-value pair or array
                        last_complete = json_str.rfind('},')
                        if last_complete > 0:
                            json_str = json_str[:last_complete + 1] + '}'
                        elif json_str.rfind('],') > 0:
                            last_array = json_str.rfind('],')
                            json_str = json_str[:last_array + 1] + '}'
                        else:
                            # Try to close any open strings and objects
                            json_str = json_str + '"}'
                    
                    # Remove any trailing incomplete content
                    json_str = re.sub(r',\s*\}', '}', json_str)
                    json_str = re.sub(r',\s*\]', ']', json_str)
                    
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict) and ('main_dart' in parsed_json or 'files' in parsed_json):
                        logger.info("Successfully extracted JSON structure from text")
                        return parsed_json
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse extracted JSON attempt: {e}")
                    continue
        
        # Strategy 2: Extract individual file sections more carefully
        # Look for file markers like "lib/screens/home_screen.dart": "content"
        file_pattern = r'"(lib/[^"]+\.dart|pubspec\.yaml)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        file_matches = re.findall(file_pattern, text, re.DOTALL)
        
        for file_path, content in file_matches:
            # Unescape the content
            content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            
            if file_path.endswith('main.dart'):
                result["main_dart"] = content
                result["files"]["lib/main.dart"] = content
            elif file_path.endswith('pubspec.yaml'):
                result["pubspec_yaml"] = content
                result["files"]["pubspec.yaml"] = content
            else:
                result["files"][file_path] = content
        
        # Strategy 3: Extract code blocks by file type/content hints
        if not result["files"]:
            # Find all code blocks
            code_block_pattern = r'```(?:dart|yaml)?\s*(?:\n?(?:// )?(?:File: )?([\w/_.]+))?\n(.*?)```'
            matches = re.findall(code_block_pattern, text, re.DOTALL)
            
            processed_files = set()
            
            for file_path, code_content in matches:
                code_content = code_content.strip()
                
                # Skip if we already processed this exact content
                content_hash = hash(code_content)
                if content_hash in processed_files:
                    continue
                processed_files.add(content_hash)
                
                if not file_path:
                    # Determine file type from content
                    if "void main(" in code_content and "runApp(" in code_content:
                        file_path = "lib/main.dart"
                    elif "name:" in code_content and "dependencies:" in code_content and "flutter:" in code_content:
                        file_path = "pubspec.yaml"
                    elif "class " in code_content and "extends State" in code_content:
                        # Extract class name for screen files
                        class_match = re.search(r'class\s+(\w+)', code_content)
                        if class_match:
                            class_name = class_match.group(1)
                            if "Screen" in class_name:
                                file_path = f"lib/screens/{class_name.lower()}.dart"
                            elif "Widget" in class_name or "State" in class_name:
                                file_path = f"lib/widgets/{class_name.lower().replace('state', '').replace('widget', '')}.dart"
                            else:
                                file_path = f"lib/widgets/{class_name.lower()}.dart"
                    elif "class " in code_content and ("Bloc" in code_content or "Cubit" in code_content):
                        class_match = re.search(r'class\s+(\w+)', code_content)
                        if class_match:
                            class_name = class_match.group(1)
                            file_path = f"lib/state/{class_name.lower()}.dart"
                    elif "class " in code_content and "Service" in code_content:
                        class_match = re.search(r'class\s+(\w+)', code_content)
                        if class_match:
                            class_name = class_match.group(1)
                            file_path = f"lib/services/{class_name.lower()}.dart"
                    elif "class " in code_content:
                        class_match = re.search(r'class\s+(\w+)', code_content)
                        if class_match:
                            class_name = class_match.group(1)
                            file_path = f"lib/models/{class_name.lower()}.dart"
                    else:
                        continue  # Skip unidentifiable content
                
                # Clean up file path
                file_path = file_path.strip("/ ")
                
                # Only add if content is substantial and not a duplicate
                if len(code_content) > 50 and file_path not in result["files"]:
                    if file_path.endswith("main.dart"):
                        result["main_dart"] = code_content
                        result["files"]["lib/main.dart"] = code_content
                    elif file_path.endswith("pubspec.yaml"):
                        result["pubspec_yaml"] = code_content
                        result["files"]["pubspec.yaml"] = code_content
                    else:
                        result["files"][file_path] = code_content
        
        # Strategy 4: Extract direct field values if they exist
        if not result["main_dart"]:
            main_dart_pattern = r'"main_dart"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            main_match = re.search(main_dart_pattern, text, re.DOTALL)
            if main_match:
                main_content = main_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                if len(main_content) > 50:
                    result["main_dart"] = main_content
                    result["files"]["lib/main.dart"] = main_content
        
        if not result["pubspec_yaml"]:
            pubspec_pattern = r'"pubspec_yaml"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            pubspec_match = re.search(pubspec_pattern, text, re.DOTALL)
            if pubspec_match:
                pubspec_content = pubspec_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                if "dependencies:" in pubspec_content and len(pubspec_content) > 50:
                    result["pubspec_yaml"] = pubspec_content
                    result["files"]["pubspec.yaml"] = pubspec_content
        
        logger.info(f"Extracted {len(result['files'])} files from text response")
        
        # Log the extracted files for debugging
        for file_path in result["files"].keys():
            logger.info(f"Extracted file: {file_path}")
        
        return result

