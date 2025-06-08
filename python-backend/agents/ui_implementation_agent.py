"""
UI Implementation Agent - Converts designs into Flutter UI code.
Specializes in widget composition, layouts, and visual implementation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent
from core.agent_types import AgentType, AgentResponse, WorkflowState, TaskStatus

logger = logging.getLogger(__name__)


class UIImplementationAgent(BaseAgent):
    """
    Specialized agent for UI implementation in Flutter.
    Converts wireframes and designs into Flutter widget code with proper styling and responsiveness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.UI_IMPLEMENTATION, config)
        
        # UI implementation configuration
        self.design_systems = config.get("design_systems", ["material", "cupertino"])
        self.responsive_breakpoints = config.get("responsive_breakpoints", {
            "mobile": 480,
            "tablet": 768,
            "desktop": 1024
        })
        self.theme_configuration = config.get("theme_configuration", {
            "color_scheme": "light",
            "typography": "default",
            "custom_components": True
        })
        
        # Widget library preferences
        self.widget_preferences = {
            "layout_widgets": ["Column", "Row", "Stack", "Container", "Padding"],
            "ui_widgets": ["Card", "ListTile", "AppBar", "BottomNavigationBar"],
            "input_widgets": ["TextField", "DropdownButton", "Checkbox", "Switch"],
            "display_widgets": ["Text", "Image", "Icon", "CircularProgressIndicator"]
        }
        
        # Code generation templates
        self.code_templates = {
            "screen_template": self._get_screen_template(),
            "widget_template": self._get_widget_template(),
            "theme_template": self._get_theme_template()
        }
        
        logger.info(f"UIImplementationAgent initialized with design systems: {self.design_systems}")
    
    def _define_capabilities(self) -> List[str]:
        return [
            "widget_tree_composition",
            "layout_implementation",
            "material_design_implementation",
            "cupertino_design_implementation",
            "responsive_layout_creation",
            "theme_application",
            "custom_widget_development",
            "ui_component_library_creation",
            "animation_integration",
            "gesture_handling_implementation",
            "form_validation_ui",
            "list_and_grid_layouts",
            "navigation_ui_implementation",
            "accessibility_widget_configuration"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process UI implementation tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'ui_implementation')
            
            logger.info(f"Processing {task_type} task for UI implementation")
            
            if task_type == "ui_implementation":
                return await self._handle_ui_implementation(state)
            elif task_type == "widget_creation":
                return await self._handle_widget_creation(state)
            elif task_type == "theme_implementation":
                return await self._handle_theme_implementation(state)
            elif task_type == "responsive_layout":
                return await self._handle_responsive_layout(state)
            elif task_type == "component_library":
                return await self._handle_component_library_creation(state)
            elif task_type == "screen_implementation":
                return await self._handle_screen_implementation(state)
            else:
                return await self._handle_generic_ui_task(state)
                
        except Exception as e:
            logger.error(f"Error processing task in UIImplementationAgent: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                message=f"UI implementation failed: {str(e)}",
                data={"error": str(e)},
                updated_state=state
            )
    
    async def _handle_ui_implementation(self, state: WorkflowState) -> AgentResponse:
        """Handle comprehensive UI implementation from wireframes."""
        try:
            wireframes = state.project_context.get("wireframes", {})
            design_system = state.project_context.get("design_system", "material")
            target_platforms = state.project_context.get("target_platforms", ["android", "ios"])
            
            # Generate screen implementations
            screen_implementations = await self._generate_screen_implementations(
                wireframes, design_system, target_platforms
            )
            
            # Create reusable components
            reusable_components = await self._create_reusable_components(screen_implementations)
            
            # Generate theme configuration
            theme_config = await self._generate_theme_configuration(design_system)
            
            # Create responsive layout utilities
            responsive_utilities = await self._create_responsive_utilities()
            
            # Generate navigation UI
            navigation_ui = await self._generate_navigation_ui(wireframes, design_system)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="UI implementation completed successfully",
                data={
                    "screen_implementations": screen_implementations,
                    "reusable_components": reusable_components,
                    "theme_configuration": theme_config,
                    "responsive_utilities": responsive_utilities,
                    "navigation_ui": navigation_ui,
                    "implementation_guidelines": await self._generate_implementation_guidelines()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"UI implementation failed: {e}")
            raise
    
    async def _handle_widget_creation(self, state: WorkflowState) -> AgentResponse:
        """Handle creation of custom Flutter widgets."""
        try:
            widget_specifications = state.project_context.get("widget_specifications", [])
            design_system = state.project_context.get("design_system", "material")
            
            created_widgets = []
            for spec in widget_specifications:
                widget_code = await self._generate_custom_widget(spec, design_system)
                created_widgets.append({
                    "widget_name": spec.get("name", "CustomWidget"),
                    "widget_code": widget_code,
                    "usage_example": await self._generate_widget_usage_example(spec),
                    "documentation": await self._generate_widget_documentation(spec)
                })
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Custom widgets created successfully",
                data={
                    "created_widgets": created_widgets,
                    "widget_library_structure": await self._generate_widget_library_structure()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Widget creation failed: {e}")
            raise
    
    async def _handle_theme_implementation(self, state: WorkflowState) -> AgentResponse:
        """Handle theme and styling implementation."""
        try:
            design_tokens = state.project_context.get("design_tokens", {})
            design_system = state.project_context.get("design_system", "material")
            brand_colors = state.project_context.get("brand_colors", {})
            
            # Generate theme data
            theme_data = await self._generate_theme_data(design_tokens, brand_colors, design_system)
            
            # Create color schemes
            color_schemes = await self._create_color_schemes(brand_colors)
            
            # Generate typography theme
            typography_theme = await self._generate_typography_theme(design_tokens)
            
            # Create component themes
            component_themes = await self._create_component_themes(design_system)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Theme implementation completed",
                data={
                    "theme_data": theme_data,
                    "color_schemes": color_schemes,
                    "typography_theme": typography_theme,
                    "component_themes": component_themes,
                    "theme_usage_guide": await self._generate_theme_usage_guide()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Theme implementation failed: {e}")
            raise
    
    async def _handle_responsive_layout(self, state: WorkflowState) -> AgentResponse:
        """Handle responsive layout implementation."""
        try:
            breakpoints = state.project_context.get("responsive_breakpoints", self.responsive_breakpoints)
            target_platforms = state.project_context.get("target_platforms", ["android", "ios"])
            
            # Generate responsive layout utilities
            responsive_utils = await self._generate_responsive_utilities(breakpoints)
            
            # Create adaptive layouts
            adaptive_layouts = await self._create_adaptive_layouts(breakpoints, target_platforms)
            
            # Generate screen size helpers
            screen_helpers = await self._generate_screen_size_helpers()
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Responsive layout implementation completed",
                data={
                    "responsive_utilities": responsive_utils,
                    "adaptive_layouts": adaptive_layouts,
                    "screen_helpers": screen_helpers,
                    "responsive_guidelines": await self._generate_responsive_guidelines()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Responsive layout implementation failed: {e}")
            raise
    
    async def _handle_component_library_creation(self, state: WorkflowState) -> AgentResponse:
        """Handle creation of a comprehensive component library."""
        try:
            component_specs = state.project_context.get("component_specifications", [])
            design_system = state.project_context.get("design_system", "material")
            
            # Generate base components
            base_components = await self._generate_base_components(design_system)
            
            # Create composite components
            composite_components = await self._generate_composite_components(component_specs)
            
            # Generate component documentation
            component_docs = await self._generate_component_documentation(
                base_components + composite_components
            )
            
            # Create component showcase
            component_showcase = await self._create_component_showcase()
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Component library created successfully",
                data={
                    "base_components": base_components,
                    "composite_components": composite_components,
                    "component_documentation": component_docs,
                    "component_showcase": component_showcase,
                    "library_structure": await self._generate_library_structure()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Component library creation failed: {e}")
            raise
    
    async def _handle_screen_implementation(self, state: WorkflowState) -> AgentResponse:
        """Handle implementation of specific screens."""
        try:
            screen_wireframes = state.project_context.get("screen_wireframes", {})
            design_system = state.project_context.get("design_system", "material")
            
            implemented_screens = {}
            for screen_name, wireframe in screen_wireframes.items():
                screen_code = await self._implement_screen(screen_name, wireframe, design_system)
                implemented_screens[screen_name] = screen_code
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Screen implementation completed",
                data={
                    "implemented_screens": implemented_screens,
                    "screen_navigation": await self._generate_screen_navigation_code(),
                    "screen_tests": await self._generate_screen_tests(implemented_screens)
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Screen implementation failed: {e}")
            raise
    
    # Helper methods for UI implementation
    
    async def _generate_screen_implementations(self, wireframes: Dict, design_system: str, platforms: List[str]) -> Dict[str, Any]:
        """Generate Flutter code for screen implementations."""
        implementations = {}
        
        for screen_name, wireframe_data in wireframes.items():
            screen_code = await self._create_screen_code(screen_name, wireframe_data, design_system)
            implementations[screen_name] = {
                "dart_code": screen_code,
                "imports": self._generate_imports(design_system),
                "dependencies": self._get_screen_dependencies(wireframe_data)
            }
        
        return implementations
    
    async def _create_screen_code(self, screen_name: str, wireframe: Dict, design_system: str) -> str:
        """Create Flutter code for a specific screen."""
        class_name = self._to_pascal_case(screen_name)
        
        # Generate basic screen structure
        screen_code = f"""
import 'package:flutter/material.dart';

class {class_name}Screen extends StatefulWidget {{
  const {class_name}Screen({{Key? key}}) : super(key: key);

  @override
  State<{class_name}Screen> createState() => _{class_name}ScreenState();
}}

class _{class_name}ScreenState extends State<{class_name}Screen> {{
  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: {self._generate_app_bar_code(wireframe, design_system)},
      body: {self._generate_body_code(wireframe, design_system)},
      {self._generate_additional_scaffold_properties(wireframe)}
    );
  }}
  
  {self._generate_helper_methods(wireframe)}
}}
"""
        return screen_code
    
    async def _create_reusable_components(self, screen_implementations: Dict) -> List[Dict[str, Any]]:
        """Identify and create reusable components from screen implementations."""
        components = []
        
        # Common components to generate
        common_components = [
            "CustomButton",
            "CustomCard",
            "CustomTextField",
            "CustomAppBar",
            "LoadingIndicator",
            "ErrorWidget",
            "EmptyStateWidget"
        ]
        
        for component_name in common_components:
            component_code = await self._generate_component_code(component_name)
            components.append({
                "name": component_name,
                "code": component_code,
                "usage_example": self._generate_component_usage(component_name),
                "props": self._get_component_props(component_name)
            })
        
        return components
    
    async def _generate_theme_configuration(self, design_system: str) -> Dict[str, Any]:
        """Generate theme configuration for the Flutter app."""
        if design_system == "material":
            return await self._generate_material_theme()
        elif design_system == "cupertino":
            return await self._generate_cupertino_theme()
        else:
            return await self._generate_custom_theme()
    
    async def _generate_material_theme(self) -> Dict[str, Any]:
        """Generate Material Design theme configuration."""
        return {
            "theme_code": """
ThemeData(
  useMaterial3: true,
  colorScheme: ColorScheme.fromSeed(
    seedColor: const Color(0xFF6750A4),
    brightness: Brightness.light,
  ),
  textTheme: const TextTheme(
    displayLarge: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
    displayMedium: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
    displaySmall: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
    headlineLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
    headlineMedium: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
    headlineSmall: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
    bodyLarge: TextStyle(fontSize: 16),
    bodyMedium: TextStyle(fontSize: 14),
    bodySmall: TextStyle(fontSize: 12),
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      minimumSize: const Size(88, 36),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(2)),
      ),
    ),
  ),
  inputDecorationTheme: const InputDecorationTheme(
    filled: true,
    border: OutlineInputBorder(),
    contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
  ),
)
""",
            "dark_theme_code": """
ThemeData(
  useMaterial3: true,
  colorScheme: ColorScheme.fromSeed(
    seedColor: const Color(0xFF6750A4),
    brightness: Brightness.dark,
  ),
)
"""
        }
    
    def _generate_imports(self, design_system: str) -> List[str]:
        """Generate necessary imports for the screen."""
        base_imports = [
            "package:flutter/material.dart"
        ]
        
        if design_system == "cupertino":
            base_imports.append("package:flutter/cupertino.dart")
        
        return base_imports
    
    def _generate_app_bar_code(self, wireframe: Dict, design_system: str) -> str:
        """Generate AppBar code based on wireframe."""
        title = wireframe.get("title", "Screen")
        has_back_button = wireframe.get("has_back_button", True)
        
        if design_system == "material":
            return f"""AppBar(
        title: const Text('{title}'),
        centerTitle: true,
        automaticallyImplyLeading: {str(has_back_button).lower()},
      )"""
        else:
            return f"""CupertinoNavigationBar(
        middle: const Text('{title}'),
        automaticallyImplyLeading: {str(has_back_button).lower()},
      )"""
    
    def _generate_body_code(self, wireframe: Dict, design_system: str) -> str:
        """Generate body code based on wireframe layout."""
        layout_type = wireframe.get("layout_type", "column")
        
        if layout_type == "column":
            return """SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // TODO: Add your widgets here
              const Text('Content goes here'),
            ],
          ),
        ),
      )"""
        elif layout_type == "list":
            return """ListView.builder(
        itemCount: 10, // TODO: Replace with actual item count
        itemBuilder: (context, index) {
          return ListTile(
            title: Text('Item \$index'),
            // TODO: Customize list item
          );
        },
      )"""
        else:
            return """const Center(
        child: Text('Content goes here'),
      )"""
    
    def _generate_additional_scaffold_properties(self, wireframe: Dict) -> str:
        """Generate additional Scaffold properties."""
        properties = []
        
        if wireframe.get("has_floating_action_button", False):
            properties.append("floatingActionButton: FloatingActionButton(onPressed: () {}, child: const Icon(Icons.add))")
        
        if wireframe.get("has_bottom_navigation", False):
            properties.append("""bottomNavigationBar: BottomNavigationBar(
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.search), label: 'Search'),
        ],
      )""")
        
        return ",\n      ".join(properties)
    
    def _generate_helper_methods(self, wireframe: Dict) -> str:
        """Generate helper methods for the screen."""
        return """
  void _onButtonPressed() {
    // TODO: Implement button press logic
  }
  
  void _showDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Dialog Title'),
        content: const Text('Dialog content'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
"""
    
    def _to_pascal_case(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        return ''.join(word.capitalize() for word in snake_str.split('_'))
    
    def _get_screen_template(self) -> str:
        """Get basic screen template."""
        return """
import 'package:flutter/material.dart';

class {class_name}Screen extends StatefulWidget {{
  const {class_name}Screen({{Key? key}}) : super(key: key);

  @override
  State<{class_name}Screen> createState() => _{class_name}ScreenState();
}}

class _{class_name}ScreenState extends State<{class_name}Screen> {{
  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      appBar: {app_bar},
      body: {body},
    );
  }}
}}
"""
    
    def _get_widget_template(self) -> str:
        """Get basic widget template."""
        return """
import 'package:flutter/material.dart';

class {widget_name} extends StatelessWidget {{
  const {widget_name}({{Key? key}}) : super(key: key);

  @override
  Widget build(BuildContext context) {{
    return {widget_body};
  }}
}}
"""
    
    def _get_theme_template(self) -> str:
        """Get theme template."""
        return """
import 'package:flutter/material.dart';

class AppTheme {{
  static ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
  );
  
  static ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Colors.blue, 
      brightness: Brightness.dark
    ),
  );
}}
"""
    
    async def _handle_generic_ui_task(self, state: WorkflowState) -> AgentResponse:
        """Handle generic UI implementation tasks."""
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            message="Generic UI task completed",
            data={"task_type": "generic_ui"},
            updated_state=state
        )
    
    # Additional placeholder methods for completeness
    async def _create_responsive_utilities(self) -> Dict[str, Any]:
        return {"utilities": "responsive_helpers"}
    
    async def _generate_navigation_ui(self, wireframes: Dict, design_system: str) -> Dict[str, Any]:
        return {"navigation_ui": "navigation_components"}
    
    async def _generate_implementation_guidelines(self) -> List[str]:
        return ["Follow Flutter best practices", "Use consistent naming conventions"]
    
    async def _generate_custom_widget(self, spec: Dict, design_system: str) -> str:
        return "// Custom widget code here"
    
    async def _generate_widget_usage_example(self, spec: Dict) -> str:
        return "// Widget usage example"
    
    async def _generate_widget_documentation(self, spec: Dict) -> str:
        return "/// Widget documentation"
