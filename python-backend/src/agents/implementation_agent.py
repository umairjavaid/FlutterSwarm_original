"""
Implementation Agent for FlutterSwarm Multi-Agent System.

This agent specializes in code generation, feature development,
and implementation of Flutter applications based on specifications.
"""

import json
import uuid
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import yaml

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..core.tools.tool_registry import ToolRegistry
from ..models.agent_models import AgentMessage, TaskResult
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..models.code_models import (
    CodeGeneration, CodeUnderstanding, ProjectContext as CodeProjectContext,
    GenerationEntry, CodePattern, ProjectStructure, CodeType, ArchitectureStyle,
    CodeConvention, CodeAnalysisResult, CodeExample, IntegrationPlan,
    GeneratedCode, PlacementResult, UpdateResult
)
from ..models.tool_models import ToolUsagePlan, ToolOperation, TaskOutcome, ToolStatus
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
        
        # Tool integration
        self.tool_registry = ToolRegistry.instance()
        self.flutter_tool = None
        self.file_tool = None
        self.process_tool = None
        
        # NEW: Project-aware code generation attributes
        self.project_context: Optional[CodeProjectContext] = None
        self.code_patterns: Dict[str, CodePattern] = {}
        self.project_structure: Optional[ProjectStructure] = None
        self.generation_history: List[GenerationEntry] = []
        
        # Project awareness (legacy - keeping for compatibility)
        self.current_project_context = None
        self.project_structure_cache = {}
        self.code_understanding_cache = {}
        
        # Development session state
        self.active_hot_reload_session = None
        self.last_build_artifacts = {}
        
        logger.info(f"Implementation Agent {self.agent_id} initialized with tool integration")

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

    async def initialize_tools(self):
        """Initialize and discover available tools for implementation work."""
        await self.discover_available_tools()
        
        # Get specific tools we need
        self.flutter_tool = self.tool_registry.get_tool("flutter_sdk_tool")
        self.file_tool = self.tool_registry.get_tool("file_system_tool")
        self.process_tool = self.tool_registry.get_tool("process_tool")
        
        if not self.flutter_tool:
            logger.warning("Flutter SDK tool not available - limited functionality")
        
        # Understand tool capabilities
        if self.flutter_tool:
            await self.analyze_tool_capability(self.flutter_tool)

    async def understand_existing_code(self, file_path: str) -> CodeUnderstanding:
        """
        Analyze existing Flutter code to understand patterns, conventions, and structure.
        
        This method uses LLM reasoning to understand code patterns and architectural
        decisions in existing Flutter projects, enabling intelligent code generation
        that follows project conventions.
        
        Args:
            file_path: Path to the Flutter file to analyze
            
        Returns:
            CodeUnderstanding: Comprehensive analysis of the code file
        """
        try:
            # Validate file path and ensure it exists
            if not file_path or not Path(file_path).exists():
                raise ValueError(f"File path does not exist: {file_path}")
            
            # Check cache first
            if file_path in self.code_understanding_cache:
                cached_understanding = self.code_understanding_cache[file_path]
                # Return cached if recent (within last hour)
                if (datetime.utcnow() - cached_understanding.analyzed_at).seconds < 3600:
                    return cached_understanding
            
            logger.info(f"Analyzing existing code: {file_path}")
            
            # Use file system tool to read the code
            if not self.file_tool:
                await self.initialize_tools()
            
            read_result = await self.use_tool(
                "file_system",
                "read_file",
                {"file_path": file_path},
                f"Reading Flutter code file for analysis: {file_path}"
            )
            
            if read_result.status != ToolStatus.SUCCESS:
                raise Exception(f"Failed to read file: {read_result.error_message}")
            
            file_content = read_result.data.get("content", "")
            file_info = read_result.data
            
            # Determine code type from file structure and content
            code_type = self._detect_code_type(file_path, file_content)
            
            # Use LLM to analyze code patterns and structure
            analysis_prompt = self._create_code_analysis_prompt(
                file_path, file_content, code_type
            )
            
            llm_analysis = await self._llm_call(
                system_prompt=await self._get_code_analysis_system_prompt(),
                user_prompt=analysis_prompt,
                context={
                    "file_path": file_path,
                    "code_type": code_type.value if code_type else "unknown",
                    "file_size": len(file_content)
                },
                structured_output=True
            )
            
            # Parse LLM analysis and extract structured information
            structure = self._extract_code_structure(file_content, llm_analysis)
            patterns = self._extract_code_patterns(file_content, llm_analysis)
            conventions = self._extract_conventions(file_content, llm_analysis)
            dependencies = self._extract_dependencies(file_content, file_info)
            relationships = self._extract_relationships(file_content, llm_analysis)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(file_content)
            
            # Generate quality indicators
            quality_indicators = self._assess_code_quality(
                file_content, llm_analysis, complexity_metrics
            )
            
            # Generate improvement suggestions
            suggestions = self._generate_code_suggestions(
                file_content, llm_analysis, quality_indicators
            )
            
            # Create CodeUnderstanding object
            understanding = CodeUnderstanding(
                file_path=file_path,
                code_type=code_type,
                structure=structure,
                patterns=patterns,
                conventions=conventions,
                dependencies=dependencies,
                relationships=relationships,
                complexity_metrics=complexity_metrics,
                quality_indicators=quality_indicators,
                suggestions=suggestions,
                analyzed_at=datetime.utcnow()
            )
            
            # Cache the result
            self.code_understanding_cache[file_path] = understanding
            
            # Update project-level pattern knowledge
            await self._update_project_patterns(patterns)
            
            logger.info(
                f"Code analysis completed for {file_path}: "
                f"found {len(patterns)} patterns, {len(dependencies)} dependencies"
            )
            
            return understanding
            
        except Exception as e:
            logger.error(f"Failed to understand code in {file_path}: {e}")
            # Return minimal understanding with error info
            return CodeUnderstanding(
                file_path=file_path,
                structure={"error": str(e)},
                patterns=[],
                conventions={},
                dependencies=[],
                relationships={},
                complexity_metrics={},
                quality_indicators={"analysis_error": str(e)},
                suggestions=[f"Could not analyze code: {str(e)}"],
                analyzed_at=datetime.utcnow()
            )
    
    def _detect_code_type(self, file_path: str, content: str) -> Optional[CodeType]:
        """Detect the type of Flutter code from file path and content."""
        path = Path(file_path)
        
        # Analyze file path patterns
        if "test" in str(path):
            return CodeType.TEST
        
        if path.suffix != ".dart":
            return None
            
        # Analyze content patterns
        if re.search(r'class\s+\w+\s+extends\s+StatelessWidget', content):
            return CodeType.WIDGET
        elif re.search(r'class\s+\w+\s+extends\s+StatefulWidget', content):
            return CodeType.WIDGET
        elif re.search(r'class\s+\w+\s+extends\s+(Bloc|Cubit)', content):
            return CodeType.BLOC if "Bloc" in content else CodeType.CUBIT
        elif re.search(r'class\s+\w+\s+with\s+ChangeNotifier', content):
            return CodeType.PROVIDER
        elif re.search(r'class\s+\w+Repository', content):
            return CodeType.REPOSITORY
        elif re.search(r'class\s+\w+Service', content):
            return CodeType.SERVICE
        elif re.search(r'class\s+\w+Controller', content):
            return CodeType.CONTROLLER
        elif "main(" in content and "runApp(" in content:
            return CodeType.CONFIGURATION
        elif re.search(r'class\s+\w+\s*{', content):
            return CodeType.MODEL
        
        return CodeType.UTILITY
    
    def _create_code_analysis_prompt(
        self, file_path: str, content: str, code_type: Optional[CodeType]
    ) -> str:
        """Create a prompt for LLM analysis of Flutter code."""
        return f"""
Analyze this Flutter/Dart code file and provide detailed insights:

FILE: {file_path}
TYPE: {code_type.value if code_type else 'unknown'}

CODE:
```dart
{content[:4000]}  # Truncate for token limits
```

Please provide a comprehensive analysis including:

1. STRUCTURE ANALYSIS:
   - Main classes, functions, and their purposes
   - Widget hierarchy (if applicable)  
   - State management approach
   - Architecture pattern used

2. CODE PATTERNS:
   - Naming conventions used
   - File organization patterns
   - Import/export patterns
   - Error handling patterns
   - State management patterns

3. CONVENTIONS:
   - Coding style and formatting
   - Documentation practices
   - Comment patterns
   - Variable naming conventions

4. DEPENDENCIES & RELATIONSHIPS:
   - External package dependencies
   - Internal file dependencies
   - Widget composition relationships
   - Data flow patterns

5. QUALITY ASSESSMENT:
   - Code complexity
   - Maintainability indicators
   - Performance considerations
   - Security considerations

6. SUGGESTIONS:
   - Potential improvements
   - Refactoring opportunities
   - Performance optimizations
   - Best practice recommendations

Respond with structured, detailed analysis focused on understanding the developer's intent and patterns.
"""
    
    async def _get_code_analysis_system_prompt(self) -> str:
        """Get system prompt for code analysis."""
        return """
You are an expert Flutter/Dart code analyst with deep understanding of:

- Flutter framework architecture and best practices
- Dart language features and conventions
- Common Flutter design patterns (BLoC, Provider, Riverpod, etc.)
- Mobile app development patterns
- Code quality and maintainability principles

Your role is to analyze Flutter code with the expertise of a senior Flutter developer,
identifying patterns, conventions, and architectural decisions that should be followed
in future code generation.

Focus on understanding the developer's intent, established patterns, and project conventions
so that new code can be generated that seamlessly integrates with the existing codebase.

Provide detailed, accurate analysis that captures both explicit and implicit patterns.
"""

    def _extract_code_structure(
        self, content: str, llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structural information from code analysis."""
        structure = {}
        
        # Parse classes
        class_matches = re.findall(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', content)
        if class_matches:
            structure["classes"] = [
                {"name": match[0], "extends": match[1] if match[1] else None}
                for match in class_matches
            ]
        
        # Parse functions
        function_matches = re.findall(r'(?:Future<\w+>|void|\w+)\s+(\w+)\s*\(', content)
        if function_matches:
            structure["functions"] = function_matches
        
        # Parse imports
        import_matches = re.findall(r"import\s+['\"]([^'\"]+)['\"]", content)
        if import_matches:
            structure["imports"] = import_matches
        
        # Add LLM analysis if available
        if llm_analysis and "structure" in llm_analysis:
            structure.update(llm_analysis["structure"])
            
        return structure
    
    def _extract_code_patterns(
        self, content: str, llm_analysis: Dict[str, Any]
    ) -> List[CodePattern]:
        """Extract code patterns from analysis."""
        patterns = []
        
        # Widget patterns
        if "StatelessWidget" in content:
            patterns.append(CodePattern(
                pattern_id=f"stateless_widget_{uuid.uuid4().hex[:8]}",
                pattern_type="widget_pattern",
                description="Uses StatelessWidget for UI components",
                examples=[],
                frequency=content.count("StatelessWidget"),
                confidence=0.9
            ))
        
        if "StatefulWidget" in content:
            patterns.append(CodePattern(
                pattern_id=f"stateful_widget_{uuid.uuid4().hex[:8]}",
                pattern_type="widget_pattern", 
                description="Uses StatefulWidget for stateful UI components",
                examples=[],
                frequency=content.count("StatefulWidget"),
                confidence=0.9
            ))
        
        # Add patterns from LLM analysis
        if llm_analysis and "patterns" in llm_analysis:
            for pattern_data in llm_analysis["patterns"]:
                if isinstance(pattern_data, dict):
                    patterns.append(CodePattern(
                        pattern_id=f"llm_pattern_{uuid.uuid4().hex[:8]}",
                        pattern_type=pattern_data.get("type", "unknown"),
                        description=pattern_data.get("description", ""),
                        examples=pattern_data.get("examples", []),
                        frequency=pattern_data.get("frequency", 1),
                        confidence=pattern_data.get("confidence", 0.7)
                    ))
        
        return patterns
    
    def _extract_conventions(
        self, content: str, llm_analysis: Dict[str, Any]
    ) -> Dict[CodeConvention, str]:
        """Extract coding conventions from analysis."""
        conventions = {}
        
        # Naming convention analysis
        if re.search(r'[a-z]+[A-Z]', content):  # camelCase
            conventions[CodeConvention.NAMING_CONVENTION] = "camelCase"
        elif re.search(r'[a-z]+_[a-z]+', content):  # snake_case
            conventions[CodeConvention.NAMING_CONVENTION] = "snake_case"
        
        # Import style
        if "package:" in content:
            conventions[CodeConvention.IMPORT_STYLE] = "package_imports"
        
        # Add LLM-detected conventions
        if llm_analysis and "conventions" in llm_analysis:
            for conv_key, conv_value in llm_analysis["conventions"].items():
                try:
                    convention_enum = CodeConvention(conv_key)
                    conventions[convention_enum] = conv_value
                except ValueError:
                    # Skip unknown conventions
                    pass
        
        return conventions
    
    def _extract_dependencies(
        self, content: str, file_info: Dict[str, Any]
    ) -> List[str]:
        """Extract dependencies from imports and analysis."""
        dependencies = []
        
        # Extract from imports
        import_matches = re.findall(r"import\s+['\"]([^'\"]+)['\"]", content)
        for import_path in import_matches:
            if import_path.startswith("package:"):
                package_name = import_path.split("/")[0].replace("package:", "")
                if package_name not in dependencies:
                    dependencies.append(package_name)
            elif import_path.startswith("dart:"):
                dart_lib = import_path.replace("dart:", "")
                if dart_lib not in dependencies:
                    dependencies.append(f"dart:{dart_lib}")
        
        return dependencies
    
    def _extract_relationships(
        self, content: str, llm_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract code relationships from analysis."""
        relationships = {}
        
        # Widget composition relationships
        widget_matches = re.findall(r'(\w+)\(\s*child:', content)
        if widget_matches:
            relationships["widget_composition"] = widget_matches
        
        # Class inheritance relationships  
        extends_matches = re.findall(r'class\s+(\w+)\s+extends\s+(\w+)', content)
        if extends_matches:
            relationships["inheritance"] = [f"{child} extends {parent}" for child, parent in extends_matches]
        
        # Add LLM analysis
        if llm_analysis and "relationships" in llm_analysis:
            relationships.update(llm_analysis["relationships"])
            
        return relationships
    
    def _calculate_complexity_metrics(self, content: str) -> Dict[str, float]:
        """Calculate code complexity metrics."""
        metrics = {}
        
        lines = content.split('\n')
        metrics["lines_of_code"] = len([line for line in lines if line.strip()])
        metrics["comment_ratio"] = len([line for line in lines if line.strip().startswith('//')]) / len(lines)
        
        # Cyclomatic complexity (simplified)
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', '&&', '||']
        complexity_count = sum(content.count(keyword) for keyword in complexity_keywords)
        metrics["cyclomatic_complexity"] = complexity_count
        
        # Nesting depth (simplified)
        max_depth = 0
        current_depth = 0
        for char in content:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        metrics["max_nesting_depth"] = max_depth
        
        return metrics
    
    def _assess_code_quality(
        self, content: str, llm_analysis: Dict[str, Any], metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess code quality indicators."""
        indicators = {}
        
        # Basic quality checks
        indicators["has_documentation"] = "///" in content or "/**" in content
        indicators["has_error_handling"] = "try" in content and "catch" in content
        indicators["complexity_score"] = min(10, max(1, 10 - metrics.get("cyclomatic_complexity", 0) / 5))
        
        # Add LLM quality assessment
        if llm_analysis and "quality" in llm_analysis:
            indicators.update(llm_analysis["quality"])
            
        return indicators
    
    def _generate_code_suggestions(
        self, content: str, llm_analysis: Dict[str, Any], quality: Dict[str, Any]
    ) -> List[str]:
        """Generate code improvement suggestions."""
        suggestions = []
        
        # Basic suggestions based on metrics
        if not quality.get("has_documentation"):
            suggestions.append("Add documentation comments for better code understanding")
        
        if not quality.get("has_error_handling"):
            suggestions.append("Consider adding error handling for robustness")
        
        if quality.get("complexity_score", 10) < 5:
            suggestions.append("Consider refactoring to reduce complexity")
        
        # Add LLM suggestions
        if llm_analysis and "suggestions" in llm_analysis:
            if isinstance(llm_analysis["suggestions"], list):
                suggestions.extend(llm_analysis["suggestions"])
        
        return suggestions
    
    async def _update_project_patterns(self, patterns: List[CodePattern]) -> None:
        """Update project-level pattern knowledge."""
        for pattern in patterns:
            if pattern.pattern_id in self.code_patterns:
                # Update existing pattern
                existing = self.code_patterns[pattern.pattern_id]
                existing.frequency += pattern.frequency
                existing.confidence = (existing.confidence + pattern.confidence) / 2
            else:
                # Add new pattern
                self.code_patterns[pattern.pattern_id] = pattern

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

    async def generate_contextual_code(
        self, 
        feature_request: str, 
        project_context: "ProjectContext"
    ) -> CodeGeneration:
        """
        Generate intelligent, contextual code that seamlessly integrates with existing project structure.
        
        This is the main entry point for intelligent code generation that:
        1. Analyzes existing project structure and patterns
        2. Identifies architectural style and conventions
        3. Finds similar existing code for reference
        4. Plans integration with minimal disruption
        5. Generates code matching project style
        6. Validates syntax, imports, and compatibility
        
        Args:
            feature_request: Natural language description of what to implement
            project_context: Complete project context information
            
        Returns:
            CodeGeneration: Generated code with integration plan and metadata
        """
        generation_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting contextual code generation for: {feature_request}")
            
            # Step 1: Analyze existing project structure
            project_structure = await self._analyze_project_structure(project_context.path)
            
            # Step 2: Identify architectural style from existing code
            architectural_style = await self._identify_architectural_style(project_context)
            
            # Step 3: Find similar existing code for reference
            similar_code = await self._find_similar_code(feature_request, project_context)
            
            # Step 4: Plan code integration
            integration_plan = await self._plan_code_integration(
                feature_request, project_structure, architectural_style
            )
            
            # Step 5: Generate contextual code
            generated_code = await self._generate_code_with_context(
                feature_request=feature_request,
                project_context=project_context,
                architectural_style=architectural_style,
                similar_code=similar_code,
                integration_plan=integration_plan
            )
            
            # Step 6: Validate generated code
            validation_result = await self._validate_generated_code(
                generated_code, project_context, integration_plan
            )
            
            # Create CodeGeneration result
            code_generation = CodeGeneration(
                generation_id=generation_id,
                request_description=feature_request,
                generated_code=generated_code.get("files", {}),
                target_files=integration_plan.new_files + integration_plan.affected_files,
                dependencies=integration_plan.dependencies_to_add,
                integration_points=[point.__dict__ for point in integration_plan.integration_points],
                imports_required=generated_code.get("imports", []),
                configuration_changes=integration_plan.configuration_changes,
                test_requirements=integration_plan.testing_requirements,
                documentation=generated_code.get("documentation"),
                metadata={
                    "architectural_style": architectural_style.value if architectural_style else None,
                    "integration_plan_id": integration_plan.plan_id,
                    "similar_code_count": len(similar_code),
                    "validation_result": validation_result,
                    "complexity_estimate": integration_plan.estimated_complexity
                }
            )
            
            # Add to generation history
            generation_entry = GenerationEntry(
                entry_id=generation_id,
                timestamp=datetime.utcnow(),
                request=feature_request,
                generated_code=code_generation,
                files_affected=integration_plan.affected_files + [
                    file["path"] for file in integration_plan.new_files
                ],
                success=validation_result.get("valid", True),
                error_message=validation_result.get("error"),
                agent_id=self.agent_id,
                project_context=project_context.__dict__ if hasattr(project_context, '__dict__') else None
            )
            
            self.generation_history.append(generation_entry)
            
            logger.info(
                f"Contextual code generation completed: "
                f"{len(generated_code.get('files', {}))} files, "
                f"{code_generation.get_total_lines()} lines"
            )
            
            return code_generation
            
        except Exception as e:
            logger.error(f"Contextual code generation failed: {e}")
            
            # Return error result
            return CodeGeneration(
                generation_id=generation_id,
                request_description=feature_request,
                generated_code={},
                metadata={
                    "error": str(e),
                    "generation_failed": True
                }
            )

    async def _analyze_project_structure(self, project_path: str) -> ProjectStructure:
        """
        Analyze project structure to understand organization patterns and conventions.
        
        Uses file_system_tool to scan the project and identify:
        - Directory structure and organization
        - File naming patterns
        - Architecture layers
        - Module dependencies
        - Configuration files
        
        Args:
            project_path: Root path of the Flutter project
            
        Returns:
            ProjectStructure: Analyzed project structure information
        """
        try:
            logger.info(f"Analyzing project structure: {project_path}")
            
            # Use file_system_tool to scan project directory
            if not self.file_tool:
                await self.initialize_tools()
            
            # Get project directory structure
            structure_result = await self.use_tool(
                "file_system",
                "list_directory",
                {
                    "path": project_path,
                    "recursive": True,
                    "include_hidden": False,
                    "filter_patterns": ["*.dart", "*.yaml", "*.json"]
                },
                f"Scanning project structure for analysis: {project_path}"
            )
            
            if structure_result.status != ToolStatus.SUCCESS:
                raise Exception(f"Failed to scan project structure: {structure_result.error_message}")
            
            file_listing = structure_result.data.get("files", [])
            
            # Build structure map
            structure_map = self._build_structure_map(file_listing)
            
            # Identify key directories
            key_directories = self._identify_key_directories(structure_map)
            
            # Identify architecture layers
            architecture_layers = self._identify_architecture_layers(key_directories, file_listing)
            
            # Analyze module dependencies
            module_dependencies = await self._analyze_module_dependencies(file_listing, project_path)
            
            # Identify entry points
            entry_points = self._identify_entry_points(file_listing)
            
            # Find configuration files
            configuration_files = self._find_configuration_files(file_listing)
            
            return ProjectStructure(
                root_path=project_path,
                structure_map=structure_map,
                key_directories=key_directories,
                architecture_layers=architecture_layers,
                module_dependencies=module_dependencies,
                entry_points=entry_points,
                configuration_files=configuration_files,
                analyzed_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Project structure analysis failed: {e}")
            # Return minimal structure
            return ProjectStructure(
                root_path=project_path,
                structure_map={"error": str(e)},
                analyzed_at=datetime.utcnow()
            )

    async def _identify_architectural_style(self, project_context: "ProjectContext") -> Optional[ArchitectureStyle]:
        """
        Identify the architectural style used in the project by analyzing existing code patterns.
        
        Uses LLM reasoning to examine code structure, dependencies, and patterns to determine
        the primary architectural approach being used.
        
        Args:
            project_context: Project context with dependencies and structure info
            
        Returns:
            ArchitectureStyle: Detected architectural style or None if unclear
        """
        try:
            logger.info("Identifying architectural style from project patterns")
            
            # Gather evidence from project structure and dependencies
            evidence = await self._gather_architectural_evidence(project_context)
            
            # Use LLM to analyze architectural patterns
            analysis_prompt = f"""
            Analyze this Flutter project to identify its primary architectural pattern:
            
            Project Evidence:
            {json.dumps(evidence, indent=2)}
            
            Consider:
            1. Directory structure and organization
            2. State management dependencies and patterns
            3. File naming conventions
            4. Code organization principles
            5. Dependency injection patterns
            
            Identify the primary architecture pattern from:
            - Clean Architecture (layers: presentation, domain, data)
            - BLoC Pattern (business logic components)
            - Provider Pattern (provider state management)
            - Riverpod Pattern (riverpod providers)
            - GetX Pattern (getx controllers)
            - MVC Pattern (model-view-controller)
            - MVVM Pattern (model-view-viewmodel)
            - Custom (unique or mixed approach)
            
            Return JSON with:
            {{
                "primary_style": "architectural_style_name",
                "confidence": 0.0-1.0,
                "evidence": ["supporting evidence"],
                "secondary_patterns": ["other patterns found"],
                "recommendations": ["suggestions for improvement"]
            }}
            """
            
            analysis_result = await self._llm_call(
                system_prompt="""You are an expert Flutter architect. Analyze project structure 
                and dependencies to identify architectural patterns with high accuracy.""",
                user_prompt=analysis_prompt,
                context={
                    "analysis_type": "architectural_style_identification",
                    "project_path": project_context.path
                },
                structured_output=True
            )
            
            # Map result to ArchitectureStyle enum
            style_name = analysis_result.get("primary_style", "").lower()
            confidence = analysis_result.get("confidence", 0.0)
            
            # Only return if confidence is high enough
            if confidence < 0.6:
                logger.warning(f"Low confidence in architectural style detection: {confidence}")
                return None
            
            style_mapping = {
                "clean_architecture": ArchitectureStyle.CLEAN_ARCHITECTURE,
                "bloc_pattern": ArchitectureStyle.BLOC_PATTERN,
                "provider_pattern": ArchitectureStyle.PROVIDER_PATTERN,
                "riverpod_pattern": ArchitectureStyle.RIVERPOD_PATTERN,
                "getx_pattern": ArchitectureStyle.GETX_PATTERN,
                "mvc_pattern": ArchitectureStyle.MVC_PATTERN,
                "mvvm_pattern": ArchitectureStyle.MVVM_PATTERN,
                "custom": ArchitectureStyle.CUSTOM
            }
            
            detected_style = style_mapping.get(style_name.replace(" ", "_"))
            
            if detected_style:
                logger.info(f"Identified architectural style: {detected_style.value} (confidence: {confidence})")
                return detected_style
            
            return ArchitectureStyle.CUSTOM
            
        except Exception as e:
            logger.error(f"Architectural style identification failed: {e}")
            return None

    async def _find_similar_code(
        self, 
        feature_request: str, 
        project_context: "ProjectContext"
    ) -> List[CodeExample]:
        """
        Find existing code in the project that's similar to the requested feature.
        
        This helps generate code that follows existing patterns and conventions.
        
        Args:
            feature_request: Description of feature to implement
            project_context: Project context information
            
        Returns:
            List[CodeExample]: Similar code examples from the project
        """
        try:
            logger.info(f"Finding similar code for: {feature_request}")
            
            # Use semantic search to find relevant files
            search_result = await self._semantic_code_search(feature_request, project_context.path)
            
            similar_examples = []
            
            for file_path in search_result.get("relevant_files", []):
                try:
                    # Analyze each relevant file
                    understanding = await self.understand_existing_code(file_path)
                    
                    if understanding and understanding.code_type:
                        # Extract relevant code snippets
                        snippet = await self._extract_relevant_snippet(
                            file_path, feature_request, understanding
                        )
                        
                        if snippet:
                            example = CodeExample(
                                file_path=file_path,
                                code_snippet=snippet,
                                code_type=understanding.code_type,
                                description=f"Similar to: {feature_request}",
                                patterns_used=[p.pattern_type for p in understanding.patterns],
                                conventions_followed=list(understanding.conventions.keys()),
                                similarity_score=search_result.get("scores", {}).get(file_path, 0.0)
                            )
                            similar_examples.append(example)
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    continue
            
            # Sort by similarity score
            similar_examples.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top 5 most similar examples
            return similar_examples[:5]
            
        except Exception as e:
            logger.error(f"Finding similar code failed: {e}")
            return []

    async def _plan_code_integration(
        self,
        feature_request: str,
        project_structure: ProjectStructure,
        architectural_style: Optional[ArchitectureStyle]
    ) -> IntegrationPlan:
        """
        Plan how to integrate new code into the existing project structure.
        
        Considers architectural constraints, existing patterns, and minimal disruption.
        
        Args:
            feature_request: What needs to be implemented
            project_structure: Current project organization
            architectural_style: Detected architectural pattern
            
        Returns:
            IntegrationPlan: Detailed plan for code integration
        """
        try:
            plan_id = str(uuid.uuid4())
            logger.info(f"Planning code integration for: {feature_request}")
            
            # Use LLM to create integration plan
            planning_prompt = f"""
            Create a detailed integration plan for implementing this feature in a Flutter project:
            
            Feature Request: {feature_request}
            
            Project Structure:
            {json.dumps(project_structure.__dict__ if hasattr(project_structure, '__dict__') else {}, indent=2)}
            
            Architectural Style: {architectural_style.value if architectural_style else 'Unknown'}
            
            Plan should include:
            1. Which existing files need modification
            2. New files to create with their purposes
            3. Dependencies to add
            4. Integration points with existing code
            5. Required configuration changes
            6. Testing requirements
            7. Implementation order
            8. Risk assessment
            
            Follow these principles:
            - Minimize changes to existing stable code
            - Respect architectural boundaries
            - Follow established naming conventions
            - Maintain test coverage
            - Consider backwards compatibility
            
            Return JSON with detailed integration plan.
            """
            
            plan_result = await self._llm_call(
                system_prompt="""You are an expert Flutter developer creating integration plans. 
                Focus on clean, maintainable solutions that respect existing project structure.""",
                user_prompt=planning_prompt,
                context={
                    "feature_request": feature_request,
                    "architectural_style": architectural_style.value if architectural_style else None
                },
                structured_output=True
            )
            
            # Create IntegrationPlan from LLM result
            from ..models.code_models import IntegrationPlan
            
            integration_plan = IntegrationPlan(
                plan_id=plan_id,
                feature_description=feature_request,
                affected_files=plan_result.get("affected_files", []),
                new_files=plan_result.get("new_files", []),
                dependencies_to_add=plan_result.get("dependencies", []),
                integration_points=plan_result.get("integration_points", []),
                required_modifications=plan_result.get("modifications", []),
                testing_requirements=plan_result.get("testing", []),
                configuration_changes=plan_result.get("configuration", []),
                architectural_impact=plan_result.get("architectural_impact", {}),
                estimated_complexity=plan_result.get("complexity", "medium"),
                risk_assessment=plan_result.get("risks", {}),
                implementation_order=plan_result.get("implementation_order", [])
            )
            
            logger.info(
                f"Integration plan created: {len(integration_plan.new_files)} new files, "
                f"{len(integration_plan.affected_files)} modifications"
            )
            
            return integration_plan
            
        except Exception as e:
            logger.error(f"Integration planning failed: {e}")
            
            # Return minimal plan
            from ..models.code_models import IntegrationPlan
            return IntegrationPlan(
                plan_id=str(uuid.uuid4()),
                feature_description=feature_request,
                metadata={"error": str(e)}
            )

    async def _generate_code_with_context(
        self,
        feature_request: str,
        project_context: "ProjectContext",
        architectural_style: Optional[ArchitectureStyle],
        similar_code: List[CodeExample],
        integration_plan: IntegrationPlan
    ) -> Dict[str, Any]:
        """
        Generate code using full context awareness.
        
        This method creates the actual code using all gathered intelligence about
        the project structure, patterns, and integration requirements.
        """
        try:
            logger.info("Generating contextual code with full project awareness")
            
            # Prepare context for code generation
            generation_context = {
                "feature_request": feature_request,
                "architectural_style": architectural_style.value if architectural_style else None,
                "project_patterns": [example.to_dict() for example in similar_code],
                "integration_plan": integration_plan.to_dict(),
                "project_conventions": getattr(project_context, 'conventions', {})
            }
            
            # Create detailed generation prompt
            generation_prompt = f"""
            Generate Flutter code for this feature request using full project context:
            
            Feature Request: {feature_request}
            Architectural Style: {architectural_style.value if architectural_style else 'Standard Flutter'}
            
            Similar Code Examples:
            {json.dumps([example.to_dict() for example in similar_code[:3]], indent=2)}
            
            Integration Plan:
            {json.dumps(integration_plan.to_dict(), indent=2)}
            
            Requirements:
            1. Follow the detected architectural style exactly
            2. Match naming conventions from similar code
            3. Use the same import organization patterns
            4. Follow established error handling patterns
            5. Maintain consistency with existing widget structures
            6. Include proper documentation following project style
            7. Add appropriate tests following existing test patterns
            
            Generate complete, production-ready Flutter code that seamlessly integrates
            with the existing project. Include all necessary imports, proper error handling,
            and follow all detected conventions.
            
            Return JSON with:
            {{
                "files": {{
                    "file_path": "complete_file_content",
                    ...
                }},
                "imports": ["required_imports"],
                "documentation": "implementation_documentation",
                "integration_notes": ["integration_instructions"]
            }}
            """
            
            generation_result = await self._llm_call(
                system_prompt="""You are an expert Flutter developer who generates production-quality 
                code that perfectly matches existing project patterns and conventions. Always create 
                complete, working solutions.""",
                user_prompt=generation_prompt,
                context=generation_context,
                structured_output=True
            )
            
            return generation_result
            
        except Exception as e:
            logger.error(f"Contextual code generation failed: {e}")
            return {"files": {}, "error": str(e)}

    async def _validate_generated_code(
        self,
        generated_code: Dict[str, Any],
        project_context: "ProjectContext", 
        integration_plan: IntegrationPlan
    ) -> Dict[str, Any]:
        """
        Validate generated code for syntax, imports, and compatibility.
        """
        try:
            validation_results = {
                "valid": True,
                "syntax_errors": [],
                "import_errors": [],
                "compatibility_issues": [],
                "warnings": []
            }
            
            files = generated_code.get("files", {})
            
            for file_path, content in files.items():
                # Basic syntax validation
                syntax_check = self._validate_dart_syntax(content)
                if not syntax_check["valid"]:
                    validation_results["syntax_errors"].extend(syntax_check["errors"])
                    validation_results["valid"] = False
                
                # Import validation
                import_check = self._validate_imports(content, project_context)
                if not import_check["valid"]:
                    validation_results["import_errors"].extend(import_check["errors"])
                    validation_results["valid"] = False
                
                # Convention compliance
                convention_check = self._validate_conventions(content, project_context)
                validation_results["warnings"].extend(convention_check.get("warnings", []))
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return {"valid": False, "error": str(e)}

    def _build_structure_map(self, file_listing: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a hierarchical map of project structure."""
        structure_map = {}
        
        for file_info in file_listing:
            path_parts = file_info.get("path", "").split("/")
            current_level = structure_map
            
            for part in path_parts[:-1]:  # Navigate to parent directory
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            # Add file to final directory
            filename = path_parts[-1]
            current_level[filename] = file_info
        
        return structure_map

    def _identify_key_directories(self, structure_map: Dict[str, Any]) -> Dict[str, str]:
        """Identify key project directories like lib, test, assets, etc."""
        key_directories = {}
        
        # Common Flutter directory patterns
        flutter_patterns = {
            "lib": "lib",
            "test": "test", 
            "assets": "assets",
            "web": "web",
            "android": "android",
            "ios": "ios",
            "windows": "windows",
            "macos": "macos",
            "linux": "linux"
        }
        
        for key, pattern in flutter_patterns.items():
            if pattern in structure_map:
                key_directories[key] = pattern
        
        return key_directories

    def _identify_architecture_layers(
        self, 
        key_directories: Dict[str, str], 
        file_listing: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Identify architecture layers from directory structure."""
        layers = {
            "presentation": [],
            "domain": [], 
            "data": [],
            "shared": []
        }
        
        # Analyze lib directory structure
        lib_files = [f for f in file_listing if f.get("path", "").startswith("lib/")]
        
        for file_info in lib_files:
            path = file_info.get("path", "")
            
            # Map directories to layers
            if any(term in path for term in ["ui", "screen", "page", "widget", "presentation"]):
                layers["presentation"].append(path)
            elif any(term in path for term in ["usecase", "repository", "domain", "service"]):
                layers["domain"].append(path)
            elif any(term in path for term in ["data", "model", "api", "database"]):
                layers["data"].append(path)
            else:
                layers["shared"].append(path)
        
        return layers

    async def _analyze_module_dependencies(
        self, 
        file_listing: List[Dict[str, Any]], 
        project_path: str
    ) -> Dict[str, List[str]]:
        """Analyze dependencies between modules."""
        dependencies = {}
        
        # This would involve parsing import statements across files
        # For now, return basic structure
        dart_files = [f for f in file_listing if f.get("path", "").endswith(".dart")]
        
        for file_info in dart_files[:10]:  # Limit analysis for performance
            file_path = file_info.get("path", "")
            try:
                # Read file and analyze imports
                read_result = await self.use_tool(
                    "file_system",
                    "read_file",
                    {"file_path": f"{project_path}/{file_path}"},
                    f"Analyzing dependencies in {file_path}"
                )
                
                if read_result.status == ToolStatus.SUCCESS:
                    content = read_result.data.get("content", "")
                    imports = self._extract_local_imports(content)
                    dependencies[file_path] = imports
                    
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies for {file_path}: {e}")
                continue
        
        return dependencies

    def _identify_entry_points(self, file_listing: List[Dict[str, Any]]) -> List[str]:
        """Identify main entry points of the application."""
        entry_points = []
        
        for file_info in file_listing:
            path = file_info.get("path", "")
            if path.endswith("main.dart") or "main.dart" in path:
                entry_points.append(path)
        
        return entry_points

    def _find_configuration_files(self, file_listing: List[Dict[str, Any]]) -> List[str]:
        """Find configuration files like pubspec.yaml, analysis_options.yaml, etc."""
        config_files = []
        config_patterns = [".yaml", ".json", ".xml", ".plist", ".gradle"]
        
        for file_info in file_listing:
            path = file_info.get("path", "")
            if any(pattern in path for pattern in config_patterns):
                config_files.append(path)
        
        return config_files

    async def _gather_architectural_evidence(self, project_context: "ProjectContext") -> Dict[str, Any]:
        """Gather evidence about architectural patterns used in the project."""
        evidence = {
            "dependencies": getattr(project_context, 'dependencies', []),
            "directory_structure": {},
            "file_patterns": [],
            "state_management_clues": []
        }
        
        # Analyze dependencies for architectural clues
        deps = getattr(project_context, 'dependencies', [])
        
        if any("bloc" in dep for dep in deps):
            evidence["state_management_clues"].append("bloc_pattern")
        if any("provider" in dep for dep in deps):
            evidence["state_management_clues"].append("provider_pattern")
        if any("riverpod" in dep for dep in deps):
            evidence["state_management_clues"].append("riverpod_pattern")
        if any("get" in dep for dep in deps):
            evidence["state_management_clues"].append("getx_pattern")
        
        return evidence

    async def _semantic_code_search(self, feature_request: str, project_path: str) -> Dict[str, Any]:
        """Search for semantically similar code in the project."""
        try:
            # Use file_system_tool to find Dart files
            search_result = await self.use_tool(
                "file_system",
                "find_files",
                {
                    "path": project_path,
                    "pattern": "*.dart",
                    "recursive": True
                },
                f"Finding Dart files for similarity search"
            )
            
            if search_result.status != ToolStatus.SUCCESS:
                return {"relevant_files": [], "scores": {}}
            
            dart_files = search_result.data.get("files", [])
            
            # Simple heuristic-based relevance scoring
            relevant_files = []
            scores = {}
            
            feature_keywords = feature_request.lower().split()
            
            for file_path in dart_files[:20]:  # Limit for performance
                score = 0
                filename = Path(file_path).name.lower()
                
                # Score based on filename similarity
                for keyword in feature_keywords:
                    if keyword in filename:
                        score += 2
                    if keyword in file_path.lower():
                        score += 1
                
                if score > 0:
                    relevant_files.append(file_path)
                    scores[file_path] = score / len(feature_keywords)
            
            # Sort by relevance
            relevant_files.sort(key=lambda x: scores.get(x, 0), reverse=True)
            
            return {
                "relevant_files": relevant_files[:10],
                "scores": scores
            }
            
        except Exception as e:
            logger.error(f"Semantic code search failed: {e}")
            return {"relevant_files": [], "scores": {}}

    async def _extract_relevant_snippet(
        self, 
        file_path: str, 
        feature_request: str,
        understanding: CodeUnderstanding
    ) -> Optional[str]:
        """Extract relevant code snippet from a file."""
        try:
            # Read the file
            read_result = await self.use_tool(
                "file_system",
                "read_file",
                {"file_path": file_path},
                f"Reading code snippet from {file_path}"
            )
            
            if read_result.status != ToolStatus.SUCCESS:
                return None
            
            content = read_result.data.get("content", "")
            lines = content.split('\n')
            
            # Extract main class or widget (simplified)
            in_class = False
            class_lines = []
            brace_count = 0
            
            for line in lines:
                if 'class ' in line and ('extends' in line or 'implements' in line):
                    in_class = True
                    brace_count = 0
                    class_lines = [line]
                elif in_class:
                    class_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and '}' in line:
                        break
            
            if class_lines and len(class_lines) < 200:  # Reasonable size
                return '\n'.join(class_lines)
            
            # Fallback: return first 50 lines
            return '\n'.join(lines[:50])
            
        except Exception as e:
            logger.error(f"Failed to extract snippet from {file_path}: {e}")
            return None

    def _extract_local_imports(self, content: str) -> List[str]:
        """Extract local import statements from Dart code."""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') and not line.startswith('import \'dart:') and not line.startswith('import \'package:'):
                # Extract local import path
                import_match = re.search(r"import\s+['\"]([^'\"]+)['\"]", line)
                if import_match:
                    imports.append(import_match.group(1))
        
        return imports

    def _validate_dart_syntax(self, content: str) -> Dict[str, Any]:
        """Basic Dart syntax validation."""
        errors = []
        
        # Check for basic syntax issues
        if content.count('{') != content.count('}'):
            errors.append("Mismatched braces")
        if content.count('(') != content.count(')'):
            errors.append("Mismatched parentheses")
        if content.count('[') != content.count(']'):
            errors.append("Mismatched brackets")
        
        # Check for unterminated strings
        single_quotes = content.count("'") - content.count("\\'")
        double_quotes = content.count('"') - content.count('\\"')
        
        if single_quotes % 2 != 0:
            errors.append("Unterminated single quote string")
        if double_quotes % 2 != 0:
            errors.append("Unterminated double quote string")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_imports(self, content: str, project_context: "ProjectContext") -> Dict[str, Any]:
        """Validate import statements."""
        errors = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import '):
                # Check for malformed import statements
                if not (line.strip().endswith(';') or line.strip().endswith("'")):
                    errors.append(f"Line {i+1}: Import statement not properly terminated")
                
                # Basic import path validation
                if "import ''" in line or 'import ""' in line:
                    errors.append(f"Line {i+1}: Empty import path")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_conventions(self, content: str, project_context: "ProjectContext") -> Dict[str, Any]:
        """Validate code follows project conventions."""
        warnings = []
        
        # Check naming conventions (simplified)
        if not re.search(r'class\s+[A-Z][a-zA-Z0-9]*', content):
            warnings.append("Class names should be PascalCase")
        
        # Check for documentation
        if 'class ' in content and '///' not in content:
            warnings.append("Public classes should have documentation")
        
        return {"warnings": warnings}

    # File Operations for Intelligent Code Placement
    
    def _determine_optimal_location(self, code_type: str, project_structure: ProjectStructure) -> str:
        """
        Determine optimal file location based on architecture pattern and project conventions.
        
        Args:
            code_type: Type of code being placed (widget, bloc, service, etc.)
            project_structure: Current project structure analysis
            
        Returns:
            str: Optimal file path for placement
        """
        try:
            logger.info(f"Determining optimal location for code type: {code_type}")
            
            # Convert string to CodeType enum if needed
            if isinstance(code_type, str):
                try:
                    code_type_enum = CodeType(code_type.lower())
                except ValueError:
                    logger.warning(f"Unknown code type: {code_type}, defaulting to UTILITY")
                    code_type_enum = CodeType.UTILITY
            else:
                code_type_enum = code_type
            
            # Use project structure to suggest location
            if project_structure:
                suggested_location = project_structure.suggest_file_location(code_type_enum, "")
                if suggested_location:
                    return suggested_location
            
            # Fallback to standard Flutter project structure
            location_mapping = {
                CodeType.WIDGET: "lib/widgets",
                CodeType.SCREEN: "lib/screens", 
                CodeType.PAGE: "lib/pages",
                CodeType.BLOC: "lib/bloc",
                CodeType.CUBIT: "lib/cubit",
                CodeType.PROVIDER: "lib/providers",
                CodeType.REPOSITORY: "lib/repositories",
                CodeType.SERVICE: "lib/services",
                CodeType.MODEL: "lib/models",
                CodeType.CONTROLLER: "lib/controllers",
                CodeType.UTILITY: "lib/utils",
                CodeType.CONFIGURATION: "lib/config",
                CodeType.TEST: "test"
            }
            
            optimal_location = location_mapping.get(code_type_enum, "lib")
            logger.info(f"Determined optimal location: {optimal_location}")
            
            return optimal_location
            
        except Exception as e:
            logger.error(f"Failed to determine optimal location: {e}")
            return "lib"  # Safe fallback

    async def _create_file_structure(self, file_path: str, generated_code: GeneratedCode) -> bool:
        """
        Create file structure and write generated code to the appropriate location.
        
        Args:
            file_path: Target file path
            generated_code: Generated code content and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Creating file structure for: {file_path}")
            
            # Ensure file system tool is available
            if not self.file_tool:
                await self.initialize_tools()
            
            # Create directory structure if needed
            directory_path = str(Path(file_path).parent)
            
            create_dir_result = await self.use_tool(
                "file_system",
                "create_directory",
                {
                    "path": directory_path,
                    "recursive": True
                },
                f"Creating directory structure: {directory_path}"
            )
            
            if create_dir_result.status != ToolStatus.SUCCESS:
                logger.error(f"Failed to create directory: {create_dir_result.error_message}")
                return False
            
            # Write the generated code to file
            write_result = await self.use_tool(
                "file_system",
                "write_file",
                {
                    "file_path": file_path,
                    "content": generated_code.content,
                    "encoding": "utf-8"
                },
                f"Writing generated code to: {file_path}"
            )
            
            if write_result.status != ToolStatus.SUCCESS:
                logger.error(f"Failed to write file: {write_result.error_message}")
                return False
            
            logger.info(f"Successfully created file structure and wrote code to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create file structure: {e}")
            return False

    async def _update_barrel_exports(self, directory: str, new_files: List[str]) -> bool:
        """
        Update or create barrel export files in the specified directory.
        
        Args:
            directory: Directory containing the files to export
            new_files: List of new files to add to barrel exports
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Updating barrel exports in directory: {directory}")
            
            if not new_files:
                logger.info("No new files to export")
                return True
            
            # Ensure file system tool is available
            if not self.file_tool:
                await self.initialize_tools()
            
            barrel_file_path = f"{directory}/index.dart"
            existing_exports = []
            
            # Check if barrel file already exists
            read_result = await self.use_tool(
                "file_system",
                "read_file",
                {"file_path": barrel_file_path},
                f"Reading existing barrel file: {barrel_file_path}"
            )
            
            if read_result.status == ToolStatus.SUCCESS:
                content = read_result.data.get("content", "")
                # Extract existing exports
                import re
                existing_exports = re.findall(r"export\s+['\"]([^'\"]+)['\"];", content)
            
            # Add new files to exports
            all_exports = set(existing_exports)
            for file_path in new_files:
                # Convert absolute path to relative export path
                relative_path = Path(file_path).stem
                export_statement = f"./{relative_path}.dart"
                all_exports.add(export_statement)
            
            # Generate barrel file content
            export_lines = [f"export '{export_path}';" for export_path in sorted(all_exports)]
            barrel_content = "\n".join(export_lines) + "\n"
            
            # Write updated barrel file
            write_result = await self.use_tool(
                "file_system",
                "write_file",
                {
                    "file_path": barrel_file_path,
                    "content": barrel_content,
                    "encoding": "utf-8"
                },
                f"Writing updated barrel file: {barrel_file_path}"
            )
            
            if write_result.status != ToolStatus.SUCCESS:
                logger.error(f"Failed to write barrel file: {write_result.error_message}")
                return False
            
            logger.info(f"Successfully updated barrel exports: {barrel_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update barrel exports: {e}")
            return False

    async def _update_related_files(self, affected_files: List[str], new_dependencies: List[str]) -> UpdateResult:
        """
        Update related files with imports and exports for new dependencies.
        
        Args:
            affected_files: List of files that need to be updated
            new_dependencies: List of new dependencies to add
            
        Returns:
            UpdateResult: Result of the update operation
        """
        update_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Updating related files with new dependencies")
            
            result = UpdateResult(
                update_id=update_id,
                success=True
            )
            
            if not affected_files or not new_dependencies:
                logger.info("No files to update or no new dependencies")
                return result
            
            # Ensure file system tool is available
            if not self.file_tool:
                await self.initialize_tools()
            
            for file_path in affected_files:
                try:
                    # Read existing file
                    read_result = await self.use_tool(
                        "file_system",
                        "read_file",
                        {"file_path": file_path},
                        f"Reading file for dependency update: {file_path}"
                    )
                    
                    if read_result.status != ToolStatus.SUCCESS:
                        error_msg = f"Failed to read file {file_path}: {read_result.error_message}"
                        result.errors.append(error_msg)
                        continue
                    
                    original_content = read_result.data.get("content", "")
                    
                    # Store original content for rollback
                    result.rollback_data[file_path] = original_content
                    
                    # Add new imports
                    updated_content = self._add_imports_to_file(original_content, new_dependencies)
                    
                    if updated_content != original_content:
                        # Write updated file
                        write_result = await self.use_tool(
                            "file_system",
                            "write_file",
                            {
                                "file_path": file_path,
                                "content": updated_content,
                                "encoding": "utf-8"
                            },
                            f"Writing updated file with new imports: {file_path}"
                        )
                        
                        if write_result.status == ToolStatus.SUCCESS:
                            result.files_updated.append(file_path)
                            result.imports_added[file_path] = new_dependencies
                            logger.info(f"Successfully updated imports in: {file_path}")
                        else:
                            error_msg = f"Failed to write updated file {file_path}: {write_result.error_message}"
                            result.errors.append(error_msg)
                    
                except Exception as e:
                    error_msg = f"Error updating file {file_path}: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
            
            # Check for any errors
            if result.errors:
                result.success = False
                logger.warning(f"Update completed with {len(result.errors)} errors")
            else:
                logger.info(f"Successfully updated {len(result.files_updated)} files")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update related files: {e}")
            return UpdateResult(
                update_id=update_id,
                success=False,
                errors=[str(e)]
            )

    def _add_imports_to_file(self, content: str, new_dependencies: List[str]) -> str:
        """
        Add import statements to a Dart file while maintaining proper order.
        
        Args:
            content: Original file content
            new_dependencies: List of import paths to add
            
        Returns:
            str: Updated file content with new imports
        """
        lines = content.split('\n')
        
        # Find existing imports
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('export '):
                if in_imports:
                    import_lines.append(line)
                else:
                    other_lines.append(line)
            elif stripped == '' and in_imports:
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Add new imports if they don't already exist
        existing_imports = set()
        for line in import_lines:
            import re
            match = re.search(r"import\s+['\"]([^'\"]+)['\"]", line)
            if match:
                existing_imports.add(match.group(1))
        
        new_import_lines = []
        for dependency in new_dependencies:
            if dependency not in existing_imports:
                new_import_lines.append(f"import '{dependency}';")
        
        # Combine imports and sort
        all_import_lines = import_lines + new_import_lines
        if new_import_lines:
            # Sort imports: dart: first, package: second, relative last
            dart_imports = [line for line in all_import_lines if 'dart:' in line]
            package_imports = [line for line in all_import_lines if 'package:' in line]
            relative_imports = [line for line in all_import_lines if not ('dart:' in line or 'package:' in line) and line.strip()]
            
            sorted_imports = sorted(dart_imports) + sorted(package_imports) + sorted(relative_imports)
            
            # Rebuild content
            if other_lines and other_lines[0].strip() == '':
                other_lines = other_lines[1:]  # Remove empty line after imports
            
            return '\n'.join(sorted_imports + [''] + other_lines)
        
        return content

    async def place_code_intelligently(self, generated_code: GeneratedCode, project_structure: ProjectStructure) -> PlacementResult:
        """
        Intelligently place generated code in the optimal location within the project structure.
        
        This method:
        1. Determines optimal file location based on architecture pattern
        2. Follows project conventions and dependency requirements
        3. Creates directories and maintains project organization
        4. Generates barrel exports and updates part files automatically
        5. Updates related files with imports and exports
        6. Preserves functionality with backup and validation
        
        Args:
            generated_code: Generated code with metadata
            project_structure: Current project structure analysis
            
        Returns:
            PlacementResult: Result of the placement operation with details
        """
        placement_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting intelligent code placement for {generated_code.file_type.value}")
            
            result = PlacementResult(
                placement_id=placement_id,
                success=True
            )
            
            # Step 1: Determine optimal location
            optimal_directory = self._determine_optimal_location(generated_code.file_type, project_structure)
            
            # Step 2: Generate appropriate file name
            file_name = self._generate_file_name(generated_code)
            target_file_path = f"{optimal_directory}/{file_name}"
            
            # Step 3: Create backup if file exists
            backup_path = await self._create_backup(target_file_path)
            if backup_path:
                result.backup_paths.append(backup_path)
            
            # Step 4: Create file structure and write code
            success = await self._create_file_structure(target_file_path, generated_code)
            if not success:
                result.success = False
                result.error_message = f"Failed to create file structure for {target_file_path}"
                return result
            
            result.placed_files.append(target_file_path)
            result.created_directories.append(optimal_directory)
            
            # Step 5: Update barrel exports if needed
            if generated_code.requires_barrel_export:
                barrel_success = await self._update_barrel_exports(optimal_directory, [target_file_path])
                if barrel_success:
                    result.barrel_exports_updated.append(f"{optimal_directory}/index.dart")
                else:
                    result.warnings.append(f"Failed to update barrel exports for {optimal_directory}")
            
            # Step 6: Update related files with imports
            if generated_code.dependencies:
                # Find files that might need to import this new code
                affected_files = await self._find_files_needing_imports(generated_code, project_structure)
                
                if affected_files:
                    update_result = await self._update_related_files(affected_files, [target_file_path])
                    
                    if update_result.success:
                        result.updated_files.extend(update_result.files_updated)
                        result.imports_added.update(update_result.imports_added)
                    else:
                        result.warnings.extend(update_result.errors)
            
            # Step 7: Validate placement
            validation_success = await self._validate_placement(result)
            if not validation_success:
                result.warnings.append("Placement validation found potential issues")
            
            result.metadata = {
                "code_type": generated_code.file_type.value,
                "target_path": target_file_path,
                "optimal_directory": optimal_directory,
                "architecture_layer": generated_code.architecture_layer,
                "requires_state_management": generated_code.requires_state_management()
            }
            
            logger.info(f"Intelligent code placement completed successfully: {target_file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Intelligent code placement failed: {e}")
            return PlacementResult(
                placement_id=placement_id,
                success=False,
                error_message=str(e)
            )

    def _generate_file_name(self, generated_code: GeneratedCode) -> str:
        """Generate appropriate file name based on code content and conventions."""
        # Extract class name or use purpose
        class_name = generated_code.get_main_class_name()
        if class_name:
            # Convert PascalCase to snake_case
            import re
            snake_case = re.sub('([A-Z]+)', r'_\1', class_name).lower().strip('_')
            return f"{snake_case}.dart"
        
        # Fallback to purpose or generic name
        if generated_code.purpose:
            safe_purpose = re.sub(r'[^a-zA-Z0-9_]', '_', generated_code.purpose.lower())
            return f"{safe_purpose}.dart"
        
        return f"generated_{generated_code.file_type.value}.dart"

    async def _create_backup(self, file_path: str) -> Optional[str]:
        """Create backup of existing file if it exists."""
        try:
            if not self.file_tool:
                await self.initialize_tools()
            
            # Check if file exists
            read_result = await self.use_tool(
                "file_system",
                "read_file",
                {"file_path": file_path},
                f"Checking if file exists for backup: {file_path}"
            )
            
            if read_result.status != ToolStatus.SUCCESS:
                # File doesn't exist, no backup needed
                return None
            
            # Create backup
            backup_path = f"{file_path}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            backup_result = await self.use_tool(
                "file_system",
                "copy_file",
                {
                    "source_path": file_path,
                    "destination_path": backup_path
                },
                f"Creating backup: {backup_path}"
            )
            
            if backup_result.status == ToolStatus.SUCCESS:
                logger.info(f"Created backup: {backup_path}")
                return backup_path
            else:
                logger.warning(f"Failed to create backup: {backup_result.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    async def _find_files_needing_imports(self, generated_code: GeneratedCode, project_structure: ProjectStructure) -> List[str]:
        """Find files that might need to import the newly generated code."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd analyze the project to find files that would benefit from the new code
            
            affected_files = []
            
            # Look for files in the same architecture layer
            if generated_code.architecture_layer and project_structure:
                layer_files = project_structure.architecture_layers.get(generated_code.architecture_layer, [])
                # Add some logic to determine which files actually need the import
                # For now, we'll return an empty list to avoid unnecessary modifications
                pass
            
            return affected_files
            
        except Exception as e:
            logger.error(f"Error finding files needing imports: {e}")
            return []

    async def _validate_placement(self, placement_result: PlacementResult) -> bool:
        """Validate that the code placement was successful and follows conventions."""
        try:
            # Check that all placed files actually exist
            for file_path in placement_result.placed_files:
                if not self.file_tool:
                    await self.initialize_tools()
                
                read_result = await self.use_tool(
                    "file_system",
                    "read_file",
                    {"file_path": file_path},
                    f"Validating placed file: {file_path}"
                )
                
                if read_result.status != ToolStatus.SUCCESS:
                    logger.error(f"Validation failed: {file_path} does not exist")
                    return False
            
            logger.info("Placement validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Placement validation error: {e}")
            return False

    async def rollback_placement(self, placement_result: PlacementResult) -> bool:
        """
        Rollback a code placement operation using backup information.
        
        Args:
            placement_result: Result of the original placement operation
            
        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            logger.info(f"Rolling back placement: {placement_result.placement_id}")
            
            if not self.file_tool:
                await self.initialize_tools()
            
            rollback_success = True
            
            # Remove placed files
            for file_path in placement_result.placed_files:
                try:
                    delete_result = await self.use_tool(
                        "file_system",
                        "delete_file",
                        {"file_path": file_path},
                        f"Rolling back placed file: {file_path}"
                    )
                    
                    if delete_result.status != ToolStatus.SUCCESS:
                        logger.error(f"Failed to delete file during rollback: {file_path}")
                        rollback_success = False
                        
                except Exception as e:
                    logger.error(f"Error during file deletion rollback: {e}")
                    rollback_success = False
            
            # Restore from backups
            for backup_path in placement_result.backup_paths:
                try:
                    # Extract original path from backup path
                    original_path = backup_path.split('.backup_')[0]
                    
                    restore_result = await self.use_tool(
                        "file_system",
                        "copy_file",
                        {
                            "source_path": backup_path,
                            "destination_path": original_path
                        },
                        f"Restoring from backup: {backup_path} -> {original_path}"
                    )
                    
                    if restore_result.status != ToolStatus.SUCCESS:
                        logger.error(f"Failed to restore from backup: {backup_path}")
                        rollback_success = False
                    else:
                        # Clean up backup file
                        await self.use_tool(
                            "file_system",
                            "delete_file",
                            {"file_path": backup_path},
                            f"Cleaning up backup file: {backup_path}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error during backup restoration: {e}")
                    rollback_success = False
            
            if rollback_success:
                logger.info(f"Successfully rolled back placement: {placement_result.placement_id}")
            else:
                logger.error(f"Rollback completed with errors: {placement_result.placement_id}")
                
            return rollback_success
            
        except Exception as e:
            logger.error(f"Rollback operation failed: {e}")
            return False
