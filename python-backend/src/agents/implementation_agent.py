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
    CodeConvention, CodeAnalysisResult
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
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """
        Generate code with full project awareness using tools.
        
        This method analyzes the existing project, plans the implementation,
        generates coherent code, and validates the result through tool usage.
        """
        try:
            # 1. Analyze existing project
            project_analysis = await self._analyze_existing_project(project_context.project_path)
            
            # 2. Plan code generation
            generation_plan = await self._plan_code_generation(
                feature_request, project_context, project_analysis
            )
            
            # 3. Generate coherent code
            generated_code = await self._generate_coherent_code(
                generation_plan, project_analysis
            )
            
            # 4. Validate generation
            validation_result = await self._validate_code_generation(
                generated_code, project_context.project_path
            )
            
            return {
                "success": True,
                "generated_code": generated_code,
                "validation": validation_result,
                "project_analysis": project_analysis,
                "generation_plan": generation_plan
            }
            
        except Exception as e:
            logger.error(f"Contextual code generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generated_code": {},
                "validation": {}
            }

    async def _analyze_existing_project(self, project_path: str) -> Dict[str, Any]:
        """Deep analysis of existing project using file system tool."""
        if not self.file_tool:
            logger.warning("File system tool not available for project analysis")
            return {"error": "File system tool not available"}
        
        try:
            # Use tool to scan project structure
            structure_result = await self.use_tool(
                "file_system_tool",
                "analyze_project_structure",
                {"project_path": project_path},
                "Analyzing existing Flutter project structure"
            )
            
            if structure_result.status.value == "success":
                project_structure = structure_result.data
                
                # Cache the structure
                self.project_structure_cache[project_path] = project_structure
                
                # Understand architectural patterns
                architecture_analysis = await self._detect_architecture_pattern(
                    project_structure
                )
                
                # Find existing code patterns
                code_patterns = await self._extract_existing_patterns(
                    project_path, project_structure
                )
                
                return {
                    "structure": project_structure,
                    "architecture": architecture_analysis,
                    "patterns": code_patterns,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"Project analysis failed: {structure_result.error_message}")
                return {"error": structure_result.error_message}
                
        except Exception as e:
            logger.error(f"Project analysis error: {e}")
            return {"error": str(e)}

    async def _detect_architecture_pattern(
        self, project_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect the architectural pattern used in the project through LLM reasoning."""
        
        structure_prompt = f"""
        Analyze this Flutter project structure and identify the architectural pattern being used:
        
        Project Structure:
        {json.dumps(project_structure, indent=2)}
        
        Look for patterns like:
        - Clean Architecture (lib/domain, lib/data, lib/presentation)
        - BLoC pattern (bloc/ directories, _bloc.dart files)
        - Provider pattern (provider/, _provider.dart files)
        - MVVM (view/, viewmodel/, model/ directories)
        - Feature-based organization
        
        Provide analysis of:
        1. Primary architectural pattern
        2. State management approach
        3. Directory organization style
        4. Dependencies and layer separation
        5. Naming conventions
        """
        
        analysis_result = await self.execute_llm_task(
            user_prompt=structure_prompt,
            context={"project_structure": project_structure}
        )
        
        return analysis_result

    async def _extract_existing_patterns(
        self, project_path: str, project_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract code patterns from existing files."""
        if not self.file_tool:
            return {"error": "File system tool not available"}
        
        # Find key Dart files to analyze
        dart_files = project_structure.get("dart_files", [])
        key_files = [f for f in dart_files if any(
            pattern in f for pattern in ["main.dart", "app.dart", "_bloc.dart", "_provider.dart", "_widget.dart"]
        )][:10]  # Analyze up to 10 key files
        
        patterns = {
            "imports": [],
            "widget_patterns": [],
            "state_management": [],
            "naming_conventions": [],
            "code_style": {}
        }
        
        for file_path in key_files:
            try:
                # Read file content using tool
                read_result = await self.use_tool(
                    "file_system_tool",
                    "read_file",
                    {"file_path": file_path},
                    f"Reading {file_path} to extract patterns"
                )
                
                if read_result.status.value == "success":
                    file_content = read_result.data.get("content", "")
                    file_patterns = await self._analyze_file_patterns(file_content, file_path)
                    
                    # Merge patterns
                    for key, value in file_patterns.items():
                        if isinstance(value, list):
                            patterns[key].extend(value)
                        elif isinstance(value, dict):
                            patterns[key].update(value)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path}: {e}")
                continue
        
        return patterns

    async def _analyze_file_patterns(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze patterns in a single file using LLM reasoning."""
        
        pattern_prompt = f"""
        Analyze this Dart file and extract code patterns:
        
        File: {file_path}
        Content:
        ```dart
        {content[:2000]}  # First 2000 characters
        ```
        
        Extract:
        1. Import patterns and dependencies
        2. Widget creation patterns
        3. State management usage
        4. Naming conventions (classes, methods, variables)
        5. Code organization style
        6. Common utilities or helpers used
        
        Return as structured data.
        """
        
        analysis = await self.execute_llm_task(
            user_prompt=pattern_prompt,
            context={"file_path": file_path, "content_preview": content[:500]}
        )
        
        return analysis

    async def _plan_code_generation(
        self,
        feature_request: str,
        project_context: ProjectContext,
        project_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan the code generation strategy using LLM reasoning."""
        
        planning_prompt = f"""
        Plan the implementation of this feature request for a Flutter project:
        
        Feature Request: {feature_request}
        
        Project Context:
        - Type: {project_context.project_type.value}
        - Architecture: {project_analysis.get('architecture', {}).get('primary_pattern', 'unknown')}
        - Target Platforms: {[p.value for p in project_context.platform_targets]}
        
        Existing Project Structure:
        {json.dumps(project_analysis.get('structure', {}), indent=2)}
        
        Create a detailed implementation plan including:
        1. Files to create/modify
        2. Dependencies to add (pubspec.yaml changes)
        3. Integration points with existing code
        4. Testing strategy
        5. Step-by-step implementation sequence
        6. Potential challenges and solutions
        
        Consider:
        - Maintaining consistency with existing patterns
        - Following project conventions
        - Ensuring proper separation of concerns
        - Minimizing breaking changes
        
        Format as structured implementation plan.
        """
        
        plan_result = await self.execute_llm_task(
            user_prompt=planning_prompt,
            context={
                "feature_request": feature_request,
                "project_context": project_context.__dict__,
                "project_analysis": project_analysis
            }
        )
        
        return plan_result

    async def _generate_coherent_code(
        self,
        generation_plan: Dict[str, Any],
        project_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate coherent code that matches project patterns."""
        
        generated_files = {}
        
        for file_info in generation_plan.get("files_to_create", []):
            file_path = file_info.get("path", "")
            file_type = file_info.get("type", "")
            purpose = file_info.get("purpose", "")
            
            # Generate code for this file
            code_content = await self._generate_file_content(
                file_path, file_type, purpose, project_analysis
            )
            
            generated_files[file_path] = {
                "content": code_content,
                "type": file_type,
                "purpose": purpose
            }
        
        return {
            "files": generated_files,
            "dependencies": generation_plan.get("dependencies", []),
            "integration_points": generation_plan.get("integration_points", [])
        }

    async def _generate_file_content(
        self,
        file_path: str,
        file_type: str,
        purpose: str,
        project_analysis: Dict[str, Any]
    ) -> str:
        """Generate content for a specific file using LLM reasoning."""
        
        # Determine template and patterns based on file type
        patterns = project_analysis.get("patterns", {})
        architecture = project_analysis.get("architecture", {})
        
        generation_prompt = f"""
        Generate Flutter/Dart code for this file:
        
        File Path: {file_path}
        File Type: {file_type}
        Purpose: {purpose}
        
        Project Context:
        - Architecture Pattern: {architecture.get("primary_pattern", "unknown")}
        - State Management: {architecture.get("state_management", "unknown")}
        - Existing Patterns: {json.dumps(patterns, indent=2)}
        
        Requirements:
        1. Match existing code style and patterns
        2. Follow Dart/Flutter best practices
        3. Include proper imports
        4. Add meaningful comments
        5. Ensure type safety
        6. Follow naming conventions from project
        
        Generate complete, production-ready code.
        """
        
        code_result = await self.execute_llm_task(
            user_prompt=generation_prompt,
            context={
                "file_path": file_path,
                "file_type": file_type,
                "project_patterns": patterns,
                "architecture": architecture
            }
        )
        
        return code_result.get("code", "")

    async def _validate_code_generation(
        self,
        generated_code: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Validate generated code using Flutter analysis tools."""
        
        validation_results = {}
        
        for file_path, file_info in generated_code.get("files", {}).items():
            # Create temporary file for validation
            temp_file_path = f"{project_path}/{file_path}.temp"
            
            try:
                # Write file temporarily
                write_result = await self.use_tool(
                    "file_system_tool",
                    "write_file",
                    {
                        "path": temp_file_path,
                        "content": file_info["content"],
                        "create_backup": False
                    },
                    f"Creating temporary file for validation: {file_path}"
                )
                
                if write_result.status.value == "success":
                    # Validate syntax using Flutter tool
                    if self.flutter_tool:
                        analysis_result = await self.use_tool(
                            "flutter_sdk_tool",
                            "analyze_code",
                            {"project_path": project_path, "file_path": temp_file_path},
                            f"Validating syntax for {file_path}"
                        )
                        
                        validation_results[file_path] = {
                            "syntax_valid": analysis_result.status.value == "success",
                            "issues": analysis_result.data.get("issues", []),
                            "analysis_result": analysis_result.data
                        }
                    
                    # Clean up temporary file
                    await self.use_tool(
                        "file_system_tool",
                        "delete_file",
                        {"path": temp_file_path},
                        f"Cleaning up temporary file: {temp_file_path}"
                    )
                else:
                    validation_results[file_path] = {
                        "syntax_valid": False,
                        "error": "Could not create temporary file for validation"
                    }
                    
            except Exception as e:
                validation_results[file_path] = {
                    "syntax_valid": False,
                    "error": str(e)
                }
        
        return validation_results

    async def place_code_intelligently(
        self,
        generated_code: Dict[str, Any],
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Smart code placement with proper integration."""
        
        placement_results = {}
        project_path = project_context.project_path
        
        try:
            # 1. Determine optimal file locations
            file_placements = await self._determine_file_placements(
                generated_code, project_context
            )
            
            # 2. Handle file creation and updates
            for file_path, placement_info in file_placements.items():
                result = await self._place_single_file(
                    file_path, placement_info, project_path
                )
                placement_results[file_path] = result
            
            # 3. Update related files (imports, exports, etc.)
            update_results = await self._update_related_files(
                file_placements, project_path
            )
            
            # 4. Verify functionality
            verification_results = await self._verify_code_placement(
                file_placements, project_path
            )
            
            return {
                "success": True,
                "placements": placement_results,
                "updates": update_results,
                "verification": verification_results
            }
            
        except Exception as e:
            logger.error(f"Code placement failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "placements": placement_results
            }

    async def _determine_file_placements(
        self,
        generated_code: Dict[str, Any],
        project_context: ProjectContext
    ) -> Dict[str, Any]:
        """Determine optimal file placements using LLM reasoning."""
        
        placement_prompt = f"""
        Determine optimal file placements for this generated code in a Flutter project:
        
        Generated Files:
        {json.dumps(list(generated_code.get("files", {}).keys()), indent=2)}
        
        Project Type: {project_context.project_type.value}
        Project Path: {project_context.project_path}
        
        Consider:
        1. Flutter project conventions (lib/, test/, assets/)
        2. Feature organization (features/, modules/)
        3. Layer separation (domain/, data/, presentation/)
        4. Existing directory structure
        5. Import path optimization
        
        For each file, specify:
        - Final path location
        - Directory creation needs
        - Import updates required
        - Export file updates
        
        Format as structured placement plan.
        """
        
        placement_result = await self.execute_llm_task(
            user_prompt=placement_prompt,
            context={
                "files": list(generated_code.get("files", {}).keys()),
                "project_context": project_context.__dict__
            }
        )
        
        return placement_result.get("file_placements", {})

    async def _place_single_file(
        self,
        file_path: str,
        placement_info: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Place a single file in the project."""
        
        try:
            final_path = placement_info.get("final_path", file_path)
            content = placement_info.get("content", "")
            
            # Use file system tool to place the file
            write_result = await self.use_tool(
                "file_system_tool",
                "write_file",
                {
                    "path": f"{project_path}/{final_path}",
                    "content": content,
                    "create_backup": True
                },
                f"Placing generated file: {final_path}"
            )
            
            return {
                "success": write_result.status.value == "success",
                "path": final_path,
                "backup_path": write_result.data.get("backup_path"),
                "error": write_result.error_message
            }
            
        except Exception as e:
            return {
                "success": False,
                "path": file_path,
                "error": str(e)
            }

    async def _update_related_files(
        self,
        file_placements: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Update related files with imports, exports, etc."""
        
        update_results = {}
        
        # Update barrel exports
        barrel_updates = await self._update_barrel_exports(file_placements, project_path)
        update_results["barrel_exports"] = barrel_updates
        
        # Update navigation if needed
        nav_updates = await self._update_navigation_files(file_placements, project_path)
        update_results["navigation"] = nav_updates
        
        # Update dependency injection if needed
        di_updates = await self._update_dependency_injection(file_placements, project_path)
        update_results["dependency_injection"] = di_updates
        
        return update_results

    async def validate_code_continuously(
        self,
        project_path: str,
        changed_files: List[str] = None
    ) -> Dict[str, Any]:
        """Continuous validation during development."""
        
        try:
            validation_results = {}
            
            # 1. Syntax validation using Flutter analyze
            if self.flutter_tool:
                syntax_result = await self.use_tool(
                    "flutter_sdk_tool",
                    "analyze_code",
                    {"project_path": project_path},
                    "Running Flutter analyzer for syntax validation"
                )
                
                validation_results["syntax"] = {
                    "valid": syntax_result.status.value == "success",
                    "issues": syntax_result.data.get("issues", []),
                    "warnings": syntax_result.data.get("warnings", [])
                }
            
            # 2. Architecture compliance check
            architecture_result = await self._check_architecture_compliance(
                project_path, changed_files
            )
            validation_results["architecture"] = architecture_result
            
            # 3. Performance validation
            performance_result = await self._check_performance_issues(
                project_path, changed_files
            )
            validation_results["performance"] = performance_result
            
            # 4. Style compliance
            style_result = await self._check_style_compliance(project_path)
            validation_results["style"] = style_result
            
            return {
                "success": True,
                "validation_results": validation_results,
                "overall_valid": all(
                    result.get("valid", True) for result in validation_results.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Continuous validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_results": {}
            }

    async def manage_project_dependencies(
        self,
        required_features: List[str],
        project_path: str
    ) -> Dict[str, Any]:
        """Intelligent package management using LLM reasoning."""
        
        try:
            # 1. Analyze current dependencies
            current_deps = await self._analyze_current_dependencies(project_path)
            
            # 2. Map features to packages using LLM
            package_mapping = await self._map_features_to_packages(
                required_features, current_deps
            )
            
            # 3. Update pubspec.yaml
            pubspec_result = await self._update_pubspec_yaml(
                package_mapping, project_path
            )
            
            # 4. Run pub get
            if self.flutter_tool and pubspec_result.get("success"):
                pub_result = await self.use_tool(
                    "flutter_sdk_tool",
                    "pub_get",
                    {"project_path": project_path},
                    "Installing new dependencies"
                )
                
                pubspec_result["pub_get"] = {
                    "success": pub_result.status.value == "success",
                    "output": pub_result.data
                }
            
            # 5. Configure packages
            config_result = await self._configure_new_packages(
                package_mapping, project_path
            )
            
            return {
                "success": pubspec_result.get("success", False),
                "dependencies_added": package_mapping,
                "pubspec_update": pubspec_result,
                "configuration": config_result
            }
            
        except Exception as e:
            logger.error(f"Dependency management failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def develop_with_hot_reload(
        self,
        project_path: str,
        development_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Seamless hot reload development experience."""
        
        try:
            config = development_config or {}
            platform = config.get("platform", "web")
            port = config.get("port", 3000)
            
            # 1. Start development server
            if self.process_tool:
                server_result = await self.use_tool(
                    "process_tool",
                    "start_dev_server",
                    {
                        "project_path": project_path,
                        "platform": platform,
                        "port": port,
                        "hot_reload": True,
                        "debug_mode": True
                    },
                    f"Starting Flutter development server for {platform}"
                )
                
                if server_result.status.value != "success":
                    return {
                        "success": False,
                        "error": f"Failed to start dev server: {server_result.error_message}"
                    }
                
                process_id = server_result.data.get("process_id")
                
                # 2. Set up file watching for hot reload
                watch_result = await self._setup_file_watching(project_path, process_id)
                
                # 3. Monitor hot reload performance
                monitoring_result = await self._setup_hot_reload_monitoring(process_id)
                
                return {
                    "success": True,
                    "process_id": process_id,
                    "server_info": server_result.data,
                    "file_watching": watch_result,
                    "monitoring": monitoring_result
                }
            else:
                return {
                    "success": False,
                    "error": "Process tool not available for development server"
                }
                
        except Exception as e:
            logger.error(f"Hot reload development setup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _setup_file_watching(
        self,
        project_path: str,
        process_id: str
    ) -> Dict[str, Any]:
        """Set up intelligent file watching for hot reload."""
        
        try:
            # Monitor key directories for changes
            watch_directories = [
                f"{project_path}/lib",
                f"{project_path}/assets",
                f"{project_path}/pubspec.yaml"
            ]
            
            # Start file watching (simplified implementation)
            # In a full implementation, this would use file system events
            return {
                "watching": True,
                "directories": watch_directories,
                "process_id": process_id
            }
            
        except Exception as e:
            return {
                "watching": False,
                "error": str(e)
            }

    async def trigger_hot_reload(
        self,
        process_id: str,
        restart: bool = False
    ) -> Dict[str, Any]:
        """Trigger hot reload or restart."""
        
        try:
            if self.process_tool:
                reload_result = await self.use_tool(
                    "process_tool",
                    "hot_reload",
                    {
                        "process_id": process_id,
                        "restart": restart
                    },
                    f"Triggering hot {'restart' if restart else 'reload'}"
                )
                
                return {
                    "success": reload_result.status.value == "success",
                    "reload_type": "restart" if restart else "reload",
                    "result": reload_result.data,
                    "error": reload_result.error_message
                }
            else:
                return {
                    "success": False,
                    "error": "Process tool not available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _analyze_current_dependencies(self, project_path: str) -> Dict[str, Any]:
        """Analyze current project dependencies."""
        
        try:
            # Read pubspec.yaml
            pubspec_result = await self.use_tool(
                "file_system_tool",
                "read_file",
                {"path": f"{project_path}/pubspec.yaml"},
                "Reading current dependencies from pubspec.yaml"
            )
            
            if pubspec_result.status.value == "success":
                import yaml
                pubspec_content = yaml.safe_load(pubspec_result.data["content"])
                
                return {
                    "dependencies": pubspec_content.get("dependencies", {}),
                    "dev_dependencies": pubspec_content.get("dev_dependencies", {}),
                    "flutter_version": pubspec_content.get("environment", {}).get("flutter"),
                    "dart_version": pubspec_content.get("environment", {}).get("sdk")
                }
            else:
                return {"error": "Could not read pubspec.yaml"}
                
        except Exception as e:
            return {"error": str(e)}

    async def _map_features_to_packages(
        self,
        required_features: List[str],
        current_deps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map required features to Flutter packages using LLM reasoning."""
        
        mapping_prompt = f"""
        Map these required features to appropriate Flutter packages:
        
        Required Features: {required_features}
        
        Current Dependencies: {json.dumps(current_deps.get("dependencies", {}), indent=2)}
        
        For each feature, recommend:
        1. Primary package to add
        2. Package version (use latest stable)
        3. Additional dependencies if needed
        4. Configuration requirements
        5. Whether it conflicts with existing packages
        
        Consider:
        - Package popularity and maintenance
        - Flutter/Dart compatibility
        - Performance implications
        - Team preferences and standards
        
        Features to Package mapping:
        - state_management: bloc, provider, riverpod, getx
        - navigation: go_router, auto_route
        - networking: dio, http
        - local_storage: hive, sqflite, shared_preferences
        - animations: flutter_animate, rive
        - ui_components: flutter_screenutil, cached_network_image
        
        Format as structured package mapping.
        """
        
        mapping_result = await self.execute_llm_task(
            user_prompt=mapping_prompt,
            context={
                "features": required_features,
                "current_dependencies": current_deps
            }
        )
        
        return mapping_result.get("package_mapping", {})

    async def _update_pubspec_yaml(
        self,
        package_mapping: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Update pubspec.yaml with new dependencies."""
        
        try:
            # Read current pubspec
            pubspec_result = await self.use_tool(
                "file_system_tool",
                "read_file",
                {"path": f"{project_path}/pubspec.yaml"},
                "Reading pubspec.yaml for dependency update"
            )
            
            if pubspec_result.status.value != "success":
                return {"success": False, "error": "Could not read pubspec.yaml"}
            
            import yaml
            pubspec_content = yaml.safe_load(pubspec_result.data["content"])
            
            # Add new dependencies
            dependencies = pubspec_content.setdefault("dependencies", {})
            dev_dependencies = pubspec_content.setdefault("dev_dependencies", {})
            
            for feature, package_info in package_mapping.items():
                package_name = package_info.get("package")
                version = package_info.get("version")
                is_dev = package_info.get("dev_dependency", False)
                
                if package_name and version:
                    target_deps = dev_dependencies if is_dev else dependencies
                    target_deps[package_name] = version
            
            # Write updated pubspec
            updated_yaml = yaml.dump(pubspec_content, default_flow_style=False)
            
            write_result = await self.use_tool(
                "file_system_tool",
                "write_file",
                {
                    "path": f"{project_path}/pubspec.yaml",
                    "content": updated_yaml,
                    "create_backup": True
                },
                "Updating pubspec.yaml with new dependencies"
            )
            
            return {
                "success": write_result.status.value == "success",
                "backup_path": write_result.data.get("backup_path"),
                "added_packages": package_mapping
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _configure_new_packages(
        self,
        package_mapping: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Configure newly added packages."""
        
        config_results = {}
        
        for feature, package_info in package_mapping.items():
            package_name = package_info.get("package")
            config_needs = package_info.get("configuration", {})
            
            if config_needs:
                # Generate configuration files
                config_result = await self._generate_package_config(
                    package_name, config_needs, project_path
                )
                config_results[package_name] = config_result
        
        return config_results

    async def _generate_package_config(
        self,
        package_name: str,
        config_needs: Dict[str, Any],
        project_path: str
    ) -> Dict[str, Any]:
        """Generate configuration for a specific package."""
        
        config_prompt = f"""
        Generate configuration code for Flutter package: {package_name}
        
        Configuration Requirements: {json.dumps(config_needs, indent=2)}
        
        Generate:
        1. Initialization code (main.dart additions)
        2. Service setup files
        3. Provider/Bloc setup if needed
        4. Configuration classes
        5. Example usage patterns
        
        Follow Flutter best practices for package integration.
        """
        
        config_result = await self.execute_llm_task(
            user_prompt=config_prompt,
            context={
                "package_name": package_name,
                "config_requirements": config_needs
            }
        )
        
        # Apply configuration if files were generated
        generated_files = config_result.get("configuration_files", {})
        
        for file_path, content in generated_files.items():
            await self.use_tool(
                "file_system_tool",
                "write_file",
                {
                    "path": f"{project_path}/{file_path}",
                    "content": content,
                    "create_backup": True
                },
                f"Creating configuration file for {package_name}: {file_path}"
            )
        
        return {
            "package": package_name,
            "files_created": list(generated_files.keys()),
            "configuration": config_result
        }

    # Helper methods for validation and architecture compliance
    async def _check_architecture_compliance(
        self,
        project_path: str,
        changed_files: List[str] = None
    ) -> Dict[str, Any]:
        """Check if code follows architectural patterns."""
        
        compliance_prompt = f"""
        Check architectural compliance for Flutter project changes:
        
        Project Path: {project_path}
        Changed Files: {changed_files or "All files"}
        
        Check for:
        1. Proper layer separation (UI, Business Logic, Data)
        2. Dependency direction (outer layers depend on inner)
        3. Single Responsibility Principle
        4. Consistent naming conventions
        5. Proper import organization
        
        Provide compliance report with violations and suggestions.
        """
        
        compliance_result = await self.execute_llm_task(
            user_prompt=compliance_prompt,
            context={
                "project_path": project_path,
                "changed_files": changed_files
            }
        )
        
        return {
            "compliant": compliance_result.get("overall_compliant", True),
            "violations": compliance_result.get("violations", []),
            "suggestions": compliance_result.get("suggestions", [])
        }

    async def _check_performance_issues(
        self,
        project_path: str,
        changed_files: List[str] = None
    ) -> Dict[str, Any]:
        """Check for potential performance issues."""
        
        # This would integrate with performance analysis tools
        # For now, return a basic structure
        return {
            "performance_warnings": [],
            "suggestions": [],
            "overall_score": 0.9
        }

    async def _check_style_compliance(self, project_path: str) -> Dict[str, Any]:
        """Check code style compliance."""
        
        if self.flutter_tool:
            # Use dartfmt to check formatting
            format_result = await self.use_tool(
                "flutter_sdk_tool",
                "format_code",
                {"project_path": project_path},
                "Checking code formatting"
            )
            
            return {
                "properly_formatted": format_result.status.value == "success",
                "format_issues": format_result.data.get("issues", [])
            }
        
        return {"properly_formatted": True}
