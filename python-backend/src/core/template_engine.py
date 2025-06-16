"""
Template Engine for FlutterSwarm Multi-Agent System.

Provides dynamic template loading, processing, and code generation
capabilities with support for Jinja2 templating.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import dataclass
from enum import Enum

from ..config import get_logger

logger = get_logger("template_engine")


class TemplateType(Enum):
    """Template types for different Flutter components."""
    APP = "app"
    WIDGET = "widget"
    SCREEN = "screen"
    MODEL = "model"
    BLOC = "bloc"
    REPOSITORY = "repository"
    SERVICE = "service"
    TEST = "test"


class ArchitecturalPattern(Enum):
    """Supported architectural patterns."""
    CLEAN_ARCHITECTURE = "clean"
    BLOC_PATTERN = "bloc"
    PROVIDER_PATTERN = "provider"
    RIVERPOD_PATTERN = "riverpod"
    GETX_PATTERN = "getx"
    MVC_PATTERN = "mvc"
    BASIC_PATTERN = "basic"


@dataclass
class TemplateContext:
    """Context for template rendering."""
    app_name: str
    app_description: str
    architectural_pattern: ArchitecturalPattern = ArchitecturalPattern.BASIC_PATTERN
    features: List[str] = None
    platform_targets: List[str] = None
    state_management: str = "stateful"
    theme_mode: str = "light"
    custom_variables: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.platform_targets is None:
            self.platform_targets = ["android", "ios"]
        if self.custom_variables is None:
            self.custom_variables = {}


class TemplateEngine:
    """Dynamic template engine for Flutter code generation."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            # Default to templates directory relative to this file
            current_dir = Path(__file__).parent.parent
            templates_dir = current_dir / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self._register_custom_filters()
        
        logger.info(f"Template engine initialized with templates dir: {self.templates_dir}")
    
    def _register_custom_filters(self):
        """Register custom Jinja2 filters for Flutter code generation."""
        
        def snake_case(text: str) -> str:
            """Convert text to snake_case."""
            import re
            text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', text).lower()
        
        def pascal_case(text: str) -> str:
            """Convert text to PascalCase."""
            return ''.join(word.capitalize() for word in text.replace('_', ' ').split())
        
        def camel_case(text: str) -> str:
            """Convert text to camelCase."""
            pascal = pascal_case(text)
            return pascal[0].lower() + pascal[1:] if pascal else ''
        
        def dart_string_literal(text: str) -> str:
            """Convert text to Dart string literal."""
            return f"'{text.replace('\'', '\\\'')}'"
        
        self.env.filters['snake_case'] = snake_case
        self.env.filters['pascal_case'] = pascal_case
        self.env.filters['camel_case'] = camel_case
        self.env.filters['dart_string'] = dart_string_literal
    
    def get_template(
        self, 
        template_type: TemplateType, 
        architectural_pattern: ArchitecturalPattern = ArchitecturalPattern.BASIC_PATTERN,
        variant: Optional[str] = None
    ) -> Optional[Template]:
        """Get a template by type and architectural pattern."""
        
        # Build template path
        template_path = f"{template_type.value}"
        
        if architectural_pattern != ArchitecturalPattern.BASIC_PATTERN:
            template_path = f"{architectural_pattern.value}/{template_type.value}"
        
        if variant:
            template_path = f"{template_path}_{variant}"
        
        template_path += ".dart.j2"
        
        try:
            template = self.env.get_template(template_path)
            logger.debug(f"Loaded template: {template_path}")
            return template
        except Exception as e:
            logger.warning(f"Failed to load template {template_path}: {e}")
            
            # Fallback to basic template
            if architectural_pattern != ArchitecturalPattern.BASIC_PATTERN:
                fallback_path = f"{template_type.value}.dart.j2"
                try:
                    template = self.env.get_template(fallback_path)
                    logger.debug(f"Using fallback template: {fallback_path}")
                    return template
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback template {fallback_path}: {fallback_e}")
            
            return None
    
    def render_template(
        self, 
        template_type: TemplateType, 
        context: TemplateContext,
        variant: Optional[str] = None
    ) -> Optional[str]:
        """Render a template with the given context."""
        
        template = self.get_template(template_type, context.architectural_pattern, variant)
        if not template:
            logger.error(f"No template found for {template_type.value}")
            return None
        
        try:
            # Convert context to dictionary for template rendering
            template_vars = {
                'app_name': context.app_name,
                'app_description': context.app_description,
                'app_name_snake': context.app_name.lower().replace(' ', '_'),
                'app_name_pascal': ''.join(word.capitalize() for word in context.app_name.split()),
                'app_name_camel': context.app_name.replace(' ', ''),
                'architectural_pattern': context.architectural_pattern.value,
                'features': context.features,
                'platform_targets': context.platform_targets,
                'state_management': context.state_management,
                'theme_mode': context.theme_mode,
                **context.custom_variables
            }
            
            rendered = template.render(**template_vars)
            logger.debug(f"Successfully rendered {template_type.value} template")
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template {template_type.value}: {e}")
            return None
    
    def generate_project_structure(self, context: TemplateContext) -> Dict[str, str]:
        """Generate complete project structure with all necessary files."""
        
        project_files = {}
        
        # Generate main app structure
        main_dart = self.render_template(TemplateType.APP, context, "main")
        if main_dart:
            project_files[f"lib/main.dart"] = main_dart
        
        # Generate pubspec.yaml
        pubspec_template = self.get_template_file("pubspec.yaml.j2")
        if pubspec_template:
            pubspec_content = self.env.from_string(pubspec_template).render(
                app_name=context.app_name.lower().replace(' ', '_'),
                app_description=context.app_description,
                features=context.features
            )
            project_files["pubspec.yaml"] = pubspec_content
        
        # Generate basic screens based on features
        for feature in context.features:
            screen_content = self.render_template(
                TemplateType.SCREEN, 
                context, 
                feature.lower()
            )
            if screen_content:
                screen_name = feature.lower().replace(' ', '_')
                project_files[f"lib/screens/{screen_name}_screen.dart"] = screen_content
        
        # Generate models, repositories, etc. based on architectural pattern
        if context.architectural_pattern in [ArchitecturalPattern.CLEAN_ARCHITECTURE, ArchitecturalPattern.BLOC_PATTERN]:
            # Generate data layer
            model_content = self.render_template(TemplateType.MODEL, context)
            if model_content:
                project_files[f"lib/models/app_model.dart"] = model_content
            
            # Generate repository
            repository_content = self.render_template(TemplateType.REPOSITORY, context)
            if repository_content:
                project_files[f"lib/repositories/app_repository.dart"] = repository_content
        
        if context.architectural_pattern == ArchitecturalPattern.BLOC_PATTERN:
            # Generate BLoC
            bloc_content = self.render_template(TemplateType.BLOC, context)
            if bloc_content:
                project_files[f"lib/blocs/app_bloc.dart"] = bloc_content
        
        return project_files
    
    def get_template_file(self, filename: str) -> Optional[str]:
        """Read a template file directly."""
        try:
            file_path = self.templates_dir / filename
            if file_path.exists():
                return file_path.read_text()
            return None
        except Exception as e:
            logger.error(f"Failed to read template file {filename}: {e}")
            return None
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates by type."""
        templates = {}
        
        for template_type in TemplateType:
            templates[template_type.value] = []
            
            # Look for templates in base directory
            pattern = f"{template_type.value}*.dart.j2"
            for template_file in self.templates_dir.glob(pattern):
                templates[template_type.value].append(template_file.stem)
            
            # Look for templates in architectural pattern directories
            for arch_pattern in ArchitecturalPattern:
                arch_dir = self.templates_dir / arch_pattern.value
                if arch_dir.exists():
                    for template_file in arch_dir.glob(pattern):
                        template_name = f"{arch_pattern.value}/{template_file.stem}"
                        templates[template_type.value].append(template_name)
        
        return templates


# Global template engine instance
_template_engine = None

def get_template_engine() -> TemplateEngine:
    """Get the global template engine instance."""
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
    return _template_engine
