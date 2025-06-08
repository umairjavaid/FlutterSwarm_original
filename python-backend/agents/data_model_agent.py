"""
Data Model Agent - Creates and manages data models for Flutter applications.
Specializes in entity design, serialization, validation, and data model relationships.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent
from core.agent_types import AgentType, AgentResponse, WorkflowState, TaskStatus

logger = logging.getLogger(__name__)


class DataModelAgent(BaseAgent):
    """
    Specialized agent for creating and managing data models in Flutter applications.
    Handles entity creation, serialization, validation, and data relationships.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.DATA_MODEL, config)
        
        # Serialization libraries
        self.serialization_libraries = {
            "json_annotation": {
                "package": "json_annotation",
                "dev_dependency": "json_serializable",
                "build_runner": "build_runner",
                "features": ["json_serialization", "custom_converters"]
            },
            "freezed": {
                "package": "freezed_annotation",
                "dev_dependency": "freezed",
                "build_runner": "build_runner",
                "features": ["immutable_classes", "union_types", "json_serialization"]
            },
            "built_value": {
                "package": "built_value",
                "dev_dependency": "built_value_generator",
                "build_runner": "build_runner",
                "features": ["immutable_classes", "serialization"]
            }
        }
        
        # Data types mapping
        self.dart_type_mappings = {
            "string": "String",
            "integer": "int",
            "double": "double",
            "boolean": "bool",
            "datetime": "DateTime",
            "list": "List",
            "map": "Map",
            "object": "Object"
        }
        
        # Validation libraries
        self.validation_libraries = {
            "formz": {
                "package": "formz",
                "features": ["input_validation", "form_state_management"]
            },
            "form_validator": {
                "package": "form_validator",
                "features": ["simple_validation"]
            }
        }
        
        # Model generation templates
        self.model_templates = {
            "basic_model": self._get_basic_model_template(),
            "freezed_model": self._get_freezed_model_template(),
            "json_model": self._get_json_model_template(),
            "validation_model": self._get_validation_model_template()
        }
        
        logger.info(f"DataModelAgent initialized with serialization libraries: {list(self.serialization_libraries.keys())}")
    
    def _define_capabilities(self) -> List[str]:
        return [
            "entity_model_creation",
            "data_transfer_object_creation",
            "json_serialization_setup",
            "model_validation_implementation",
            "data_class_generation",
            "immutable_model_creation",
            "model_relationship_mapping",
            "custom_converter_creation",
            "enum_model_generation",
            "nested_model_handling",
            "model_inheritance_setup",
            "data_transformation_logic",
            "model_factory_creation",
            "type_safe_model_design",
            "model_documentation_generation"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process data model creation and management tasks."""
        try:
            task_type = state.project_context.get("task_type", "data_model_setup")
            
            if task_type == "data_model_setup":
                return await self._handle_data_model_setup(state)
            elif task_type == "entity_creation":
                return await self._handle_entity_creation(state)
            elif task_type == "dto_creation":
                return await self._handle_dto_creation(state)
            elif task_type == "serialization_setup":
                return await self._handle_serialization_setup(state)
            elif task_type == "validation_setup":
                return await self._handle_validation_setup(state)
            elif task_type == "model_relationships":
                return await self._handle_model_relationships(state)
            elif task_type == "model_generation":
                return await self._handle_model_generation(state)
            else:
                return await self._handle_generic_data_model(state)
                
        except Exception as e:
            logger.error(f"DataModelAgent task processing failed: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                message=f"Data model task failed: {str(e)}",
                data={"error": str(e)},
                updated_state=state
            )
    
    async def _handle_data_model_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle comprehensive data model setup."""
        try:
            entities = state.project_context.get("entities", [])
            requirements = state.project_context.get("requirements", {})
            
            # Analyze data modeling requirements
            modeling_analysis = await self._analyze_data_modeling_requirements(entities, requirements)
            
            # Choose serialization strategy
            serialization_strategy = await self._choose_serialization_strategy(modeling_analysis)
            
            # Setup data model architecture
            model_architecture = await self._setup_model_architecture(serialization_strategy)
            
            # Generate folder structure
            folder_structure = await self._generate_model_folder_structure()
            
            # Create base model classes
            base_models = await self._create_base_model_classes(serialization_strategy)
            
            # Setup validation framework
            validation_setup = await self._setup_validation_framework(modeling_analysis)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Data model setup completed successfully",
                data={
                    "modeling_analysis": modeling_analysis,
                    "serialization_strategy": serialization_strategy,
                    "model_architecture": model_architecture,
                    "folder_structure": folder_structure,
                    "base_models": base_models,
                    "validation_setup": validation_setup,
                    "dependencies": await self._get_required_dependencies(serialization_strategy),
                    "implementation_guide": await self._generate_model_implementation_guide()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Data model setup failed: {e}")
            raise
    
    async def _handle_entity_creation(self, state: WorkflowState) -> AgentResponse:
        """Handle entity model creation."""
        try:
            entities = state.project_context.get("entities", [])
            serialization_strategy = state.project_context.get("serialization_strategy", "json_annotation")
            
            # Create entity models
            entity_models = await self._create_entity_models(entities, serialization_strategy)
            
            # Generate entity relationships
            entity_relationships = await self._generate_entity_relationships(entities)
            
            # Create entity validation
            entity_validation = await self._create_entity_validation(entities)
            
            # Generate entity factories
            entity_factories = await self._generate_entity_factories(entities)
            
            # Create entity tests
            entity_tests = await self._generate_entity_tests(entities)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Entity models created successfully",
                data={
                    "entity_models": entity_models,
                    "entity_relationships": entity_relationships,
                    "entity_validation": entity_validation,
                    "entity_factories": entity_factories,
                    "entity_tests": entity_tests,
                    "build_configuration": await self._generate_build_configuration()
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Entity creation failed: {e}")
            raise
    
    async def _handle_dto_creation(self, state: WorkflowState) -> AgentResponse:
        """Handle Data Transfer Object creation."""
        try:
            api_endpoints = state.project_context.get("api_endpoints", [])
            entities = state.project_context.get("entities", [])
            
            # Create request DTOs
            request_dtos = await self._create_request_dtos(api_endpoints)
            
            # Create response DTOs
            response_dtos = await self._create_response_dtos(api_endpoints)
            
            # Create entity-DTO mappers
            dto_mappers = await self._create_dto_mappers(entities, api_endpoints)
            
            # Generate DTO validation
            dto_validation = await self._create_dto_validation(api_endpoints)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="DTO models created successfully",
                data={
                    "request_dtos": request_dtos,
                    "response_dtos": response_dtos,
                    "dto_mappers": dto_mappers,
                    "dto_validation": dto_validation,
                    "dto_documentation": await self._generate_dto_documentation(api_endpoints)
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"DTO creation failed: {e}")
            raise
    
    async def _handle_serialization_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle serialization framework setup."""
        try:
            serialization_strategy = state.project_context.get("serialization_strategy", "json_annotation")
            entities = state.project_context.get("entities", [])
            
            # Setup serialization dependencies
            serialization_deps = await self._setup_serialization_dependencies(serialization_strategy)
            
            # Create custom converters
            custom_converters = await self._create_custom_converters(entities)
            
            # Setup build configuration
            build_config = await self._setup_build_configuration(serialization_strategy)
            
            # Generate serialization helpers
            serialization_helpers = await self._create_serialization_helpers()
            
            # Create serialization tests
            serialization_tests = await self._create_serialization_tests(entities)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Serialization setup completed successfully",
                data={
                    "serialization_dependencies": serialization_deps,
                    "custom_converters": custom_converters,
                    "build_configuration": build_config,
                    "serialization_helpers": serialization_helpers,
                    "serialization_tests": serialization_tests
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Serialization setup failed: {e}")
            raise
    
    async def _handle_validation_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle model validation setup."""
        try:
            entities = state.project_context.get("entities", [])
            validation_rules = state.project_context.get("validation_rules", [])
            
            # Setup validation framework
            validation_framework = await self._setup_validation_framework_impl(validation_rules)
            
            # Create validation rules
            validation_implementations = await self._create_validation_implementations(validation_rules)
            
            # Create form validation
            form_validation = await self._create_form_validation(entities)
            
            # Generate validation helpers
            validation_helpers = await self._create_validation_helpers()
            
            # Create validation tests
            validation_tests = await self._create_validation_tests(validation_rules)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Validation setup completed successfully",
                data={
                    "validation_framework": validation_framework,
                    "validation_implementations": validation_implementations,
                    "form_validation": form_validation,
                    "validation_helpers": validation_helpers,
                    "validation_tests": validation_tests
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Validation setup failed: {e}")
            raise
    
    async def _handle_model_relationships(self, state: WorkflowState) -> AgentResponse:
        """Handle model relationship implementation."""
        try:
            entities = state.project_context.get("entities", [])
            relationships = state.project_context.get("relationships", [])
            
            # Create relationship models
            relationship_models = await self._create_relationship_models(relationships)
            
            # Generate foreign key references
            foreign_keys = await self._generate_foreign_key_references(relationships)
            
            # Create relationship helpers
            relationship_helpers = await self._create_relationship_helpers(relationships)
            
            # Generate cascade operations
            cascade_operations = await self._generate_cascade_operations(relationships)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Model relationships implemented successfully",
                data={
                    "relationship_models": relationship_models,
                    "foreign_keys": foreign_keys,
                    "relationship_helpers": relationship_helpers,
                    "cascade_operations": cascade_operations,
                    "relationship_documentation": await self._generate_relationship_documentation(relationships)
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Model relationships handling failed: {e}")
            raise
    
    async def _handle_model_generation(self, state: WorkflowState) -> AgentResponse:
        """Handle automated model generation."""
        try:
            schema_source = state.project_context.get("schema_source", "manual")
            schema_data = state.project_context.get("schema_data", {})
            
            # Parse schema
            parsed_schema = await self._parse_schema(schema_source, schema_data)
            
            # Generate models from schema
            generated_models = await self._generate_models_from_schema(parsed_schema)
            
            # Create model documentation
            model_documentation = await self._create_model_documentation(generated_models)
            
            # Generate model tests
            model_tests = await self._generate_model_tests_from_schema(parsed_schema)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Models generated successfully from schema",
                data={
                    "generated_models": generated_models,
                    "model_documentation": model_documentation,
                    "model_tests": model_tests,
                    "schema_analysis": parsed_schema
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise
    
    # Helper methods for data model analysis and generation
    
    async def _analyze_data_modeling_requirements(self, entities: List, requirements: Dict) -> Dict[str, Any]:
        """Analyze data modeling requirements."""
        complexity_factors = []
        recommended_strategies = []
        
        # Analyze entity complexity
        total_entities = len(entities)
        complex_entities = sum(1 for entity in entities if len(entity.get("properties", [])) > 10)
        
        if complex_entities > 0:
            complexity_factors.append("Complex entities with many properties")
            recommended_strategies.append("freezed")
        
        # Analyze relationships
        relationships = sum(1 for entity in entities for prop in entity.get("properties", []) 
                         if prop.get("type", "").startswith("relationship"))
        
        if relationships > 5:
            complexity_factors.append("Many entity relationships")
            recommended_strategies.append("json_annotation")
        
        # Analyze serialization needs
        has_api = requirements.get("has_api", False)
        has_local_storage = requirements.get("has_local_storage", False)
        
        if has_api or has_local_storage:
            complexity_factors.append("Serialization required")
            if not recommended_strategies:
                recommended_strategies.append("json_annotation")
        
        # Determine complexity level
        complexity_score = len(complexity_factors) + (total_entities // 5)
        if complexity_score <= 2:
            complexity_level = "simple"
        elif complexity_score <= 5:
            complexity_level = "medium"
        else:
            complexity_level = "complex"
        
        return {
            "complexity_level": complexity_level,
            "complexity_factors": complexity_factors,
            "entity_count": total_entities,
            "complex_entity_count": complex_entities,
            "relationship_count": relationships,
            "recommended_strategies": recommended_strategies or ["json_annotation"],
            "serialization_needed": has_api or has_local_storage
        }
    
    async def _choose_serialization_strategy(self, analysis: Dict) -> str:
        """Choose the best serialization strategy."""
        recommended = analysis.get("recommended_strategies", [])
        complexity = analysis.get("complexity_level", "medium")
        
        if "freezed" in recommended and complexity in ["medium", "complex"]:
            return "freezed"
        elif "json_annotation" in recommended:
            return "json_annotation"
        else:
            return "json_annotation"  # Default choice
    
    async def _setup_model_architecture(self, strategy: str) -> Dict[str, Any]:
        """Setup data model architecture."""
        strategy_config = self.serialization_libraries.get(strategy, {})
        
        return {
            "strategy": strategy,
            "features": strategy_config.get("features", []),
            "structure": {
                "models": "lib/models",
                "entities": "lib/models/entities",
                "dtos": "lib/models/dtos",
                "validators": "lib/models/validators",
                "converters": "lib/models/converters"
            },
            "conventions": {
                "naming": "snake_case_files",
                "class_naming": "PascalCase",
                "property_naming": "camelCase"
            }
        }
    
    async def _generate_model_folder_structure(self) -> Dict[str, Any]:
        """Generate folder structure for models."""
        return {
            "lib/models": {
                "entities": [],
                "dtos": {
                    "request": [],
                    "response": []
                },
                "validators": [],
                "converters": [],
                "base": []
            }
        }
    
    async def _create_base_model_classes(self, strategy: str) -> Dict[str, Any]:
        """Create base model classes."""
        base_classes = {}
        
        if strategy == "freezed":
            base_classes["base_entity"] = await self._generate_freezed_base_entity()
            base_classes["base_dto"] = await self._generate_freezed_base_dto()
        else:
            base_classes["base_entity"] = await self._generate_json_base_entity()
            base_classes["base_dto"] = await self._generate_json_base_dto()
        
        return base_classes
    
    async def _setup_validation_framework(self, analysis: Dict) -> Dict[str, Any]:
        """Setup validation framework."""
        complexity = analysis.get("complexity_level", "medium")
        
        if complexity in ["medium", "complex"]:
            return {
                "framework": "formz",
                "features": ["input_validation", "form_state_management"],
                "setup": await self._generate_formz_setup()
            }
        else:
            return {
                "framework": "built_in",
                "features": ["basic_validation"],
                "setup": await self._generate_basic_validation_setup()
            }
    
    # Code generation template methods
    
    def _get_basic_model_template(self) -> str:
        return '''
class {model_name} {{
  {properties}
  
  {model_name}({{
    {constructor_parameters}
  }});
  
  {methods}
}}
'''
    
    def _get_freezed_model_template(self) -> str:
        return '''
import 'package:freezed_annotation/freezed_annotation.dart';

part '{file_name}.freezed.dart';
part '{file_name}.g.dart';

@freezed
class {model_name} with _${model_name} {{
  const factory {model_name}({{
    {properties}
  }}) = _{model_name};
  
  factory {model_name}.fromJson(Map<String, dynamic> json) => 
      _${model_name}FromJson(json);
}}
'''
    
    def _get_json_model_template(self) -> str:
        return '''
import 'package:json_annotation/json_annotation.dart';

part '{file_name}.g.dart';

@JsonSerializable()
class {model_name} {{
  {properties}
  
  {model_name}({{
    {constructor_parameters}
  }});
  
  factory {model_name}.fromJson(Map<String, dynamic> json) => 
      _${model_name}FromJson(json);
  
  Map<String, dynamic> toJson() => _${model_name}ToJson(this);
}}
'''
    
    def _get_validation_model_template(self) -> str:
        return '''
import 'package:formz/formz.dart';

enum {validation_name}ValidationError {{ invalid }}

class {validation_name} extends FormzInput<String, {validation_name}ValidationError> {{
  const {validation_name}.pure() : super.pure('');
  const {validation_name}.dirty([String value = '']) : super.dirty(value);
  
  @override
  {validation_name}ValidationError? validator(String value) {{
    {validation_logic}
  }}
}}
'''
    
    # Implementation methods (placeholders for detailed implementations)
    
    async def _get_required_dependencies(self, strategy: str) -> Dict[str, Any]:
        """Get required dependencies for strategy."""
        strategy_config = self.serialization_libraries.get(strategy, {})
        return {
            "dependencies": {
                strategy_config.get("package"): "latest"
            },
            "dev_dependencies": {
                strategy_config.get("dev_dependency"): "latest",
                "build_runner": "latest"
            }
        }
    
    async def _generate_model_implementation_guide(self) -> Dict[str, Any]:
        """Generate implementation guide for models."""
        return {
            "steps": [
                "1. Define entity models",
                "2. Setup serialization",
                "3. Create validation rules",
                "4. Generate code with build_runner"
            ],
            "best_practices": [
                "Use immutable models",
                "Implement proper validation",
                "Document model relationships"
            ]
        }
    
    async def _create_entity_models(self, entities: List, strategy: str) -> Dict[str, Any]:
        """Create entity model implementations."""
        return {"placeholder": "entity_model_implementations"}
    
    async def _generate_entity_relationships(self, entities: List) -> Dict[str, Any]:
        """Generate entity relationships."""
        return {"placeholder": "entity_relationships"}
    
    async def _create_entity_validation(self, entities: List) -> Dict[str, Any]:
        """Create entity validation."""
        return {"placeholder": "entity_validation"}
    
    async def _generate_entity_factories(self, entities: List) -> Dict[str, Any]:
        """Generate entity factories."""
        return {"placeholder": "entity_factories"}
    
    async def _generate_entity_tests(self, entities: List) -> Dict[str, Any]:
        """Generate entity tests."""
        return {"placeholder": "entity_tests"}
    
    async def _generate_build_configuration(self) -> Dict[str, Any]:
        """Generate build configuration."""
        return {"placeholder": "build_configuration"}
    
    async def _create_request_dtos(self, endpoints: List) -> Dict[str, Any]:
        """Create request DTOs."""
        return {"placeholder": "request_dtos"}
    
    async def _create_response_dtos(self, endpoints: List) -> Dict[str, Any]:
        """Create response DTOs."""
        return {"placeholder": "response_dtos"}
    
    async def _create_dto_mappers(self, entities: List, endpoints: List) -> Dict[str, Any]:
        """Create DTO mappers."""
        return {"placeholder": "dto_mappers"}
    
    async def _create_dto_validation(self, endpoints: List) -> Dict[str, Any]:
        """Create DTO validation."""
        return {"placeholder": "dto_validation"}
    
    async def _generate_dto_documentation(self, endpoints: List) -> Dict[str, Any]:
        """Generate DTO documentation."""
        return {"placeholder": "dto_documentation"}
    
    async def _setup_serialization_dependencies(self, strategy: str) -> Dict[str, Any]:
        """Setup serialization dependencies."""
        return {"placeholder": "serialization_dependencies"}
    
    async def _create_custom_converters(self, entities: List) -> Dict[str, Any]:
        """Create custom converters."""
        return {"placeholder": "custom_converters"}
    
    async def _setup_build_configuration(self, strategy: str) -> Dict[str, Any]:
        """Setup build configuration."""
        return {"placeholder": "build_configuration"}
    
    async def _create_serialization_helpers(self) -> Dict[str, Any]:
        """Create serialization helpers."""
        return {"placeholder": "serialization_helpers"}
    
    async def _create_serialization_tests(self, entities: List) -> Dict[str, Any]:
        """Create serialization tests."""
        return {"placeholder": "serialization_tests"}
    
    async def _setup_validation_framework_impl(self, rules: List) -> Dict[str, Any]:
        """Setup validation framework implementation."""
        return {"placeholder": "validation_framework"}
    
    async def _create_validation_implementations(self, rules: List) -> Dict[str, Any]:
        """Create validation implementations."""
        return {"placeholder": "validation_implementations"}
    
    async def _create_form_validation(self, entities: List) -> Dict[str, Any]:
        """Create form validation."""
        return {"placeholder": "form_validation"}
    
    async def _create_validation_helpers(self) -> Dict[str, Any]:
        """Create validation helpers."""
        return {"placeholder": "validation_helpers"}
    
    async def _create_validation_tests(self, rules: List) -> Dict[str, Any]:
        """Create validation tests."""
        return {"placeholder": "validation_tests"}
    
    async def _create_relationship_models(self, relationships: List) -> Dict[str, Any]:
        """Create relationship models."""
        return {"placeholder": "relationship_models"}
    
    async def _generate_foreign_key_references(self, relationships: List) -> Dict[str, Any]:
        """Generate foreign key references."""
        return {"placeholder": "foreign_keys"}
    
    async def _create_relationship_helpers(self, relationships: List) -> Dict[str, Any]:
        """Create relationship helpers."""
        return {"placeholder": "relationship_helpers"}
    
    async def _generate_cascade_operations(self, relationships: List) -> Dict[str, Any]:
        """Generate cascade operations."""
        return {"placeholder": "cascade_operations"}
    
    async def _generate_relationship_documentation(self, relationships: List) -> Dict[str, Any]:
        """Generate relationship documentation."""
        return {"placeholder": "relationship_documentation"}
    
    async def _parse_schema(self, source: str, data: Dict) -> Dict[str, Any]:
        """Parse schema from various sources."""
        return {"placeholder": "parsed_schema"}
    
    async def _generate_models_from_schema(self, schema: Dict) -> Dict[str, Any]:
        """Generate models from schema."""
        return {"placeholder": "generated_models"}
    
    async def _create_model_documentation(self, models: Dict) -> Dict[str, Any]:
        """Create model documentation."""
        return {"placeholder": "model_documentation"}
    
    async def _generate_model_tests_from_schema(self, schema: Dict) -> Dict[str, Any]:
        """Generate model tests from schema."""
        return {"placeholder": "model_tests"}
    
    async def _generate_freezed_base_entity(self) -> Dict[str, Any]:
        """Generate freezed base entity."""
        return {"placeholder": "freezed_base_entity"}
    
    async def _generate_freezed_base_dto(self) -> Dict[str, Any]:
        """Generate freezed base DTO."""
        return {"placeholder": "freezed_base_dto"}
    
    async def _generate_json_base_entity(self) -> Dict[str, Any]:
        """Generate JSON base entity."""
        return {"placeholder": "json_base_entity"}
    
    async def _generate_json_base_dto(self) -> Dict[str, Any]:
        """Generate JSON base DTO."""
        return {"placeholder": "json_base_dto"}
    
    async def _generate_formz_setup(self) -> Dict[str, Any]:
        """Generate formz setup."""
        return {"placeholder": "formz_setup"}
    
    async def _generate_basic_validation_setup(self) -> Dict[str, Any]:
        """Generate basic validation setup."""
        return {"placeholder": "basic_validation_setup"}
    
    async def _handle_generic_data_model(self, state: WorkflowState) -> AgentResponse:
        """Handle generic data model tasks."""
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            message="Generic data model task completed",
            data={"task_type": "generic_data_model"},
            updated_state=state
        )
