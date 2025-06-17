"""
Tool Models for FlutterSwarm Multi-Agent System.

This module provides supporting data structures and utilities for the tool system,
including operation schemas, results, and validation functions.
"""

import json
import jsonschema
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4


class ToolResultStatus(Enum):
    """Status enumeration for tool operation results."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """
    Result of a tool operation execution.
    
    This class encapsulates the outcome of any tool operation, providing
    structured information about success/failure, data, errors, and metadata.
    """
    
    # Core result information
    status: ToolResultStatus
    operation_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Operation output
    data: Optional[Dict[str, Any]] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Execution metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization to set default timestamps."""
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        
        # Calculate duration if end_time is set
        if self.end_time and self.start_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000
    
    def mark_completed(self) -> None:
        """Mark the operation as completed and calculate duration."""
        self.end_time = datetime.utcnow()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000
    
    def add_error(self, error_message: str) -> None:
        """Add an error message and update status if needed."""
        self.errors.append(error_message)
        if self.status == ToolResultStatus.SUCCESS:
            self.status = ToolResultStatus.ERROR
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message and update status if needed."""
        self.warnings.append(warning_message)
        if self.status == ToolResultStatus.SUCCESS:
            self.status = ToolResultStatus.WARNING
    
    def is_successful(self) -> bool:
        """Check if the operation was successful."""
        return self.status in [ToolResultStatus.SUCCESS, ToolResultStatus.WARNING]
    
    def has_errors(self) -> bool:
        """Check if the operation has any errors."""
        return len(self.errors) > 0 or self.status == ToolResultStatus.ERROR
    
    def has_warnings(self) -> bool:
        """Check if the operation has any warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create ToolResult from dictionary."""
        return cls(
            operation_id=data.get("operation_id", str(uuid4())),
            status=ToolResultStatus(data.get("status", "error")),
            data=data.get("data"),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            duration_ms=data.get("duration_ms")
        )


@dataclass
class OperationSchema:
    """
    Schema definition for a tool operation.
    
    This class defines the structure, parameters, and validation rules
    for a specific tool operation, enabling proper parameter validation
    and documentation generation.
    """
    
    # Basic operation information
    name: str
    description: str
    
    # Parameter schema (JSON Schema format)
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    
    # Optional fields
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Response schema
    response_schema: Optional[Dict[str, Any]] = None
    
    # Operation metadata
    deprecated: bool = False
    version: str = "1.0.0"
    
    def get_json_schema(self) -> Dict[str, Any]:
        """Get complete JSON schema for parameter validation."""
        schema = {
            "type": "object",
            "properties": self.parameters,
            "required": self.required_params,
            "additionalProperties": False
        }
        
        # Add description and examples
        if self.description:
            schema["description"] = self.description
        
        if self.examples:
            schema["examples"] = self.examples
        
        return schema
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters against the schema.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Use jsonschema for validation
            schema = self.get_json_schema()
            jsonschema.validate(params, schema)
            return True, []
            
        except jsonschema.ValidationError as e:
            errors.append(f"Validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
        
        return False, errors
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific parameter."""
        return self.parameters.get(param_name)
    
    def is_required_parameter(self, param_name: str) -> bool:
        """Check if a parameter is required."""
        return param_name in self.required_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required_params": self.required_params,
            "examples": self.examples,
            "tags": self.tags,
            "response_schema": self.response_schema,
            "deprecated": self.deprecated,
            "version": self.version
        }


@dataclass
class ToolCapabilities:
    """
    Complete capability description for a tool.
    
    This class aggregates all the information about what a tool can do,
    including available operations, schemas, constraints, and performance
    characteristics.
    """
    
    # Tool identification
    tool_name: str
    tool_version: str
    
    # Operations
    operations: List[OperationSchema] = field(default_factory=list)
    
    # Constraints and requirements
    constraints: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    
    # Performance characteristics
    performance_info: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def get_operation_schema(self, operation_name: str) -> Optional[OperationSchema]:
        """Get schema for a specific operation."""
        for op in self.operations:
            if op.name == operation_name:
                return op
        return None
    
    def supports_operation(self, operation_name: str) -> bool:
        """Check if tool supports a specific operation."""
        return any(op.name == operation_name for op in self.operations)
    
    def get_operation_names(self) -> List[str]:
        """Get list of all supported operation names."""
        return [op.name for op in self.operations]
    
    def validate_operation_params(self, operation_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameters for a specific operation."""
        schema = self.get_operation_schema(operation_name)
        if not schema:
            return False, [f"Operation '{operation_name}' not supported"]
        
        return schema.validate_parameters(params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "operations": [op.to_dict() for op in self.operations],
            "constraints": self.constraints,
            "requirements": self.requirements,
            "performance_info": self.performance_info,
            "resource_requirements": self.resource_requirements
        }


@dataclass
class ToolOperation:
    """
    Describes an operation that a tool can perform.
    
    This includes metadata about the operation, its parameters,
    expected results, and usage examples.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    example_usage: Optional[Dict[str, Any]] = None
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate that required parameters are provided."""
        missing_params = [p for p in self.required_params if p not in params]
        if missing_params:
            return False, f"Missing required parameters: {', '.join(missing_params)}"
        return True, None


# Utility Functions

def create_success_result(
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    operation_id: Optional[str] = None
) -> ToolResult:
    """
    Create a successful tool result.
    
    Args:
        data: Operation output data
        metadata: Additional metadata
        operation_id: Optional operation ID
        
    Returns:
        ToolResult with success status
    """
    result = ToolResult(
        status=ToolResultStatus.SUCCESS,
        data=data or {},
        metadata=metadata or {},
        operation_id=operation_id or str(uuid4())
    )
    result.mark_completed()
    return result


def create_error_result(
    error_message: str,
    error_code: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    operation_id: Optional[str] = None
) -> ToolResult:
    """
    Create an error tool result.
    
    Args:
        error_message: Error description
        error_code: Optional error code
        metadata: Additional metadata
        operation_id: Optional operation ID
        
    Returns:
        ToolResult with error status
    """
    result_metadata = metadata or {}
    if error_code:
        result_metadata["error_code"] = error_code
    
    result = ToolResult(
        status=ToolResultStatus.ERROR,
        errors=[error_message],
        metadata=result_metadata,
        operation_id=operation_id or str(uuid4())
    )
    result.mark_completed()
    return result


def create_warning_result(
    data: Optional[Dict[str, Any]] = None,
    warning_message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    operation_id: Optional[str] = None
) -> ToolResult:
    """
    Create a warning tool result.
    
    Args:
        data: Operation output data
        warning_message: Warning description
        metadata: Additional metadata
        operation_id: Optional operation ID
        
    Returns:
        ToolResult with warning status
    """
    result = ToolResult(
        status=ToolResultStatus.WARNING,
        data=data or {},
        warnings=[warning_message] if warning_message else [],
        metadata=metadata or {},
        operation_id=operation_id or str(uuid4())
    )
    result.mark_completed()
    return result


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate data against JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        jsonschema.validate(data, schema)
        return True, []
        
    except jsonschema.ValidationError as e:
        errors.append(f"Validation error at {e.json_path}: {e.message}")
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
    
    return False, errors


def merge_schemas(*schemas: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple JSON schemas into one.
    
    Args:
        schemas: JSON schemas to merge
        
    Returns:
        Combined schema
    """
    merged = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for schema in schemas:
        if isinstance(schema, dict):
            # Merge properties
            if "properties" in schema:
                merged["properties"].update(schema["properties"])
            
            # Merge required fields
            if "required" in schema:
                merged["required"].extend(schema["required"])
    
    # Remove duplicates from required
    merged["required"] = list(set(merged["required"]))
    
    return merged


def format_operation_help(schema: OperationSchema) -> str:
    """
    Format operation schema as human-readable help text.
    
    Args:
        schema: Operation schema
        
    Returns:
        Formatted help text
    """
    help_text = f"Operation: {schema.name}\n"
    help_text += f"Description: {schema.description}\n\n"
    
    if schema.parameters:
        help_text += "Parameters:\n"
        for param_name, param_info in schema.parameters.items():
            required = " (required)" if param_name in schema.required_params else ""
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            help_text += f"  - {param_name} ({param_type}){required}: {param_desc}\n"
    
    if schema.examples:
        help_text += "\nExamples:\n"
        for i, example in enumerate(schema.examples[:3]):  # Show first 3 examples
            help_text += f"  Example {i+1}:\n"
            help_text += f"    {json.dumps(example, indent=4)}\n"
    
    return help_text


def create_basic_operation_schema(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    required_params: Optional[List[str]] = None
) -> OperationSchema:
    """
    Create a basic operation schema with common defaults.
    
    Args:
        name: Operation name
        description: Operation description
        parameters: Parameter definitions
        required_params: Required parameter names
        
    Returns:
        OperationSchema instance
    """
    return OperationSchema(
        name=name,
        description=description,
        parameters=parameters or {},
        required_params=required_params or []
    )


# Common parameter schemas for reuse
COMMON_PARAMETERS = {
    "path": {
        "type": "string",
        "description": "File or directory path",
        "pattern": r"^[^<>:\"|?*]+$"
    },
    "content": {
        "type": "string",
        "description": "Text content"
    },
    "encoding": {
        "type": "string",
        "description": "Text encoding",
        "default": "utf-8",
        "enum": ["utf-8", "ascii", "latin-1"]
    },
    "project_path": {
        "type": "string",
        "description": "Path to Flutter project root",
        "pattern": r"^[^<>:\"|?*]+$"
    },
    "timeout": {
        "type": "number",
        "description": "Operation timeout in seconds",
        "minimum": 0,
        "default": 30
    },
    "verbose": {
        "type": "boolean",
        "description": "Enable verbose output",
        "default": False
    }
}


def get_common_parameter(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a common parameter definition.
    
    Args:
        name: Parameter name
        
    Returns:
        Parameter definition or None if not found
    """
    return COMMON_PARAMETERS.get(name)


# Exception classes for tool operations
class ToolValidationError(Exception):
    """Raised when tool parameter validation fails."""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    
    def __init__(self, message: str, operation: str = "", tool_name: str = ""):
        super().__init__(message)
        self.operation = operation
        self.tool_name = tool_name


class ToolSchemaError(Exception):
    """Raised when tool schema is invalid."""
    pass

