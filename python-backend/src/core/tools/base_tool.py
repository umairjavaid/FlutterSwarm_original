"""
Base Tool Framework for FlutterSwarm Multi-Agent System.

This module provides the abstract base class that all specialized tools inherit from.
Every tool provides structured capabilities for agents to perform operations through LLM reasoning.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from ...config import get_logger

logger = get_logger("base_tool")


class ToolStatus(Enum):
    """Status of tool operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolPermission(Enum):
    """Required permissions for tool operations."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    NETWORK_ACCESS = "network_access"
    PROCESS_SPAWN = "process_spawn"
    PROCESS_KILL = "process_kill"
    SYSTEM_INFO = "system_info"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"


@dataclass
class ToolOperation:
    """Definition of a tool operation with parameters and schemas."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    required_permissions: List[ToolPermission] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    error_codes: Dict[str, str] = field(default_factory=dict)
    is_destructive: bool = False
    estimated_duration: Optional[int] = None  # seconds
    supports_cancellation: bool = False


@dataclass
class ToolResult:
    """Result of a tool operation execution."""
    operation: str
    status: ToolStatus
    data: Any = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    partial_results: List[Any] = field(default_factory=list)
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def is_successful(self) -> bool:
        """Check if the operation was successful."""
        return self.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS]
    
    def mark_completed(self) -> None:
        """Mark the operation as completed."""
        self.completed_at = datetime.utcnow()
        if self.execution_time is None:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation,
            "status": self.status.value,
            "data": self.data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "partial_results": self.partial_results,
            "operation_id": self.operation_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "successful": self.is_successful()
        }


@dataclass
class ToolCapabilities:
    """Comprehensive tool capability description for LLM understanding."""
    available_operations: List[ToolOperation] = field(default_factory=list)
    input_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_codes: Dict[str, str] = field(default_factory=dict)
    usage_examples: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    supported_file_types: List[str] = field(default_factory=list)
    max_file_size: Optional[int] = None
    concurrent_operations: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "available_operations": [op.__dict__ for op in self.available_operations],
            "input_schemas": self.input_schemas,
            "output_schemas": self.output_schemas,
            "error_codes": self.error_codes,
            "usage_examples": self.usage_examples,
            "limitations": self.limitations,
            "dependencies": self.dependencies,
            "supported_file_types": self.supported_file_types,
            "max_file_size": self.max_file_size,
            "concurrent_operations": self.concurrent_operations
        }


class BaseTool(ABC):
    """
    Abstract base class for all FlutterSwarm tools.
    
    This class defines the core interface and common functionality that all
    specialized tools must implement. Tools provide structured capabilities
    for agents to perform operations while maintaining safety and traceability.
    
    Attributes:
        name: Unique tool identifier
        description: Detailed capability description for LLM understanding
        version: Tool version for compatibility tracking
        required_permissions: List of permissions needed for tool operations
        active_operations: Currently executing operations
        operation_history: History of tool operations for learning
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        required_permissions: Optional[List[ToolPermission]] = None
    ):
        """
        Initialize the base tool.
        
        Args:
            name: Unique tool identifier
            description: Detailed capability description
            version: Tool version string
            required_permissions: List of required permissions
        """
        self.name = name
        self.description = description
        self.version = version
        self.required_permissions = required_permissions or []
        self.active_operations: Dict[str, ToolResult] = {}
        self.operation_history: List[ToolResult] = []
        self.is_available = True
        self.last_health_check = datetime.utcnow()
        
        logger.info(
            f"Tool initialized: {self.name} v{self.version}",
            extra={
                "tool_name": self.name,
                "version": self.version,
                "permissions": [p.value for p in self.required_permissions]
            }
        )
    
    @abstractmethod
    async def get_capabilities(self) -> ToolCapabilities:
        """
        Get detailed capability description including operations, schemas, and examples.
        
        Returns:
            ToolCapabilities object with comprehensive tool information
        """
        pass
    
    @abstractmethod
    async def validate_params(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters against operation schema.
        
        Args:
            operation: Name of the operation to validate
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        operation: str,
        params: Dict[str, Any],
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Execute the requested operation.
        
        Args:
            operation: Name of the operation to execute
            params: Parameters for the operation
            operation_id: Optional operation ID for tracking
            
        Returns:
            ToolResult with operation outcome and data
        """
        pass
    
    @abstractmethod
    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """
        Provide example usage for agent learning.
        
        Returns:
            List of usage examples with context and expected results
        """
        pass
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel a running operation if supported.
        
        Args:
            operation_id: ID of the operation to cancel
            
        Returns:
            True if cancellation was successful
        """
        if operation_id not in self.active_operations:
            return False
        
        operation = self.active_operations[operation_id]
        if hasattr(self, '_cancel_operation_impl'):
            success = await self._cancel_operation_impl(operation_id)
            if success:
                operation.status = ToolStatus.CANCELLED
                operation.mark_completed()
                self._move_to_history(operation_id)
            return success
        
        return False
    
    async def get_operation_status(self, operation_id: str) -> Optional[ToolResult]:
        """
        Get the status of a running or completed operation.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            ToolResult if operation exists, None otherwise
        """
        if operation_id in self.active_operations:
            return self.active_operations[operation_id]
        
        # Check history
        for result in self.operation_history:
            if result.operation_id == operation_id:
                return result
        
        return None
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the tool.
        
        Returns:
            True if tool is healthy and available
        """
        try:
            self.last_health_check = datetime.utcnow()
            
            # Basic availability check
            if not self.is_available:
                return False
            
            # Tool-specific health check if implemented
            if hasattr(self, '_health_check_impl'):
                return await self._health_check_impl()
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for tool {self.name}: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get tool usage metrics and statistics.
        
        Returns:
            Dictionary containing tool metrics
        """
        total_operations = len(self.operation_history)
        successful_operations = sum(
            1 for result in self.operation_history
            if result.is_successful()
        )
        
        avg_execution_time = 0.0
        if self.operation_history:
            execution_times = [
                result.execution_time for result in self.operation_history
                if result.execution_time is not None
            ]
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "tool_name": self.name,
            "version": self.version,
            "is_available": self.is_available,
            "active_operations": len(self.active_operations),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": successful_operations / max(total_operations, 1),
            "average_execution_time": avg_execution_time,
            "last_health_check": self.last_health_check.isoformat(),
            "operation_types": self._get_operation_type_stats()
        }
    
    def _get_operation_type_stats(self) -> Dict[str, int]:
        """Get statistics by operation type."""
        stats = {}
        for result in self.operation_history:
            op_type = result.operation
            stats[op_type] = stats.get(op_type, 0) + 1
        return stats
    
    def _move_to_history(self, operation_id: str) -> None:
        """Move completed operation to history."""
        if operation_id in self.active_operations:
            result = self.active_operations.pop(operation_id)
            self.operation_history.append(result)
            
            # Keep history size manageable
            if len(self.operation_history) > 1000:
                self.operation_history = self.operation_history[-800:]
    
    def _validate_common_params(
        self,
        params: Dict[str, Any],
        required_params: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate common parameter requirements.
        
        Args:
            params: Parameters to validate
            required_params: List of required parameter names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
            
            if params[param] is None:
                return False, f"Parameter '{param}' cannot be None"
        
        return True, None
    
    def _create_error_result(
        self,
        operation: str,
        error_message: str,
        error_code: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Create a standardized error result.
        
        Args:
            operation: Name of the failed operation
            error_message: Description of the error
            error_code: Optional error code
            operation_id: Optional operation ID
            
        Returns:
            ToolResult with error information
        """
        result = ToolResult(
            operation=operation,
            status=ToolStatus.FAILURE,
            error_message=error_message,
            error_code=error_code,
            operation_id=operation_id or str(uuid.uuid4())
        )
        result.mark_completed()
        return result
    
    def _create_success_result(
        self,
        operation: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        operation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Create a standardized success result.
        
        Args:
            operation: Name of the successful operation
            data: Result data
            metadata: Optional metadata
            warnings: Optional warnings
            operation_id: Optional operation ID
            
        Returns:
            ToolResult with success information
        """
        result = ToolResult(
            operation=operation,
            status=ToolStatus.SUCCESS,
            data=data,
            metadata=metadata or {},
            warnings=warnings or [],
            operation_id=operation_id or str(uuid.uuid4())
        )
        result.mark_completed()
        return result
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name} v{self.version} - {self.description[:100]}..."
    
    def __repr__(self) -> str:
        """Detailed representation of the tool."""
        return (
            f"<{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"available={self.is_available}, "
            f"active_ops={len(self.active_operations)})>"
        )