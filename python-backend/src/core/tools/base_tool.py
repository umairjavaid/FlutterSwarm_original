"""
Base Tool Framework for FlutterSwarm Multi-Agent System.

This module provides the abstract base class that all specialized tools inherit from.
Every tool provides structured capabilities for agents to perform operations through LLM reasoning.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import uuid

from ...config import get_logger
from ...models.tool_models import (
    ToolStatus, ToolPermission, ToolOperation, ToolResult, ToolCapabilities,
    ToolUsageEntry, ToolMetrics, ToolValidation, ToolUsageExample
)

logger = get_logger("base_tool")


class ToolCategory(Enum):
    """Categories of tools for organization."""
    DEVELOPMENT = "development"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    ANALYSIS = "analysis"
    FILE_SYSTEM = "file_system"
    PROCESS = "process"
    NETWORK = "network"
    SECURITY = "security"
    MONITORING = "monitoring"
    VERSION_CONTROL = "version_control"


@dataclass
class ToolContext:
    """Context information for tool execution."""
    project_path: Optional[str] = None
    agent_id: str = ""
    session_id: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the FlutterSwarm system.
    
    Tools provide structured capabilities that agents can use to perform
    operations. All tool usage is driven by agent reasoning, not hardcoded logic.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str,
        required_permissions: List[ToolPermission],
        category: ToolCategory = ToolCategory.DEVELOPMENT
    ):
        self.name = name
        self.description = description
        self.version = version
        self.required_permissions = required_permissions
        self.category = category
        self.usage_history: List[ToolUsageEntry] = []
        self.metrics = ToolMetrics()
        self.is_available = True
        self.last_health_check = datetime.now()
        self.active_operations: Dict[str, asyncio.Task] = {}
    
    @abstractmethod
    async def get_capabilities(self) -> ToolCapabilities:
        """
        Return detailed capability description including:
        - available_operations: List of operations with parameters
        - input_schemas: JSON schemas for each operation
        - output_schemas: Expected output structures
        - error_codes: Possible errors and meanings
        - resource_requirements: CPU, memory, network needs
        - constraints: Limitations and requirements
        """
        pass
    
    @abstractmethod
    async def validate_params(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters against schema.
        
        Args:
            operation: Operation name to validate
            params: Parameters dictionary
            
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
            operation: Operation to perform
            params: Operation parameters
            operation_id: Optional operation ID for tracking
            
        Returns:
            ToolResult with status, data, and metadata
        """
        pass
    
    @abstractmethod
    async def get_usage_examples(self) -> List[Dict[str, Any]]:
        """
        Provide example usage for agent learning.
        
        Returns:
            List of example usage scenarios with context, parameters, and expected outcomes
        """
        pass
    
    async def check_health(self) -> bool:
        """
        Check if tool is healthy and available.
        
        Returns:
            True if tool is healthy, False otherwise
        """
        try:
            # Basic health check - subclasses can override
            self.last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Health check failed for tool {self.name}: {e}")
            self.is_available = False
            return False
    
    async def record_usage(
        self,
        agent_id: str,
        operation: str,
        params: Dict[str, Any],
        result: ToolResult,
        context: Dict[str, Any]
    ) -> None:
        """Record tool usage for learning and optimization."""
        usage_entry = ToolUsageEntry(
            agent_id=agent_id,
            tool_name=self.name,
            operation=operation,
            parameters=params,
            execution_time=result.execution_time,
            success=(result.status == ToolStatus.SUCCESS),
            error_details=result.error_message,
            context=context,
            outcome_quality=self._calculate_outcome_quality(result),
            agent_satisfaction=context.get("satisfaction", 0.0)
        )
        
        self.usage_history.append(usage_entry)
        await self._update_metrics()
    
    def _calculate_outcome_quality(self, result: ToolResult) -> float:
        """Calculate quality score for the operation outcome."""
        if result.status == ToolStatus.SUCCESS:
            # Base quality score for success
            quality = 0.8
            
            # Adjust based on execution time (faster is better, up to a point)
            if result.execution_time < 1.0:
                quality += 0.1
            elif result.execution_time > 10.0:
                quality -= 0.1
            
            # Adjust based on warnings
            if result.warnings:
                quality -= len(result.warnings) * 0.05
            
            return min(1.0, max(0.0, quality))
        
        elif result.status == ToolStatus.PARTIAL_SUCCESS:
            return 0.5
        else:
            return 0.0
    
    async def _update_metrics(self) -> None:
        """Update tool performance metrics."""
        if not self.usage_history:
            return
        
        recent_history = self.usage_history[-100:]  # Last 100 uses
        
        self.metrics.total_uses = len(self.usage_history)
        self.metrics.success_rate = sum(
            1 for entry in recent_history if entry.success
        ) / len(recent_history)
        
        self.metrics.average_execution_time = sum(
            entry.execution_time for entry in recent_history
        ) / len(recent_history)
        
        # Update error frequency
        error_counts = {}
        for entry in recent_history:
            if entry.error_details:
                error_type = entry.error_details.split(':')[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        self.metrics.error_frequency = error_counts
        
        # Calculate resource efficiency (inverse of average execution time)
        self.metrics.resource_efficiency = 1.0 / (self.metrics.average_execution_time + 0.1)
        
        self.metrics.last_updated = datetime.now()
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for agent decision-making."""
        return {
            "tool_name": self.name,
            "is_available": self.is_available,
            "success_rate": self.metrics.success_rate,
            "average_execution_time": self.metrics.average_execution_time,
            "total_uses": self.metrics.total_uses,
            "recent_errors": list(self.metrics.error_frequency.keys())[:3],
            "resource_efficiency": self.metrics.resource_efficiency,
            "last_health_check": self.last_health_check.isoformat(),
            "recommended_use_cases": await self._get_recommended_use_cases()
        }
    
    async def _get_recommended_use_cases(self) -> List[str]:
        """Get recommended use cases based on success patterns."""
        if not self.usage_history:
            return ["General purpose usage"]
        
        # Analyze successful usage patterns
        successful_operations = [
            entry.operation for entry in self.usage_history
            if entry.success and entry.outcome_quality > 0.7
        ]
        
        # Return most common successful operations
        operation_counts = {}
        for op in successful_operations:
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        return sorted(operation_counts.keys(), key=operation_counts.get, reverse=True)[:5]
    
    def get_schema_for_operation(self, operation: str) -> Dict[str, Any]:
        """Get JSON schema for a specific operation (to be overridden by subclasses)."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def prepare_for_operation(
        self,
        operation: str,
        context: ToolContext
    ) -> bool:
        """
        Prepare tool for operation execution.
        
        Args:
            operation: Operation to prepare for
            context: Execution context
            
        Returns:
            True if preparation successful, False otherwise
        """
        # Basic preparation - subclasses can override
        return await self.check_health()
    
    async def cleanup_after_operation(
        self,
        operation: str,
        result: ToolResult
    ) -> None:
        """Cleanup after operation execution."""
        # Basic cleanup - subclasses can override
        pass
    
    def supports_operation(self, operation: str) -> bool:
        """Check if tool supports a specific operation."""
        # Default implementation - subclasses should override
        return operation in ["help", "status", "info"]
    
    async def get_help_for_operation(self, operation: str) -> str:
        """Get help text for a specific operation."""
        capabilities = await self.get_capabilities()
        for op in capabilities.available_operations:
            if op.get("name") == operation:
                return op.get("description", f"No help available for {operation}")
        return f"Operation '{operation}' not found"
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation if supported."""
        if operation_id in self.active_operations:
            task = self.active_operations[operation_id]
            if not task.done():
                task.cancel()
                return True
        return False
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.category.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
    
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