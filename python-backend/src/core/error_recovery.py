"""
Error Recovery System for FlutterSwarm Multi-Agent System.

This module provides comprehensive error handling, recovery mechanisms,
and fault tolerance for the FlutterSwarm workflow system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from ..config import get_logger
from ..models.agent_models import AgentMessage, TaskResult
from ..core.langsmith_integration import get_tracer

logger = get_logger("error_recovery")


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryAction:
    """Recovery action to be taken."""
    strategy: RecoveryStrategy
    action_data: Dict[str, Any]
    estimated_time: float
    success_probability: float
    fallback_strategy: Optional[RecoveryStrategy] = None


class ErrorRecoveryManager:
    """
    Manages error recovery and fault tolerance for the FlutterSwarm system.
    
    Provides intelligent error analysis, recovery strategy selection,
    and automated recovery execution.
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.tracer = get_tracer()
        
        # Register default recovery handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error recovery handlers."""
        self.recovery_handlers.update({
            "llm_timeout": self._handle_llm_timeout,
            "llm_rate_limit": self._handle_llm_rate_limit,
            "agent_communication_error": self._handle_communication_error,
            "task_execution_error": self._handle_task_execution_error,
            "memory_error": self._handle_memory_error,
            "system_overload": self._handle_system_overload,
            "validation_error": self._handle_validation_error
        })
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle an error with intelligent recovery strategy selection.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            Tuple of (recovery_attempted, recovery_action)
        """
        try:
            # Create error context
            error_context = await self._create_error_context(error, context)
            
            # Log error to history
            self.error_history.append(error_context)
            
            # Trace error if tracer is available
            if self.tracer:
                await self.tracer.trace_error(
                    error_type=error_context.error_type,
                    error_message=error_context.error_message,
                    context=error_context.context_data,
                    agent_id=error_context.agent_id,
                    task_id=error_context.task_id
                )
            
            # Analyze error patterns
            pattern_analysis = await self._analyze_error_patterns(error_context)
            
            # Select recovery strategy
            recovery_action = await self._select_recovery_strategy(
                error_context, pattern_analysis
            )
            
            if recovery_action:
                # Execute recovery
                success = await self._execute_recovery(error_context, recovery_action)
                return success, recovery_action
            
            return False, None
            
        except Exception as recovery_error:
            logger.error(f"Error in error recovery system: {recovery_error}")
            return False, None
    
    async def _create_error_context(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> ErrorContext:
        """Create error context from exception and context."""
        import uuid
        import traceback
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine severity based on error type and context
        severity = self._determine_error_severity(error_type, context)
        
        return ErrorContext(
            error_id=str(uuid.uuid4()),
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            agent_id=context.get("agent_id"),
            task_id=context.get("task_id"),
            workflow_id=context.get("workflow_id"),
            stack_trace=traceback.format_exc(),
            context_data=context,
            retry_count=context.get("retry_count", 0),
            max_retries=context.get("max_retries", 3)
        )
    
    def _determine_error_severity(
        self,
        error_type: str,
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        critical_errors = [
            "SystemExit", "KeyboardInterrupt", "MemoryError",
            "SystemError", "ImportError"
        ]
        
        high_severity_errors = [
            "ConnectionError", "TimeoutError", "PermissionError",
            "FileNotFoundError", "DatabaseError"
        ]
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_severity_errors:
            return ErrorSeverity.HIGH
        elif context.get("retry_count", 0) > 2:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    async def _analyze_error_patterns(
        self,
        error_context: ErrorContext
    ) -> Dict[str, Any]:
        """Analyze error patterns to inform recovery strategy."""
        recent_errors = [
            err for err in self.error_history[-10:]  # Last 10 errors
            if (datetime.utcnow() - err.timestamp) < timedelta(minutes=30)
        ]
        
        # Analyze patterns
        error_frequency = {}
        agent_error_counts = {}
        
        for error in recent_errors:
            error_frequency[error.error_type] = error_frequency.get(error.error_type, 0) + 1
            if error.agent_id:
                agent_error_counts[error.agent_id] = agent_error_counts.get(error.agent_id, 0) + 1
        
        return {
            "recent_error_count": len(recent_errors),
            "error_frequency": error_frequency,
            "agent_error_counts": agent_error_counts,
            "is_recurring": error_frequency.get(error_context.error_type, 0) > 2,
            "system_health": self._assess_system_health(recent_errors)
        }
    
    def _assess_system_health(self, recent_errors: List[ErrorContext]) -> str:
        """Assess overall system health based on recent errors."""
        if len(recent_errors) > 20:
            return "poor"
        elif len(recent_errors) > 10:
            return "degraded"
        elif len(recent_errors) > 5:
            return "fair"
        else:
            return "good"
    
    async def _select_recovery_strategy(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> Optional[RecoveryAction]:
        """Select the best recovery strategy for the error."""
        
        # Check if we have a specific handler for this error type
        handler_key = self._get_handler_key(error_context.error_type)
        if handler_key in self.recovery_handlers:
            return await self.recovery_handlers[handler_key](error_context, pattern_analysis)
        
        # Default strategy selection based on severity and patterns
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                action_data={"reason": "Critical error requires manual intervention"},
                estimated_time=0,
                success_probability=0
            )
        
        elif error_context.retry_count < error_context.max_retries:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action_data={
                    "delay": min(2 ** error_context.retry_count, 30),  # Exponential backoff
                    "max_retries": error_context.max_retries
                },
                estimated_time=min(2 ** error_context.retry_count, 30),
                success_probability=0.7 - (0.2 * error_context.retry_count)
            )
        
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action_data={"fallback_method": "simplified_execution"},
                estimated_time=10,
                success_probability=0.4
            )
    
    def _get_handler_key(self, error_type: str) -> str:
        """Get handler key for error type."""
        error_mapping = {
            "TimeoutError": "llm_timeout",
            "RateLimitError": "llm_rate_limit",
            "ConnectionError": "agent_communication_error",
            "TaskExecutionError": "task_execution_error",
            "MemoryError": "memory_error",
            "ValidationError": "validation_error"
        }
        return error_mapping.get(error_type, error_type.lower())
    
    async def _execute_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> bool:
        """Execute the selected recovery action."""
        try:
            logger.info(f"Executing recovery strategy: {recovery_action.strategy.value}")
            
            if recovery_action.strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry(error_context, recovery_action)
            
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback(error_context, recovery_action)
            
            elif recovery_action.strategy == RecoveryStrategy.SKIP:
                return await self._execute_skip(error_context, recovery_action)
            
            elif recovery_action.strategy == RecoveryStrategy.ABORT:
                return await self._execute_abort(error_context, recovery_action)
            
            else:
                logger.warning(f"Unknown recovery strategy: {recovery_action.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
    
    async def _execute_retry(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> bool:
        """Execute retry recovery strategy."""
        delay = recovery_action.action_data.get("delay", 5)
        await asyncio.sleep(delay)
        
        # Update retry count in context
        error_context.retry_count += 1
        
        logger.info(f"Retrying after {delay}s (attempt {error_context.retry_count})")
        return True
    
    async def _execute_fallback(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> bool:
        """Execute fallback recovery strategy."""
        fallback_method = recovery_action.action_data.get("fallback_method")
        logger.info(f"Using fallback method: {fallback_method}")
        
        # Implement fallback logic based on method
        if fallback_method == "simplified_execution":
            return await self._simplified_execution_fallback(error_context)
        
        return True
    
    async def _execute_skip(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> bool:
        """Execute skip recovery strategy."""
        logger.info(f"Skipping failed operation: {error_context.task_id}")
        return True
    
    async def _execute_abort(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> bool:
        """Execute abort recovery strategy."""
        reason = recovery_action.action_data.get("reason", "Critical error")
        logger.error(f"Aborting operation: {reason}")
        return False
    
    async def _simplified_execution_fallback(
        self,
        error_context: ErrorContext
    ) -> bool:
        """Implement simplified execution fallback."""
        # This would implement a simplified version of the failed operation
        logger.info("Executing simplified fallback operation")
        return True
    
    # Specific error handlers
    async def _handle_llm_timeout(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle LLM timeout errors."""
        if error_context.retry_count < 2:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action_data={"delay": 10, "timeout_increase": 50},
                estimated_time=15,
                success_probability=0.8
            )
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action_data={"fallback_method": "cached_response"},
                estimated_time=5,
                success_probability=0.6
            )
    
    async def _handle_llm_rate_limit(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle LLM rate limit errors."""
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action_data={"delay": 60, "provider_switch": True},
            estimated_time=65,
            success_probability=0.9
        )
    
    async def _handle_communication_error(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle agent communication errors."""
        if pattern_analysis.get("is_recurring"):
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action_data={"fallback_method": "direct_execution"},
                estimated_time=10,
                success_probability=0.7
            )
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action_data={"delay": 5},
                estimated_time=10,
                success_probability=0.8
            )
    
    async def _handle_task_execution_error(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle task execution errors."""
        if error_context.retry_count < 1:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action_data={"delay": 3, "simplified_mode": True},
                estimated_time=8,
                success_probability=0.7
            )
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                action_data={"partial_completion": True},
                estimated_time=1,
                success_probability=1.0
            )
    
    async def _handle_memory_error(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle memory-related errors."""
        return RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action_data={"cleanup_memory": True, "reduced_context": True},
            estimated_time=5,
            success_probability=0.8
        )
    
    async def _handle_system_overload(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle system overload errors."""
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action_data={"delay": 30, "reduced_concurrency": True},
            estimated_time=35,
            success_probability=0.9
        )
    
    async def _handle_validation_error(
        self,
        error_context: ErrorContext,
        pattern_analysis: Dict[str, Any]
    ) -> RecoveryAction:
        """Handle validation errors."""
        return RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action_data={"relaxed_validation": True},
            estimated_time=2,
            success_probability=0.6
        )
    
    def register_recovery_handler(
        self,
        error_type: str,
        handler: Callable
    ):
        """Register a custom recovery handler."""
        self.recovery_handlers[error_type] = handler
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        recent_errors = [
            err for err in self.error_history
            if (datetime.utcnow() - err.timestamp) < timedelta(hours=24)
        ]
        
        error_types = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors_24h": len(recent_errors),
            "error_types": error_types,
            "severity_distribution": severity_counts,
            "recovery_success_rate": self._calculate_recovery_success_rate()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate the recovery success rate."""
        # This would track successful recoveries vs failures
        # For now, return a placeholder
        return 0.85


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time and
            datetime.utcnow().timestamp() - self.last_failure_time > self.timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Global error recovery manager instance
_global_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager


async def handle_system_error(
    error: Exception,
    context: Dict[str, Any]
) -> Tuple[bool, Optional[RecoveryAction]]:
    """Convenience function for handling system errors."""
    manager = get_recovery_manager()
    return await manager.handle_error(error, context)
