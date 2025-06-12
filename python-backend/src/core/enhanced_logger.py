"""
Enhanced Logging System for FlutterSwarm Multi-Agent System.

This module provides comprehensive logging capabilities with:
- Colored output with emoji prefixes
- Structured logging with JSON format
- Distributed tracing support
- Performance metrics logging
- Log rotation and archiving
- Real-time log streaming
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback for when colorama is not available
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""

import psutil


# Agent-specific colors and emojis
AGENT_COLORS = {
    "orchestrator": Fore.MAGENTA,
    "architecture": Fore.BLUE,
    "implementation": Fore.GREEN,
    "testing": Fore.YELLOW,
    "security": Fore.RED,
    "devops": Fore.CYAN,
    "documentation": Fore.WHITE,
    "performance": Fore.LIGHTBLUE_EX,
    "supervisor": Fore.LIGHTMAGENTA_EX
}

AGENT_EMOJIS = {
    "orchestrator": "🎯",
    "architecture": "🏗️",
    "implementation": "💻",
    "testing": "🧪",
    "security": "🔒",
    "devops": "🚀",
    "documentation": "📚",
    "performance": "⚡",
    "supervisor": "👁️"
}

# Module-specific colors
MODULE_COLORS = {
    "event_bus": Fore.LIGHTMAGENTA_EX,
    "memory_manager": Fore.LIGHTCYAN_EX,
    "api": Fore.LIGHTGREEN_EX,
    "core": Fore.LIGHTYELLOW_EX,
    "llm_client": Fore.LIGHTBLUE_EX,
    "system": Fore.LIGHTWHITE_EX
}

MODULE_EMOJIS = {
    "event_bus": "📡",
    "memory_manager": "🧠",
    "api": "🌐",
    "core": "⚙️",
    "llm_client": "🤖",
    "system": "🖥️"
}

# Operation emojis
OPERATION_EMOJIS = {
    "task_start": "▶️",
    "task_complete": "✅",
    "task_failed": "❌",
    "task_assigned": "📋",
    "message_sent": "📤",
    "message_received": "📥",
    "memory_stored": "💾",
    "memory_retrieved": "📂",
    "llm_call": "🤖",
    "llm_response": "💬",
    "agent_init": "🎬",
    "agent_shutdown": "🔚",
    "collaboration": "🤝",
    "error": "⚠️",
    "warning": "⚡",
    "info": "ℹ️",
    "debug": "🔍",
    "critical": "🚨",
    "performance": "📊",
    "resource_usage": "📈"
}

# Log level emojis and colors
LEVEL_EMOJIS = {
    "DEBUG": "🔍",
    "INFO": "ℹ️",
    "WARNING": "⚡",
    "ERROR": "❌",
    "CRITICAL": "🚨"
}

LEVEL_COLORS = {
    "DEBUG": Fore.LIGHTBLACK_EX,
    "INFO": Fore.LIGHTWHITE_EX,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Back.YELLOW
}


@dataclass
class LogContext:
    """Context information for logging."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    task_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    operation: Optional[str] = None
    module: Optional[str] = None
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "correlation_id": self.correlation_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "operation": self.operation,
            "module": self.module,
            "metadata": self.metadata
        }
        
        if self.start_time:
            result["execution_time"] = time.time() - self.start_time
        
        return {k: v for k, v in result.items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Get context from record
        context = getattr(record, 'context', LogContext())
        
        # Get system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": context.to_dict(),
            "system_metrics": {
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads()
            }
        }
        
        # Add exception info if present
        if record.exc_info and record.exc_info != True:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text',
                          'stack_info', 'getMessage', 'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter with emojis."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and emojis."""
        # Get context from record
        context = getattr(record, 'context', LogContext())
        
        # Determine colors and emojis
        level_emoji = LEVEL_EMOJIS.get(record.levelname, "")
        level_color = LEVEL_COLORS.get(record.levelname, "")
        
        agent_emoji = ""
        agent_color = ""
        module_emoji = ""
        module_color = ""
        operation_emoji = ""
        
        if context.agent_type:
            agent_emoji = AGENT_EMOJIS.get(context.agent_type, "🤖")
            agent_color = AGENT_COLORS.get(context.agent_type, Fore.WHITE)
        
        if context.module:
            module_emoji = MODULE_EMOJIS.get(context.module, "⚙️")
            module_color = MODULE_COLORS.get(context.module, Fore.WHITE)
        
        if context.operation:
            operation_emoji = OPERATION_EMOJIS.get(context.operation, "")
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # Build formatted message
        parts = []
        
        # Timestamp and level
        parts.append(f"{Fore.LIGHTBLACK_EX}[{timestamp}]{Style.RESET_ALL}")
        parts.append(f"{level_color}{level_emoji} {record.levelname}{Style.RESET_ALL}")
        
        # Agent information
        if context.agent_type:
            agent_display = f"{agent_emoji} {context.agent_type.upper()}"
            if context.agent_id:
                agent_id_short = context.agent_id.split('_')[-1][:6]
                agent_display += f":{agent_id_short}"
            parts.append(f"{agent_color}[{agent_display}]{Style.RESET_ALL}")
        
        # Module information
        if context.module:
            module_display = f"{module_emoji} {context.module.upper()}"
            parts.append(f"{module_color}[{module_display}]{Style.RESET_ALL}")
        
        # Operation
        if operation_emoji:
            parts.append(operation_emoji)
        
        # Main message
        message = record.getMessage()
        parts.append(message)
        
        # Context information
        context_parts = []
        if context.task_id:
            context_parts.append(f"task:{context.task_id[:8]}")
        if context.correlation_id and context.correlation_id != context.task_id:
            context_parts.append(f"correlation:{context.correlation_id[:8]}")
        if context.start_time:
            execution_time = time.time() - context.start_time
            context_parts.append(f"duration:{execution_time:.3f}s")
        
        if context_parts:
            context_str = " | ".join(context_parts)
            parts.append(f"{Fore.LIGHTBLACK_EX}({context_str}){Style.RESET_ALL}")
        
        # Metadata
        if context.metadata:
            metadata_str = " | ".join([f"{k}:{v}" for k, v in context.metadata.items() if v])
            if metadata_str:
                parts.append(f"{Fore.LIGHTBLACK_EX}[{metadata_str}]{Style.RESET_ALL}")
        
        formatted = " ".join(parts)
        
        # Add exception info if present
        if record.exc_info and record.exc_info != True:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


class EnhancedLogger:
    """Enhanced logger with context support and performance tracking."""
    
    def __init__(self, name: str, agent_type: Optional[str] = None, agent_id: Optional[str] = None):
        self.name = name
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.logger = logging.getLogger(name)
        self._context_stack: List[LogContext] = []
        
        # Set up formatters if not already configured
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up log handlers."""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        
        # File handler for structured logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}.json.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # Set logger level
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
    
    def _get_current_context(self) -> LogContext:
        """Get current logging context."""
        if self._context_stack:
            return self._context_stack[-1]
        return LogContext(
            agent_type=self.agent_type,
            agent_id=self.agent_id
        )
    
    def _log_with_context(self, level: int, message: str, operation: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Log message with context."""
        context = self._get_current_context()
        
        if operation:
            context.operation = operation
        
        if metadata:
            context.metadata.update(metadata)
        
        # Extract exc_info from kwargs if present
        exc_info = kwargs.pop('exc_info', None)
        
        # Create log record
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), exc_info
        )
        record.context = context
        
        # Add any extra kwargs to record
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def debug(self, message: str, operation: Optional[str] = None, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, operation, **kwargs)
    
    def info(self, message: str, operation: Optional[str] = None, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, operation, **kwargs)
    
    def warning(self, message: str, operation: Optional[str] = None, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, operation, **kwargs)
    
    def error(self, message: str, operation: Optional[str] = None, exc_info: bool = True, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, operation, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, operation: Optional[str] = None, exc_info: bool = True, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, operation, exc_info=exc_info, **kwargs)
    
    @contextmanager
    def context(self, correlation_id: Optional[str] = None, task_id: Optional[str] = None,
                operation: Optional[str] = None, **kwargs):
        """Context manager for logging context."""
        context = LogContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            task_id=task_id,
            operation=operation,
            start_time=time.time(),
            **kwargs
        )
        
        self._context_stack.append(context)
        try:
            yield context
        finally:
            self._context_stack.pop()
    
    def log_operation_start(self, operation: str, task_id: Optional[str] = None, **metadata):
        """Log operation start."""
        self.info(
            f"Starting {operation}",
            operation="task_start",
            metadata={"target_operation": operation, "task_id": task_id, **metadata}
        )
    
    def log_operation_complete(self, operation: str, execution_time: float, **metadata):
        """Log operation completion."""
        self.info(
            f"Completed {operation}",
            operation="task_complete",
            metadata={"target_operation": operation, "execution_time": execution_time, **metadata}
        )
    
    def log_operation_failed(self, operation: str, error: str, **metadata):
        """Log operation failure."""
        self.error(
            f"Failed {operation}: {error}",
            operation="task_failed",
            metadata={"target_operation": operation, "error": error, **metadata}
        )
    
    def log_llm_interaction(self, prompt_tokens: int, response_tokens: int, model: str,
                           execution_time: float, cost: Optional[float] = None, **metadata):
        """Log LLM interaction."""
        llm_metadata = {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "model": model,
            "execution_time": execution_time
        }
        
        if cost:
            llm_metadata["cost"] = cost
        
        llm_metadata.update(metadata)
        
        self.info(
            f"LLM interaction completed - {prompt_tokens + response_tokens} tokens, {execution_time:.3f}s",
            operation="llm_call",
            metadata=llm_metadata
        )
    
    def log_memory_operation(self, operation: str, entries_count: int = None,
                            similarity_score: float = None, **metadata):
        """Log memory operation."""
        memory_metadata = {"memory_operation": operation}
        
        if entries_count is not None:
            memory_metadata["entries_count"] = entries_count
        if similarity_score is not None:
            memory_metadata["similarity_score"] = similarity_score
        
        memory_metadata.update(metadata)
        
        self.info(
            f"Memory {operation}",
            operation="memory_stored" if "store" in operation else "memory_retrieved",
            metadata=memory_metadata
        )
    
    def log_event_communication(self, event_type: str, topic: str, payload_size: Optional[int] = None,
                               target_agent: Optional[str] = None, **metadata):
        """Log event bus communication."""
        event_metadata = {
            "event_type": event_type,
            "topic": topic,
            "target_agent": target_agent
        }
        
        if payload_size:
            event_metadata["payload_size"] = payload_size
        
        event_metadata.update(metadata)
        
        operation = "message_sent" if event_type == "publish" else "message_received"
        
        self.info(
            f"Event {event_type}: {topic}",
            operation=operation,
            metadata=event_metadata
        )
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.info(
            f"Performance metrics for {operation}",
            operation="performance",
            metadata={"target_operation": operation, **metrics}
        )
    
    def log_resource_usage(self, memory_mb: float, cpu_percent: float, **additional_metrics):
        """Log resource usage."""
        metrics = {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            **additional_metrics
        }
        
        self.info(
            f"Resource usage - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%",
            operation="resource_usage",
            metadata=metrics
        )


# Decorators for common logging patterns
def log_operation(operation: str, include_timing: bool = True, include_args: bool = False):
    """Decorator to log operation execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get logger from self if available
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            start_time = time.time()
            
            # Prepare metadata
            metadata = {"function": func.__name__}
            if include_args:
                metadata["args_count"] = len(args)
                metadata["kwargs_keys"] = list(kwargs.keys())
            
            logger.log_operation_start(operation, **metadata)
            
            try:
                result = await func(*args, **kwargs)
                
                if include_timing:
                    execution_time = time.time() - start_time
                    logger.log_operation_complete(operation, execution_time, **metadata)
                
                return result
                
            except Exception as e:
                logger.log_operation_failed(operation, str(e), **metadata)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get logger from self if available
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            start_time = time.time()
            
            # Prepare metadata
            metadata = {"function": func.__name__}
            if include_args:
                metadata["args_count"] = len(args)
                metadata["kwargs_keys"] = list(kwargs.keys())
            
            logger.log_operation_start(operation, **metadata)
            
            try:
                result = func(*args, **kwargs)
                
                if include_timing:
                    execution_time = time.time() - start_time
                    logger.log_operation_complete(operation, execution_time, **metadata)
                
                return result
                
            except Exception as e:
                logger.log_operation_failed(operation, str(e), **metadata)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_llm_interaction(include_tokens: bool = True, include_cost: bool = False):
    """Decorator to log LLM interactions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Extract token information if available
                if include_tokens and hasattr(result, 'usage'):
                    usage = result.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    response_tokens = getattr(usage, 'completion_tokens', 0)
                    model = kwargs.get('model', 'unknown')
                    
                    cost = None
                    if include_cost:
                        # Simple cost calculation (would need real pricing)
                        cost = (prompt_tokens * 0.0001 + response_tokens * 0.0002) / 1000
                    
                    logger.log_llm_interaction(
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        model=model,
                        execution_time=execution_time,
                        cost=cost,
                        function=func.__name__
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"LLM interaction failed: {e}", operation="llm_call")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            # For sync functions, create a sync wrapper
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def log_memory_operation(include_similarity_scores: bool = False):
    """Decorator to log memory operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            operation_name = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract operation details
                entries_count = None
                similarity_score = None
                
                if isinstance(result, list):
                    entries_count = len(result)
                    if include_similarity_scores and result:
                        # Assume tuples of (entry, score)
                        if isinstance(result[0], tuple) and len(result[0]) == 2:
                            similarity_score = result[0][1]
                
                logger.log_memory_operation(
                    operation=operation_name,
                    entries_count=entries_count,
                    similarity_score=similarity_score,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Memory operation failed: {e}", operation="memory_operation")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def log_event_communication(include_payload_size: bool = False):
    """Decorator to log event bus communications."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            # Extract event details from arguments
            topic = kwargs.get('topic') or (args[1] if len(args) > 1 else 'unknown')
            event_type = func.__name__  # publish, subscribe, etc.
            
            payload_size = None
            if include_payload_size and len(args) > 2:
                payload = args[2]
                if hasattr(payload, '__len__'):
                    payload_size = len(str(payload))
            
            try:
                result = await func(*args, **kwargs)
                
                logger.log_event_communication(
                    event_type=event_type,
                    topic=topic,
                    payload_size=payload_size,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Event communication failed: {e}", operation="event_communication")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def log_performance_metrics(include_memory: bool = True, include_cpu: bool = True):
    """Decorator to log performance metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = EnhancedLogger(func.__module__)
            
            # Capture initial metrics
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024 if include_memory else None
            start_cpu = process.cpu_percent() if include_cpu else None
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Capture final metrics
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024 if include_memory else None
                end_cpu = process.cpu_percent() if include_cpu else None
                
                metrics = {
                    "execution_time": end_time - start_time,
                    "function": func.__name__
                }
                
                if include_memory and start_memory and end_memory:
                    metrics["memory_delta_mb"] = end_memory - start_memory
                    metrics["peak_memory_mb"] = end_memory
                
                if include_cpu and start_cpu is not None and end_cpu is not None:
                    metrics["avg_cpu_percent"] = (start_cpu + end_cpu) / 2
                
                logger.log_performance_metrics(func.__name__, metrics)
                
                return result
                
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}", operation="performance_monitoring")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global logger cache
_logger_cache: Dict[str, EnhancedLogger] = {}


def get_enhanced_logger(name: str, agent_type: Optional[str] = None, 
                       agent_id: Optional[str] = None) -> EnhancedLogger:
    """Get or create an enhanced logger instance."""
    cache_key = f"{name}_{agent_type}_{agent_id}"
    
    if cache_key not in _logger_cache:
        _logger_cache[cache_key] = EnhancedLogger(name, agent_type, agent_id)
    
    return _logger_cache[cache_key]


# Convenience function for backward compatibility
def get_logger(name: str, agent_type: Optional[str] = None, 
               agent_id: Optional[str] = None) -> EnhancedLogger:
    """Get enhanced logger (convenience function)."""
    return get_enhanced_logger(name, agent_type, agent_id)
