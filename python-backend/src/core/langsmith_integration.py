"""
LangSmith Integration for FlutterSwarm Multi-Agent System.

This module provides comprehensive logging, tracing, and debugging
capabilities using LangSmith for the FlutterSwarm workflow system.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from contextlib import asynccontextmanager

from langsmith import Client
from langsmith.run_helpers import traceable
from langchain_core.tracers import LangChainTracer

from ..config import get_logger, settings

logger = get_logger("langsmith_integration")


class LangSmithWorkflowTracer:
    """
    Enhanced tracer for FlutterSwarm workflows with LangSmith integration.
    
    Provides comprehensive logging and debugging capabilities for
    multi-agent workflow execution.
    """
    
    def __init__(
        self,
        project_name: str = "flutterswarm",
        session_name: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize LangSmith tracer."""
        self.project_name = project_name
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enabled = enabled and self._check_langsmith_config()
        
        self.client = None
        self.tracer = None
        
        if self.enabled:
            try:
                self.client = Client()
                self.tracer = LangChainTracer(
                    project_name=self.project_name,
                    client=self.client
                )
                logger.info(f"LangSmith tracer initialized for project: {self.project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith tracer: {e}")
                self.enabled = False
        else:
            logger.info("LangSmith tracing disabled")
    
    def _check_langsmith_config(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(
            os.getenv("LANGCHAIN_API_KEY") and
            os.getenv("LANGCHAIN_TRACING_V2") == "true"
        )
    
    @asynccontextmanager
    async def trace_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        inputs: Dict[str, Any]
    ):
        """Context manager for tracing entire workflows."""
        run_id = str(uuid.uuid4())
        
        if self.enabled:
            try:
                # Start workflow trace
                await self._log_workflow_start(workflow_id, workflow_name, inputs, run_id)
                
                yield run_id
                
                # End workflow trace
                await self._log_workflow_end(workflow_id, run_id, "success")
                
            except Exception as e:
                await self._log_workflow_end(workflow_id, run_id, "error", str(e))
                raise
        else:
            yield run_id
    
    async def trace_agent_execution(
        self,
        agent_id: str,
        task_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float,
        status: str = "success",
        error: Optional[str] = None
    ):
        """Trace individual agent task execution."""
        if not self.enabled:
            return
        
        try:
            trace_data = {
                "agent_id": agent_id,
                "task_id": task_id,
                "execution_time": execution_time,
                "status": status,
                "inputs": self._sanitize_data(inputs),
                "outputs": self._sanitize_data(outputs),
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                "session": self.session_name
            }
            
            # Log to LangSmith
            self.client.create_run(
                name=f"agent_execution_{agent_id}",
                run_type="llm",
                inputs={"inputs": inputs},
                outputs={"outputs": outputs},
                session_name=self.session_name,
                extra=trace_data
            )
            
        except Exception as e:
            logger.error(f"Failed to trace agent execution: {e}")
    
    async def trace_llm_call(
        self,
        agent_id: str,
        prompt: str,
        response: str,
        model: str,
        tokens_used: int,
        latency: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Trace LLM API calls."""
        if not self.enabled:
            return
        
        try:
            llm_data = {
                "agent_id": agent_id,
                "model": model,
                "tokens_used": tokens_used,
                "latency": latency,
                "context": self._sanitize_data(context or {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.client.create_run(
                name=f"llm_call_{agent_id}",
                run_type="llm",
                inputs={"prompt": prompt[:1000]},  # Truncate long prompts
                outputs={"response": response[:1000]},  # Truncate long responses
                session_name=self.session_name,
                extra=llm_data
            )
            
        except Exception as e:
            logger.error(f"Failed to trace LLM call: {e}")
    
    async def trace_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Trace errors for debugging."""
        if not self.enabled:
            return
        
        try:
            error_data = {
                "error_type": error_type,
                "error_message": error_message,
                "agent_id": agent_id,
                "task_id": task_id,
                "context": self._sanitize_data(context),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.client.create_run(
                name=f"error_{error_type}",
                run_type="chain",
                inputs={"context": context},
                outputs={"error": error_message},
                session_name=self.session_name,
                extra=error_data
            )
            
        except Exception as e:
            logger.error(f"Failed to trace error: {e}")
    
    async def log_workflow_metrics(
        self,
        workflow_id: str,
        metrics: Dict[str, Any]
    ):
        """Log workflow performance metrics."""
        if not self.enabled:
            return
        
        try:
            metrics_data = {
                "workflow_id": workflow_id,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "session": self.session_name
            }
            
            self.client.create_run(
                name=f"workflow_metrics_{workflow_id}",
                run_type="chain",
                inputs={"workflow_id": workflow_id},
                outputs={"metrics": metrics},
                session_name=self.session_name,
                extra=metrics_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log workflow metrics: {e}")
    
    async def _log_workflow_start(
        self,
        workflow_id: str,
        workflow_name: str,
        inputs: Dict[str, Any],
        run_id: str
    ):
        """Log workflow start."""
        try:
            start_data = {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "run_id": run_id,
                "inputs": self._sanitize_data(inputs),
                "started_at": datetime.utcnow().isoformat(),
                "session": self.session_name
            }
            
            self.client.create_run(
                name=f"workflow_start_{workflow_name}",
                run_type="chain",
                inputs={"inputs": inputs},
                session_name=self.session_name,
                extra=start_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log workflow start: {e}")
    
    async def _log_workflow_end(
        self,
        workflow_id: str,
        run_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Log workflow completion."""
        try:
            end_data = {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "status": status,
                "error": error,
                "ended_at": datetime.utcnow().isoformat(),
                "session": self.session_name
            }
            
            self.client.create_run(
                name=f"workflow_end_{status}",
                run_type="chain",
                inputs={"workflow_id": workflow_id},
                outputs={"status": status, "error": error},
                session_name=self.session_name,
                extra=end_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log workflow end: {e}")
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data for logging (remove sensitive information)."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str) and len(data) > 10000:
            return data[:10000] + "... [TRUNCATED]"
        else:
            return data


# Decorator for tracing agent tasks
def trace_agent_task(
    agent_id: str,
    task_id: str,
    task_type: str,
    tracer: Optional[LangSmithWorkflowTracer] = None
):
    """Decorator for tracing agent task execution."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                if tracer:
                    await tracer.trace_agent_execution(
                        agent_id=agent_id,
                        task_id=task_id,
                        inputs={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
                        outputs={"result": str(result)[:500] if result else "None"},
                        execution_time=execution_time,
                        status="success"
                    )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                if tracer:
                    await tracer.trace_agent_execution(
                        agent_id=agent_id,
                        task_id=task_id,
                        inputs={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
                        outputs={},
                        execution_time=execution_time,
                        status="error",
                        error=str(e)
                    )
                
                raise
        
        return wrapper
    return decorator


# Global tracer instance
_global_tracer: Optional[LangSmithWorkflowTracer] = None


def get_tracer() -> Optional[LangSmithWorkflowTracer]:
    """Get the global tracer instance."""
    return _global_tracer


def initialize_tracer(
    project_name: str = "flutterswarm",
    session_name: Optional[str] = None,
    enabled: bool = True
) -> LangSmithWorkflowTracer:
    """Initialize the global tracer."""
    global _global_tracer
    _global_tracer = LangSmithWorkflowTracer(
        project_name=project_name,
        session_name=session_name,
        enabled=enabled
    )
    return _global_tracer


# Convenience function for tracing
async def trace_agent_task(
    agent_id: str,
    task_id: str,
    task_type: str,
    func: Callable,
    *args,
    **kwargs
):
    """Convenience function for tracing agent tasks."""
    tracer = get_tracer()
    start_time = datetime.utcnow()
    
    try:
        result = await func(*args, **kwargs)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        if tracer:
            await tracer.trace_agent_execution(
                agent_id=agent_id,
                task_id=task_id,
                inputs={"task_type": task_type},
                outputs={"status": "completed"},
                execution_time=execution_time,
                status="success"
            )
        
        return result
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        if tracer:
            await tracer.trace_agent_execution(
                agent_id=agent_id,
                task_id=task_id,
                inputs={"task_type": task_type},
                outputs={},
                execution_time=execution_time,
                status="error",
                error=str(e)
            )
        
        raise
