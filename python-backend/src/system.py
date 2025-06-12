"""
System Initialization for FlutterSwarm Multi-Agent System.

This module handles the initialization and coordination of all core components
including LLM clients, memory managers, event bus, and agent orchestration.
"""

import asyncio
from typing import Any, Dict, Optional

from .core.llm_client import LLMClient
from .core.memory_manager import MemoryManager
from .core.event_bus import EventBus
from .agents.base_agent import BaseAgent, AgentConfig
from .agents.orchestrator_agent import OrchestratorAgent
from .agents.architecture_agent import ArchitectureAgent
from .agents.implementation_agent import ImplementationAgent
from .agents.testing_agent import TestingAgent
from .agents.devops_agent import DevOpsAgent
from .agents.security_agent import SecurityAgent
from .agents.documentation_agent import DocumentationAgent
from .config.settings import settings
from .config import setup_logging, get_logger
from src.config.agent_configs import agent_config_manager

logger = get_logger("system")


# LangGraph Integration Functions
async def create_langgraph_workflow(
    project_name: str = "flutterswarm",
    checkpointer_backend: str = "memory",
    redis_url: Optional[str] = None,
    enable_langsmith: bool = False
):
    """Initialize LangGraph workflow with enhanced logging."""
    
    # Set up logging first
    setup_logging()
    
    logger.info(
        "Initializing LangGraph workflow",
        operation="system_init",
        metadata={
            "project_name": project_name,
            "checkpointer_backend": checkpointer_backend,
            "enable_langsmith": enable_langsmith
        }
    )
    
    from .agents.langgraph_supervisor import SupervisorAgent
    from .core.langsmith_integration import initialize_tracer
    
    # Initialize LangSmith tracing if enabled
    tracer = None
    if enable_langsmith:
        tracer = initialize_tracer(
            project_name=project_name,
            enabled=enable_langsmith
        )
    
    # Create supervisor agent
    supervisor = SupervisorAgent(
        checkpointer_backend=checkpointer_backend,
        redis_url=redis_url
    )
    
    return supervisor, tracer


async def execute_langgraph_workflow(
    task_description: str,
    project_context: Dict[str, Any],
    workflow_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a task using the LangGraph workflow."""
    try:
        # Create workflow
        supervisor, tracer = await create_langgraph_workflow(
            enable_langsmith=workflow_config.get("enable_langsmith", False) if workflow_config else False
        )
        
        # Execute workflow
        result = await supervisor.execute_workflow(
            task_description=task_description,
            project_context=project_context,
            workflow_id=workflow_config.get("workflow_id") if workflow_config else None,
            thread_id=workflow_config.get("thread_id") if workflow_config else None
        )
        
        return result
        
    except Exception as e:
        logger.error(f"LangGraph workflow execution failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "workflow_id": workflow_config.get("workflow_id", "unknown") if workflow_config else "unknown"
        }


# Convenience function for external usage
async def initialize_langgraph_system(
    project_name: str = "flutterswarm",
    checkpointer_backend: str = "memory",
    redis_url: Optional[str] = None,
    enable_langsmith: bool = False
) -> Any:
    """Initialize the LangGraph-based FlutterSwarm system."""
    supervisor, tracer = await create_langgraph_workflow(
        project_name=project_name,
        checkpointer_backend=checkpointer_backend,
        redis_url=redis_url,
        enable_langsmith=enable_langsmith
    )
    return supervisor


def get_langgraph_system() -> Any:
    """Get the LangGraph system instance."""
    # This would return a cached instance in a real implementation
    return None


# System management functions for CLI compatibility
_system_instance = None


async def initialize_system(**kwargs) -> Any:
    """Initialize the system (alias for initialize_langgraph_system)."""
    global _system_instance
    _system_instance = await initialize_langgraph_system(**kwargs)
    return _system_instance


async def start_system(**kwargs) -> Any:
    """Start the system."""
    if _system_instance is None:
        await initialize_system(**kwargs)
    return _system_instance


async def stop_system():
    """Stop the system."""
    global _system_instance
    if _system_instance:
        # Add cleanup logic here if needed
        _system_instance = None
    logger.info("System stopped")


def get_system() -> Any:
    """Get the current system instance."""
    return _system_instance
