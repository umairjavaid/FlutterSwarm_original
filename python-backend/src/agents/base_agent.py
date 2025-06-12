"""
Base Agent Framework for FlutterSwarm Multi-Agent System.

This module provides the abstract base class that all specialized agents inherit from.
Every agent uses LLM calls for reasoning and decision-making, with no hardcoded logic.
"""

import logging
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import AgentError, LLMError

from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, AgentStatus, TaskResult
from ..models.task_models import TaskContext
from ..config.agent_configs import agent_config_manager
from ..core.enhanced_logger import get_logger, log_operation, log_llm_interaction, log_performance_metrics
from ..config import get_logger as get_config_logger


logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enumeration of agent capabilities."""
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    SECURITY_ANALYSIS = "security_analysis"
    DOCUMENTATION = "documentation"
    ORCHESTRATION = "orchestration"
    FILE_OPERATIONS = "file_operations"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    MONITORING = "monitoring"
    INFRASTRUCTURE = "infrastructure"
    QUALITY_ASSURANCE = "quality_assurance"


@dataclass
class AgentConfig:
    """Configuration for agent initialization."""
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    max_concurrent_tasks: int = None
    llm_model: str = None
    temperature: float = None
    max_tokens: int = None
    timeout: int = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set defaults from global config if not provided."""
        from ..config.settings import settings
        if self.max_concurrent_tasks is None:
            self.max_concurrent_tasks = settings.agent.max_concurrent_tasks
        if self.llm_model is None:
            self.llm_model = settings.llm.default_model
        if self.temperature is None:
            self.temperature = settings.llm.temperature
        if self.max_tokens is None:
            self.max_tokens = settings.llm.max_tokens
        if self.timeout is None:
            self.timeout = settings.llm.timeout


class BaseAgent(ABC):
    """
    Abstract base class for all FlutterSwarm agents.
    
    This class defines the core interface and common functionality that all
    specialized agents must implement. Every agent uses LLM reasoning for
    decision-making and communicates through the event bus system.
    
    Attributes:
        agent_id: Unique identifier for this agent instance
        agent_type: Type/role of the agent (e.g., "architecture", "implementation")
        config: Agent configuration parameters
        status: Current operational status of the agent
        llm_client: LLM client for AI reasoning
        memory_manager: Shared memory management system
        event_bus: Communication system for inter-agent messaging
        active_tasks: Currently executing tasks
        capabilities: List of agent capabilities
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        """
        Initialize the base agent with required dependencies.
        """
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.config = config
        self.status = AgentStatus.IDLE
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.event_bus = event_bus
        self.active_tasks: Dict[str, TaskContext] = {}
        self.capabilities = config.capabilities
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Load agent-specific configuration
        self.agent_config = agent_config_manager.get_agent_config(self.agent_type)
        
        # Initialize enhanced logging
        self.logger = get_config_logger(
            f"agents.{self.agent_type}",
            agent_type=self.agent_type,
            agent_id=self.agent_id
        )
        
        # Log agent initialization
        self.logger.info(
            f"Agent initialized with capabilities: {[cap.value for cap in self.capabilities]}",
            operation="agent_init",
            metadata={
                "capabilities": [cap.value for cap in self.capabilities],
                "max_concurrent_tasks": config.max_concurrent_tasks,
                "llm_model": config.llm_model
            }
        )
        
        # Register for relevant events
        asyncio.create_task(self._register_event_handlers())
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers for this agent."""
        try:
            await self.event_bus.subscribe(
                f"task.assigned.{self.agent_id}",
                self._handle_task_assignment
            )
            await self.event_bus.subscribe(
                f"agent.shutdown.{self.agent_id}",
                self._handle_shutdown
            )
            
            self.logger.info(
                "Event handlers registered successfully",
                operation="agent_init",
                metadata={
                    "topics": [f"task.assigned.{self.agent_id}", f"agent.shutdown.{self.agent_id}"]
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to register event handlers: {e}", operation="agent_init")
            raise
    
    async def get_system_prompt(self) -> str:
        """
        Get the system prompt from configuration.
        
        Returns:
            Comprehensive system prompt string for LLM initialization
        """
        from ..config.agent_configs import agent_config_manager
        
        config_prompt = agent_config_manager.get_system_prompt(self.agent_type)
        if config_prompt:
            return config_prompt
        else:
            # Fallback to default if no configuration found
            return await self._get_default_system_prompt()
    
    @abstractmethod
    async def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for this agent type.
        This is a fallback when no configuration prompt is available.
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """
        Get a list of specific capabilities this agent provides.
        
        Returns:
            List of capability descriptions that can be used by other agents
            to understand what this agent can accomplish
        """
        pass
    
    @log_llm_interaction(include_tokens=True, include_cost=True)
    @log_performance_metrics(include_memory=True, include_cpu=True)
    async def execute_llm_task(
        self,
        user_prompt: str,
        context: Dict[str, Any],
        structured_output: bool = False,
        max_retries: int = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute a task using LLM reasoning with comprehensive context.
        
        Args:
            user_prompt: The specific task or question for the LLM
            context: Additional context including project state, constraints, etc.
            structured_output: Whether to expect JSON-formatted response
            max_retries: Maximum number of retry attempts on failure (uses config if None)
        
        Returns:
            LLM response as string or parsed JSON dictionary
        """
        from ..config.agent_configs import agent_config_manager
        
        # Get agent configuration
        agent_config = agent_config_manager.get_agent_config(self.agent_type)
        
        if max_retries is None:
            max_retries = agent_config.model_config.max_retries if agent_config else 3
        
        system_prompt = await self.get_system_prompt()
        memory_context = await self.memory_manager.get_relevant_context(
            user_prompt,
            agent_id=self.agent_id
        )
        
        full_prompt = await self._construct_full_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            memory_context=memory_context,
            structured_output=structured_output,
            agent_config=agent_config
        )
        
        with self.logger.context(
            correlation_id=context.get('correlation_id'),
            task_id=context.get('task_id'),
            operation="llm_task_execution"
        ) as log_ctx:
            
            self.logger.info(
                f"Starting LLM task execution with {len(user_prompt)} character prompt",
                operation="llm_call",
                metadata={
                    "prompt_length": len(user_prompt),
                    "structured_output": structured_output,
                    "max_retries": max_retries or 3
                }
            )
            
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"LLM execution attempt {attempt + 1}/{max_retries}")
                    
                    # Use configuration for model parameters
                    model = agent_config.model_config.model if agent_config else self.config.llm_model
                    temperature = agent_config.model_config.temperature if agent_config else self.config.temperature
                    max_tokens = agent_config.model_config.max_tokens if agent_config else self.config.max_tokens
                    timeout = agent_config.model_config.timeout if agent_config else self.config.timeout
                    provider = agent_config.model_config.provider if agent_config else None
                    
                    response = await self.llm_client.generate(
                        prompt=full_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        provider=provider,
                        agent_id=self.agent_id,
                        correlation_id=context.get('correlation_id', '')
                    )
                    
                    # Store interaction in memory
                    await self.memory_manager.store_interaction(
                        agent_id=self.agent_id,
                        prompt=user_prompt,
                        response=response,
                        context=context
                    )
                    
                    self.logger.info(
                        f"LLM task completed successfully on attempt {attempt + 1}",
                        operation="llm_response",
                        metadata={
                            "attempt": attempt + 1,
                            "response_length": len(response),
                            "model": model,
                            "temperature": temperature
                        }
                    )
                    
                    if structured_output:
                        try:
                            parsed_response = json.loads(response.strip())
                            return parsed_response
                        except json.JSONDecodeError as e:
                            self.logger.warning(
                                f"Failed to parse JSON response: {e}",
                                metadata={"attempt": attempt + 1, "response_preview": response[:200]}
                            )
                            if attempt == max_retries - 1:
                                raise AgentError(f"Invalid JSON response after {max_retries} attempts")
                            continue
                    
                    return response.strip()
                    
                except Exception as e:
                    self.logger.error(
                        f"LLM execution failed (attempt {attempt + 1}): {e}",
                        operation="llm_call",
                        metadata={"attempt": attempt + 1, "error_type": type(e).__name__}
                    )
                    
                    # Try fallback provider if configured
                    if (attempt == 0 and agent_config and 
                        agent_config.model_config.fallback_provider):
                        try:
                            self.logger.info("Attempting fallback provider", metadata={"provider": agent_config.model_config.fallback_provider})
                            
                            fallback_response = await self.llm_client.generate(
                                prompt=full_prompt,
                                model=agent_config.model_config.fallback_model or model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                timeout=timeout,
                                provider=agent_config.model_config.fallback_provider,
                                agent_id=self.agent_id,
                                correlation_id=context.get('correlation_id', '')
                            )
                            
                            self.logger.info("Fallback provider succeeded")
                            
                            if structured_output:
                                try:
                                    return json.loads(fallback_response.strip())
                                except json.JSONDecodeError:
                                    pass
                            
                            return fallback_response.strip()
                            
                        except Exception as fallback_error:
                            self.logger.warning(f"Fallback provider also failed: {fallback_error}")
                    
                    if attempt == max_retries - 1:
                        raise LLMError(f"LLM execution failed after {max_retries} attempts: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            raise LLMError("Unexpected error in LLM execution")
    
    async def _construct_full_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Dict[str, Any],
        memory_context: str,
        structured_output: bool = False,
        agent_config: Optional[Any] = None
    ) -> str:
        """Construct a comprehensive prompt with all necessary context."""
        # Add custom instructions from configuration
        custom_instructions = ""
        constraint_instructions = ""
        response_format_instructions = ""
        
        if agent_config and agent_config.prompt_config:
            if agent_config.prompt_config.custom_instructions:
                custom_instructions = "\n\nCUSTOM INSTRUCTIONS:\n" + \
                    "\n".join(f"- {instruction}" for instruction in agent_config.prompt_config.custom_instructions)
            
            if agent_config.prompt_config.constraint_instructions:
                constraint_instructions = "\n\nCONSTRAINT INSTRUCTIONS:\n" + \
                    "\n".join(f"- {instruction}" for instruction in agent_config.prompt_config.constraint_instructions)
            
            if agent_config.prompt_config.response_format_instructions:
                response_format_instructions = f"\n\nRESPONSE FORMAT:\n{agent_config.prompt_config.response_format_instructions}"
        
        output_format = ""
        if structured_output and not response_format_instructions:
            output_format = """
            
RESPONSE FORMAT:
Provide your response as valid JSON with the following structure:
{
    "status": "success|error|partial",
    "result": "detailed response or analysis",
    "reasoning": "step-by-step reasoning process",
    "next_steps": ["recommended", "actions"],
    "confidence": 0.95,
    "metadata": {}
}
"""
        
        return f"""SYSTEM: {system_prompt}

AGENT CONTEXT:
- Agent ID: {self.agent_id}
- Agent Type: {self.agent_type}
- Capabilities: {[cap.value for cap in self.capabilities]}
- Current Status: {self.status.value}

PROJECT CONTEXT:
{json.dumps(context.get('project', {}), indent=2)}

MEMORY CONTEXT:
{memory_context}
{custom_instructions}
{constraint_instructions}

CURRENT TASK:
{user_prompt}
{response_format_instructions or output_format}

Provide a detailed, actionable response with clear reasoning and specific implementation guidance."""
    
    @log_operation("task_processing", include_timing=True)
    async def process_task(self, task_context: TaskContext) -> TaskResult:
        """
        Process a task using LLM reasoning and agent-specific logic.
        
        This is the main entry point for task processing. It uses LLM calls
        to analyze the task, plan the approach, and execute the solution.
        
        Args:
            task_context: Complete context for the task including requirements,
                         constraints, and expected deliverables
        
        Returns:
            TaskResult containing the outcome, deliverables, and metadata
        
        Raises:
            AgentError: If task processing fails
        """
        from ..models.agent_models import AgentError
        
        task_id = task_context.task_id
        correlation_id = task_context.correlation_id
        
        with self.logger.context(
            correlation_id=correlation_id,
            task_id=task_id,
            operation="task_processing"
        ) as log_ctx:
            
            try:
                self.logger.info(
                    f"Starting task processing",
                    operation="task_start",
                    metadata={
                        "task_type": task_context.task_type.value,
                        "priority": task_context.priority.value,
                        "requirements_count": len(task_context.requirements),
                        "deliverables_count": len(task_context.expected_deliverables)
                    }
                )
                
                # Update agent status
                old_status = self.status
                self.status = AgentStatus.PROCESSING
                self.logger.info(
                    f"Agent status changed: {old_status.value} -> {self.status.value}",
                    operation="status_change",
                    metadata={"old_status": old_status.value, "new_status": self.status.value}
                )
                
                # Add task to active tasks
                self.active_tasks[task_id] = task_context
                self.last_activity = datetime.utcnow()
                
                # Execute LLM-based task analysis
                analysis_result = await self.execute_llm_task(
                    user_prompt=f"Analyze and process this task: {task_context.description}",
                    context={
                        "task": task_context.to_dict(),
                        "requirements": task_context.requirements,
                        "expected_deliverables": task_context.expected_deliverables,
                        "correlation_id": correlation_id,
                        "task_id": task_id
                    },
                    structured_output=True
                )
                
                # Execute specialized processing if implemented
                specialized_result = {}
                if hasattr(self, '_execute_specialized_processing'):
                    self.logger.debug("Executing specialized processing")
                    specialized_result = await self._execute_specialized_processing(
                        task_context, analysis_result
                    )
                
                # Create task result
                task_result = TaskResult(
                    task_id=task_id,
                    agent_id=self.agent_id,
                    status="completed",
                    result=analysis_result.get("result", "Task completed successfully"),
                    deliverables={
                        "analysis": analysis_result,
                        "specialized_output": specialized_result
                    },
                    metadata={
                        "agent_type": self.agent_type,
                        "capabilities_used": [cap.value for cap in self.capabilities],
                        "processing_time": (datetime.utcnow() - task_context.created_at).total_seconds()
                    },
                    correlation_id=correlation_id
                )
                
                # Store task result in memory
                await self.memory_manager.store_memory(
                    content=f"Task completed: {task_result.result}",
                    metadata={
                        "type": "task_result",
                        "task_id": task_id,
                        "status": task_result.status
                    },
                    correlation_id=correlation_id,
                    importance=0.8
                )
                
                self.logger.info(
                    "Task processing completed successfully",
                    operation="task_complete",
                    metadata={
                        "deliverables_count": len(task_result.deliverables),
                        "processing_time": task_result.metadata.get("processing_time")
                    }
                )
                
                self.status = AgentStatus.IDLE
                self.logger.info(
                    f"Agent status changed: PROCESSING -> IDLE",
                    operation="status_change",
                    metadata={"old_status": "PROCESSING", "new_status": "IDLE"}
                )
                
                return task_result
                
            except Exception as e:
                self.logger.error(
                    f"Task processing failed: {e}",
                    operation="task_failed",
                    metadata={
                        "error_type": type(e).__name__,
                        "task_type": task_context.task_type.value
                    }
                )
                
                # Create error result
                error_result = TaskResult(
                    task_id=task_id,
                    agent_id=self.agent_id,
                    status="failed",
                    error=str(e),
                    correlation_id=correlation_id
                )
                
                self.status = AgentStatus.IDLE
                return error_result
                
            finally:
                # Remove from active tasks
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the agent.
        
        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            self.logger.debug("Performing health check", operation="health_check")
            
            # Test LLM connectivity
            test_response = await self.execute_llm_task(
                user_prompt="Respond with 'OK' if you can process this message.",
                context={"correlation_id": str(uuid.uuid4())},
                max_retries=1
            )
            
            is_healthy = "OK" in test_response.upper()
            
            self.logger.info(
                f"Health check {'passed' if is_healthy else 'failed'}",
                operation="health_check",
                metadata={"healthy": is_healthy, "response": test_response[:50]}
            )
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", operation="health_check")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_tasks": len(self.active_tasks),
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "uptime": (datetime.utcnow() - self.created_at).total_seconds()
        }

    async def _handle_task_assignment(self, message: AgentMessage) -> None:
        """Handle task assignment events."""
        try:
            self.logger.info(
                "Received task assignment",
                operation="message_received",
                metadata={
                    "message_type": message.type,
                    "sender": message.source,
                    "correlation_id": message.correlation_id
                }
            )
            
            task_data = message.payload
            if "task_context" in task_data:
                task_context = TaskContext.from_dict(task_data["task_context"])
                await self.process_task(task_context)
        except Exception as e:
            self.logger.error(f"Failed to handle task assignment: {e}", operation="message_handling")

    async def _handle_shutdown(self, message: AgentMessage) -> None:
        """Handle shutdown events."""
        self.logger.info(
            f"Agent received shutdown signal",
            operation="agent_shutdown",
            metadata={"active_tasks": len(self.active_tasks)}
        )
        
        self.status = AgentStatus.OFFLINE
        
        # Cancel active tasks gracefully
        for task_id in list(self.active_tasks.keys()):
            self.logger.info(f"Cancelling active task: {task_id}", operation="task_cancellation")
            del self.active_tasks[task_id]
        
        self.logger.info("Agent shutdown complete", operation="agent_shutdown")
