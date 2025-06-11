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

from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, AgentStatus, TaskResult
from ..models.task_models import TaskContext
from ..config.agent_configs import agent_config_manager


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
    max_concurrent_tasks: int = 5
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        
        Args:
            config: Agent configuration parameters
            llm_client: LLM client for AI reasoning
            memory_manager: Shared memory management system
            event_bus: Communication system for inter-agent messaging
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
        
        # Initialize agent-specific logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_type}.{self.agent_id}")
        
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
            self.logger.info(f"Agent {self.agent_id} registered event handlers")
        except Exception as e:
            self.logger.error(f"Failed to register event handlers: {e}")
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
                
                if structured_output:
                    try:
                        parsed_response = json.loads(response.strip())
                        return parsed_response
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON response: {e}")
                        if attempt == max_retries - 1:
                            raise AgentError(f"Invalid JSON response after {max_retries} attempts")
                        continue
                
                return response.strip()
                
            except Exception as e:
                self.logger.error(f"LLM execution failed (attempt {attempt + 1}): {e}")
                
                # Try fallback provider if configured
                if (attempt == 0 and agent_config and 
                    agent_config.model_config.fallback_provider):
                    try:
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
        
        try:
            self.logger.info(f"Starting task processing: {task_id}")
            self.status = AgentStatus.PROCESSING
            
            # Add task to active tasks
            self.active_tasks[task_id] = task_context
            self.last_activity = datetime.utcnow()
            
            # Execute LLM-based task analysis
            analysis_result = await self.execute_llm_task(
                user_prompt=f"Analyze and process this task: {task_context.description}",
                context={
                    "task": task_context.to_dict(),
                    "requirements": task_context.requirements,
                    "expected_deliverables": task_context.expected_deliverables
                },
                structured_output=True
            )
            
            # Execute specialized processing if implemented
            specialized_result = {}
            if hasattr(self, '_execute_specialized_processing'):
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
            
            self.status = AgentStatus.IDLE
            return task_result
            
        except Exception as e:
            self.logger.error(f"Task processing failed for {task_id}: {e}")
            
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
            # Test LLM connectivity
            test_response = await self.execute_llm_task(
                user_prompt="Respond with 'OK' if you can process this message.",
                context={},
                max_retries=1
            )
            
            return "OK" in test_response.upper()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
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
            task_data = message.payload
            if "task_context" in task_data:
                task_context = TaskContext.from_dict(task_data["task_context"])
                await self.process_task(task_context)
        except Exception as e:
            self.logger.error(f"Failed to handle task assignment: {e}")

    async def _handle_shutdown(self, message: AgentMessage) -> None:
        """Handle shutdown events."""
        self.logger.info(f"Agent {self.agent_id} received shutdown signal")
        self.status = AgentStatus.OFFLINE
        
        # Cancel active tasks gracefully
        for task_id in list(self.active_tasks.keys()):
            self.logger.info(f"Cancelling active task: {task_id}")
            del self.active_tasks[task_id]
