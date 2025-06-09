"""
Base Agent Framework for FlutterSwarm Multi-Agent System.

This module provides the abstract base class that all specialized agents inherit from.
Every agent uses LLM calls for reasoning and decision-making, with no hardcoded logic.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from core.event_bus import EventBus
from core.memory_manager import MemoryManager
from models.agent_models import AgentMessage, AgentStatus, TaskResult
from models.task_models import TaskContext


logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enumeration of agent capabilities."""
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    ORCHESTRATION = "orchestration"


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
    
    @abstractmethod
    async def get_system_prompt(self) -> str:
        """
        Get the system prompt that defines this agent's role and capabilities.
        
        This prompt is used to instruct the LLM about the agent's specific
        domain expertise, responsibilities, and expected behavior patterns.
        
        Returns:
            Comprehensive system prompt string for LLM initialization
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
        max_retries: int = 3
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute a task using LLM reasoning with comprehensive context.
        
        This method constructs a complete prompt including system instructions,
        project context, memory context, and specific task requirements.
        
        Args:
            user_prompt: The specific task or question for the LLM
            context: Additional context including project state, constraints, etc.
            structured_output: Whether to expect JSON-formatted response
            max_retries: Maximum number of retry attempts on failure
        
        Returns:
            LLM response as string or parsed JSON dictionary
        
        Raises:
            LLMError: If LLM reasoning fails after retries
            AgentError: If response format is invalid
        """
        from models.agent_models import LLMError, AgentError
        
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
            structured_output=structured_output
        )
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"LLM execution attempt {attempt + 1}/{max_retries}")
                
                response = await self.llm_client.generate(
                    prompt=full_prompt,
                    model=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
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
        structured_output: bool = False
    ) -> str:
        """Construct a comprehensive prompt with all necessary context."""
        output_format = ""
        if structured_output:
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

TASK CONSTRAINTS:
- Follow Flutter/Dart best practices
- Ensure code is production-ready and well-documented
- Include comprehensive error handling
- Consider performance and scalability
- Follow established project patterns and conventions
- Maintain security best practices

CURRENT TASK:
{user_prompt}
{output_format}

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
        from models.agent_models import AgentError
        
        task_id = task_context.task_id
        correlation_id = task_context.correlation_id
        
        try:
            self.logger.info(f"Starting task processing: {task_id}")
            self.status = AgentStatus.PROCESSING
            self.active_tasks[task_id] = task_context
            self.last_activity = datetime.utcnow()
            
            # Notify start of processing
            await self.event_bus.publish(
                "task.started",
                AgentMessage(
                    type="task_started",
                    source=self.agent_id,
                    target="orchestrator",
                    payload={"task_id": task_id, "agent_id": self.agent_id},
                    correlation_id=correlation_id
                )
            )
            
            # Use LLM to analyze and process the task
            analysis_result = await self.execute_llm_task(
                user_prompt=f"""
                Analyze and process this task:
                
                Task Description: {task_context.description}
                Task Type: {task_context.task_type}
                Requirements: {json.dumps(task_context.requirements, indent=2)}
                Expected Deliverables: {task_context.expected_deliverables}
                
                Please provide a comprehensive analysis and execution plan.
                """,
                context={
                    "task": task_context.to_dict(),
                    "project": task_context.project_context,
                    "agent_capabilities": [cap.value for cap in self.capabilities]
                },
                structured_output=True
            )
            
            # Execute agent-specific processing
            result = await self._execute_specialized_processing(task_context, analysis_result)
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                status="completed",
                result=result,
                deliverables=result.get("deliverables", {}),
                metadata={
                    "processing_time": (datetime.utcnow() - task_context.created_at).total_seconds(),
                    "llm_analysis": analysis_result,
                    "agent_type": self.agent_type
                },
                correlation_id=correlation_id
            )
            
            # Notify completion
            await self.event_bus.publish(
                "task.completed",
                AgentMessage(
                    type="task_completed",
                    source=self.agent_id,
                    target="orchestrator",
                    payload=task_result.to_dict(),
                    correlation_id=correlation_id
                )
            )
            
            self.logger.info(f"Task completed successfully: {task_id}")
            return task_result
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {task_id}, error: {e}")
            
            # Create error result
            error_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                status="error",
                result={"error": str(e)},
                error=str(e),
                correlation_id=correlation_id
            )
            
            # Notify failure
            await self.event_bus.publish(
                "task.failed",
                AgentMessage(
                    type="task_failed",
                    source=self.agent_id,
                    target="orchestrator",
                    payload=error_result.to_dict(),
                    correlation_id=correlation_id
                )
            )
            
            return error_result
            
        finally:
            self.active_tasks.pop(task_id, None)
            self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.PROCESSING
    
    @abstractmethod
    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agent-specific processing logic.
        
        This method must be implemented by each specialized agent to handle
        their domain-specific tasks using the LLM analysis as guidance.
        
        Args:
            task_context: Complete task context
            llm_analysis: Analysis and plan from the LLM
        
        Returns:
            Dictionary containing the processing results and deliverables
        """
        pass
    
    async def _handle_task_assignment(self, message: AgentMessage) -> None:
        """Handle task assignment from orchestrator."""
        try:
            task_data = message.payload
            task_context = TaskContext.from_dict(task_data)
            
            # Process task asynchronously
            asyncio.create_task(self.process_task(task_context))
            
        except Exception as e:
            self.logger.error(f"Failed to handle task assignment: {e}")
    
    async def _handle_shutdown(self, message: AgentMessage) -> None:
        """Handle shutdown signal."""
        self.logger.info(f"Received shutdown signal for agent {self.agent_id}")
        self.status = AgentStatus.SHUTTING_DOWN
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            self.logger.info(f"Cancelling active task: {task_id}")
            self.active_tasks.pop(task_id, None)
        
        self.status = AgentStatus.OFFLINE
        self.logger.info(f"Agent {self.agent_id} shutdown complete")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_tasks": len(self.active_tasks),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "config": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "llm_model": self.config.llm_model
            }
        }
    
    async def health_check(self) -> bool:
        """Perform health check on agent components."""
        try:
            # Check LLM client
            test_response = await self.llm_client.generate(
                "Health check test",
                model=self.config.llm_model,
                max_tokens=10,
                timeout=5
            )
            
            # Check memory manager
            await self.memory_manager.health_check()
            
            # Check event bus
            await self.event_bus.health_check()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
