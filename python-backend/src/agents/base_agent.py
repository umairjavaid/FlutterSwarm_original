"""
Base Agent Framework for FlutterSwarm Multi-Agent System.

This module provides the abstract base class that all specialized agents inherit from.
Every agent uses LLM calls for reasoning and decision-making, with no hardcoded logic.
"""

import logging
import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Coroutine

from ..core.exceptions import AgentError, LLMError

from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, AgentStatus, TaskResult, TaskStatus, MessageType
from ..models.task_models import TaskContext
from ..core.enhanced_logger import get_logger, log_operation, log_llm_interaction, log_performance_metrics
from ..config import get_logger as get_config_logger
from ..core.tools.base_tool import BaseTool
from ..models.tool_models import (
    ToolUsageEntry, ToolLearningModel, ToolMetrics, ToolUnderstanding,
    ToolResult, ToolUsagePlan, ToolOperation, ToolDiscovery, TaskOutcome, ToolStatus,
    AsyncTask
)
from ..models.learning_models import AdvancedToolWorkflowMixin


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
    # Additional capabilities found in configurations
    DESIGN_PATTERNS = "design_patterns"
    COORDINATION = "coordination"
    ADAPTATION = "adaptation"
    RESOURCE_MANAGEMENT = "resource_management"
    PROJECT_STRUCTURE = "project_structure"


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
    max_retries: int = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tool integration attributes
    available_tools: Dict[str, BaseTool] = field(default_factory=dict)
    tool_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    tool_usage_history: List[ToolUsageEntry] = field(default_factory=list)
    tool_learning_model: Optional[ToolLearningModel] = None
    tool_performance_metrics: Dict[str, ToolMetrics] = field(default_factory=dict)
    active_tool_operations: Dict[str, AsyncTask] = field(default_factory=dict)
    
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
        if self.max_retries is None:
            self.max_retries = getattr(settings.llm, 'max_retries', 3)


class BaseAgent(ABC, AdvancedToolWorkflowMixin):
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
        
        # Tool integration attributes:
        available_tools: Dict[str, BaseTool] - Tools discovered and available for this agent
        tool_capabilities: Dict[str, List[str]] - Cached tool capabilities by tool name
        tool_usage_history: List[ToolUsageEntry] - Detailed usage tracking for learning
        tool_performance_metrics: Dict[str, ToolMetrics] - Performance metrics by tool name
        active_tool_operations: Dict[str, AsyncTask] - Currently running tool operations
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: 'MemoryManager',
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
        try:
            from ..config.agent_configs import agent_config_manager
            self.agent_config = agent_config_manager.get_agent_config(self.agent_type)
        except ImportError as e:
            self.logger.warning(f"Could not load agent configuration: {e}")
            self.agent_config = None
        
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
        
        # --- Tool integration ---
        self.tool_registry = None
        self.available_tools: Dict[str, BaseTool] = {}
        self.tool_capabilities: Dict[str, List[str]] = {}
        self.tool_usage_history: List[ToolUsageEntry] = []
        self.tool_learning_model: Optional[ToolLearningModel] = None
        self.tool_performance_metrics: Dict[str, ToolMetrics] = {}
        self.active_tool_operations: Dict[str, AsyncTask] = {}
        
        # --- Advanced workflow management (from AdvancedToolWorkflowMixin) ---
        self.operation_status: Dict[str, str] = {}
        self._operation_resources: Dict[str, List[Any]] = {}
        self._workflow_monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize tool system
        asyncio.create_task(self._initialize_tool_system())

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
        try:
            # Try to get the configured prompt - avoid circular import
            if hasattr(self, 'agent_config') and self.agent_config:
                config_prompt = getattr(self.agent_config, 'system_prompt', None)
                if config_prompt:
                    return config_prompt
        except Exception as e:
            self.logger.warning(f"Failed to get configured system prompt: {e}")
        
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
        max_retries: int = None,
        include_tools: bool = True
    ) -> dict:
        """
        Enhanced LLM execution with tool awareness:
        1. Include available tools in system prompt
        2. Add tool usage examples from history
        3. Parse tool usage intentions from response
        4. Execute planned tool operations
        5. Feed results back to LLM if needed
        6. Iterate until task complete
        
        Args:
            user_prompt: The specific task or question for the LLM
            context: Additional context including project state, constraints, etc.
            structured_output: Whether to expect JSON-formatted response
            max_retries: Maximum number of retry attempts on failure (uses config if None)
        
        Returns:
            LLM response as string or parsed JSON dictionary
        """
        # 1. Compose tool info for prompt
        tool_info = ""
        if include_tools and self.available_tools:
            tool_info = "\n".join(
                f"- {name}: {', '.join(self.tool_capabilities.get(name, []))}"
                for name in self.available_tools
            )
        usage_examples = ""
        if include_tools and self.tool_usage_history:
            usage_examples = "\n".join(
                f"{entry.tool_name}.{entry.operation}({entry.parameters}) => {entry.status}"
                for entry in self.tool_usage_history[-3:]
            )
        # 2. Build prompt
        system_prompt = await self.get_system_prompt()
        full_prompt = (
            f"{system_prompt}\n"
            f"Available tools:\n{tool_info}\n"
            f"Recent tool usage:\n{usage_examples}\n"
            f"Project context:\n{context.get('project', {})}\n"
            f"Task:\n{user_prompt}\n"
            "If tool usage is needed, specify tool, operation, and parameters."
        )
        # 3. Call LLM with timeout
        try:
            response = await asyncio.wait_for(
                self.llm_client.generate(
                    prompt=full_prompt,
                    model=self.config.llm_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    agent_id=self.agent_id
                ),
                timeout=self.config.timeout or 60  # Use config timeout or default to 60 seconds
            )
        except asyncio.TimeoutError:
            self.logger.error(f"LLM task execution timed out for agent {self.agent_id}")
            return {
                "error": "LLM execution timeout",
                "status": "timeout",
                "response": "Task execution timed out"
            }
        # 4. Parse tool usage intentions
        tool_plan = response.get("tool_plan") if isinstance(response, dict) else None
        if include_tools and tool_plan:
            # 5. Execute planned tool operations
            tool_results = []
            for op in tool_plan.get("operations", []):
                tool_result = await self.use_tool(
                    tool_name=op.get("tool"),
                    operation=op.get("operation"),
                    parameters=op.get("parameters", {}),
                    reasoning=op.get("reasoning", "")
                )
                tool_results.append(tool_result)
            # 6. Feed results back if needed
            response["tool_results"] = [tr.to_dict() for tr in tool_results]
        return response
    
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
        from ..core.exceptions import AgentError
        
        task_id = task_context.task_id
        correlation_id = task_context.metadata.get("correlation_id", str(uuid.uuid4()))
        
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
                        "priority": task_context.metadata.get("priority", "normal"),
                        "requirements_count": len(task_context.metadata.get("requirements", [])),
                        "deliverables_count": len(task_context.metadata.get("expected_deliverables", []))
                    }
                )
                
                # Update agent status
                old_status = self.status
                self.status = AgentStatus.BUSY
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
                        "requirements": task_context.metadata.get("requirements", []),
                        "expected_deliverables": task_context.metadata.get("expected_deliverables", []),
                        "correlation_id": correlation_id,
                        "task_id": task_id
                    },
                    structured_output=True
                )
                
                # Execute specialized processing if implemented
                specialized_result = {}
                if hasattr(self, '_execute_specialized_processing'):
                    self.logger.debug("Executing specialized processing")
                    # Assuming analysis_result is defined earlier in the actual code flow
                    # If analysis_result is not available here, this call needs to be adjusted
                    # For now, proceeding with the assumption it's available from prior LLM call
                    if 'analysis_result' not in locals() and 'analysis_result' not in globals():
                        # Fallback if analysis_result is not in scope, though it should be.
                        # This indicates a potential logic flow issue to be reviewed.
                        self.logger.warning("analysis_result not in local scope for _execute_specialized_processing")
                        # Creating a placeholder or fetching it if necessary
                        # For now, let's assume it might be part of task_context or self
                        # This part needs clarification based on the broader context of analysis_result
                        # If it's from the LLM call just before, it should be in scope.
                        # If the LLM call was:
                        # analysis_result = await self.execute_llm_task(...)
                        # then it should be available.

                        # Placeholder for where analysis_result should be defined
                        # analysis_result = await self.execute_llm_task(...) # This should be earlier

                        # If analysis_result is truly needed and not available, we might pass None or an empty dict
                        # depending on how _execute_specialized_processing handles it.
                        # For now, let's assume it's available from the preceding LLM call.
                        # If not, the method signature or logic needs adjustment.
                        # This is a common source of bugs - relying on variables that might not be in scope.
                        # The original code snippet was:
                        # specialized_result = await self._execute_specialized_processing(
                        #     task_context, analysis_result  <-- this analysis_result
                        # )
                        # So, it must be defined before this block.
                        # The provided context shows:
                        # analysis_result = await self.execute_llm_task(...)
                        # So it should be fine.

                    specialized_result = await self._execute_specialized_processing(
                        task_context, analysis_result # analysis_result should be in scope from LLM call
                    )

            except Exception as e:
                self.logger.error(f"Error during task analysis or specialized processing: {e}", exc_info=True)
                # Ensure task_result is created with an error status
                task_result = TaskResult(
                    task_id=task_id,
                    agent_id=self.agent_id,
                    status=TaskStatus.FAILED,
                    result=f"Failed to process task: {str(e)}",
                    error_message=str(e)
                )
                # No re-raise here, finally block will handle cleanup
            
            finally:
                # This 'finally' block is for the main try/except of process_task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                # Log task completion or failure
                if 'task_result' in locals() and task_result.status == TaskStatus.COMPLETED:
                    self.logger.info(f"Task {task_id} completed successfully by agent {self.agent_id}")
                elif 'task_result' in locals(): # Covers error status
                    self.logger.error(f"Task {task_id} failed for agent {self.agent_id}. Status: {task_result.status}, Error: {task_result.error_message if task_result.error_message else task_result.result}")
                else:
                    # This case should ideally not be reached if task_result is always initialized.
                    # It implies an error before task_result could be set, even to an error state.
                    self.logger.error(f"Task {task_id} processing concluded for agent {self.agent_id}, but task_result was not defined. This indicates a severe error early in processing.")
                    # Define a fallback task_result if it's missing entirely
                    task_result = TaskResult(
                        task_id=task_id,
                        agent_id=self.agent_id,
                        status=TaskStatus.FAILED,
                        result="Critical error in task processing, result not generated.",
                        error_message="task_result was not defined by the end of process_task"
                    )

                # Emit event for task completion/failure
                await self.event_bus.publish(
                    topic=f"agent.{self.agent_id}.task.finished",
                    message=AgentMessage(
                        message_id=str(uuid.uuid4()),
                        sender_id=self.agent_id,
                        receiver_id="system",
                        message_type=MessageType.NOTIFICATION,
                        content={
                            "task_id": task_id,
                            "status": task_result.status.value,
                            "result_summary": task_result.result[:200] if task_result.result else "N/A", # Summary
                        },
                        correlation_id=correlation_id
                    )
                )
                
                return task_result
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task - LangGraph compatibility method.
        
        This method provides a simple interface for LangGraph supervisor to execute tasks.
        It converts the task_data to TaskContext and calls process_task.
        
        Args:
            task_data: Dictionary containing task information
            
        Returns:
            Dictionary containing task result
        """
        try:
            # Convert task_data to TaskContext
            from ..models.task_models import TaskContext, TaskType
            
            task_context = TaskContext(
                task_id=task_data.get("task_id", f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                description=task_data.get("description", ""),
                task_type=TaskType.ANALYSIS,  # Default type
                metadata={
                    "priority": task_data.get("priority", "normal"),
                    "correlation_id": task_data.get("correlation_id", str(uuid.uuid4())),
                    "project_id": task_data.get("project_id", "default_project"),
                    **task_data.get("metadata", {})
                }
            )
            
            # Process the task
            result = await self.process_task(task_context)
            
            # Convert TaskResult to dict format expected by LangGraph
            return {
                "task_id": result.task_id,
                "agent_id": result.agent_id,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "result": result.result,
                "deliverables": getattr(result, 'deliverables', {}),
                "metadata": getattr(result, 'metadata', {}),
                "error_message": getattr(result, 'error_message', None)
            }
            
        except Exception as e:
            self.logger.error(f"Error in execute_task: {e}", exc_info=True)
            return {
                "task_id": task_data.get("task_id", "unknown"),
                "agent_id": self.agent_id,
                "status": "failed",
                "result": None,
                "error_message": str(e),
                "deliverables": {},
                "metadata": {}
            }
    
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
    
    # --- Tool Discovery and Understanding ---

    async def _initialize_tool_system(self):
        """Initialize the tool system for this agent."""
        try:
            from ..core.tools.tool_registry import ToolRegistry
            self.tool_registry = ToolRegistry.instance()
            
            # Ensure registry is initialized
            if not self.tool_registry.is_initialized:
                await self.tool_registry.initialize()
            
            # Discover available tools
            await self.discover_available_tools()
            
            logger.info(f"Tool system initialized for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize tool system: {e}")

    # ==========================================
    # Tool Integration Methods
    # ==========================================
    
    async def discover_available_tools(self) -> List[BaseTool]:
        """
        Discover all available tools from the tool registry.
        
        Returns:
            List of available tools
        """
        try:
            if not self.tool_registry:
                from ..core.tools.tool_registry import ToolRegistry
                self.tool_registry = ToolRegistry.instance()
            
            available_tools = self.tool_registry.get_available_tools()
            
            # Cache tools for quick access
            for tool in available_tools:
                self.available_tools[tool.name] = tool
            
            self.logger.info(
                f"Discovered {len(available_tools)} tools",
                operation="tool_discovery",
                metadata={"tool_names": [tool.name for tool in available_tools]}
            )
            
            return available_tools
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}", operation="tool_discovery")
            return []
    
    async def analyze_tool_capabilities(self, tool: BaseTool) -> ToolUnderstanding:
        """
        Analyze a tool's capabilities using LLM reasoning.
        
        Args:
            tool: The tool to analyze
            
        Returns:
            Tool understanding object with analysis results
        """
        try:
            # Generate analysis prompt
            prompt = await self.generate_tool_analysis_prompt(tool)
            
            # Get LLM analysis
            analysis_response = await self.llm_client.generate(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.3,  # Lower temperature for consistent analysis
                agent_id=self.agent_id,
                structured_output=True
            )
            
            # Parse and validate response
            analysis_result = await self.parse_tool_analysis_response(analysis_response)
            
            # Create tool understanding object
            understanding = ToolUnderstanding(
                tool_name=tool.name,
                agent_id=self.agent_id,
                understanding_timestamp=datetime.utcnow(),
                confidence_score=analysis_result.get("confidence", 0.7),
                analysis_result=analysis_result,
                usage_patterns=analysis_result.get("usage_scenarios", []),
                performance_expectations=analysis_result.get("performance_expectations", {}),
                error_patterns=analysis_result.get("failure_patterns", []),
                integration_notes=analysis_result.get("integration_notes", "")
            )
            
            # Store understanding
            await self.store_tool_understanding(understanding)
            
            self.logger.info(
                f"Analyzed tool {tool.name} with confidence {understanding.confidence_score:.2f}",
                operation="tool_analysis",
                metadata={
                    "tool_name": tool.name,
                    "confidence": understanding.confidence_score,
                    "scenarios_count": len(understanding.usage_patterns)
                }
            )
            
            return understanding
            
        except Exception as e:
            self.logger.error(f"Tool analysis failed for {tool.name}: {e}", operation="tool_analysis")
            
            # Return basic understanding as fallback
            return await self._generate_basic_understanding(tool)
    
    async def build_tool_understanding(self, tools: List[BaseTool]) -> Dict[str, ToolUnderstanding]:
        """
        Build understanding for multiple tools.
        
        Args:
            tools: List of tools to analyze
            
        Returns:
            Dictionary mapping tool names to their understanding
        """
        understanding_map = {}
        
        # Analyze tools in parallel for efficiency
        analysis_tasks = [
            self.analyze_tool_capabilities(tool) for tool in tools
        ]
        
        try:
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            for tool, result in zip(tools, results):
                if isinstance(result, ToolUnderstanding):
                    understanding_map[tool.name] = result
                else:
                    self.logger.warning(f"Failed to analyze {tool.name}: {result}")
                    # Create basic understanding for failed analysis
                    understanding_map[tool.name] = await self._generate_basic_understanding(tool)
            
        except Exception as e:
            self.logger.error(f"Batch tool analysis failed: {e}", operation="tool_analysis")
        
        return understanding_map
    
    async def update_tool_preferences(self, tool_usage_data: List[ToolUsageEntry]) -> None:
        """
        Update tool preferences based on usage data.
        
        Args:
            tool_usage_data: List of tool usage entries for learning
        """
        try:
            # Group usage data by tool
            tool_performance = defaultdict(list)
            for entry in tool_usage_data:
                tool_performance[entry.tool_name].append(entry)
            
            # Update preferences for each tool
            for tool_name, usage_entries in tool_performance.items():
                await self._update_single_tool_preference(tool_name, usage_entries)
            
            self.logger.info(
                f"Updated preferences for {len(tool_performance)} tools",
                operation="preference_update",
                metadata={"tools_updated": list(tool_performance.keys())}
            )
            
        except Exception as e:
            self.logger.error(f"Tool preference update failed: {e}", operation="preference_update")
    
    async def _update_single_tool_preference(self, tool_name: str, usage_entries: List[ToolUsageEntry]) -> None:
        """Update preference for a single tool based on its usage history."""
        if not usage_entries:
            return
        
        # Calculate performance metrics
        success_rates = [entry.success_rate for entry in usage_entries]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        # Get current preference
        current_preference = await self.get_tool_preference(tool_name)
        
        # Adjust preference based on performance
        if avg_success_rate > 0.9:
            new_preference = min(1.0, current_preference + 0.1)
        elif avg_success_rate < 0.6:
            new_preference = max(0.1, current_preference - 0.1)
        else:
            new_preference = current_preference
        
        # Store updated preference
        await self.set_tool_preference(tool_name, new_preference)
    
    async def generate_tool_analysis_prompt(self, tool: BaseTool) -> str:
        """
        Generate LLM prompt for tool capability analysis.
        
        Args:
            tool: Tool to generate prompt for
            
        Returns:
            Analysis prompt string
        """
        capabilities = await tool.get_capabilities()
        
        prompt = f"""
Analyze the following development tool for Flutter application development:

Tool Name: {tool.name}
Description: {tool.description}
Category: {tool.category.value}
Version: {tool.version}

Available Operations:
{chr(10).join([f"- {op.name}: {op.description}" for op in capabilities.available_operations])}

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "summary": "Brief summary of the tool's primary purpose",
    "usage_scenarios": [
        "List of specific scenarios where this tool would be used in Flutter development"
    ],
    "parameter_patterns": {{
        "parameter_name": "Description of what this parameter represents"
    }},
    "success_indicators": [
        "Indicators that suggest successful tool operation"
    ],
    "failure_patterns": [
        "Common failure scenarios and error patterns"
    ],
    "performance_expectations": {{
        "typical_execution_time": 0.0,
        "memory_usage": "low/medium/high",
        "cpu_intensive": false
    }},
    "decision_factors": [
        "Factors that should influence when to use this tool"
    ],
    "integration_notes": "Any special considerations for integrating this tool",
    "confidence": 0.8
}}

Focus on practical usage within Flutter development workflows and be specific about when and how this tool should be used.
"""
        
        return prompt
    
    async def parse_tool_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate LLM tool analysis response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Validated analysis result
        """
        try:
            # Ensure required fields exist
            required_fields = ["summary", "usage_scenarios", "confidence"]
            for field in required_fields:
                if field not in response:
                    response[field] = self._get_default_field_value(field)

            # Validate and normalize confidence score
            confidence = response.get("confidence", 0.5)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5

            response["confidence"] = max(0.0, min(1.0, confidence))

            # Ensure lists are actually lists
            list_fields = ["usage_scenarios", "success_indicators", "failure_patterns", "decision_factors"]
            for field in list_fields:
                if field in response and not isinstance(response[field], list):
                    response[field] = [str(response[field])]

            return response
            
        except Exception as e:
            self.logger.warning(f"Failed to parse tool analysis response: {e}")
            return self._get_default_analysis_response()
    
    async def select_best_tool(
        self,
        task_description: str,
        requirements: List[str],
        available_tools: Optional[List[BaseTool]] = None
    ) -> Optional[BaseTool]:
        """
        Select the most appropriate tool for a given task.
        
        Args:
            task_description: Description of the task
            requirements: List of task requirements
            available_tools: Optional list of tools to choose from
            
        Returns:
            Selected tool or None if no suitable tool found
        """
        try:
            if available_tools is None:
                available_tools = await self.discover_available_tools()
            
            if not available_tools:
                return None
            
            # Generate tool selection prompt
            selection_prompt = await self.generate_tool_selection_prompt(
                {"description": task_description, "requirements": requirements},
                available_tools
            )
            
            # Get LLM selection
            selection_response = await self.llm_client.generate(
                prompt=selection_prompt,
                model=self.config.llm_model,
                temperature=0.3,
                agent_id=self.agent_id,
                structured_output=True
            )
            
            # Parse selection
            parsed_selection = await self.parse_tool_selection_response(selection_response)
            
            # Find selected tool
            selected_tool_name = parsed_selection.get("tool_name")
            if selected_tool_name:
                for tool in available_tools:
                    if tool.name == selected_tool_name:
                        return tool
            
            return None
            
        except Exception as e:
            self.logger.error(f"Tool selection failed: {e}", operation="tool_selection")
            return None
    
    async def generate_tool_selection_prompt(self, task_context: Dict, available_tools: List[BaseTool]) -> str:
        """Generate prompt for tool selection."""
        tool_summaries = []
        
        for tool in available_tools:
            understanding = await self.get_tool_understanding(tool.name)
            if understanding:
                tool_summaries.append({
                    "name": tool.name,
                    "description": tool.description,
                    "scenarios": understanding.usage_patterns[:3],  # Top 3 scenarios
                    "confidence": understanding.confidence_score
                })
            else:
                capabilities = await tool.get_capabilities()
                tool_summaries.append({
                    "name": tool.name,
                    "description": tool.description,
                    "operations": [op.name for op in capabilities.available_operations[:3]],
                    "confidence": 0.5
                })
        
        prompt = f"""
Select the most appropriate tool for the following task:

Task Description: {task_context.get('description', '')}
Requirements: {', '.join(task_context.get('requirements', []))}

Available Tools:
{json.dumps(tool_summaries, indent=2)}

Please respond in JSON format:
{{
    "tool_name": "selected_tool_name",
    "confidence": 0.8,
    "reasoning": "explanation of why this tool was selected"
}}

Consider:
1. Tool capabilities match task requirements
2. Tool reliability and performance
3. Appropriateness for Flutter development
"""
        
        return prompt
    
    async def parse_tool_selection_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool selection response."""
        try:
            if "tool_name" not in response:
                return {"tool_name": None, "confidence": 0.0}
            
            confidence = response.get("confidence", 0.5)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5
            
            return {
                "tool_name": response["tool_name"],
                "confidence": max(0.0, min(1.0, confidence)),
                "reasoning": response.get("reasoning", "")
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse tool selection response: {e}")
            return {"tool_name": None, "confidence": 0.0}
    
    async def use_tool(
        self,
        tool: BaseTool,
        operation: str,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """
        Use a tool to perform an operation.
        
        Args:
            tool: Tool to use
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            is_valid, validation_error = await tool.validate_params(operation, parameters)
            if not is_valid:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error_message=f"Parameter validation failed: {validation_error}",
                    execution_time=time.time() - start_time
                )
            
            # Execute tool operation
            result = await tool.execute(operation, parameters)
            
            # Record usage for learning
            usage_entry = ToolUsageEntry(
                tool_name=tool.name,
                operation=operation,
                timestamp=datetime.utcnow(),
                success_rate=1.0 if result.status == ToolStatus.SUCCESS else 0.0,
                performance_metrics={
                    "execution_time": result.execution_time,
                    "memory_usage": result.data.get("memory_usage", 0) if result.data else 0
                },
                parameters_used=parameters
            )
            
            await self.update_tool_usage_patterns(usage_entry)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool execution failed: {e}", operation="tool_execution")
            
            return ToolResult(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                execution_time=execution_time
            )
    
    # Helper methods for tool integration
    
    async def get_tool_understanding(self, tool_name: str) -> Optional[ToolUnderstanding]:
        """Get stored understanding for a tool."""
        try:
            understanding_data = await self.memory_manager.retrieve_memory(
                f"tool_understanding_{tool_name}_{self.agent_id}"
            )
            return understanding_data
        except Exception:
            return None
    
    async def store_tool_understanding(self, understanding: ToolUnderstanding) -> None:
        """Store tool understanding in memory."""
        try:
            await self.memory_manager.store_memory(
                f"tool_understanding_{understanding.tool_name}_{self.agent_id}",
                understanding,
                category="tool_understanding"
            )
        except Exception as e:
            self.logger.error(f"Failed to store tool understanding: {e}")
    
    async def get_tool_preference(self, tool_name: str) -> float:
        """Get preference score for a tool (0.0 to 1.0)."""
        try:
            preference_data = await self.memory_manager.retrieve_memory(
                f"tool_preference_{tool_name}_{self.agent_id}"
            )
            return preference_data.get("preference", 0.5) if preference_data else 0.5
        except Exception:
            return 0.5
    
    async def set_tool_preference(self, tool_name: str, preference: float) -> None:
        """Set preference score for a tool."""
        try:
            await self.memory_manager.store_memory(
                f"tool_preference_{tool_name}_{self.agent_id}",
                {"preference": preference, "updated_at": datetime.utcnow()},
                category="tool_preferences"
            )
        except Exception as e:
            self.logger.error(f"Failed to store tool preference: {e}")
    
    async def _generate_basic_understanding(self, tool: BaseTool) -> ToolUnderstanding:
        """Generate basic understanding when LLM analysis fails."""
        capabilities = await tool.get_capabilities()
        
        return ToolUnderstanding(
            tool_name=tool.name,
            agent_id=self.agent_id,
            understanding_timestamp=datetime.utcnow(),
            confidence_score=0.5,
            analysis_result={
                "summary": f"Basic understanding of {tool.name}",
                "usage_scenarios": [f"Use {tool.name} for {tool.category.value} operations"],
                "confidence": 0.5
            },
            usage_patterns=[f"{tool.category.value}_operations"],
            performance_expectations={"execution_time": 1.0},
            error_patterns=["Unknown error patterns"],
            integration_notes="Generated from basic tool metadata"
        )
    
    def _get_default_field_value(self, field: str) -> Any:
        """Get default value for missing analysis fields."""
        defaults = {
            "summary": "Tool analysis summary",
            "usage_scenarios": ["General usage"],
            "confidence": 0.5,
            "parameter_patterns": {},
            "success_indicators": ["Operation completed"],
            "failure_patterns": ["Operation failed"],
            "decision_factors": ["Task requirements match tool capabilities"]
        }
        return defaults.get(field, None)
    
    def _get_default_analysis_response(self) -> Dict[str, Any]:
        """Get default analysis response when parsing fails."""
        return {
            "summary": "Default tool analysis",
            "usage_scenarios": ["General development tasks"],
            "parameter_patterns": {},
            "success_indicators": ["Operation successful"],
            "failure_patterns": ["Operation failed"],
            "performance_expectations": {"execution_time": 1.0},
            "decision_factors": ["Task requirements"],
            "integration_notes": "Default analysis",
            "confidence": 0.5
        }
