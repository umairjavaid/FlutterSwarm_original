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
from typing import Any, Dict, List, Optional, Union, Coroutine

from ..core.exceptions import AgentError, LLMError

from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, AgentStatus, TaskResult, MessageType
from ..models.task_models import TaskContext
from ..config.agent_configs import agent_config_manager
from ..core.enhanced_logger import get_logger, log_operation, log_llm_interaction, log_performance_metrics
from ..config import get_logger as get_config_logger
from ..core.tools.base_tool import BaseTool
from ..models.tool_models import (
    ToolUsageEntry, ToolLearningModel, ToolMetrics, ToolUnderstanding,
    ToolResult, ToolUsagePlan, ToolOperation, ToolDiscovery, TaskOutcome, ToolStatus
)


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
    
    # Tool integration attributes
    available_tools: Dict[str, Any] = field(default_factory=dict)
    tool_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    tool_usage_history: List[Any] = field(default_factory=list)
    tool_learning_model: Optional[Any] = None
    tool_performance_metrics: Dict[str, Any] = field(default_factory=dict)
    active_tool_operations: Dict[str, Any] = field(default_factory=dict)
    
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
        
        # --- Tool integration ---
        self.tool_registry = None
        self.available_tools: Dict[str, Any] = {}
        self.tool_capabilities: Dict[str, List[str]] = {}
        self.tool_usage_history: List[ToolUsageEntry] = []
        self.tool_learning_model: Optional[ToolLearningModel] = None
        self.tool_performance_metrics: Dict[str, ToolMetrics] = {}
        self.active_tool_operations: Dict[str, asyncio.Task] = {}
        
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
        # 3. Call LLM
        response = await self.llm_client.generate(
            prompt=full_prompt,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            agent_id=self.agent_id,
            structured_output=structured_output,
            max_retries=max_retries or self.config.max_retries
        )
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

    async def discover_available_tools(self) -> None:
        """
        Discover and understand available tools.
        
        This method queries the tool registry, analyzes each tool's capabilities,
        and builds the agent's understanding of what tools can do.
        """
        if not self.tool_registry:
            logger.warning("Tool registry not available for tool discovery")
            return
        
        try:
            # Get all available tools
            available_tools = self.tool_registry.get_available_tools()
            
            for tool in available_tools:
                # Analyze tool capability
                understanding = await self.analyze_tool_capability(tool)
                
                # Store tool information
                self.available_tools[tool.name] = tool
                self.tool_capabilities[tool.name] = understanding.usage_scenarios
                
                # Initialize performance metrics
                self.tool_performance_metrics[tool.name] = ToolMetrics()
                
            logger.info(f"Discovered {len(self.available_tools)} tools for agent {self.agent_id}")
            
            # Share discovery insights with other agents
            discovery_insight = ToolDiscovery(
                agent_id=self.agent_id,
                discovery_type="tool_discovery",
                tool_names=list(self.available_tools.keys()),
                description=f"Agent {self.agent_id} discovered {len(self.available_tools)} available tools",
                confidence=0.9,
                applicability=[self.config.agent_type]
            )
            
            await self.share_tool_discovery(discovery_insight)
            
        except Exception as e:
            logger.error(f"Tool discovery failed for agent {self.agent_id}: {e}")

    async def analyze_tool_capability(self, tool: BaseTool) -> ToolUnderstanding:
        """
        Deep analysis of a tool's capabilities through LLM reasoning.
        
        Args:
            tool: Tool to analyze
            
        Returns:
            ToolUnderstanding with structured analysis
        """
        try:
            # Get tool capabilities
            capabilities = await tool.get_capabilities()
            usage_examples = await tool.get_usage_examples()
            
            # Convert capabilities to JSON-serializable format
            def serialize_capabilities(caps):
                """Convert capabilities to JSON-serializable format."""
                try:
                    serializable = {}
                    for key, value in caps.__dict__.items():
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            # Handle lists/tuples that might contain enums or objects
                            serializable[key] = []
                            for item in value:
                                if hasattr(item, 'value'):  # Enum
                                    serializable[key].append(item.value)
                                elif isinstance(item, dict):
                                    # Handle nested dicts (like operations)
                                    serialized_item = {}
                                    for k, v in item.items():
                                        if k == 'required_permissions' and hasattr(v, '__iter__'):
                                            # Handle permissions list - ensure we get the value from enums
                                            try:
                                                serialized_item[k] = [
                                                    getattr(p, 'value', str(p)) for p in v
                                                ]
                                            except (AttributeError, TypeError):
                                                # Fallback if serialization fails
                                                serialized_item[k] = [str(p) for p in v]
                                        elif hasattr(v, 'value'):  # Enum
                                            serialized_item[k] = v.value
                                        else:
                                            serialized_item[k] = v
                                    serializable[key].append(serialized_item)
                                else:
                                    # For other types, try to convert to string if needed
                                    try:
                                        serializable[key].append(item)
                                    except (TypeError, AttributeError):
                                        serializable[key].append(str(item))
                        elif hasattr(value, 'value'):  # Enum
                            serializable[key] = value.value
                        else:
                            serializable[key] = value
                    return serializable
                except Exception as e:
                    logger.warning(f"Failed to serialize capabilities, using basic info: {e}")
                    return {
                        "tool_name": getattr(caps, 'name', 'unknown'),
                        "available_operations": [],
                        "error": f"Serialization failed: {str(e)}"
                    }

            serializable_capabilities = serialize_capabilities(capabilities)
            
            # Use LLM to understand the tool
            analysis_prompt = f"""
            Analyze this tool and understand its capabilities:
            
            Tool Name: {tool.name}
            Description: {tool.description}
            Version: {tool.version}
            Category: {tool.category.value}
            
            Capabilities:
            {json.dumps(serializable_capabilities.get('available_operations', [])[:5], indent=2)}  # First 5 operations
            
            Usage Examples:
            {json.dumps(usage_examples[:3], indent=2)}  # First 3 examples
            
            Provide analysis including:
            1. Summary of what this tool does
            2. When to use this tool (usage scenarios)
            3. What parameters it typically needs
            4. What success looks like
            5. Common failure patterns to watch for
            6. How this tool relates to my role as a {self.config.agent_type} agent
            
            Format response as structured data.
            """
            
            analysis_result = await self.execute_llm_task(
                user_prompt=analysis_prompt,
                context={
                    "tool_name": tool.name,
                    "agent_type": self.config.agent_type,
                    "capabilities": serializable_capabilities
                }
            )
            
            # Create tool understanding
            understanding = ToolUnderstanding(
                tool_name=tool.name,
                agent_id=self.agent_id,
                capability_summary=analysis_result.get("summary", ""),
                usage_scenarios=analysis_result.get("usage_scenarios", []),
                parameter_patterns=analysis_result.get("parameter_patterns", {}),
                success_indicators=analysis_result.get("success_indicators", []),
                failure_patterns=analysis_result.get("failure_patterns", []),
                confidence_level=0.8
            )
            
            # Store in memory for future reference
            await self.memory_manager.store_memory(
                content=f"Tool analysis: {tool.name} - {understanding.capability_summary}",
                metadata={
                    "type": "tool_understanding",
                    "tool_name": tool.name,
                    "agent_type": self.config.agent_type
                },
                importance=0.7,
                long_term=True
            )
            
            return understanding
            
        except Exception as e:
            logger.error(f"Tool capability analysis failed for {tool.name}: {e}")
            return ToolUnderstanding(
                tool_name=tool.name,
                agent_id=self.agent_id,
                capability_summary=f"Analysis failed: {str(e)}",
                confidence_level=0.0
            )

    # --- Intelligent Tool Usage ---

    async def use_tool(
        self,
        tool_name: str,
        operation: str,
        parameters: Dict[str, Any],
        reasoning: str
    ) -> ToolResult:
        """
        Execute tool with full context and learning.
        
        Args:
            tool_name: Name of the tool to use
            operation: Operation to perform
            parameters: Operation parameters
            reasoning: Reasoning for using this tool
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.now()
        
        try:
            # Validate tool availability
            if tool_name not in self.available_tools:
                return ToolResult(
                    operation_id=f"{tool_name}_{operation}_{int(start_time.timestamp())}",
                    status=ToolStatus.FAILURE,
                    error_message=f"Tool {tool_name} not available"
                )
            
            tool = self.available_tools[tool_name]
            
            # Validate parameters
            is_valid, error_msg = await tool.validate_params(operation, parameters)
            if not is_valid:
                return ToolResult(
                    operation_id=f"{tool_name}_{operation}_{int(start_time.timestamp())}",
                    status=ToolStatus.FAILURE,
                    error_message=f"Parameter validation failed: {error_msg}"
                )
            
            # Log reasoning
            logger.info(f"Agent {self.agent_id} using tool {tool_name}.{operation}: {reasoning}")
            
            # Execute tool operation
            result = await tool.execute(operation, parameters)
            
            # Record usage for learning
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_tool_usage(
                tool_name, operation, parameters, result, reasoning, execution_time
            )
            
            # Learn from the outcome
            await self.learn_from_tool_usage(
                tool_name, operation, parameters, result,
                TaskOutcome(
                    task_id=f"tool_usage_{int(start_time.timestamp())}",
                    success=(result.status == ToolStatus.SUCCESS),
                    quality_score=0.8 if result.status == ToolStatus.SUCCESS else 0.2,
                    efficiency_score=min(1.0, 10.0 / execution_time) if execution_time > 0 else 1.0,
                    tools_used=[tool_name],
                    total_time=execution_time
                )
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Tool usage failed: {e}")
            
            error_result = ToolResult(
                operation_id=f"{tool_name}_{operation}_{int(start_time.timestamp())}",
                status=ToolStatus.FAILURE,
                error_message=str(e),
                execution_time=execution_time
            )
            
            # Still record for learning
            await self._record_tool_usage(
                tool_name, operation, parameters, error_result, reasoning, execution_time
            )
            
            return error_result

    async def plan_tool_usage(self, task: str) -> ToolUsagePlan:
        """
        Create tool usage plan using LLM reasoning.
        
        Args:
            task: Task description
            
        Returns:
            ToolUsagePlan with detailed execution plan
        """
        try:
            # Get available tools and their capabilities
            available_tools_info = {}
            for tool_name, tool in self.available_tools.items():
                tool_info = await self.tool_registry.query_capabilities(tool_name)
                available_tools_info[tool_name] = tool_info
            
            planning_prompt = f"""
            Create a detailed plan for using tools to accomplish this task:
            
            Task: {task}
            
            Available Tools:
            {json.dumps(available_tools_info, indent=2)}
            
            My role: {self.config.agent_type} agent
            My capabilities: {self.config.capabilities}
            
            Create a plan including:
            1. Tool sequence (order of tool operations)
            2. Expected duration for each operation
            3. Resource requirements
            4. Fallback strategies if tools fail
            5. Success criteria for each step
            6. Risk assessment
            
            Consider:
            - Tool dependencies and prerequisites
            - Error handling and recovery
            - Efficiency and optimization
            - Resource constraints
            
            Format as structured tool usage plan.
            """
            
            plan_result = await self.execute_llm_task(
                user_prompt=planning_prompt,
                context={
                    "task": task,
                    "available_tools": list(self.available_tools.keys()),
                    "agent_type": self.config.agent_type
                }
            )
            
            # Convert to ToolUsagePlan
            tool_operations = []
            for step in plan_result.get("tool_sequence", []):
                operation = ToolOperation(
                    tool_name=step.get("tool_name", ""),
                    operation=step.get("operation", ""),
                    parameters=step.get("parameters", {}),
                    reasoning=step.get("reasoning", ""),
                    context={"step_number": step.get("step", 0)}
                )
                tool_operations.append(operation)
            
            return ToolUsagePlan(
                task_description=task,
                tool_sequence=tool_operations,
                estimated_duration=plan_result.get("estimated_duration", 300),
                resource_requirements=plan_result.get("resource_requirements", {}),
                fallback_strategies=plan_result.get("fallback_strategies", []),
                success_criteria=plan_result.get("success_criteria", []),
                risk_assessment=plan_result.get("risk_assessment", {})
            )
            
        except Exception as e:
            logger.error(f"Tool usage planning failed: {e}")
            return ToolUsagePlan(
                task_description=task,
                tool_sequence=[],
                estimated_duration=0,
                fallback_strategies=[f"Manual completion due to planning error: {str(e)}"]
            )

    async def learn_from_tool_usage(
        self,
        tool_name: str,
        operation: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        task_outcome: TaskOutcome
    ) -> None:
        """
        Learn from each tool usage to improve future decisions.
        
        Args:
            tool_name: Name of tool used
            operation: Operation performed
            parameters: Parameters used
            result: Tool execution result
            task_outcome: Overall task outcome
        """
        try:
            # Analyze the correlation between tool usage and outcome
            learning_prompt = f"""
            Analyze this tool usage and learn from the outcome:
            
            Tool Used: {tool_name}
            Operation: {operation}
            Parameters: {json.dumps(parameters, indent=2)}
            
            Tool Result:
            - Status: {result.status.value}
            - Execution Time: {result.execution_time}
            - Success: {result.status == ToolStatus.SUCCESS}
            
            Task Outcome:
            - Overall Success: {task_outcome.success}
            - Quality Score: {task_outcome.quality_score}
            - Efficiency Score: {task_outcome.efficiency_score}
            
            Learning Analysis:
            1. Was this tool the right choice for the task?
            2. Were the parameters optimal?
            3. What could be improved next time?
            4. Are there better tool alternatives?
            5. What patterns emerge from this usage?
            
            Provide insights for future tool selection and usage.
            """
            
            learning_analysis = await self.execute_llm_task(
                user_prompt=learning_prompt,
                context={
                    "tool_name": tool_name,
                    "agent_type": self.config.agent_type,
                    "result": result.__dict__,
                    "outcome": task_outcome.__dict__
                }
            )
            
            # Update tool preferences based on learning
            if tool_name not in self.tool_performance_metrics:
                self.tool_performance_metrics[tool_name] = ToolMetrics()
            
            metrics = self.tool_performance_metrics[tool_name]
            metrics.total_uses += 1
            
            # Update success rate
            if result.status == ToolStatus.SUCCESS:
                metrics.success_rate = (metrics.success_rate * (metrics.total_uses - 1) + 1.0) / metrics.total_uses
            else:
                metrics.success_rate = (metrics.success_rate * (metrics.total_uses - 1)) / metrics.total_uses
            
            # Update average execution time
            if result.execution_time:
                if metrics.average_execution_time == 0:
                    metrics.average_execution_time = result.execution_time
                else:
                    metrics.average_execution_time = (
                        metrics.average_execution_time * (metrics.total_uses - 1) + result.execution_time
                    ) / metrics.total_uses
            
            # Store learning insights in memory
            await self.memory_manager.store_memory(
                content=f"Tool learning: {tool_name} - {learning_analysis.get('summary', '')}",
                metadata={
                    "type": "tool_learning",
                    "tool_name": tool_name,
                    "operation": operation,
                    "success": task_outcome.success,
                    "agent_type": self.config.agent_type
                },
                importance=0.6,
                long_term=True
            )
            
        except Exception as e:
            logger.error(f"Learning from tool usage failed: {e}")

    async def share_tool_discovery(self, discovery: ToolDiscovery) -> None:
        """
        Share tool insights with other agents via event bus.
        
        Args:
            discovery: Tool discovery information to share
        """
        try:
            # Convert discovery to dict manually to handle serialization
            discovery_dict = {
                "discovery_id": discovery.discovery_id,
                "agent_id": discovery.agent_id,
                "discovery_type": discovery.discovery_type,
                "description": discovery.description,
                "tool_names": discovery.tool_names,
                "evidence": discovery.evidence,
                "confidence": discovery.confidence,
                "applicability": discovery.applicability,
                "timestamp": discovery.timestamp.isoformat()
            }
            
            # Create an AgentMessage object for the event bus
            discovery_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id="broadcast",
                message_type=MessageType.NOTIFICATION,
                content={
                    "agent_id": self.agent_id,
                    "discovery": discovery_dict,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            await self.event_bus.publish("tool_discovery", discovery_message)
            
            logger.debug(f"Shared tool discovery: {discovery.description}")
            
        except Exception as e:
            logger.error(f"Failed to share tool discovery: {e}")

    async def _record_tool_usage(
        self,
        tool_name: str,
        operation: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        reasoning: str,
        execution_time: float
    ) -> None:
        """Record tool usage for learning and optimization."""
        usage_entry = ToolUsageEntry(
            agent_id=self.agent_id,
            tool_name=tool_name,
            operation=operation,
            parameters=parameters,
            execution_time=execution_time,
            success=(result.status == ToolStatus.SUCCESS),
            error_details=result.error_message,
            context={
                "reasoning": reasoning,
                "agent_type": self.config.agent_type,
                "task_context": "agent_task"
            },
            outcome_quality=0.8 if result.status == ToolStatus.SUCCESS else 0.2
        )
        
        self.tool_usage_history.append(usage_entry)
        
        # Also record with the tool itself
        tool = self.available_tools.get(tool_name)
        if tool:
            await tool.record_usage(
                self.agent_id, operation, parameters, result, usage_entry.context
            )
