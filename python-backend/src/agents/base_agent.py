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
from ..models.agent_models import AgentMessage, AgentStatus, TaskResult
from ..models.task_models import TaskContext
from ..config.agent_configs import agent_config_manager
from ..core.enhanced_logger import get_logger, log_operation, log_llm_interaction, log_performance_metrics
from ..config import get_logger as get_config_logger
from ..models.tool_models import (
    BaseTool, ToolUsageEntry, ToolLearningModel, ToolMetrics, ToolUnderstanding,
    ToolResult, ToolUsagePlan, ToolOperation, ToolDiscovery, TaskOutcome
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
        
        # --- Tool-related attributes ---
        self.available_tools: Dict[str, BaseTool] = {}
        self.tool_capabilities: Dict[str, List[str]] = {}
        self.tool_usage_history: List[ToolUsageEntry] = []
        self.tool_learning_model: ToolLearningModel = ToolLearningModel()
        self.tool_performance_metrics: Dict[str, ToolMetrics] = {}
        self.active_tool_operations: Dict[str, Coroutine] = {}

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
        response = await super().execute_llm_task(
            user_prompt=full_prompt,
            context=context,
            structured_output=True,
            max_retries=max_retries
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

    async def discover_available_tools(self) -> None:
        """
        Discover and understand available tools:
        1. Query tool registry for all registered tools
        2. For each tool, call get_capabilities() and analyze
        3. Use LLM to understand tool descriptions and capabilities
        4. Store structured understanding in memory
        5. Subscribe to tool availability changes
        6. Build tool preference based on agent type
        """
        # 1. Query tool registry (assume registry is available via event_bus or memory_manager)
        tool_registry = await self.memory_manager.get_tool_registry()
        for tool_name, tool in tool_registry.items():
            self.available_tools[tool_name] = tool
            # 2. Get capabilities
            capabilities = await tool.get_capabilities()
            self.tool_capabilities[tool_name] = capabilities
            # 3. Analyze tool
            understanding = await self.analyze_tool_capability(tool)
            # 4. Store in memory
            await self.memory_manager.store_memory(
                content=f"Tool analyzed: {tool_name}",
                metadata={"type": "tool_understanding", "tool": tool_name, "understanding": understanding},
                importance=0.7
            )
        # 5. Subscribe to tool availability changes
        await self.event_bus.subscribe("tool.availability", self._handle_tool_availability)
        # 6. Build tool preference (simple: prefer tools matching agent_type)
        self.tool_learning_model.build_preferences(self.agent_type, self.tool_capabilities)

    async def analyze_tool_capability(self, tool: BaseTool) -> ToolUnderstanding:
        """
        Deep analysis of a tool's capabilities:
        1. Parse capability description
        2. Generate usage scenarios
        3. Identify parameter patterns
        4. Map to agent's responsibilities
        5. Create mental model of tool usage
        """
        description = await tool.get_description()
        capabilities = await tool.get_capabilities()
        # Use LLM to analyze
        prompt = (
            f"Tool: {tool.name}\n"
            f"Description: {description}\n"
            f"Capabilities: {capabilities}\n"
            "Analyze this tool for usage scenarios, parameter patterns, and mapping to agent responsibilities."
        )
        analysis = await self.execute_llm_task(prompt, context={"tool": tool.name}, include_tools=False)
        return ToolUnderstanding.from_dict(analysis if isinstance(analysis, dict) else {})

    # --- Intelligent Tool Usage ---

    async def use_tool(
        self, 
        tool_name: str, 
        operation: str,
        parameters: Dict[str, Any],
        reasoning: str
    ) -> ToolResult:
        """
        Execute tool with full context:
        1. Validate tool availability
        2. Check parameter validity
        3. Log reasoning for decision
        4. Execute with timeout and error handling
        5. Interpret results
        6. Update usage history
        7. Learn from outcome
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        tool = self.available_tools[tool_name]
        # 2. Validate parameters (assume tool provides validate_parameters)
        if hasattr(tool, "validate_parameters"):
            valid, msg = tool.validate_parameters(operation, parameters)
            if not valid:
                raise ValueError(f"Invalid parameters for {tool_name}.{operation}: {msg}")
        # 3. Log reasoning
        self.logger.info(
            f"Using tool '{tool_name}' for operation '{operation}'",
            operation="tool_usage",
            metadata={"parameters": parameters, "reasoning": reasoning}
        )
        # 4. Execute with timeout/error handling
        try:
            result = await asyncio.wait_for(
                tool.execute(operation, parameters),
                timeout=tool.timeout if hasattr(tool, "timeout") else 60
            )
            status = "success"
        except Exception as e:
            result = {"error": str(e)}
            status = "error"
        # 5. Interpret results (assume ToolResult)
        tool_result = ToolResult(
            tool_name=tool_name,
            operation=operation,
            parameters=parameters,
            result=result,
            status=status,
            reasoning=reasoning
        )
        # 6. Update usage history
        entry = ToolUsageEntry(
            tool_name=tool_name,
            operation=operation,
            parameters=parameters,
            result=result,
            status=status,
            reasoning=reasoning,
            timestamp=datetime.utcnow()
        )
        self.tool_usage_history.append(entry)
        # 7. Learn from outcome
        await self.learn_from_tool_usage(tool_name, operation, parameters, tool_result, TaskOutcome(status=status))
        return tool_result

    async def plan_tool_usage(self, task: str) -> ToolUsagePlan:
        """
        Create tool usage plan:
        1. Analyze task requirements
        2. Identify required tools
        3. Determine operation sequence
        4. Consider dependencies
        5. Estimate resource needs
        6. Plan fallback strategies
        """
        prompt = (
            f"Task: {task}\n"
            f"Available tools: {list(self.available_tools.keys())}\n"
            "Plan tool usage: required tools, operation sequence, dependencies, resources, fallbacks."
        )
        plan = await self.execute_llm_task(prompt, context={"task": task}, include_tools=True)
        return ToolUsagePlan.from_dict(plan if isinstance(plan, dict) else {})

    # --- Tool Learning and Adaptation ---

    async def learn_from_tool_usage(
        self,
        tool_name: str,
        operation: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        task_outcome: TaskOutcome
    ) -> None:
        """
        Learn from each tool usage:
        1. Correlate tool usage with task success
        2. Identify effective parameter patterns
        3. Update tool preference scores
        4. Share insights via event bus
        5. Adjust future tool selection
        6. Build tool combination patterns
        """
        self.tool_learning_model.update(tool_name, operation, parameters, result, task_outcome)
        # Share insights
        discovery = ToolDiscovery(
            tool_name=tool_name,
            operation=operation,
            parameters=parameters,
            result=result,
            task_outcome=task_outcome
        )
        await self.share_tool_discovery(discovery)

    async def share_tool_discovery(self, discovery: ToolDiscovery) -> None:
        """
        Share tool insights with other agents:
        1. Publish successful tool patterns
        2. Warn about tool limitations
        3. Suggest tool combinations
        4. Share performance metrics
        """
        await self.event_bus.publish("tool.discovery", discovery.to_dict())

    # --- Tool Operation Management ---

    async def execute_tool_workflow(
        self,
        workflow: List[ToolOperation]
    ) -> "WorkflowResult":
        """
        Execute complex multi-tool workflows:
        1. Validate workflow feasibility
        2. Manage tool operation dependencies
        3. Handle parallel operations where possible
        4. Rollback on failure if needed
        5. Collect comprehensive results
        """
        # 1. Validate feasibility (simple: check all tools available)
        for op in workflow:
            if op.tool_name not in self.available_tools:
                raise ValueError(f"Tool '{op.tool_name}' not available for workflow")
        results = []
        # 2. Manage dependencies (sequential for now)
        for op in workflow:
            result = await self.use_tool(
                tool_name=op.tool_name,
                operation=op.operation,
                parameters=op.parameters,
                reasoning=op.reasoning
            )
            results.append(result)
            if result.status != "success":
                # 4. Rollback if needed (not implemented)
                break
        # 5. Collect results
        from ..models.langgraph_models import WorkflowResult
        return WorkflowResult(
            workflow_id="tool_workflow_" + str(uuid.uuid4()),
            status="completed" if all(r.status == "success" for r in results) else "partial",
            deliverables={f"{r.tool_name}.{r.operation}": r.result for r in results},
            task_results={f"{r.tool_name}.{r.operation}": r.to_dict() for r in results},
            execution_metrics={},
            error_message=None if all(r.status == "success" for r in results) else "Some tool operations failed"
        )

    async def monitor_tool_operations(self) -> None:
        """
        Continuous monitoring of active operations:
        1. Track operation progress
        2. Detect stuck operations
        3. Handle timeouts gracefully
        4. Update operation status
        5. Trigger retries if needed
        """
        while True:
            for op_id, task in list(self.active_tool_operations.items()):
                if task.done():
                    result = task.result() if not task.cancelled() else None
                    self.logger.info(
                        f"Tool operation {op_id} completed",
                        operation="tool_monitor",
                        metadata={"result": result}
                    )
                    del self.active_tool_operations[op_id]
                elif task._state == "PENDING":
                    # Detect stuck (simple: log if pending too long)
                    self.logger.warning(
                        f"Tool operation {op_id} may be stuck",
                        operation="tool_monitor"
                    )
            await asyncio.sleep(5)

    async def _handle_tool_availability(self, message: Any) -> None:
        """Handle tool availability events (add/remove tools)."""
        # Example: message = {"tool": "tool_name", "status": "available|unavailable"}
        tool = message.get("tool")
        status = message.get("status")
        if status == "available":
            # Re-discover tool
            await self.discover_available_tools()
        elif status == "unavailable" and tool in self.available_tools:
            del self.available_tools[tool]
            self.logger.info(f"Tool '{tool}' removed from available_tools", operation="tool_availability")
