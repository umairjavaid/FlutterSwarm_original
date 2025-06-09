"""
System Initialization for FlutterSwarm Multi-Agent System.

This module handles the initialization and coordination of all core components
including LLM clients, memory managers, event bus, and agent orchestration.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any

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
from .agents.performance_agent import PerformanceAgent
from .agents.documentation_agent import DocumentationAgent
from .config.settings import settings
from .config import get_logger

logger = get_logger("system_init")


class FlutterSwarmSystem:
    """
    Main system coordinator for the FlutterSwarm multi-agent framework.
    
    This class manages the lifecycle of all system components and provides
    the main interface for interacting with the multi-agent system.
    """
    
    def __init__(self):
        self.llm_client: Optional[LLMClient] = None
        self.event_bus: Optional[EventBus] = None
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.agents: Dict[str, BaseAgent] = {}
        self.memory_managers: Dict[str, MemoryManager] = {}
        self.is_initialized = False
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize all system components."""
        try:
            logger.info("Initializing FlutterSwarm system...")
            
            # Validate configuration
            settings.validate()
            
            # Initialize core components
            await self._initialize_llm_client()
            await self._initialize_event_bus()
            await self._initialize_orchestrator()
            await self._initialize_specialized_agents()
            
            # Setup system-wide event handlers
            await self._setup_system_events()
            
            self.is_initialized = True
            logger.info("FlutterSwarm system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FlutterSwarm system: {e}")
            raise
    
    async def _initialize_llm_client(self) -> None:
        """Initialize the LLM client with configured providers."""
        logger.info("Initializing LLM client...")
        
        self.llm_client = LLMClient()
        
        # Test LLM connectivity
        try:
            test_response = await self.llm_client.generate(
                prompt="Hello, this is a connectivity test.",
                max_tokens=10,
                temperature=0.1
            )
            logger.info(f"LLM client test successful: {test_response[:50]}...")
        except Exception as e:
            logger.warning(f"LLM client test failed: {e}")
            # Continue initialization - LLM may still work for actual tasks
    
    async def _initialize_event_bus(self) -> None:
        """Initialize the event bus system."""
        logger.info("Initializing event bus...")
        
        self.event_bus = EventBus(
            max_history=settings.event_bus.buffer_size,
            max_retry_attempts=3
        )
        
        # Test event bus
        test_received = False
        
        async def test_handler(message):
            nonlocal test_received
            test_received = True
        
        await self.event_bus.subscribe("test.ping", test_handler)
        
        from .models.agent_models import AgentMessage
        await self.event_bus.publish(
            "test.ping", 
            AgentMessage(
                type="test",
                source="system",
                target="system",
                payload={"test": True}
            )
        )
        
        # Give it a moment to process
        await asyncio.sleep(0.1)
        
        if test_received:
            logger.info("Event bus test successful")
        else:
            logger.warning("Event bus test failed - messages may not be delivered")
    
    async def _initialize_orchestrator(self) -> None:
        """Initialize the orchestrator agent."""
        logger.info("Initializing orchestrator agent...")
        
        # Create memory manager for orchestrator
        orchestrator_memory = MemoryManager(
            agent_id="orchestrator",
            llm_client=self.llm_client
        )
        self.memory_managers["orchestrator"] = orchestrator_memory
        
        # Create orchestrator config
        orchestrator_config = AgentConfig(
            agent_id="orchestrator",
            agent_type="orchestrator",
            capabilities=[],
            max_concurrent_tasks=settings.agent.max_concurrent_tasks,
            llm_model=settings.llm.default_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout
        )
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(
            config=orchestrator_config,
            llm_client=self.llm_client,
            memory_manager=orchestrator_memory,
            event_bus=self.event_bus
        )
        
        self.agents["orchestrator"] = self.orchestrator
        
        # Test orchestrator health
        if await self.orchestrator.health_check():
            logger.info("Orchestrator agent initialized successfully")
        else:
            logger.warning("Orchestrator health check failed")
    
    async def _initialize_specialized_agents(self) -> None:
        """Initialize all specialized agents in the system."""
        logger.info("Initializing specialized agents...")
        
        # Define agent configurations for each specialized agent
        agent_configs = [
            {
                "class": ArchitectureAgent,
                "agent_id": "architecture_agent",
                "agent_type": "architecture",
                "description": "Architecture planning and system design agent"
            },
            {
                "class": ImplementationAgent,
                "agent_id": "implementation_agent", 
                "agent_type": "implementation",
                "description": "Code generation and feature implementation agent"
            },
            {
                "class": TestingAgent,
                "agent_id": "testing_agent",
                "agent_type": "testing", 
                "description": "Testing strategy and test automation agent"
            },
            {
                "class": DevOpsAgent,
                "agent_id": "devops_agent",
                "agent_type": "devops",
                "description": "CI/CD and deployment automation agent"
            },
            {
                "class": SecurityAgent,
                "agent_id": "security_agent",
                "agent_type": "security",
                "description": "Security assessment and compliance agent"
            },
            {
                "class": PerformanceAgent,
                "agent_id": "performance_agent",
                "agent_type": "performance", 
                "description": "Performance analysis and optimization agent"
            },
            {
                "class": DocumentationAgent,
                "agent_id": "documentation_agent",
                "agent_type": "documentation",
                "description": "Documentation generation and knowledge management agent"
            }
        ]
        
        # Initialize each specialized agent
        for agent_info in agent_configs:
            try:
                logger.info(f"Initializing {agent_info['description']}...")
                
                # Create memory manager for the agent
                memory_manager = MemoryManager(
                    agent_id=agent_info["agent_id"],
                    llm_client=self.llm_client
                )
                self.memory_managers[agent_info["agent_id"]] = memory_manager
                
                # Create agent configuration
                agent_config = AgentConfig(
                    agent_id=agent_info["agent_id"],
                    agent_type=agent_info["agent_type"],
                    capabilities=[],  # Will be populated by the agent itself
                    max_concurrent_tasks=settings.agent.max_concurrent_tasks,
                    llm_model=settings.llm.default_model,
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens,
                    timeout=settings.llm.timeout
                )
                
                # Initialize the agent
                agent = agent_info["class"](
                    config=agent_config,
                    llm_client=self.llm_client,
                    memory_manager=memory_manager,
                    event_bus=self.event_bus
                )
                
                # Register the agent
                self.agents[agent_info["agent_id"]] = agent
                
                # Test agent health
                if await agent.health_check():
                    logger.info(f"{agent_info['description']} initialized successfully")
                else:
                    logger.warning(f"{agent_info['description']} health check failed")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {agent_info['description']}: {e}")
                # Continue with other agents even if one fails
                continue
        
        logger.info(f"Specialized agents initialization complete. {len(self.agents)} total agents registered.")

    async def _setup_system_events(self) -> None:
        """Setup system-wide event handlers."""
        logger.info("Setting up system event handlers...")
        
        # System shutdown handler
        await self.event_bus.subscribe(
            "system.shutdown",
            self._handle_system_shutdown
        )
        
        # Agent registration handler
        await self.event_bus.subscribe(
            "agent.register",
            self._handle_agent_registration
        )
        
        # Agent unregistration handler
        await self.event_bus.subscribe(
            "agent.unregister",
            self._handle_agent_unregistration
        )
        
        # System health check handler
        await self.event_bus.subscribe(
            "system.health_check",
            self._handle_health_check
        )
    
    async def start(self) -> None:
        """Start the FlutterSwarm system."""
        if not self.is_initialized:
            await self.initialize()
        
        logger.info("Starting FlutterSwarm system...")
        
        try:
            # Start orchestrator
            if self.orchestrator:
                self.orchestrator.status = self.orchestrator.status.__class__.IDLE
            
            self.is_running = True
            logger.info("FlutterSwarm system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FlutterSwarm system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the FlutterSwarm system gracefully."""
        logger.info("Stopping FlutterSwarm system...")
        
        try:
            # Stop all agents
            for agent_id, agent in self.agents.items():
                logger.info(f"Stopping agent: {agent_id}")
                await self.event_bus.publish(
                    f"agent.shutdown.{agent_id}",
                    self._create_system_message("shutdown", {"agent_id": agent_id})
                )
            
            # Give agents time to shutdown gracefully
            await asyncio.sleep(2)
            
            # Stop event bus
            if self.event_bus:
                await self.event_bus.shutdown()
            
            self.is_running = False
            logger.info("FlutterSwarm system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            raise
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the system."""
        logger.info(f"Registering agent: {agent.agent_id} ({agent.agent_type})")
        
        # Add to agents registry
        self.agents[agent.agent_id] = agent
        
        # Create memory manager for agent if needed
        if agent.agent_id not in self.memory_managers:
            memory_manager = MemoryManager(
                agent_id=agent.agent_id,
                llm_client=self.llm_client
            )
            self.memory_managers[agent.agent_id] = memory_manager
            
            # Update agent's memory manager if it doesn't have LLM client
            if not hasattr(agent.memory_manager, 'llm_client') or not agent.memory_manager.llm_client:
                agent.memory_manager.llm_client = self.llm_client
        
        # Register with orchestrator
        if self.orchestrator:
            from .models.agent_models import AgentCapabilityInfo
            capability_info = AgentCapabilityInfo(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                capabilities=[cap.value for cap in agent.capabilities],
                specializations=await agent.get_capabilities(),
                performance_metrics={},
                availability=True,
                current_load=0,
                max_concurrent_tasks=agent.config.max_concurrent_tasks
            )
            await self.orchestrator.register_agent(capability_info)
        
        logger.info(f"Agent {agent.agent_id} registered successfully")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the system."""
        logger.info(f"Unregistering agent: {agent_id}")
        
        # Remove from agents registry
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        # Unregister from orchestrator
        if self.orchestrator:
            await self.orchestrator.unregister_agent(agent_id)
        
        logger.info(f"Agent {agent_id} unregistered successfully")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a high-level request through the multi-agent system.
        
        This is the main entry point for external requests.
        """
        if not self.is_running:
            raise RuntimeError("FlutterSwarm system is not running")
        
        logger.info(f"Processing request: {request.get('type', 'unknown')}")
        
        try:
            # Create task context from request
            from .models.task_models import TaskContext, TaskType, TaskPriority
            from .models.project_models import ProjectContext
            
            task_context = TaskContext(
                task_id=request.get('task_id', f"task_{asyncio.get_event_loop().time()}"),
                description=request.get('description', ''),
                task_type=TaskType(request.get('task_type', 'analysis')),
                requirements=request.get('requirements', []),
                expected_deliverables=request.get('expected_deliverables', []),
                project_context=request.get('project_context', {}),
                priority=TaskPriority[request.get('priority', 'NORMAL').upper()],
                metadata=request.get('metadata', {})
            )
            
            # Process through orchestrator
            result = await self.orchestrator.process_task(task_context)
            
            return {
                "success": result.is_successful(),
                "task_id": result.task_id,
                "status": result.status,
                "result": result.result,
                "deliverables": result.deliverables,
                "error": result.error,
                "execution_time": result.execution_time,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "system": {
                "initialized": self.is_initialized,
                "running": self.is_running,
                "agents_count": len(self.agents),
                "memory_managers_count": len(self.memory_managers)
            },
            "agents": {},
            "event_bus": {},
            "llm_client": {}
        }
        
        # Agent statuses
        for agent_id, agent in self.agents.items():
            try:
                status["agents"][agent_id] = await agent.get_status()
            except Exception as e:
                status["agents"][agent_id] = {"error": str(e)}
        
        # Event bus status
        if self.event_bus:
            try:
                status["event_bus"] = await self.event_bus.get_metrics()
            except Exception as e:
                status["event_bus"] = {"error": str(e)}
        
        # LLM client status
        if self.llm_client:
            try:
                status["llm_client"] = self.llm_client.get_statistics()
            except Exception as e:
                status["llm_client"] = {"error": str(e)}
        
        return status
    
    def _create_system_message(self, message_type: str, payload: Dict[str, Any]):
        """Create a system message for internal communication."""
        from .models.agent_models import AgentMessage
        return AgentMessage(
            type=message_type,
            source="system",
            target="broadcast",
            payload=payload
        )
    
    async def _handle_system_shutdown(self, message) -> None:
        """Handle system shutdown request."""
        logger.info("Received system shutdown request")
        await self.stop()
    
    async def _handle_agent_registration(self, message) -> None:
        """Handle agent registration request."""
        payload = message.payload
        agent_id = payload.get("agent_id")
        logger.info(f"Received agent registration request for: {agent_id}")
        # Agent registration logic would go here
    
    async def _handle_agent_unregistration(self, message) -> None:
        """Handle agent unregistration request."""
        payload = message.payload
        agent_id = payload.get("agent_id")
        await self.unregister_agent(agent_id)
    
    async def _handle_health_check(self, message) -> None:
        """Handle system health check request."""
        logger.info("Performing system health check")
        status = await self.get_system_status()
        
        # Respond with health status
        response = self._create_system_message("health_check_response", status)
        await self.event_bus.publish("system.health_check.response", response)


# Global system instance
flutter_swarm_system = FlutterSwarmSystem()


# Convenience functions for external usage
async def initialize_system() -> FlutterSwarmSystem:
    """Initialize and return the FlutterSwarm system."""
    await flutter_swarm_system.initialize()
    return flutter_swarm_system


async def start_system() -> FlutterSwarmSystem:
    """Start the FlutterSwarm system."""
    await flutter_swarm_system.start()
    return flutter_swarm_system


async def stop_system() -> None:
    """Stop the FlutterSwarm system."""
    await flutter_swarm_system.stop()


async def get_system() -> FlutterSwarmSystem:
    """Get the current system instance."""
    return flutter_swarm_system
