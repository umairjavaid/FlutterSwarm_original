import logging
import asyncio
from .config import settings
from .core.llm_client import LLMClient
from .core.event_bus import EventBus
from .core.memory_manager import MemoryManager
from .agents.orchestrator_agent import OrchestratorAgent
from .agents.base_agent import AgentConfig, AgentCapability

logger = logging.getLogger("flutterswarm.system_init")

async def initialize_flutterswarm() -> bool:
    """Initialize the FlutterSwarm system."""
    try:
        logger.info("Initializing FlutterSwarm system...")
        
        # Validate configuration first
        settings.validate()
        
        # Check available LLM providers
        available_providers = settings.llm.get_available_providers()
        if not available_providers:
            raise ValueError("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        
        default_model = settings.llm.default_model
        default_provider = settings.llm.get_default_provider()
        
        logger.info(f"Available LLM providers: {available_providers}")
        logger.info(f"Using default model: {default_model} (provider: {default_provider})")
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = LLMClient()
        
        # Test LLM connectivity with correct model
        try:
            test_response = await llm_client.generate(
                prompt="Respond with 'System OK' if you can process this message.",
                model=default_model
            )
            
            if "System OK" in test_response:
                logger.info("LLM connectivity test successful")
            else:
                logger.warning(f"LLM test response unexpected: {test_response}")
        except Exception as e:
            logger.warning(f"LLM client test failed: {e}")
        
        # Initialize event bus
        logger.info("Initializing event bus...")
        event_bus = EventBus()
        
        # Test event bus
        test_received = asyncio.Event()
        
        async def test_handler(message):
            test_received.set()
        
        await event_bus.subscribe("test.ping", test_handler)
        await event_bus.publish("test.ping", {"test": "data"})
        
        try:
            await asyncio.wait_for(test_received.wait(), timeout=1.0)
            logger.info("Event bus test successful")
        except asyncio.TimeoutError:
            logger.warning("Event bus test timeout")
        
        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager()
        
        # Initialize orchestrator agent
        logger.info("Initializing orchestrator agent...")
        orchestrator_config = AgentConfig(
            agent_id="orchestrator",
            agent_type="orchestrator",
            capabilities=[AgentCapability.ORCHESTRATION],
            llm_model=default_model  # Use the correct default model
        )
        
        orchestrator = OrchestratorAgent(
            config=orchestrator_config,
            llm_client=llm_client,
            memory_manager=memory_manager,
            event_bus=event_bus
        )
        
        # Initialize other system components here...
        # (database, redis, other agents, etc.)
        
        logger.info("FlutterSwarm system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize FlutterSwarm system: {e}")
        return False
