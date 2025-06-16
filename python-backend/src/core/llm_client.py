"""
LLM Client for FlutterSwarm Multi-Agent System.

This module provides a unified interface for interacting with Anthropic's Claude models.
It handles authentication, rate limiting, retries, and logging for all LLM interactions.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp

from ..config import settings
from ..models.agent_models import LLMInteraction, LLMError

logger = logging.getLogger(__name__)

# Import for Anthropic
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMRequest:
    """Request structure for LLM calls."""
    prompt: str
    model: str
    temperature: float = None
    max_tokens: int = None
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = None
    agent_id: str = ""
    correlation_id: str = ""
    
    def __post_init__(self):
        """Set defaults from config if not provided."""
        from ..config.settings import settings
        if self.temperature is None:
            self.temperature = settings.llm.temperature
        if self.max_tokens is None:
            self.max_tokens = settings.llm.max_tokens


@dataclass
class LLMResponse:
    """Response structure from LLM calls."""
    content: str
    model: str
    tokens_used: int
    response_time: float
    finish_reason: str
    metadata: Dict[str, Any] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, rate_limit: int = 60):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._request_times: List[float] = []
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        # Check if we've exceeded the rate limit
        if len(self._request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(now)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) LLM provider implementation."""
    
    def __init__(self, api_key: str, rate_limit: int = 60):
        super().__init__(api_key, rate_limit)
        if not ANTHROPIC_AVAILABLE:
            raise LLMError("Anthropic library not installed. Install with: pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic's API with timeout handling."""
        await self._check_rate_limit()
        
        start_time = time.time()
        
        try:
            # Format messages for Claude API
            messages = []
            
            # Add system message if provided
            if request.system_prompt:
                messages.append({"role": "user", "content": f"System: {request.system_prompt}\n\nUser: {request.prompt}"})
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            # Use asyncio.wait_for to add timeout
            timeout = 180  # Increased from 60 to 180 seconds
            
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                model=request.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                response_time=response_time,
                finish_reason=response.stop_reason or "stop",
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            logger.error(f"LLM request timed out after {timeout} seconds")
            raise LLMError(f"Request timed out after {timeout} seconds")
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Anthropic API error after {response_time:.2f}s: {str(e)}")
            raise LLMError(f"Anthropic API error: {str(e)}")


class LocalFallbackProvider(BaseLLMProvider):
    """
    Local fallback provider that generates realistic Flutter development responses
    when no API key is available. This enables the system to function for demonstration.
    """
    
    def __init__(self):
        super().__init__("local_fallback", rate_limit=100)
        self.flutter_templates = {
            "architecture": {
                "clean_architecture": """
                ## Clean Architecture for Music Streaming App

                ### Recommended Architecture:
                - **Clean Architecture** with dependency injection using GetIt
                - **BLoC Pattern** for state management 
                - **Repository Pattern** for data abstraction

                ### Project Structure:
                ```
                lib/
                ├── core/
                │   ├── di/               # Dependency injection
                │   ├── error/           # Error handling
                │   ├── constants/       # App constants
                │   └── utils/          # Utilities
                ├── data/
                │   ├── repositories/    # Repository implementations
                │   ├── datasources/     # Local/remote data sources
                │   └── models/         # Data models
                ├── domain/
                │   ├── entities/        # Business entities
                │   ├── repositories/    # Repository interfaces
                │   └── usecases/       # Business logic
                └── presentation/
                    ├── blocs/          # BLoC state management
                    ├── pages/          # Screen widgets
                    └── widgets/        # Reusable widgets
                ```

                ### Dependencies:
                - flutter_bloc: ^8.1.3
                - get_it: ^7.6.4
                - injectable: ^2.3.2
                - audio_service: ^0.18.12
                - just_audio: ^0.9.35
                """,
                "implementation": """
                I'll implement the Flutter music streaming app with the following approach:

                **Tool Usage Required:**
                1. flutter_sdk: create_project - Initialize new Flutter project
                2. file_system: create_from_template - Generate clean architecture structure
                3. flutter_sdk: add_dependencies - Add audio and state management packages

                **Implementation Plan:**
                1. Create project structure with clean architecture
                2. Set up dependency injection with GetIt
                3. Implement audio player with just_audio
                4. Create BLoC states for music player
                5. Design modern UI with Material 3
                6. Add playlist and library management
                7. Implement background audio service

                **Key Features to Implement:**
                - Audio playback controls (play/pause/skip)
                - Playlist management with local storage
                - Modern UI with dark/light themes
                - Background audio service
                - Search and library organization
                """
            }
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a realistic Flutter development response based on the prompt."""
        prompt_lower = request.prompt.lower()
        
        # Determine response content based on prompt
        response_content = ""
        
        # Architecture-related responses
        if any(keyword in prompt_lower for keyword in ["architecture", "design", "structure", "pattern"]):
            if "music" in prompt_lower or "audio" in prompt_lower:
                response_content = self.flutter_templates["architecture"]["clean_architecture"]
            else:
                response_content = """
            ## Flutter App Architecture Recommendation

            ### Recommended Approach:
            - **Clean Architecture** with clear separation of concerns
            - **BLoC Pattern** for predictable state management
            - **Repository Pattern** for data abstraction
            - **Dependency Injection** using GetIt for testability

            ### Project Structure:
            ```
            lib/
            ├── core/              # Core functionality
            ├── data/              # Data layer
            ├── domain/            # Business logic
            └── presentation/      # UI layer
            ```
            """

        # Implementation-related responses  
        elif any(keyword in prompt_lower for keyword in ["implement", "create", "build", "develop"]):
            if "music" in prompt_lower or "audio" in prompt_lower:
                response_content = self.flutter_templates["architecture"]["implementation"]
            else:
                response_content = """
            I'll implement the Flutter application following clean architecture principles.

            **Tool Usage Required:**
            1. flutter_sdk: create_project - Initialize Flutter project
            2. file_system: create_from_template - Generate project structure
            3. flutter_sdk: add_dependencies - Add required packages

            **Implementation Steps:**
            1. Set up project structure
            2. Configure dependencies
            3. Implement core features
            4. Create UI components
            5. Add state management
            6. Implement tests
            """

        # Tool usage responses
        elif "tool" in prompt_lower:
            response_content = """
            **Available Tools:**
            - flutter_sdk: Flutter SDK operations (create, build, test)
            - file_system: File operations (create, read, write)
            - process_tool: Process management (run commands)

            **Recommended Usage:**
            1. Use flutter_sdk for Flutter-specific operations
            2. Use file_system for template-based code generation
            3. Use process_tool for running external commands
            """
        else:
            # Default response
            response_content = """
        I understand the task requirements. I'll proceed with implementing the solution following Flutter best practices and clean architecture principles.

        **Next Steps:**
        1. Analyze requirements in detail
        2. Design appropriate architecture
        3. Implement core functionality
        4. Create comprehensive tests
        5. Ensure production readiness
        """
        
        return LLMResponse(
            content=response_content,
            model="local_fallback",
            tokens_used=100,
            response_time=0.1,
            finish_reason="completed",
            metadata={"source": "local_fallback"}
        )


class LLMClient:
    """
    LLM client that manages Anthropic provider and handles
    routing, retries, fallbacks, and logging.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = None
        self.interactions: List[LLMInteraction] = []
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initializes LLM providers. Attempts to use configured API-based providers first.
        Falls back to LocalFallbackProvider if API-based providers are not available,
        not configured, or fail to initialize.
        """
        primary_provider_initialized = False
        try:
            # Attempt to initialize Anthropic provider
            if hasattr(settings.llm, 'anthropic_api_key') and settings.llm.anthropic_api_key:
                if ANTHROPIC_AVAILABLE:
                    logger.info("Anthropic API key found. Initializing Anthropic provider.")
                    self.providers['anthropic'] = AnthropicProvider(
                        api_key=settings.llm.anthropic_api_key,
                        rate_limit=getattr(settings.llm, 'anthropic_rate_limit', 60)
                    )
                    self.default_provider = 'anthropic'
                    primary_provider_initialized = True
                    logger.info("Anthropic provider initialized and set as default.")
                else:
                    logger.error("Anthropic API key provided but 'anthropic' library not installed. "
                                 "Install with: pip install anthropic. Will attempt fallback.")
            else:
                logger.info("Anthropic API key not found or not configured in settings.llm. "
                             "Will attempt other providers or fallback.")
            
            # Add other primary provider initializations here if any in the future.
            # Example:
            # if hasattr(settings.llm, 'openai_api_key') and settings.llm.openai_api_key:
            #     if OPENAI_AVAILABLE: # Check if library for openai is available
            #         logger.info("OpenAI API key found. Initializing OpenAI provider.")
            #         self.providers['openai'] = OpenAIProvider(api_key=settings.llm.openai_api_key) # Assuming OpenAIProvider exists
            #         if not primary_provider_initialized: # Set as default if no other primary was set
            #             self.default_provider = 'openai'
            #             primary_provider_initialized = True
            #             logger.info("OpenAI provider initialized and set as default.")
            #     else:
            #         logger.error("OpenAI API key provided but library not installed.")


        except Exception as e:
            logger.error(f"Exception during primary LLM provider initialization: {e}. "
                         "Will use fallback provider.")
            primary_provider_initialized = False # Ensure fallback is used

        # Fallback initialization if no primary provider was successfully initialized
        if not primary_provider_initialized:
            logger.warning("No primary LLM provider was successfully initialized or configured. "
                         "Using local fallback provider.")
            # Ensure LocalFallbackProvider is in providers list
            if 'local_fallback' not in self.providers:
                try:
                    self.providers['local_fallback'] = LocalFallbackProvider()
                except Exception as fallback_init_e:
                    # This is a critical failure if even LocalFallbackProvider can't be initialized.
                    raise LLMError(f"CRITICAL: Failed to initialize LocalFallbackProvider: {fallback_init_e}")
            
            self.default_provider = 'local_fallback'
            logger.info("LocalFallbackProvider is set as the default provider.")
        
        # Final check: if default_provider is somehow None despite the logic above
        if not self.default_provider:
            if self.providers: # If there are any providers, pick the first one
                self.default_provider = list(self.providers.keys())[0]
                logger.warning(f"Default provider was not set, defaulting to first available: {self.default_provider}")
            else:
                # This state should ideally not be reached if LocalFallbackProvider initialization is robust.
                # As an absolute last resort, try to create LocalFallbackProvider again if providers is empty.
                try:
                    logger.error("CRITICAL: No providers available and default_provider is None. "
                                 "Attempting to force LocalFallbackProvider one last time.")
                    self.providers['local_fallback'] = LocalFallbackProvider()
                    self.default_provider = 'local_fallback'
                except Exception as final_emergency_e:
                    raise LLMError(f"CRITICAL: Failed to initialize ANY LLM provider, including emergency fallback: {final_emergency_e}")

        logger.info(f"LLMClient initialization finished. Default provider: '{self.default_provider}'. "
                    f"Available providers: {list(self.providers.keys())}")
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: str = "",
        correlation_id: str = "",
        provider: Optional[str] = None,
        retry_count: int = None
    ) -> str:
        """
        Generate text using the specified or default LLM provider.
        
        Args:
            prompt: Input prompt for the LLM
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt for context
            context: Additional context for the request
            agent_id: ID of requesting agent
            correlation_id: Request correlation ID
            provider: LLM provider to use
            retry_count: Number of retries on failure
            
        Returns:
            Generated text response
        """
        from ..config.settings import settings
        
        # Use config defaults if not provided
        if model is None:
            model = settings.llm.default_model
        if temperature is None:
            temperature = settings.llm.temperature
        if max_tokens is None:
            max_tokens = settings.llm.max_tokens
        if retry_count is None:
            retry_count = settings.llm.max_retries
        
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            context=context or {},
            agent_id=agent_id,
            correlation_id=correlation_id
        )
        
        # Determine which provider to use
        provider_name = provider or self._select_provider(model)
        if provider_name not in self.providers:
            raise LLMError(f"Provider '{provider_name}' not available")
        
        provider_instance = self.providers[provider_name]
        
        # Attempt generation with retries
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                start_time = time.time()
                response = await provider_instance.generate(request)
                
                # Log the interaction
                interaction = LLMInteraction(
                    agent_id=agent_id,
                    prompt=prompt,
                    response=response.content,
                    context=context or {},
                    model=model,
                    temperature=temperature,
                    tokens_used=response.tokens_used,
                    response_time=response.response_time,
                    correlation_id=correlation_id,
                    success=True
                )
                self.interactions.append(interaction)
                
                return response.content
                
            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    # Log failed interaction
                    interaction = LLMInteraction(
                        agent_id=agent_id,
                        prompt=prompt,
                        response="",
                        context=context or {},
                        model=model,
                        temperature=temperature,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        correlation_id=correlation_id,
                        success=False,
                        error=str(e)
                    )
                    self.interactions.append(interaction)
                    raise LLMError(f"Failed to generate response after {retry_count} retries: {last_error}")
    
    def _select_provider(self, model: str) -> str:
        """Select appropriate provider based on model name."""
        model_lower = model.lower()
        
        # Anthropic models
        if any(name in model_lower for name in ['claude', 'anthropic']):
            if 'anthropic' in self.providers:
                return 'anthropic'
        
        # Fallback to default provider
        return self.default_provider or 'anthropic'
    
    def get_interactions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LLMInteraction]:
        """Get recent LLM interactions, optionally filtered by agent."""
        interactions = self.interactions
        
        if agent_id:
            interactions = [i for i in interactions if i.agent_id == agent_id]
        
        return interactions[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        total_interactions = len(self.interactions)
        successful_interactions = sum(1 for i in self.interactions if i.success)
        total_tokens = sum(i.tokens_used for i in self.interactions)
        
        return {
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "failed_interactions": total_interactions - successful_interactions,
            "success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0,
            "total_tokens_used": total_tokens,
            "average_response_time": sum(i.response_time for i in self.interactions) / total_interactions if total_interactions > 0 else 0,
            "providers_available": list(self.providers.keys()),
            "default_provider": self.default_provider
        }
