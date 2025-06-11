"""
LLM Client for FlutterSwarm Multi-Agent System.

This module provides a unified interface for interacting with various LLM providers
including OpenAI, Anthropic, and others. It handles authentication, rate limiting,
retries, and logging for all LLM interactions.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp
import openai

from ..config import settings
from ..models.agent_models import LLMInteraction, LLMError

logger = logging.getLogger(__name__)

# Optional import for Anthropic
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
    temperature: float = 0.7
    max_tokens: int = 4000
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = None
    agent_id: str = ""
    correlation_id: str = ""


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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str, rate_limit: int = 60):
        super().__init__(api_key, rate_limit)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI's API."""
        await self._check_rate_limit()
        
        start_time = time.time()
        
        try:
            messages = []
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            # Add user prompt
            messages.append({"role": "user", "content": request.prompt})
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")
    

class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) LLM provider implementation."""
    
    def __init__(self, api_key: str, rate_limit: int = 60):
        super().__init__(api_key, rate_limit)
        if not ANTHROPIC_AVAILABLE:
            raise LLMError("Anthropic library not installed. Install with: pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic's API."""
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
            
            # Use the correct Anthropic API endpoint
            response = await self.client.messages.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
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
            
        except Exception as e:
            raise LLMError(f"Anthropic API error: {str(e)}")


class LLMClient:
    """
    Unified LLM client that manages multiple providers and handles
    routing, retries, fallbacks, and logging.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = None
        self.interactions: List[LLMInteraction] = []
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize LLM providers based on configuration."""
        try:
            # OpenAI provider
            if hasattr(settings.llm, 'openai_api_key') and settings.llm.openai_api_key:
                self.providers['openai'] = OpenAIProvider(
                    api_key=settings.llm.openai_api_key,
                    rate_limit=getattr(settings.llm, 'openai_rate_limit', 60)
                )
                if not self.default_provider:
                    self.default_provider = 'openai'
            
            # Anthropic provider
            if hasattr(settings.llm, 'anthropic_api_key') and settings.llm.anthropic_api_key:
                if ANTHROPIC_AVAILABLE:
                    self.providers['anthropic'] = AnthropicProvider(
                        api_key=settings.llm.anthropic_api_key,
                        rate_limit=getattr(settings.llm, 'anthropic_rate_limit', 60)
                    )
                    if not self.default_provider:
                        self.default_provider = 'anthropic'
                else:
                    logger.warning("Anthropic API key provided but anthropic library not installed")
                    
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM providers: {e}")
        
        if not self.providers:
            raise LLMError("No LLM providers configured")
    
    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: str = "",
        correlation_id: str = "",
        provider: Optional[str] = None,
        retry_count: int = 3
    ) -> str:
        """
        Generate text using the specified or default LLM provider.
        
        Args:
            prompt: Input prompt for the LLM
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            context: Additional context information
            agent_id: ID of the requesting agent
            correlation_id: Correlation ID for tracking
            provider: Specific provider to use (optional)
            retry_count: Number of retries on failure
            
        Returns:
            Generated text response
        """
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
        
        # OpenAI models
        if any(name in model_lower for name in ['gpt', 'davinci', 'curie', 'babbage', 'ada']):
            if 'openai' in self.providers:
                return 'openai'
        
        # Anthropic models
        if any(name in model_lower for name in ['claude', 'anthropic']):
            if 'anthropic' in self.providers:
                return 'anthropic'
        
        # Fallback to default provider
        return self.default_provider
    
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
