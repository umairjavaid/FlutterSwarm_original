import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    """Configuration for LLM providers and models."""
    
    # Available models by provider
    OPENAI_MODELS = [
        "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"
    ]
    
    ANTHROPIC_MODELS = [
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620"  # Corrected model name
    ]
    
    GROQ_MODELS = [
        "mixtral-8x7b-32768", "llama2-70b-4096"
    ]
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
    def get_available_providers(self) -> List[str]:
        """Get list of available providers based on API keys."""
        providers = []
        if self.openai_key:
            providers.append("openai")
        if self.anthropic_key:
            providers.append("anthropic")
        if self.groq_key:
            providers.append("groq")
        return providers
    
    def get_default_model(self) -> str:
        """Get the default model based on available API keys."""
        if self.anthropic_key:
            return "claude-3-5-sonnet-20240620"
        elif self.openai_key:
            return "gpt-4"
        elif self.groq_key:
            return "mixtral-8x7b-32768"
        else:
            raise ValueError("No API keys configured")
    
    def get_provider_for_model(self, model: str) -> str:
        """Get the provider for a specific model."""
        if model in self.OPENAI_MODELS:
            return "openai"
        elif model in self.ANTHROPIC_MODELS:
            return "anthropic"
        elif model in self.GROQ_MODELS:
            return "groq"
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def validate_model_availability(self, model: str) -> bool:
        """Check if a model is available based on API keys."""
        try:
            provider = self.get_provider_for_model(model)
            available_providers = self.get_available_providers()
            return provider in available_providers
        except ValueError:
            # Handle unknown models gracefully
            return False
    
    def get_model_config(self, model: Optional[str] = None) -> Dict[str, str]:
        """Get model configuration including provider and API key."""
        if model is None:
            model = self.get_default_model()
        
        provider = self.get_provider_for_model(model)
        
        if not self.validate_model_availability(model):
            raise ValueError(f"Model {model} is not available. Missing API key for {provider}")
        
        config = {
            "model": model,
            "provider": provider
        }
        
        if provider == "openai":
            config["api_key"] = self.openai_key
        elif provider == "anthropic":
            config["api_key"] = self.anthropic_key
        elif provider == "groq":
            config["api_key"] = self.groq_key
        
        return config

# Global instance
llm_config = LLMConfig()
