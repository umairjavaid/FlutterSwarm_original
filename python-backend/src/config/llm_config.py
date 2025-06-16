import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    """Configuration for LLM providers and models."""
    
    # Available models by provider
    ANTHROPIC_MODELS = [
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-sonnet-4-20250514"
    ]
    
    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
    def get_available_providers(self) -> List[str]:
        """Get list of available providers based on API keys."""
        providers = []
        if self.anthropic_key:
            providers.append("anthropic")
        return providers
    
    def get_default_model(self) -> str:
        """Get the default model based on available API keys."""
        if self.anthropic_key:
            return "claude-sonnet-4-20250514"
        else:
            raise ValueError("No Anthropic API key configured")
    
    def get_provider_for_model(self, model: str) -> str:
        """Get the provider for a specific model."""
        if model in self.ANTHROPIC_MODELS:
            return "anthropic"
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def validate_model_availability(self, model: str) -> bool:
        """Check if a model is available based on API keys."""
        try:
            provider = self.get_provider_for_model(model)
            available_providers = self.get_available_providers()
            return provider in available_providers
        except ValueError:
            return False
    
    def get_model_config(self, model: Optional[str] = None) -> Dict[str, str]:
        """Get model configuration including provider and API key."""
        if model is None:
            model = self.get_default_model()
        
        provider = self.get_provider_for_model(model)
        
        if not self.validate_model_availability(model):
            raise ValueError(f"Model {model} is not available. Missing Anthropic API key")
        
        config = {
            "model": model,
            "provider": provider,
            "api_key": self.anthropic_key
        }
        
        return config

# Global instance
llm_config = LLMConfig()
