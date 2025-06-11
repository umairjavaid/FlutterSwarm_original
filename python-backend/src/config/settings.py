"""
Configuration management for FlutterSwarm.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "postgresql://user:pass@localhost/flutterswarm"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    def __post_init__(self):
        self.url = os.getenv("DATABASE_URL", self.url)


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = "redis://localhost:6379"
    max_connections: int = 20
    retry_on_timeout: bool = True
    
    def __post_init__(self):
        self.url = os.getenv("REDIS_URL", self.url)


@dataclass
class AgentConfig:
    """Agent system configuration."""
    memory_ttl: int = 3600  # seconds
    max_memory_entries: int = 1000
    context_window_size: int = 8000
    max_concurrent_tasks: int = 5
    task_timeout: int = 300  # seconds
    
    def __post_init__(self):
        self.memory_ttl = int(os.getenv("AGENT_MEMORY_TTL", self.memory_ttl))


@dataclass
class EventBusConfig:
    """Event bus configuration."""
    buffer_size: int = 1000
    max_subscribers: int = 100
    message_ttl: int = 3600  # seconds
    
    def __post_init__(self):
        self.buffer_size = int(os.getenv("EVENT_BUS_BUFFER_SIZE", self.buffer_size))


@dataclass
class Settings:
    """Main application settings."""
    log_level: str = "INFO"
    debug: bool = False
    environment: str = "development"
    
    # Component configurations
    llm: LLMConfig = None
    database: DatabaseConfig = None
    redis: RedisConfig = None
    agent: AgentConfig = None
    event_bus: EventBusConfig = None
    
    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        
        # Initialize component configs
        self.llm = LLMConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.agent = AgentConfig()
        self.event_bus = EventBusConfig()
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.llm.openai_api_key and not self.llm.anthropic_api_key:
            raise ValueError("At least one LLM API key must be provided")
        
        if not self.database.url:
            raise ValueError("Database URL is required")
        
        if not self.redis.url:
            raise ValueError("Redis URL is required")


# Global settings instance
settings = Settings()
