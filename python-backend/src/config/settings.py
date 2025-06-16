"""
Configuration management for FlutterSwarm.
"""
import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration from YAML file
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "settings.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

_config = load_config()


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    anthropic_api_key: Optional[str] = None
    default_model: str = _config.get("llm", {}).get("default_model", "claude-sonnet-4-20250514")
    max_tokens: int = _config.get("llm", {}).get("max_tokens", 4000)
    temperature: float = _config.get("llm", {}).get("temperature", 0.7)
    timeout: int = _config.get("llm", {}).get("timeout", 30)
    max_retries: int = _config.get("llm", {}).get("max_retries", 3)
    rate_limit: int = _config.get("llm", {}).get("rate_limit", 60)
    
    def __post_init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = _config.get("database", {}).get("url", "postgresql://user:pass@localhost/flutterswarm")
    pool_size: int = _config.get("database", {}).get("pool_size", 10)
    max_overflow: int = _config.get("database", {}).get("max_overflow", 20)
    pool_timeout: int = _config.get("database", {}).get("pool_timeout", 30)
    
    def __post_init__(self):
        self.url = os.getenv("DATABASE_URL", self.url)


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = _config.get("redis", {}).get("url", "redis://localhost:6379")
    max_connections: int = _config.get("redis", {}).get("max_connections", 20)
    retry_on_timeout: bool = _config.get("redis", {}).get("retry_on_timeout", True)
    ttl_seconds: int = _config.get("redis", {}).get("ttl_seconds", 604800)
    max_versions: int = _config.get("redis", {}).get("max_versions", 10)
    
    def __post_init__(self):
        self.url = os.getenv("REDIS_URL", self.url)


@dataclass
class AgentConfig:
    """Agent system configuration."""
    memory_ttl: int = _config.get("agent", {}).get("memory_ttl", 3600)  # seconds
    max_memory_entries: int = _config.get("agent", {}).get("max_memory_entries", 1000)
    context_window_size: int = _config.get("agent", {}).get("context_window_size", 8000)
    max_concurrent_tasks: int = _config.get("agent", {}).get("max_concurrent_tasks", 5)
    task_timeout: int = _config.get("agent", {}).get("task_timeout", 300)  # seconds
    memory_retention_hours: int = _config.get("agent", {}).get("memory_retention_hours", 24)
    importance_threshold: float = _config.get("agent", {}).get("importance_threshold", 0.5)
    max_load_capacity: int = _config.get("agent", {}).get("max_load_capacity", 5)
    
    def __post_init__(self):
        self.memory_ttl = int(os.getenv("AGENT_MEMORY_TTL", self.memory_ttl))


@dataclass
class EventBusConfig:
    """Event bus configuration."""
    buffer_size: int = _config.get("event_bus", {}).get("buffer_size", 1000)
    max_subscribers: int = _config.get("event_bus", {}).get("max_subscribers", 100)
    message_ttl: int = _config.get("event_bus", {}).get("message_ttl", 3600)  # seconds
    max_history: int = _config.get("event_bus", {}).get("max_history", 1000)
    max_retry_attempts: int = _config.get("event_bus", {}).get("max_retry_attempts", 3)
    
    def __post_init__(self):
        self.buffer_size = int(os.getenv("EVENT_BUS_BUFFER_SIZE", self.buffer_size))


@dataclass
class FlutterConfig:
    """Flutter project configuration."""
    output_directory: str = _config.get("flutter", {}).get("output_directory", "flutter_projects")
    default_flutter_version: str = _config.get("flutter", {}).get("default_flutter_version", "3.24.0")
    default_dart_version: str = _config.get("flutter", {}).get("default_dart_version", "3.5.0")
    
    def __post_init__(self):
        self.output_directory = os.getenv("FLUTTER_OUTPUT_DIR", self.output_directory)
        self.default_flutter_version = os.getenv("FLUTTER_VERSION", self.default_flutter_version)
        self.default_dart_version = os.getenv("DART_VERSION", self.default_dart_version)


@dataclass
class Settings:
    """Main application settings."""
    log_level: str = _config.get("application", {}).get("log_level", "INFO")
    debug: bool = _config.get("application", {}).get("debug", False)
    environment: str = _config.get("application", {}).get("environment", "development")
    
    # Component configurations
    llm: LLMConfig = None
    database: DatabaseConfig = None
    redis: RedisConfig = None
    agent: AgentConfig = None
    event_bus: EventBusConfig = None
    flutter: FlutterConfig = None
    
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
        self.flutter = FlutterConfig()
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.llm.anthropic_api_key:
            raise ValueError("Anthropic API key must be provided")
        
        if not self.database.url:
            raise ValueError("Database URL is required")
        
        if not self.redis.url:
            raise ValueError("Redis URL is required")


# Global settings instance
settings = Settings()
