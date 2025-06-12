"""
Logging Configuration for FlutterSwarm Multi-Agent System.

This module provides flexible logging configuration for different environments
and use cases.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.enhanced_logger import (
    EnhancedLogger, ColoredFormatter, StructuredFormatter,
    get_enhanced_logger
)


class LoggingConfig:
    """Centralized logging configuration."""
    
    # Default log levels per agent type
    AGENT_LOG_LEVELS = {
        "orchestrator": logging.INFO,
        "architecture": logging.INFO,
        "implementation": logging.INFO,
        "testing": logging.INFO,
        "security": logging.WARNING,  # More verbose for security
        "devops": logging.INFO,
        "documentation": logging.INFO,
        "performance": logging.DEBUG,  # More verbose for performance
        "supervisor": logging.INFO
    }
    
    # Default log levels per module
    MODULE_LOG_LEVELS = {
        "event_bus": logging.INFO,
        "memory_manager": logging.INFO,
        "llm_client": logging.INFO,
        "api": logging.INFO,
        "core": logging.INFO,
        "system": logging.INFO
    }
    
    # Environment-specific configurations
    ENVIRONMENT_CONFIGS = {
        "development": {
            "console_level": logging.DEBUG,
            "file_level": logging.DEBUG,
            "structured_output": False,
            "color_output": True,
            "log_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 3
        },
        "testing": {
            "console_level": logging.WARNING,
            "file_level": logging.DEBUG,
            "structured_output": True,
            "color_output": False,
            "log_file_size": 5 * 1024 * 1024,  # 5MB
            "backup_count": 2
        },
        "production": {
            "console_level": logging.INFO,
            "file_level": logging.INFO,
            "structured_output": True,
            "color_output": False,
            "log_file_size": 50 * 1024 * 1024,  # 50MB
            "backup_count": 10
        }
    }
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config = self.ENVIRONMENT_CONFIGS.get(environment, self.ENVIRONMENT_CONFIGS["development"])
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Privacy filters for sensitive data
        self.privacy_filters = [
            r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_-]+)',
            r'password["\s]*[:=]["\s]*([a-zA-Z0-9_@!#$%^&*]+)',
            r'secret["\s]*[:=]["\s]*([a-zA-Z0-9_-]+)',
            r'token["\s]*[:=]["\s]*([a-zA-Z0-9_.-]+)'
        ]
    
    def setup_logging(self):
        """Set up logging configuration for the entire system."""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set up console handler
        self._setup_console_handler(root_logger)
        
        # Set up file handlers
        self._setup_file_handlers(root_logger)
        
        # Set specific log levels
        self._configure_logger_levels()
        
        # Set up privacy filters
        self._setup_privacy_filters()
    
    def _setup_console_handler(self, root_logger):
        """Set up console handler with appropriate formatter."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config["color_output"]:
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.config["console_level"])
        root_logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, root_logger):
        """Set up file handlers for different log types."""
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "flutterswarm.log",
            maxBytes=self.config["log_file_size"],
            backupCount=self.config["backup_count"]
        )
        
        if self.config["structured_output"]:
            main_handler.setFormatter(StructuredFormatter())
        else:
            main_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        main_handler.setLevel(self.config["file_level"])
        root_logger.addHandler(main_handler)
        
        # Error-only log
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=self.config["log_file_size"],
            backupCount=self.config["backup_count"]
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # Performance metrics log
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=self.config["log_file_size"],
            backupCount=self.config["backup_count"]
        )
        performance_handler.setFormatter(StructuredFormatter())
        performance_handler.addFilter(PerformanceLogFilter())
        root_logger.addHandler(performance_handler)
    
    def _configure_logger_levels(self):
        """Configure log levels for specific loggers."""
        # Agent loggers
        for agent_type, level in self.AGENT_LOG_LEVELS.items():
            logger = logging.getLogger(f"flutterswarm.agents.{agent_type}")
            logger.setLevel(level)
        
        # Module loggers
        for module, level in self.MODULE_LOG_LEVELS.items():
            logger = logging.getLogger(f"flutterswarm.{module}")
            logger.setLevel(level)
        
        # Third-party loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    def _setup_privacy_filters(self):
        """Set up privacy filters to redact sensitive data."""
        root_logger = logging.getLogger()
        privacy_filter = PrivacyFilter(self.privacy_filters)
        
        for handler in root_logger.handlers:
            handler.addFilter(privacy_filter)
    
    def get_agent_logger(self, agent_type: str, agent_id: Optional[str] = None) -> EnhancedLogger:
        """Get configured logger for specific agent."""
        logger_name = f"flutterswarm.agents.{agent_type}"
        return get_enhanced_logger(logger_name, agent_type, agent_id)
    
    def get_module_logger(self, module_name: str) -> EnhancedLogger:
        """Get configured logger for specific module."""
        logger_name = f"flutterswarm.{module_name}"
        return get_enhanced_logger(logger_name)


class PrivacyFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""
    
    def __init__(self, patterns):
        super().__init__()
        import re
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def filter(self, record):
        """Filter sensitive information from log record."""
        if hasattr(record, 'msg'):
            message = str(record.msg)
            for pattern in self.patterns:
                message = pattern.sub(r'\1[REDACTED]', message)
            record.msg = message
        
        return True


class PerformanceLogFilter(logging.Filter):
    """Filter to only include performance-related logs."""
    
    def filter(self, record):
        """Only allow performance-related log records."""
        if hasattr(record, 'context'):
            context = record.context
            return context.operation in ['performance', 'resource_usage', 'llm_call']
        
        return 'performance' in record.getMessage().lower()


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(environment: str = None) -> LoggingConfig:
    """Set up global logging configuration."""
    global _logging_config
    
    if environment is None:
        environment = os.getenv("FLUTTERSWARM_ENV", "development")
    
    _logging_config = LoggingConfig(environment)
    _logging_config.setup_logging()
    
    return _logging_config


def get_logging_config() -> LoggingConfig:
    """Get current logging configuration."""
    global _logging_config
    
    if _logging_config is None:
        setup_logging()
    
    return _logging_config


def get_logger(name: str, agent_type: Optional[str] = None, 
               agent_id: Optional[str] = None) -> EnhancedLogger:
    """Get enhanced logger with current configuration."""
    config = get_logging_config()
    
    if agent_type:
        return config.get_agent_logger(agent_type, agent_id)
    else:
        return config.get_module_logger(name)
