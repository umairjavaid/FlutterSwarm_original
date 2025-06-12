"""
FlutterSwarm exceptions module.
"""


class FlutterSwarmError(Exception):
    """Base exception for FlutterSwarm errors."""
    pass


class LLMError(FlutterSwarmError):
    """Raised when there's an error with LLM operations."""
    pass


class AgentError(FlutterSwarmError):
    """Raised when there's an error with agent operations."""
    pass


class ConfigurationError(FlutterSwarmError):
    """Raised when there's a configuration error."""
    pass


class TaskError(FlutterSwarmError):
    """Raised when there's a task execution error."""
    pass


class ValidationError(FlutterSwarmError):
    """Raised when validation fails."""
    pass
