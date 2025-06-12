"""
LangGraph Agent Configuration Management System.

This module provides flexible configuration for LangGraph agents in the FlutterSwarm system.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from .settings import settings


@dataclass
class LangGraphAgentModelConfig:
    """Configuration for LangGraph agent LLM model settings."""
    provider: str = "anthropic"  # anthropic, local
    model: str = None  # Will be set from config
    temperature: float = None  # Will be set from config
    max_tokens: int = None  # Will be set from config
    timeout: int = None  # Will be set from config
    max_retries: int = None  # Will be set from config
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    
    def __post_init__(self):
        """Set defaults from global config if not provided."""
        from .settings import settings
        if self.model is None:
            self.model = settings.llm.default_model
        if self.temperature is None:
            self.temperature = settings.llm.temperature
        if self.max_tokens is None:
            self.max_tokens = settings.llm.max_tokens
        if self.timeout is None:
            self.timeout = settings.llm.timeout
        if self.max_retries is None:
            self.max_retries = settings.llm.max_retries


@dataclass
class LangGraphAgentPromptConfig:
    """Configuration for agent prompts and instructions."""
    system_prompt_file: Optional[str] = None
    system_prompt_template: Optional[str] = None
    custom_instructions: List[str] = field(default_factory=list)
    constraint_instructions: List[str] = field(default_factory=list)
    response_format_instructions: Optional[str] = None


@dataclass
class LangGraphAgentBehaviorConfig:
    """Configuration for agent behavior and operational parameters."""
    max_concurrent_tasks: int = None
    task_timeout: int = None
    auto_retry_failed_tasks: bool = True
    enable_collaboration: bool = True
    enable_learning: bool = True
    memory_retention_hours: int = None
    importance_threshold: float = None
    
    def __post_init__(self):
        """Set defaults from global config if not provided."""
        from .settings import settings
        if self.max_concurrent_tasks is None:
            self.max_concurrent_tasks = settings.agent.max_concurrent_tasks
        if self.task_timeout is None:
            self.task_timeout = settings.agent.task_timeout
        if self.memory_retention_hours is None:
            self.memory_retention_hours = settings.agent.memory_retention_hours
        if self.importance_threshold is None:
            self.importance_threshold = settings.agent.importance_threshold


@dataclass
class LangGraphAgentConfig:
    """Complete configuration for LangGraph agents."""
    agent_type: str
    enabled: bool = True
    model_config: LangGraphAgentModelConfig = field(default_factory=LangGraphAgentModelConfig)
    prompt_config: LangGraphAgentPromptConfig = field(default_factory=LangGraphAgentPromptConfig)
    behavior_config: LangGraphAgentBehaviorConfig = field(default_factory=LangGraphAgentBehaviorConfig)
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class LangGraphAgentConfigManager:
    """Manages configuration for LangGraph agents."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.path.join(os.path.dirname(__file__), "agents"))
        self.config_dir.mkdir(exist_ok=True)
        
        # Prompts directory
        self.prompts_dir = self.config_dir / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        
        self._agent_configs: Dict[str, LangGraphAgentConfig] = {}
        self._prompt_cache: Dict[str, str] = {}
        
        # Load configurations
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default configurations for LangGraph agents."""
        # Load configurations from YAML files
        config_files = list(self.config_dir.glob("*_sample.yaml"))
        
        for config_file in config_files:
            try:
                agent_type = config_file.stem.replace("_sample", "")
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                config = self._create_config_from_dict(agent_type, config_data)
                self._agent_configs[agent_type] = config
                
            except Exception as e:
                print(f"Warning: Failed to load config from {config_file}: {e}")
        
        # Create default configs for standard agent types if not found
        default_agent_types = [
            "orchestrator", "architecture", "implementation", "testing", 
            "security", "devops", "documentation", "performance"
        ]
        
        for agent_type in default_agent_types:
            if agent_type not in self._agent_configs:
                self._agent_configs[agent_type] = self._create_default_config(agent_type)
    
    def _create_config_from_dict(self, agent_type: str, config_data: Dict[str, Any]) -> LangGraphAgentConfig:
        """Create agent config from dictionary data."""
        model_config_data = config_data.get("model_config", {})
        model_config = LangGraphAgentModelConfig(
            provider=model_config_data.get("provider", "anthropic"),
            model=model_config_data.get("model"),  # Will use config default if None
            temperature=model_config_data.get("temperature"),  # Will use config default if None
            max_tokens=model_config_data.get("max_tokens"),  # Will use config default if None
            timeout=model_config_data.get("timeout"),  # Will use config default if None
            max_retries=model_config_data.get("max_retries"),  # Will use config default if None
            fallback_provider=model_config_data.get("fallback_provider"),
            fallback_model=model_config_data.get("fallback_model")
        )
        
        prompt_config_data = config_data.get("prompt_config", {})
        prompt_config = LangGraphAgentPromptConfig(
            system_prompt_file=prompt_config_data.get("system_prompt_file"),
            system_prompt_template=prompt_config_data.get("system_prompt_template"),
            custom_instructions=prompt_config_data.get("custom_instructions", []),
            constraint_instructions=prompt_config_data.get("constraint_instructions", []),
            response_format_instructions=prompt_config_data.get("response_format_instructions")
        )
        
        behavior_config_data = config_data.get("behavior_config", {})
        behavior_config = LangGraphAgentBehaviorConfig(
            max_concurrent_tasks=behavior_config_data.get("max_concurrent_tasks"),  # Will use config default if None
            task_timeout=behavior_config_data.get("task_timeout"),  # Will use config default if None
            auto_retry_failed_tasks=behavior_config_data.get("auto_retry_failed_tasks", True),
            enable_collaboration=behavior_config_data.get("enable_collaboration", True),
            enable_learning=behavior_config_data.get("enable_learning", True),
            memory_retention_hours=behavior_config_data.get("memory_retention_hours"),  # Will use config default if None
            importance_threshold=behavior_config_data.get("importance_threshold")  # Will use config default if None
        )
        
        return LangGraphAgentConfig(
            agent_type=agent_type,
            enabled=config_data.get("enabled", True),
            model_config=model_config,
            prompt_config=prompt_config,
            behavior_config=behavior_config,
            capabilities=config_data.get("capabilities", []),
            specializations=config_data.get("specializations", []),
            custom_parameters=config_data.get("custom_parameters", {})
        )
    
    def _create_default_config(self, agent_type: str) -> LangGraphAgentConfig:
        """Create default configuration for an agent type."""
        from .settings import settings
        
        # Agent-specific defaults that can override global defaults
        agent_defaults = {
            "orchestrator": {
                "temperature": 0.3,
                "max_tokens": settings.llm.max_tokens * 0.75,  # 75% of default
                "capabilities": ["task_decomposition", "workflow_management", "agent_coordination"]
            },
            "architecture": {
                "temperature": 0.3,
                "max_tokens": settings.llm.max_tokens * 1.5,  # 150% of default
                "capabilities": ["architecture_analysis", "design_patterns", "project_structure"]
            },
            "implementation": {
                "temperature": 0.2,
                "max_tokens": settings.llm.max_tokens * 2,  # 200% of default
                "capabilities": ["code_generation", "feature_development", "ui_implementation"]
            },
            "testing": {
                "temperature": 0.3,
                "max_tokens": settings.llm.max_tokens * 1.5,  # 150% of default
                "capabilities": ["test_generation", "quality_assurance", "test_automation"]
            },
            "security": {
                "temperature": 0.2,
                "max_tokens": settings.llm.max_tokens * 1.75,  # 175% of default
                "capabilities": ["security_analysis", "vulnerability_assessment", "compliance"]
            },
            "devops": {
                "temperature": 0.3,
                "max_tokens": settings.llm.max_tokens * 1.75,  # 175% of default
                "capabilities": ["deployment_automation", "ci_cd_setup", "infrastructure"]
            },
            "documentation": {
                "temperature": 0.4,
                "max_tokens": settings.llm.max_tokens * 2,  # 200% of default
                "capabilities": ["documentation_generation", "api_docs", "user_guides"]
            },
            "performance": {
                "temperature": 0.3,
                "max_tokens": settings.llm.max_tokens * 1.75,  # 175% of default
                "capabilities": ["performance_analysis", "optimization", "monitoring"]
            }
        }
        
        defaults = agent_defaults.get(agent_type, {})
        
        model_config = LangGraphAgentModelConfig(
            temperature=defaults.get("temperature"),  # Will use global default if None
            max_tokens=int(defaults.get("max_tokens", settings.llm.max_tokens))
        )
        
        prompt_config = LangGraphAgentPromptConfig(
            system_prompt_file=f"{agent_type}_prompt.txt"
        )
        
        return LangGraphAgentConfig(
            agent_type=agent_type,
            model_config=model_config,
            prompt_config=prompt_config,
            capabilities=defaults.get("capabilities", [])
        )
    
    def get_agent_config(self, agent_type: str) -> Optional[LangGraphAgentConfig]:
        """Get configuration for a specific agent type."""
        return self._agent_configs.get(agent_type)
    
    def get_system_prompt(self, agent_type: str) -> str:
        """Get system prompt for an agent type."""
        config = self.get_agent_config(agent_type)
        if not config or not config.prompt_config.system_prompt_file:
            return ""
        
        prompt_file = config.prompt_config.system_prompt_file
        
        # Check cache first
        if prompt_file in self._prompt_cache:
            return self._prompt_cache[prompt_file]
        
        # Load from file
        prompt_path = self.prompts_dir / prompt_file
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
            
            # Cache the prompt
            self._prompt_cache[prompt_file] = prompt_content
            return prompt_content
            
        except FileNotFoundError:
            print(f"Warning: Prompt file not found: {prompt_path}")
            return ""
        except Exception as e:
            print(f"Error loading prompt file {prompt_path}: {e}")
            return ""
    
    def reload_prompt(self, agent_type: str) -> None:
        """Reload prompt from file, clearing cache."""
        config = self.get_agent_config(agent_type)
        if config and config.prompt_config.system_prompt_file:
            # Clear cache entry
            if config.prompt_config.system_prompt_file in self._prompt_cache:
                del self._prompt_cache[config.prompt_config.system_prompt_file]
    
    def get_all_agent_types(self) -> List[str]:
        """Get list of all configured agent types."""
        return list(self._agent_configs.keys())
    
    def is_agent_enabled(self, agent_type: str) -> bool:
        """Check if an agent type is enabled."""
        config = self.get_agent_config(agent_type)
        return config.enabled if config else False
    
    def create_sample_configs(self) -> None:
        """Create sample configuration files for all agent types."""
        sample_configs = {
            "orchestrator_sample.yaml": self._get_orchestrator_sample_config(),
            "architecture_sample.yaml": self._get_architecture_sample_config(),
            "implementation_sample.yaml": self._get_implementation_sample_config(),
            "testing_sample.yaml": self._get_testing_sample_config(),
            "security_sample.yaml": self._get_security_sample_config(),
            "devops_sample.yaml": self._get_devops_sample_config(),
            "documentation_sample.yaml": self._get_documentation_sample_config(),
            "performance_sample.yaml": self._get_performance_sample_config()
        }
        
        for filename, config_content in sample_configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    yaml.dump(config_content, f, default_flow_style=False, indent=2)
                print(f"Created sample config: {config_path}")
    
    def _get_orchestrator_sample_config(self) -> Dict[str, Any]:
        """Get orchestrator sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.3,
                "max_tokens": 3000,
                "timeout": 300,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "orchestrator_prompt.txt",
                "custom_instructions": [
                    "Always provide clear task decomposition",
                    "Consider agent capabilities when assigning tasks",
                    "Monitor workflow progress actively"
                ],
                "constraint_instructions": [
                    "Follow Flutter best practices",
                    "Ensure quality at each workflow stage",
                    "Maintain clear communication between agents"
                ]
            },
            "behavior_config": {
                "max_concurrent_tasks": 10,
                "task_timeout": 1800,
                "auto_retry_failed_tasks": True,
                "enable_collaboration": True,
                "enable_learning": True,
                "memory_retention_hours": 72,
                "importance_threshold": 0.7
            },
            "capabilities": [
                "task_decomposition",
                "workflow_management", 
                "agent_coordination",
                "progress_monitoring"
            ],
            "custom_parameters": {
                "max_workflow_depth": 5,
                "parallel_execution_limit": 3
            }
        }
    
    def _get_architecture_sample_config(self) -> Dict[str, Any]:
        """Get architecture agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.3,
                "max_tokens": 6000,
                "timeout": 600,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "architecture_prompt.txt",
                "custom_instructions": [
                    "Always provide detailed rationale for architectural decisions",
                    "Include specific Flutter/Dart implementation details",
                    "Consider scalability and maintainability"
                ],
                "constraint_instructions": [
                    "Follow SOLID principles",
                    "Ensure clean architecture patterns",
                    "Consider platform-specific requirements"
                ]
            },
            "behavior_config": {
                "max_concurrent_tasks": 3,
                "task_timeout": 900,
                "auto_retry_failed_tasks": True,
                "enable_collaboration": True,
                "enable_learning": True,
                "memory_retention_hours": 48,
                "importance_threshold": 0.8
            },
            "capabilities": [
                "architecture_analysis",
                "design_patterns",
                "project_structure",
                "technical_debt_assessment"
            ],
            "specializations": [
                "clean_architecture",
                "state_management_patterns",
                "dependency_injection"
            ]
        }
    
    def _get_implementation_sample_config(self) -> Dict[str, Any]:
        """Get implementation agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.2,
                "max_tokens": 8000,
                "timeout": 900,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "implementation_prompt.txt",
                "custom_instructions": [
                    "Generate complete, working code solutions",
                    "Include comprehensive error handling",
                    "Add meaningful comments and documentation"
                ]
            },
            "behavior_config": {
                "max_concurrent_tasks": 5,
                "task_timeout": 1200,
                "enable_collaboration": True
            },
            "capabilities": [
                "code_generation",
                "feature_development",
                "ui_implementation",
                "api_integration"
            ]
        }
    
    def _get_testing_sample_config(self) -> Dict[str, Any]:
        """Get testing agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.3,
                "max_tokens": 6000,
                "timeout": 600,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "testing_prompt.txt"
            },
            "behavior_config": {
                "max_concurrent_tasks": 4
            },
            "capabilities": [
                "test_generation",
                "quality_assurance",
                "test_automation"
            ]
        }
    
    def _get_security_sample_config(self) -> Dict[str, Any]:
        """Get security agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.2,
                "max_tokens": 7000,
                "timeout": 800,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "security_prompt.txt"
            },
            "behavior_config": {
                "max_concurrent_tasks": 3
            },
            "capabilities": [
                "security_analysis",
                "vulnerability_assessment",
                "compliance_checking"
            ]
        }
    
    def _get_devops_sample_config(self) -> Dict[str, Any]:
        """Get devops agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.3,
                "max_tokens": 7000,
                "timeout": 900,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "devops_prompt.txt"
            },
            "behavior_config": {
                "max_concurrent_tasks": 3
            },
            "capabilities": [
                "deployment_automation",
                "ci_cd_setup",
                "infrastructure_management"
            ]
        }
    
    def _get_documentation_sample_config(self) -> Dict[str, Any]:
        """Get documentation agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.4,
                "max_tokens": 8000,
                "timeout": 600,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "documentation_prompt.txt"
            },
            "behavior_config": {
                "max_concurrent_tasks": 4
            },
            "capabilities": [
                "documentation_generation",
                "api_documentation",
                "user_guides"
            ]
        }
    
    def _get_performance_sample_config(self) -> Dict[str, Any]:
        """Get performance agent sample configuration."""
        return {
            "enabled": True,
            "model_config": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20240620",
                "temperature": 0.3,
                "max_tokens": 7000,
                "timeout": 700,
                "max_retries": 3
            },
            "prompt_config": {
                "system_prompt_file": "performance_prompt.txt"
            },
            "behavior_config": {
                "max_concurrent_tasks": 3
            },
            "capabilities": [
                "performance_analysis",
                "optimization",
                "monitoring_setup"
            ]
        }


# Global configuration manager instance
langgraph_agent_config_manager = LangGraphAgentConfigManager()

# Backward compatibility alias
agent_config_manager = langgraph_agent_config_manager
