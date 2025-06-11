"""
LangGraph Agent Configuration Management System.

This module provides flexible configuration for LangGraph agents in the FlutterSwarm system.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import os

from .settings import settings


@dataclass
class LangGraphAgentModelConfig:
    """Configuration for LangGraph agent LLM model settings."""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 300
    max_retries: int = 3


@dataclass
class LangGraphAgentConfig:
    """Configuration for LangGraph agents."""
    agent_type: str
    enabled: bool = True
    model_config: LangGraphAgentModelConfig = field(default_factory=LangGraphAgentModelConfig)
    system_prompt_file: Optional[str] = None
    system_prompt_template: Optional[str] = None
    custom_instructions: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class LangGraphAgentConfigManager:
    """Manages configuration for LangGraph agents."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.path.join(os.path.dirname(__file__), "langgraph_agents"))
        self.config_dir.mkdir(exist_ok=True)
        
        self._agent_configs: Dict[str, LangGraphAgentConfig] = {}
        self._prompt_cache: Dict[str, str] = {}
        
        # Load configurations
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default configurations for LangGraph agents."""
        default_configs = {
            "supervisor": LangGraphAgentConfig(
                agent_type="supervisor",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.3,
                    max_tokens=6000,
                    timeout=600
                ),
                system_prompt_file="supervisor_prompt.txt",
                capabilities=["workflow_orchestration", "task_decomposition", "agent_coordination"],
                specializations=["flutter_development", "multi_agent_coordination"]
            ),
            "architecture": LangGraphAgentConfig(
                agent_type="architecture",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.3,
                    max_tokens=6000,
                    timeout=600
                ),
                system_prompt_file="architecture_prompt.txt",
                capabilities=["architecture_analysis", "system_design", "pattern_selection"],
                specializations=["flutter_architecture", "clean_architecture", "state_management"]
            ),
            "implementation": LangGraphAgentConfig(
                agent_type="implementation",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.2,
                    max_tokens=8000,
                    timeout=900
                ),
                system_prompt_file="implementation_prompt.txt",
                capabilities=["code_generation", "feature_implementation", "ui_development"],
                specializations=["flutter_widgets", "dart_programming", "state_management"]
            ),
            "testing": LangGraphAgentConfig(
                agent_type="testing",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.3,
                    max_tokens=6000,
                    timeout=600
                ),
                system_prompt_file="testing_prompt.txt",
                capabilities=["test_generation", "quality_assurance", "test_automation"],
                specializations=["flutter_testing", "unit_testing", "widget_testing"]
            ),
            "security": LangGraphAgentConfig(
                agent_type="security",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.2,
                    max_tokens=7000,
                    timeout=800
                ),
                system_prompt_file="security_prompt.txt",
                capabilities=["security_analysis", "vulnerability_scanning", "compliance_check"],
                specializations=["mobile_security", "flutter_security", "privacy_compliance"]
            ),
            "devops": LangGraphAgentConfig(
                agent_type="devops",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.3,
                    max_tokens=7000,
                    timeout=900
                ),
                system_prompt_file="devops_prompt.txt",
                capabilities=["deployment", "ci_cd_setup", "infrastructure_management"],
                specializations=["flutter_deployment", "mobile_deployment", "cloud_deployment"]
            ),
            "documentation": LangGraphAgentConfig(
                agent_type="documentation",
                model_config=LangGraphAgentModelConfig(
                    provider="openai",
                    model="gpt-4",
                    temperature=0.4,
                    max_tokens=8000,
                    timeout=600
                ),
                system_prompt_file="documentation_prompt.txt",
                capabilities=["documentation_generation", "api_docs", "user_guides"],
                specializations=["flutter_documentation", "dart_docs", "technical_writing"]
            )
        }
        
        self._agent_configs.update(default_configs)
    
    def get_agent_config(self, agent_type: str) -> Optional[LangGraphAgentConfig]:
        """Get configuration for a specific agent type."""
        return self._agent_configs.get(agent_type)
    
    def get_system_prompt(self, agent_type: str) -> str:
        """Get system prompt for an agent type."""
        if agent_type in self._prompt_cache:
            return self._prompt_cache[agent_type]
        
        config = self.get_agent_config(agent_type)
        if not config:
            return ""
        
        prompt = ""
        
        # Load from file if specified
        if config.system_prompt_file:
            prompt_file = self.config_dir / config.system_prompt_file
            if prompt_file.exists():
                prompt = prompt_file.read_text()
        
        # Use template if file not found
        if not prompt and config.system_prompt_template:
            prompt = config.system_prompt_template
        
        # Add custom instructions
        if config.custom_instructions:
            prompt += "\n\nADDITIONAL INSTRUCTIONS:\n"
            prompt += "\n".join(f"- {instruction}" for instruction in config.custom_instructions)
        
        # Cache the prompt
        self._prompt_cache[agent_type] = prompt
        
        return prompt
    
    def get_all_agent_types(self) -> List[str]:
        """Get list of all configured agent types."""
        return list(self._agent_configs.keys())
    
    def is_agent_enabled(self, agent_type: str) -> bool:
        """Check if an agent type is enabled."""
        config = self.get_agent_config(agent_type)
        return config.enabled if config else False


# Global configuration manager instance
langgraph_agent_config_manager = LangGraphAgentConfigManager()
