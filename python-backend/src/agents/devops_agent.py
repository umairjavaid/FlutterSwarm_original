"""
DevOps Agent for FlutterSwarm Multi-Agent System.

This agent specializes in deployment, CI/CD pipeline management,
and infrastructure automation for Flutter applications.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult
from ..models.task_models import TaskContext
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..config import get_logger

logger = get_logger("devops_agent")


class DevOpsAgent(BaseAgent):
    """
    Specialized agent for Flutter application DevOps and deployment management.
    
    This agent handles:
    - CI/CD pipeline creation and management
    - Deployment automation and orchestration
    - Infrastructure as Code (IaC) setup
    - Environment management and configuration
    - Release management and versioning
    - Monitoring and alerting setup
    - Container orchestration and scaling
    - Security and compliance automation
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: 'MemoryManager',
        event_bus: EventBus
    ):
        # Override config for DevOps-specific settings
        devops_config = AgentConfig(
            agent_id=config.agent_id or f"devops_agent_{str(uuid.uuid4())[:8]}",
            agent_type="devops",
            capabilities=[
                AgentCapability.DEPLOYMENT,
                AgentCapability.CODE_GENERATION,
                AgentCapability.INFRASTRUCTURE
            ],
            max_concurrent_tasks=3,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.3,  # Lower temperature for consistent infrastructure code
            max_tokens=7000,
            timeout=900,  # Longer timeout for complex deployment setups
            metadata=config.metadata
        )
        
        super().__init__(devops_config, llm_client, memory_manager, event_bus)
        
        # DevOps-specific state
        self.deployment_platforms = [
            "firebase_hosting", "app_store", "play_store", "web_hosting",
            "aws", "gcp", "azure", "netlify", "vercel", "github_pages"
        ]
        
        self.ci_cd_providers = {
            "github_actions": "GitHub Actions workflows",
            "gitlab_ci": "GitLab CI/CD pipelines",
            "azure_devops": "Azure DevOps pipelines",
            "bitbucket": "Bitbucket pipelines",
            "jenkins": "Jenkins pipeline scripts",
            "circleci": "CircleCI configuration"
        }
        
        self.infrastructure_tools = {
            "containerization": ["docker", "podman"],
            "orchestration": ["kubernetes", "docker-compose"],
            "iac": ["terraform", "pulumi", "cloudformation"],
            "monitoring": ["prometheus", "grafana", "datadog", "new_relic"],
            "logging": ["elk_stack", "splunk", "cloudwatch"],
            "security": ["vault", "sealed_secrets", "cert_manager"]
        }
        
        logger.info(f"DevOps Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the DevOps agent."""
        return await self._get_default_system_prompt()

    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the DevOps agent."""
        return """
You are the DevOps Agent in the FlutterSwarm multi-agent system, specializing in deployment, CI/CD, and infrastructure automation for Flutter applications.

CORE EXPERTISE:
- Flutter application deployment across multiple platforms (iOS, Android, Web, Desktop)
- CI/CD pipeline design and implementation using various providers
- Infrastructure as Code (IaC) with Terraform, Pulumi, and cloud-native tools
- Container orchestration with Docker and Kubernetes

Always provide production-ready, secure, and scalable solutions with comprehensive documentation.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of DevOps-specific capabilities."""
        return [
            "ci_cd_pipeline_creation",
            "deployment_automation",
            "infrastructure_as_code",
            "environment_management",
            "release_management",
            "monitoring_setup",
            "security_automation",
            "container_orchestration",
            "cloud_platform_management",
            "compliance_automation",
            "disaster_recovery_planning"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute DevOps-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "ci_cd_setup":
                return await self._setup_ci_cd_pipeline(task_context, llm_analysis)
            elif task_type == "deployment":
                return await self._create_deployment_config(task_context, llm_analysis)
            elif task_type == "infrastructure":
                return await self._design_infrastructure(task_context, llm_analysis)
            elif task_type == "monitoring":
                return await self._setup_monitoring(task_context, llm_analysis)
            elif task_type == "security":
                return await self._implement_security(task_context, llm_analysis)
            elif task_type == "release_management":
                return await self._setup_release_management(task_context, llm_analysis)
            else:
                # Generic DevOps processing
                return await self._process_devops_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"DevOps processing failed: {e}")
            return {
                "error": str(e),
                "configuration_files": {},
                "setup_instructions": []
            }

    async def _setup_ci_cd_pipeline(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up CI/CD pipeline for Flutter application."""
        logger.info(f"Setting up CI/CD pipeline for task: {task_context.task_id}")
        
        pipeline_prompt = self._create_ci_cd_prompt(task_context, llm_analysis)
        
        pipeline_config = await self.execute_llm_task(
            user_prompt=pipeline_prompt,
            context={
                "task": task_context.to_dict(),
                "ci_cd_providers": self.ci_cd_providers,
                "best_practices": self._get_ci_cd_best_practices()
            },
            structured_output=True
        )
        
        # Store pipeline configuration
        await self.memory_manager.store_memory(
            content=f"CI/CD pipeline setup: {json.dumps(pipeline_config)}",
            metadata={
                "type": "ci_cd_setup",
                "provider": pipeline_config.get('provider'),
                "platforms": pipeline_config.get('target_platforms', [])
            },
            correlation_id=task_context.correlation_id,
            importance=0.9,
            long_term=True
        )
        
        return {
            "pipeline_config": pipeline_config,
            "workflow_files": pipeline_config.get("workflow_files", {}),
            "scripts": pipeline_config.get("scripts", {}),
            "secrets_config": pipeline_config.get("secrets", {}),
            "environment_config": pipeline_config.get("environments", {}),
            "setup_instructions": pipeline_config.get("setup_instructions", [])
        }

    async def _create_deployment_config(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create deployment configuration for multiple platforms."""
        logger.info(f"Creating deployment config for task: {task_context.task_id}")
        
        deployment_prompt = self._create_deployment_prompt(task_context, llm_analysis)
        
        deployment_config = await self.execute_llm_task(
            user_prompt=deployment_prompt,
            context={
                "task": task_context.to_dict(),
                "platforms": self.deployment_platforms,
                "deployment_strategies": self._get_deployment_strategies()
            },
            structured_output=True
        )
        
        return {
            "deployment_config": deployment_config,
            "platform_configs": deployment_config.get("platform_configs", {}),
            "deployment_scripts": deployment_config.get("scripts", {}),
            "environment_variables": deployment_config.get("env_vars", {}),
            "certificates": deployment_config.get("certificates", {}),
            "monitoring_config": deployment_config.get("monitoring", {})
        }

    async def _design_infrastructure(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design infrastructure using IaC principles."""
        logger.info(f"Designing infrastructure for task: {task_context.task_id}")
        
        infrastructure_prompt = self._create_infrastructure_prompt(task_context, llm_analysis)
        
        infrastructure_design = await self.execute_llm_task(
            user_prompt=infrastructure_prompt,
            context={
                "task": task_context.to_dict(),
                "tools": self.infrastructure_tools,
                "cloud_patterns": self._get_cloud_patterns()
            },
            structured_output=True
        )
        
        return {
            "infrastructure_design": infrastructure_design,
            "iac_files": infrastructure_design.get("iac_files", {}),
            "container_configs": infrastructure_design.get("containers", {}),
            "networking_config": infrastructure_design.get("networking", {}),
            "security_config": infrastructure_design.get("security", {}),
            "scaling_config": infrastructure_design.get("scaling", {})
        }

    async def _setup_monitoring(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up comprehensive monitoring and alerting."""
        logger.info(f"Setting up monitoring for task: {task_context.task_id}")
        
        monitoring_prompt = self._create_monitoring_prompt(task_context, llm_analysis)
        
        monitoring_setup = await self.execute_llm_task(
            user_prompt=monitoring_prompt,
            context={
                "task": task_context.to_dict(),
                "monitoring_tools": self.infrastructure_tools["monitoring"],
                "observability_patterns": self._get_observability_patterns()
            },
            structured_output=True
        )
        
        return {
            "monitoring_setup": monitoring_setup,
            "dashboards": monitoring_setup.get("dashboards", {}),
            "alerts": monitoring_setup.get("alerts", {}),
            "metrics": monitoring_setup.get("metrics", []),
            "logging_config": monitoring_setup.get("logging", {}),
            "tracing_config": monitoring_setup.get("tracing", {})
        }

    # Prompt creation methods
    def _create_ci_cd_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for CI/CD pipeline setup."""
        return f"""
Create a comprehensive CI/CD pipeline for the following Flutter application:

PROJECT ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

REQUIREMENTS:
{task_context.description}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please design a complete CI/CD pipeline including:

1. PIPELINE ARCHITECTURE:
   - Multi-stage pipeline design (build, test, deploy)
   - Branch-based workflow strategies
   - Environment progression (dev → staging → production)
   - Parallel execution and dependency management

2. BUILD AUTOMATION:
   - Flutter build configuration for all target platforms
   - Dependency caching and optimization
   - Artifact management and versioning
   - Build matrix for multiple platforms and configurations

3. TESTING INTEGRATION:
   - Automated test execution (unit, widget, integration)
   - Code coverage reporting and thresholds
   - Quality gates and build failure conditions
   - Test result aggregation and reporting

4. SECURITY SCANNING:
   - Dependency vulnerability scanning
   - Static code analysis integration
   - Secret detection and management
   - License compliance checking

5. DEPLOYMENT STRATEGIES:
   - Platform-specific deployment processes
   - Blue-green or canary deployment patterns
   - Rollback mechanisms and strategies
   - Release approval workflows

6. MONITORING AND NOTIFICATIONS:
   - Build status notifications
   - Deployment health monitoring
   - Performance regression detection
   - Alert configuration and escalation

Provide complete pipeline configuration files and setup instructions.
"""

    def _create_deployment_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for deployment configuration."""
        return f"""
Create deployment configuration for the following Flutter application:

DEPLOYMENT ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

DEPLOYMENT REQUIREMENTS:
{task_context.description}

Please provide comprehensive deployment configuration including:

1. PLATFORM-SPECIFIC DEPLOYMENT:
   - iOS App Store deployment configuration
   - Google Play Store deployment setup
   - Web hosting deployment (Firebase, Netlify, etc.)
   - Desktop application distribution

2. ENVIRONMENT MANAGEMENT:
   - Development, staging, and production environments
   - Environment-specific configuration management
   - Secret and credential management
   - Feature flag and configuration systems

3. DEPLOYMENT AUTOMATION:
   - Automated deployment scripts and workflows
   - Database migration and schema updates
   - Asset optimization and CDN configuration
   - SSL certificate management and renewal

4. MONITORING AND HEALTH CHECKS:
   - Application health monitoring
   - Performance monitoring and alerting
   - Error tracking and crash reporting
   - User analytics and usage tracking

5. SECURITY CONFIGURATION:
   - Security headers and policies
   - Authentication and authorization setup
   - API security and rate limiting
   - Data encryption and privacy compliance

6. SCALING AND PERFORMANCE:
   - Auto-scaling configuration
   - Load balancing and traffic distribution
   - Content delivery network setup
   - Performance optimization strategies

Provide complete deployment configurations and step-by-step setup guides.
"""

    def _create_infrastructure_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for infrastructure design."""
        return f"""
Design infrastructure for the following Flutter application using Infrastructure as Code:

INFRASTRUCTURE REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

APPLICATION CONTEXT:
{task_context.description}

Please design comprehensive infrastructure including:

1. INFRASTRUCTURE ARCHITECTURE:
   - Multi-tier architecture design
   - Network topology and security groups
   - Load balancing and traffic routing
   - High availability and disaster recovery

2. CONTAINERIZATION:
   - Docker container configuration
   - Multi-stage build optimization
   - Container orchestration with Kubernetes
   - Service mesh implementation

3. CLOUD SERVICES INTEGRATION:
   - Database services and configuration
   - Storage services and CDN setup
   - Authentication and authorization services
   - API gateway and microservices architecture

4. SECURITY IMPLEMENTATION:
   - Network security and firewall rules
   - Identity and access management
   - Secret management and encryption
   - Compliance and audit logging

5. MONITORING AND OBSERVABILITY:
   - Infrastructure monitoring setup
   - Application performance monitoring
   - Log aggregation and analysis
   - Distributed tracing implementation

6. COST OPTIMIZATION:
   - Resource sizing and optimization
   - Auto-scaling policies
   - Reserved instance strategies
   - Cost monitoring and alerting

Provide complete IaC templates and deployment documentation.
"""

    def _create_monitoring_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        # Ensure llm_analysis is a valid JSON string or dict for embedding in the prompt
        try:
            if isinstance(llm_analysis, dict):
                llm_analysis_str = json.dumps(llm_analysis, indent=2)
            elif isinstance(llm_analysis, str):
                # Attempt to parse to validate, then re-serialize to ensure format
                json.loads(llm_analysis) # Validate
                llm_analysis_str = llm_analysis # Assume it's already well-formed if string
            else:
                llm_analysis_str = json.dumps({"error": "Invalid llm_analysis format provided"}, indent=2)
                logger.warning(f"Invalid llm_analysis type: {type(llm_analysis)}. Using error string.")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error processing llm_analysis for monitoring prompt: {e}")
            llm_analysis_str = json.dumps({"error": f"Failed to process llm_analysis: {str(e)}"}, indent=2)

        return f"""
Set up comprehensive monitoring and observability for the following Flutter application:

MONITORING REQUIREMENTS:
{llm_analysis_str}

APPLICATION CONTEXT:
{task_context.description}

Please design complete monitoring solution including:

1. APPLICATION MONITORING:
   - Performance metrics and KPIs
   - Error tracking and crash reporting
   - User experience monitoring
   - Business metrics and analytics

2. INFRASTRUCTURE MONITORING:
   - Server and container monitoring
   - Network and storage monitoring
   - Database performance monitoring
   - Security and compliance monitoring

3. ALERTING AND NOTIFICATIONS:
   - Alert rule configuration
   - Escalation policies and procedures
   - Multi-channel notification setup
   - Alert correlation and suppression

4. DASHBOARDS AND VISUALIZATION:
   - Executive and operational dashboards
   - Real-time monitoring displays
   - Historical trend analysis
   - Custom metric visualization

5. LOGGING AND TRACING:
   - Centralized log management
   - Distributed tracing setup
   - Error analysis and debugging
   - Audit trail and compliance logging

6. AUTOMATION AND REMEDIATION:
   - Automated incident response
   - Self-healing mechanisms
   - Capacity management automation
   - Performance optimization triggers

Provide complete monitoring configuration and setup procedures.
"""

    # Helper methods for best practices and patterns
    def _get_ci_cd_best_practices(self) -> Dict[str, Any]:
        """Get CI/CD best practices and guidelines."""
        return {
            "pipeline_design": [
                "fail_fast_principle",
                "parallel_execution",
                "artifact_caching",
                "environment_parity"
            ],
            "security": [
                "secret_management",
                "least_privilege_access",
                "vulnerability_scanning",
                "compliance_checking"
            ],
            "testing": [
                "test_pyramid",
                "quality_gates",
                "coverage_thresholds",
                "performance_testing"
            ],
            "deployment": [
                "blue_green_deployment",
                "canary_releases",
                "rollback_strategies",
                "health_checks"
            ]
        }

    def _get_deployment_strategies(self) -> Dict[str, Any]:
        """Get deployment strategies and patterns."""
        return {
            "rolling_deployment": "Gradual replacement of instances",
            "blue_green": "Switch between two identical environments",
            "canary": "Gradual traffic shifting to new version",
            "feature_flags": "Runtime feature toggling",
            "immutable_deployment": "Complete infrastructure replacement"
        }

    def _get_cloud_patterns(self) -> Dict[str, Any]:
        """Get cloud architecture patterns."""
        return {
            "microservices": "Service-oriented architecture",
            "serverless": "Function-as-a-Service patterns",
            "event_driven": "Event-based communication",
            "cqrs": "Command Query Responsibility Segregation",
            "circuit_breaker": "Fault tolerance patterns"
        }

    def _get_observability_patterns(self) -> Dict[str, Any]:
        """Get observability patterns and practices."""
        return {
            "three_pillars": ["metrics", "logs", "traces"],
            "sli_slo": "Service Level Indicators and Objectives",
            "error_budgets": "Reliability engineering practices",
            "chaos_engineering": "Resilience testing practices"
        }
