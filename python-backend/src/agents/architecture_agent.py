"""
Architecture Agent for FlutterSwarm Multi-Agent System.

This agent specializes in system design, project structure analysis,
and architectural decision-making for Flutter applications.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..core.event_bus import EventBus
from ..core.memory_manager import MemoryManager
from ..models.agent_models import AgentMessage, TaskResult, MessageType, TaskStatus # Added MessageType and TaskStatus
from ..models.task_models import TaskContext, TaskType
from ..utils.json_utils import extract_json_from_llm_response
from ..models.project_models import (
    ProjectContext, ArchitecturePattern, PlatformTarget, 
    ProjectType, CodeMetrics
)
from ..config import get_logger
from ..utils.json_utils import extract_json_from_llm_response

logger = get_logger("architecture_agent")


class ArchitectureAgent(BaseAgent):
    """
    Specialized agent for Flutter application architecture analysis and design.
    
    This agent handles:
    - Project structure analysis and recommendations
    - Architectural pattern selection and implementation
    - Dependency management and module organization
    - Design principle enforcement and guidance
    - Technical debt assessment and resolution planning
    - Scalability and maintainability planning
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: 'MemoryManager',
        event_bus: EventBus
    ):
        # Override config for architecture-specific settings
        arch_config = AgentConfig(
            agent_id=config.agent_id or f"architecture_agent_{str(uuid.uuid4())[:8]}",
            agent_type="architecture",
            capabilities=[
                AgentCapability.ARCHITECTURE_ANALYSIS,
                AgentCapability.CODE_GENERATION
            ],
            max_concurrent_tasks=3,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.3,  # Lower temperature for consistent architectural decisions
            max_tokens=6000,
            timeout=900,  # Increased from 600 to 900 seconds (15 minutes)
            metadata=config.metadata
        )
        
        super().__init__(arch_config, llm_client, memory_manager, event_bus)
        
        # Architecture-specific state
        self.supported_patterns = [
            ArchitecturePattern.CLEAN_ARCHITECTURE,
            ArchitecturePattern.BLOC,
            ArchitecturePattern.PROVIDER,
            ArchitecturePattern.RIVERPOD,
            ArchitecturePattern.GETX,
            ArchitecturePattern.MVC,
            ArchitecturePattern.MVVM
        ]
        
        self.flutter_best_practices = {
            "state_management": ["bloc", "provider", "riverpod", "getx"],
            "navigation": ["go_router", "auto_route", "flutter_navigation"],
            "dependency_injection": ["get_it", "injectable", "provider"],
            "testing": ["flutter_test", "mockito", "bloc_test"],
            "networking": ["dio", "http", "chopper"],
            "local_storage": ["hive", "sqflite", "shared_preferences"]
        }
        
        logger.info(f"Architecture Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the architecture agent."""
        try:
            from ..config.agent_configs import agent_config_manager
            config_prompt = agent_config_manager.get_system_prompt(self.agent_type)
            return config_prompt if config_prompt else await self._get_default_system_prompt()
        except ImportError as e:
            logger.warning(f"Could not load agent configuration: {e}")
            return await self._get_default_system_prompt()

    async def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the architecture agent."""
        return """
You are the Architecture Agent in the FlutterSwarm multi-agent system, specializing in Flutter application architecture design and analysis.

CORE EXPERTISE:
- Flutter/Dart ecosystem architecture patterns
- Clean Architecture, SOLID principles, and design patterns
- State management solutions (Bloc, Provider, Riverpod, GetX)
- Project structure organization and module design

Always provide detailed rationale for architectural decisions and include specific Flutter/Dart implementation details.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of architecture-specific capabilities."""
        return [
            "project_structure_analysis",
            "architectural_pattern_selection",
            "clean_architecture_design",
            "state_management_design",
            "dependency_injection_setup",
            "module_organization",
            "scalability_planning",
            "testing_architecture",
            "technical_debt_assessment",
            "migration_planning",
            "flutter_best_practices"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute architecture-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "architecture_design":
                return await self._design_architecture(task_context, llm_analysis)
            elif task_type == "analysis":
                return await self._analyze_architecture(task_context, llm_analysis)
            elif task_type == "refactoring":
                return await self._plan_refactoring(task_context, llm_analysis)
            else:
                # Generic architectural processing
                return await self._process_architectural_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"Architecture processing failed: {e}")
            return {
                "error": str(e),
                "deliverables": {},
                "recommendations": []
            }

    async def _design_architecture(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design complete architecture for a new Flutter project."""
        logger.info(f"Designing architecture for task: {task_context.task_id}")
        
        # Analyze project requirements
        project_context = task_context.metadata.get("project_context", {})
        
        # Use LLM to design architecture based on requirements
        design_prompt = self._create_architecture_design_prompt(task_context, project_context)
        
        architecture_design = await self.execute_llm_task(
            user_prompt=design_prompt,
            context={
                "task": task_context.to_dict(),
                "supported_patterns": [p.value for p in self.supported_patterns],
                "best_practices": self.flutter_best_practices
            },
            structured_output=True
        )
        
        # Store architecture decisions in memory
        await self.memory_manager.store_memory(
            content=f"Architecture design for {project_context.get('project_name', 'project')}: {json.dumps(architecture_design)}",
            metadata={
                "type": "architecture_design",
                "project": project_context.get('project_name'),
                "patterns": architecture_design.get('patterns', [])
            },
            correlation_id=task_context.metadata.get("correlation_id", task_context.task_id),
            importance=0.9,
            long_term=True
        )
        
        return {
            "architecture_design": architecture_design,
            "deliverables": {
                "project_structure": architecture_design.get("project_structure"),
                "dependency_graph": architecture_design.get("dependency_graph"),
                "implementation_plan": architecture_design.get("implementation_plan"),
                "setup_instructions": architecture_design.get("setup_instructions")
            },
            "recommendations": architecture_design.get("recommendations", []),
            "next_steps": architecture_design.get("next_steps", [])
        }

    async def _analyze_architecture(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze existing architecture and provide recommendations."""
        logger.info(f"Analyzing architecture for task: {task_context.task_id}")
        
        project_context = task_context.metadata.get("project_context", {})
        
        # Create comprehensive analysis prompt
        analysis_prompt = self._create_architecture_analysis_prompt(task_context, project_context)
        
        analysis_result = await self.execute_llm_task(
            user_prompt=analysis_prompt,
            context={
                "task": task_context.to_dict(),
                "project": project_context,
                "best_practices": self.flutter_best_practices
            },
            structured_output=True
        )
        
        # Store analysis results
        await self.memory_manager.store_memory(
            content=f"Architecture analysis: {json.dumps(analysis_result)}",
            metadata={
                "type": "architecture_analysis",
                "project": project_context.get('project_name'),
                "issues_found": len(analysis_result.get('issues', []))
            },
            correlation_id=task_context.metadata.get("correlation_id", task_context.task_id),
            importance=0.8,
            long_term=True
        )
        
        return {
            "analysis_result": analysis_result,
            "deliverables": {
                "architecture_assessment": analysis_result.get("assessment"),
                "issues_report": analysis_result.get("issues"),
                "improvement_plan": analysis_result.get("improvement_plan"),
                "compliance_report": analysis_result.get("compliance")
            },
            "recommendations": analysis_result.get("recommendations", []),
            "priority_actions": analysis_result.get("priority_actions", [])
        }

    async def _plan_refactoring(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a detailed refactoring plan for architectural improvements."""
        logger.info(f"Planning refactoring for task: {task_context.task_id}")
        
        refactoring_prompt = self._create_refactoring_prompt(task_context)
        
        refactoring_plan = await self.execute_llm_task(
            user_prompt=refactoring_prompt,
            context={
                "task": task_context.to_dict(),
                "current_analysis": llm_analysis
            },
            structured_output=True
        )
        
        # Store refactoring plan
        await self.memory_manager.store_memory(
            content=f"Refactoring plan: {json.dumps(refactoring_plan)}",
            metadata={
                "type": "refactoring_plan",
                "scope": refactoring_plan.get('scope'),
                "estimated_effort": refactoring_plan.get('estimated_effort')
            },
            correlation_id=task_context.metadata.get("correlation_id", task_context.task_id),
            importance=0.8,
            long_term=True
        )
        
        return {
            "refactoring_plan": refactoring_plan,
            "deliverables": {
                "migration_strategy": refactoring_plan.get("migration_strategy"),
                "implementation_phases": refactoring_plan.get("phases"),
                "risk_assessment": refactoring_plan.get("risks"),
                "rollback_plan": refactoring_plan.get("rollback_plan")
            },
            "timeline": refactoring_plan.get("timeline"),
            "risks": refactoring_plan.get("risks", [])
        }

    async def _process_architectural_request(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process general architectural requests."""
        logger.info(f"Processing architectural request: {task_context.task_id}")
        
        # Use LLM analysis to determine specific architectural guidance
        guidance_prompt = f"""
        Based on the following architectural request, provide detailed guidance:
        
        Request: {task_context.description}
        
        Analysis: {json.dumps(llm_analysis, indent=2)}
        
        Provide specific Flutter/Dart architectural guidance including:
        1. Recommended approach and patterns
        2. Implementation details with code examples
        3. Best practices and considerations
        4. Potential pitfalls and how to avoid them
        5. Testing strategy for the architectural solution
        """
        
        guidance = await self.execute_llm_task(
            user_prompt=guidance_prompt,
            context={"task": task_context.to_dict()},
            structured_output=True
        )
        
        return {
            "guidance": guidance,
            "deliverables": {
                "implementation_guide": guidance.get("implementation"),
                "code_examples": guidance.get("examples"),
                "best_practices": guidance.get("best_practices"),
                "testing_strategy": guidance.get("testing")
            },
            "recommendations": guidance.get("recommendations", [])
        }

    def _create_architecture_design_prompt(
        self,
        task_context: TaskContext,
        project_context: Dict[str, Any]
    ) -> str:
        """Create prompt for architecture design."""
        return f"""
Design a comprehensive Flutter application architecture based on the following requirements:

PROJECT DETAILS:
- Name: {project_context.get('project_name', 'Flutter App')}
- Type: {project_context.get('project_type', 'app')}
- Target Platforms: {project_context.get('target_platforms', ['Android', 'iOS'])}
- Expected Scale: {project_context.get('expected_scale', 'medium')}
- Team Size: {project_context.get('team_size', 'small')}

REQUIREMENTS:
{task_context.description}

SPECIFIC REQUIREMENTS:
{json.dumps([req.__dict__ if hasattr(req, '__dict__') else req for req in task_context.requirements], indent=2)}

CONSTRAINTS:
- Flutter Version: {project_context.get('flutter_version', 'latest')}
- Existing Dependencies: {project_context.get('dependencies', [])}
- Performance Requirements: {project_context.get('performance_requirements', 'standard')}
- Maintenance Requirements: {project_context.get('maintenance_requirements', 'standard')}

Please provide a comprehensive architecture design including:

1. ARCHITECTURAL PATTERN SELECTION:
   - Recommended primary pattern (Clean Architecture, BLoC, Provider, etc.)
   - Justification for the choice
   - Integration with Flutter ecosystem

2. PROJECT STRUCTURE:
   - Detailed folder and file organization
   - Module boundaries and responsibilities
   - Layer separation and dependencies

3. STATE MANAGEMENT:
   - Chosen state management solution
   - State flow and data flow patterns
   - Global vs local state strategies

4. DEPENDENCY INJECTION:
   - DI container setup and configuration
   - Service registration and lifecycle
   - Testing considerations

5. NAVIGATION ARCHITECTURE:
   - Navigation strategy and routing
   - Deep linking and URL handling
   - Platform-specific navigation patterns

6. DATA LAYER DESIGN:
   - Repository patterns and data sources
   - Caching strategies
   - Offline-first considerations

7. TESTING ARCHITECTURE:
   - Unit testing structure
   - Integration testing approach
   - Widget testing strategies

8. IMPLEMENTATION PLAN:
   - Setup steps and initial configuration
   - Development phases and milestones
   - Integration points and dependencies

Respond with a detailed JSON structure containing all architectural decisions and implementation guidance.
"""

    def _create_architecture_analysis_prompt(
        self,
        task_context: TaskContext,
        project_context: Dict[str, Any]
    ) -> str:
        """Create prompt for architecture analysis."""
        return f"""
Analyze the current Flutter application architecture and provide comprehensive assessment:

PROJECT INFORMATION:
- Name: {project_context.get('project_name', 'Flutter App')}
- Current Structure: {project_context.get('file_structure', 'Not provided')}
- Dependencies: {project_context.get('dependencies', [])}
- Platforms: {project_context.get('target_platforms', [])}

ANALYSIS REQUEST:
{task_context.description}

CURRENT CODE METRICS (if available):
{json.dumps(project_context.get('code_metrics', {}), indent=2)}

Please provide a thorough architectural analysis including:

1. CURRENT ARCHITECTURE ASSESSMENT:
   - Identified architectural patterns in use
   - Adherence to Flutter best practices
   - Code organization and structure quality
   - State management evaluation

2. TECHNICAL DEBT ANALYSIS:
   - Code smells and architectural issues
   - Maintenance and scalability concerns
   - Performance bottlenecks
   - Testing gaps

3. COMPLIANCE EVALUATION:
   - Flutter/Dart best practices compliance
   - SOLID principles adherence
   - Clean architecture principles
   - Platform-specific guidelines

4. IDENTIFIED ISSUES:
   - Critical architectural problems
   - Performance and scalability issues
   - Maintainability concerns
   - Security vulnerabilities

5. IMPROVEMENT RECOMMENDATIONS:
   - Prioritized list of improvements
   - Refactoring opportunities
   - Pattern migrations
   - Dependency updates

6. IMPLEMENTATION ROADMAP:
   - Phased improvement plan
   - Risk assessment for changes
   - Effort estimation
   - Dependencies and prerequisites

Respond with detailed JSON structure containing analysis results and actionable recommendations.
"""

    def _create_refactoring_prompt(self, task_context: TaskContext) -> str:
        """Create prompt for refactoring planning."""
        return f"""
Create a detailed refactoring plan for the following architectural improvements:

REFACTORING REQUEST:
{task_context.description}

CURRENT PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please provide a comprehensive refactoring plan including:

1. REFACTORING SCOPE:
   - Areas of code to be modified
   - Components to be restructured
   - Dependencies to be updated

2. MIGRATION STRATEGY:
   - Step-by-step migration approach
   - Backward compatibility considerations
   - Data migration requirements

3. IMPLEMENTATION PHASES:
   - Phase breakdown with clear milestones
   - Dependencies between phases
   - Parallel workstreams

4. RISK ASSESSMENT:
   - Potential risks and mitigation strategies
   - Impact on existing functionality
   - Testing requirements

5. ROLLBACK PLAN:
   - Rollback procedures for each phase
   - Backup and restoration strategies
   - Monitoring and validation

6. TIMELINE AND EFFORT:
   - Estimated effort for each phase
   - Resource requirements
   - Critical path analysis

7. TESTING STRATEGY:
   - Testing approach for refactored code
   - Regression testing requirements
   - Performance validation

Respond with detailed JSON structure containing the complete refactoring plan.
"""

    async def analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze a Flutter project's structure and provide recommendations."""
        # This would typically scan the actual project files
        # For now, return a structured analysis based on LLM reasoning
        
        analysis_prompt = f"""
        Analyze the Flutter project structure at path: {project_path}
        
        Provide analysis covering:
        1. Folder organization quality
        2. Separation of concerns
        3. Adherence to Flutter conventions
        4. Scalability considerations
        5. Specific recommendations for improvement
        
        Return a structured analysis with scores and actionable recommendations.
        """
        
        try:
            analysis = await self.execute_llm_task(
                user_prompt=analysis_prompt,
                context={"project_path": project_path},
                structured_output=True
            )
            
            return {
                "structure_quality": analysis.get("quality_score", "good"),
                "recommendations": analysis.get("recommendations", []),
                "issues": analysis.get("issues", []),
                "compliance_score": analysis.get("compliance_score", 0.8)
            }
        except Exception as e:
            logger.error(f"Project structure analysis failed: {e}")
            return {
                "structure_quality": "unknown",
                "recommendations": ["Unable to analyze - please check project structure"],
                "issues": [f"Analysis error: {str(e)}"],
                "compliance_score": 0.0
            }

    async def recommend_architecture_pattern(
        self,
        requirements: Dict[str, Any]
    ) -> ArchitecturePattern:
        """Recommend the best architecture pattern for given requirements."""
        complexity = requirements.get('complexity', 'medium')
        team_size = requirements.get('team_size', 'small')
        
        # Use LLM to make intelligent recommendation
        recommendation_prompt = f"""
        Recommend the best Flutter architecture pattern for a project with:
        - Complexity: {complexity}
        - Team Size: {team_size}
        - Requirements: {json.dumps(requirements)}
        
        Choose from: {[p.value for p in self.supported_patterns]}
        
        Consider:
        1. Development team expertise and size
        2. Project complexity and scope
        3. Maintenance requirements
        4. Scalability needs
        5. Testing requirements
        
        Provide the recommended pattern name and detailed reasoning.
        """
        
        try:
            result = await self.execute_llm_task(
                user_prompt=recommendation_prompt,
                context={"requirements": requirements},
                structured_output=True
            )
            
            # Extract recommended pattern
            recommended = result.get('recommended_pattern', 'clean_architecture')
            
            # Store recommendation in memory
            await self.memory_manager.store_memory(
                content=f"Architecture pattern recommendation: {recommended} for {json.dumps(requirements)}",
                metadata={
                    "type": "pattern_recommendation",
                    "pattern": recommended,
                    "complexity": complexity
                },
                importance=0.7
            )
            
            return ArchitecturePattern(recommended)
            
        except ValueError:
            logger.warning(f"Invalid pattern recommendation: {recommended}, using default")
            return ArchitecturePattern.CLEAN_ARCHITECTURE  # Default fallback
        except Exception as e:
            logger.error(f"Pattern recommendation failed: {e}")
            return ArchitecturePattern.CLEAN_ARCHITECTURE

    async def validate_architecture_compliance(
        self,
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate project compliance with architectural standards."""
        
        validation_prompt = f"""
        Validate the Flutter project architecture compliance:
        
        Project Context: {json.dumps(project_context, indent=2)}
        
        Check compliance against:
        1. Flutter best practices
        2. SOLID principles
        3. Clean architecture principles
        4. State management best practices
        5. Testing standards
        6. Performance guidelines
        7. Security considerations
        
        Provide detailed compliance report with:
        - Compliance score (0-100)
        - Specific violations and recommendations
        - Priority levels for fixes
        - Implementation guidance
        """
        
        try:
            compliance = await self.execute_llm_task(
                user_prompt=validation_prompt,
                context={"project_context": project_context},
                structured_output=True
            )
            
            # Store compliance results
            await self.memory_manager.store_memory(
                content=f"Architecture compliance validation: {json.dumps(compliance)}",
                metadata={
                    "type": "compliance_validation",
                    "score": compliance.get("compliance_score", 0),
                    "violations": len(compliance.get("violations", []))
                },
                importance=0.7
            )
            
            return compliance
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {
                "compliance_score": 0,
                "violations": [f"Validation error: {str(e)}"],
                "recommendations": ["Please check project structure and try again"],
                "error": str(e)
            }

    async def create_migration_plan(
        self,
        current_architecture: str,
        target_architecture: ArchitecturePattern,
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a detailed migration plan between architecture patterns."""
        
        migration_prompt = f"""
        Create a detailed migration plan from {current_architecture} to {target_architecture.value}:
        
        Current Architecture: {current_architecture}
        Target Architecture: {target_architecture.value}
        Project Context: {json.dumps(project_context, indent=2)}
        
        Provide comprehensive migration plan including:
        1. Pre-migration assessment and preparation
        2. Step-by-step migration phases
        3. Code transformation requirements
        4. Testing strategy during migration
        5. Risk mitigation strategies
        6. Rollback procedures
        7. Timeline and effort estimates
        8. Team training requirements
        
        Include specific code examples and transformation patterns.
        """
        
        try:
            migration_plan = await self.execute_llm_task(
                user_prompt=migration_prompt,
                context={
                    "current": current_architecture,
                    "target": target_architecture.value,
                    "project": project_context
                },
                structured_output=True
            )
            
            # Store migration plan
            await self.memory_manager.store_memory(
                content=f"Migration plan from {current_architecture} to {target_architecture.value}",
                metadata={
                    "type": "migration_plan",
                    "from_pattern": current_architecture,
                    "to_pattern": target_architecture.value,
                    "estimated_effort": migration_plan.get("estimated_effort")
                },
                importance=0.9,
                long_term=True
            )
            
            return migration_plan
            
        except Exception as e:
            logger.error(f"Migration plan creation failed: {e}")
            return {
                "error": f"Failed to create migration plan: {str(e)}",
                "phases": [],
                "risks": ["Migration planning failed"],
                "estimated_effort": "unknown"
            }
        
        super().__init__(config, llm_client, memory_manager, event_bus)
        
        # Register event handlers
        self.event_bus.subscribe("architecture.analyze_project", self._handle_analyze_project)
        self.event_bus.subscribe("architecture.recommend_structure", self._handle_recommend_structure)
        self.event_bus.subscribe("architecture.evaluate_design", self._handle_evaluate_design)
        
        logger.info(f"ArchitectureAgent {self.agent_id} initialized")
    
    async def _handle_analyze_project(self, message: AgentMessage) -> None:
        """Handle project architecture analysis requests."""
        try:
            # Safely create TaskContext by filtering supported parameters
            payload = message.payload
            task_context = TaskContext(
                task_id=payload.get("task_id", f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                task_type=payload.get("task_type", TaskType.ANALYSIS),
                description=payload.get("description", ""),
                parameters=payload.get("parameters", {}),
                dependencies=payload.get("dependencies", []),
                metadata={
                    **payload.get("metadata", {}),
                    "project_id": payload.get("project_id", "default_project")
                }
            )
            
            # Store the analysis request in memory
            await self.memory_manager.store_memory(
                f"analysis_request_{task_context.task_id}",
                {
                    "type": "project_analysis",
                    "project_path": task_context.context.get("project_path"),
                    "requirements": task_context.context.get("requirements", []),
                    "constraints": task_context.context.get("constraints", [])
                },
                memory_type="short_term"
            )
            
            result = await self._analyze_project_architecture(task_context)
            
            # Send result back via event bus
            await self.event_bus.publish(
                f"architecture.analysis_complete.{task_context.task_id}",
                AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_result",
                    payload=result.dict(),
                    correlation_id=message.correlation_id
                )
            )
            
        except Exception as e:
            logger.error(f"Error handling project analysis: {e}")
            await self._send_error_response(message, str(e))
    
    async def _handle_recommend_structure(self, message: AgentMessage) -> None:
        """Handle project structure recommendation requests."""
        try:
            # Safely create TaskContext by filtering supported parameters
            payload = message.payload
            task_context = TaskContext(
                task_id=payload.get("task_id", f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                task_type=payload.get("task_type", TaskType.ANALYSIS),
                description=payload.get("description", ""),
                parameters=payload.get("parameters", {}),
                dependencies=payload.get("dependencies", []),
                metadata={
                    **payload.get("metadata", {}),
                    "project_id": payload.get("project_id", "default_project")
                }
            )
            result = await self._recommend_project_structure(task_context)
            
            await self.event_bus.publish(
                f"architecture.structure_recommended.{task_context.task_id}",
                AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_result",
                    payload=result.dict(),
                    correlation_id=message.correlation_id
                )
            )
            
        except Exception as e:
            logger.error(f"Error handling structure recommendation: {e}")
            await self._send_error_response(message, str(e))
    
    async def _handle_evaluate_design(self, message: AgentMessage) -> None:
        """Handle design evaluation requests."""
        try:
            # Safely create TaskContext by filtering supported parameters
            payload = message.payload
            task_context = TaskContext(
                task_id=payload.get("task_id", f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}"),
                task_type=payload.get("task_type", TaskType.ANALYSIS),
                description=payload.get("description", ""),
                parameters=payload.get("parameters", {}),
                dependencies=payload.get("dependencies", []),
                metadata={
                    **payload.get("metadata", {}),
                    "project_id": payload.get("project_id", "default_project")
                }
            )
            result = await self._evaluate_architectural_design(task_context)
            
            await self.event_bus.publish(
                "architecture_agent",
                AgentMessage(
                    message_id=f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}",
                    agent_id=self.agent_id,
                    message_type=MessageType.TASK_RESULT,
                    payload=result.to_dict(),
                    timestamp=datetime.utcnow()
                )
            )
        except Exception as e:
            logger.error(f"Error handling design evaluation: {e}", exc_info=True)
            await self._send_error_response(message, str(e))
    
    async def _analyze_project_architecture(self, task_context: TaskContext) -> TaskResult:
        """Analyze the architecture of a Flutter project using LLM reasoning."""
        
        # Retrieve relevant context from memory
        context_memories = await self.memory_manager.get_relevant_memories(
            f"architecture analysis {task_context.context.get('project_path', '')}"
        )
        
        # Build LLM prompt for architecture analysis
        prompt = self._build_analysis_prompt(task_context, context_memories)
        
        # Get LLM analysis
        response = await self.llm_client.generate_response(
            prompt,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse LLM response into structured format
        analysis_result = await self._parse_analysis_response(response)
        
        # Store analysis results in memory
        await self.memory_manager.store_memory(
            f"architecture_analysis_{task_context.task_id}",
            analysis_result,
            memory_type="long_term"
        )
        
        return TaskResult(
            task_id=task_context.task_id,
            agent_id=self.agent_id,
            status="completed",
            result=analysis_result,
            metadata={
                "analysis_type": "architecture",
                "confidence_score": analysis_result.get("confidence", 0.8)
            }
        )
    
    async def _recommend_project_structure(self, task_context: TaskContext) -> TaskResult:
        """Recommend optimal project structure using LLM reasoning."""
        
        # Get project requirements and constraints
        requirements = task_context.context.get("requirements", [])
        constraints = task_context.context.get("constraints", [])
        project_type = task_context.context.get("project_type", "mobile_app")
        
        # Build context from memory
        relevant_memories = await self.memory_manager.get_relevant_memories(
            f"project structure {project_type} recommendations"
        )
        
        # Create LLM prompt for structure recommendation
        prompt = f"""
        As a Flutter architecture expert, recommend an optimal project structure for the following requirements:
        
        Project Type: {project_type}
        Requirements: {json.dumps(requirements, indent=2)}
        Constraints: {json.dumps(constraints, indent=2)}
        
        Previous similar recommendations:
        {self._format_memories_for_prompt(relevant_memories)}
        
        Please provide:
        1. Recommended folder structure with explanations
        2. Suggested architectural patterns (MVC, MVVM, Clean Architecture, etc.)
        3. Package organization strategy
        4. Dependency management approach
        5. Configuration and environment setup
        6. Testing structure recommendations
        7. Rationale for each recommendation
        
        Format your response as JSON with the following structure:
        {{
            "structure": {{
                "folders": [
                    {{"name": "folder_name", "purpose": "description", "subfolders": []}}
                ],
                "key_files": [
                    {{"name": "file_name", "purpose": "description", "location": "path"}}
                ]
            }},
            "patterns": {{
                "primary_pattern": "pattern_name",
                "secondary_patterns": [],
                "rationale": "explanation"
            }},
            "dependencies": {{
                "core_packages": [],
                "dev_packages": [],
                "management_strategy": "explanation"
            }},
            "configuration": {{
                "environments": ["dev", "staging", "prod"],
                "config_files": [],
                "setup_steps": []
            }},
            "testing": {{
                "structure": {{}},
                "strategies": [],
                "tools": []
            }},
            "rationale": "overall explanation for the recommendations",
            "confidence": 0.9
        }}
        """
        
        # Get LLM recommendation
        response = await self.llm_client.generate_response(
            prompt,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse and validate response
        recommendation = await self._parse_structure_response(response)
        
        # Store recommendation in memory
        await self.memory_manager.store_memory(
            f"structure_recommendation_{task_context.task_id}",
            recommendation,
            memory_type="long_term"
        )
        
        return TaskResult(
            task_id=task_context.task_id,
            agent_id=self.agent_id,
            status="completed",
            result=recommendation,
            metadata={
                "recommendation_type": "project_structure",
                "project_type": project_type
            }
        )
    
    async def _evaluate_architectural_design(self, task_context: TaskContext) -> TaskResult:
        """Evaluate existing architectural design using LLM reasoning."""
        
        design_details = task_context.context.get("design_details", {})
        evaluation_criteria = task_context.context.get("criteria", [
            "maintainability", "scalability", "performance", 
            "testability", "security", "code_quality"
        ])
        
        # Get relevant evaluation patterns from memory
        relevant_memories = await self.memory_manager.get_relevant_memories(
            f"architectural design evaluation {' '.join(evaluation_criteria)}"
        )
        
        # Build evaluation prompt
        prompt = f"""
        As a Flutter architecture expert, evaluate the following architectural design:
        
        Design Details:
        {json.dumps(design_details, indent=2)}
        
        Evaluation Criteria: {evaluation_criteria}
        
        Previous evaluations for reference:
        {self._format_memories_for_prompt(relevant_memories)}
        
        Please provide a comprehensive evaluation including:
        1. Strengths of the current design
        2. Weaknesses and potential issues
        3. Recommendations for improvement
        4. Risk assessment
        5. Scores for each criterion (1-10)
        6. Overall architectural health score
        7. Priority areas for refactoring
        
        Format as JSON:
        {{
            "overall_score": 8.5,
            "criterion_scores": {{
                "maintainability": 8,
                "scalability": 7,
                "performance": 9,
                "testability": 6,
                "security": 8,
                "code_quality": 9
            }},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [
                {{
                    "priority": "high|medium|low",
                    "area": "area_name",
                    "issue": "description",
                    "solution": "recommended_fix",
                    "effort": "estimation"
                }}
            ],
            "risks": [
                {{
                    "risk": "description",
                    "impact": "high|medium|low",
                    "likelihood": "high|medium|low",
                    "mitigation": "strategy"
                }}
            ],
            "refactoring_priorities": [],
            "confidence": 0.85
        }}
        """
        
        # Get LLM evaluation
        response = await self.llm_client.generate_response(
            prompt,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse evaluation response
        evaluation = await self._parse_evaluation_response(response)
        
        # Store evaluation in memory
        await self.memory_manager.store_memory(
            f"design_evaluation_{task_context.task_id}",
            evaluation,
            memory_type="long_term"
        )
        
        return TaskResult(
            task_id=task_context.task_id,
            agent_id=self.agent_id,
            status="completed",
            result=evaluation,
            metadata={
                "evaluation_type": "architectural_design",
                "overall_score": evaluation.get("overall_score", 0)
            }
        )
    
    def _build_analysis_prompt(
        self, 
        task_context: TaskContext, 
        context_memories: List[Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for architecture analysis."""
        
        project_path = task_context.context.get("project_path", "")
        requirements = task_context.context.get("requirements", [])
        
        return f"""
        As a Flutter architecture expert, analyze the project structure and architecture:
        
        Project Path: {project_path}
        Requirements: {json.dumps(requirements, indent=2)}
        
        Previous analysis patterns:
        {self._format_memories_for_prompt(context_memories)}
        
        Please analyze:
        1. Current project structure and organization
        2. Architectural patterns in use
        3. Dependency management approach
        4. Code organization and modularity
        5. Configuration and environment handling
        6. Testing architecture
        7. Potential issues and improvements
        8. Compliance with Flutter best practices
        
        Provide analysis as JSON:
        {{
            "structure_analysis": {{}},
            "patterns_identified": [],
            "dependencies": {{}},
            "modularity_score": 8,
            "issues": [],
            "recommendations": [],
            "best_practices_compliance": 0.85,
            "confidence": 0.9
        }}
        """
    
    async def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response into structured format with better error handling."""
        try:
            # Use robust JSON extraction
            parsed_json = extract_json_from_llm_response(response)
            if parsed_json:
                return parsed_json
            
            # If extraction failed, use LLM to convert to structured format
            structure_prompt = f"""
            Convert this architecture analysis into structured JSON format:
            
            {response}
            
            Required JSON structure:
            {{
                "structure_analysis": {{}},
                "patterns_identified": [],
                "dependencies": {{}},
                "modularity_score": 8,
                "issues": [],
                "recommendations": [],
                "best_practices_compliance": 0.85,
                "confidence": 0.9
            }}
            
            Respond with ONLY the JSON object, no additional text.
            """
            
            try:
                structured_response = await asyncio.wait_for(
                    self.llm_client.generate_response(
                        structure_prompt,
                        model=self.config.llm_model,
                        temperature=0.1,
                        max_tokens=2000
                    ),
                    timeout=60  # 1 minute timeout
                )
                
                parsed_structured = extract_json_from_llm_response(structured_response)
                if parsed_structured:
                    return parsed_structured
                else:
                    logger.error("Failed to extract JSON from structured response")
                    return self._create_fallback_analysis_response()
                    
            except asyncio.TimeoutError:
                logger.error("Architecture analysis structuring timed out")
                return self._create_fallback_analysis_response()
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return self._create_fallback_analysis_response()

    def _create_fallback_analysis_response(self) -> Dict[str, Any]:
        """Create a fallback analysis response when parsing fails."""
        return {
            "structure_analysis": {"error": "Failed to parse analysis"},
            "patterns_identified": ["Unable to analyze patterns"],
            "dependencies": {},
            "modularity_score": 5,
            "issues": ["Analysis parsing failed"],
            "recommendations": ["Re-run analysis with clearer input"],
            "best_practices_compliance": 0.5,
            "confidence": 0.3
        }
    
    async def _parse_structure_response(self, response: str) -> Dict[str, Any]:
        """Parse structure recommendation response."""
        try:
            # Use robust JSON extraction
            parsed_json = extract_json_from_llm_response(response)
            if parsed_json:
                return parsed_json
            
            # Use LLM to structure the response
            structure_prompt = f"""
            Convert this project structure recommendation into valid JSON:
            
            {response}
            
            Required format:
            {{
                "structure": {{"folders": [], "key_files": []}},
                "patterns": {{"primary_pattern": "", "secondary_patterns": [], "rationale": ""}},
                "dependencies": {{"core_packages": [], "dev_packages": [], "management_strategy": ""}},
                "configuration": {{"environments": [], "config_files": [], "setup_steps": []}},
                "testing": {{"structure": {{}}, "strategies": [], "tools": []}},
                "rationale": "",
                "confidence": 0.9
            }}
            
            Respond with ONLY the JSON object, no additional text.
            """
            
            structured = await self.llm_client.generate_response(structure_prompt, temperature=0.1)
            parsed_structured = extract_json_from_llm_response(structured)
            if parsed_structured:
                return parsed_structured
            else:
                logger.error("Failed to extract JSON from structured structure response")
                return {"error": "Structure parsing failed", "confidence": 0.1}
            
        except Exception as e:
            logger.error(f"Error parsing structure response: {e}")
            return {"error": f"Parsing failed: {str(e)}", "confidence": 0.1}
    
    async def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse design evaluation response."""
        try:
            # Use robust JSON extraction
            parsed_json = extract_json_from_llm_response(response)
            if parsed_json:
                return parsed_json
            
            # Structure the evaluation response
            structure_prompt = f"""
            Convert this architectural evaluation into structured JSON:
            
            {response}
            
            Required format:
            {{
                "overall_score": 8.5,
                "criterion_scores": {{}},
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "risks": [],
                "refactoring_priorities": [],
                "confidence": 0.85
            }}
            
            Respond with ONLY the JSON object, no additional text.
            """
            
            structured = await self.llm_client.generate_response(structure_prompt, temperature=0.1)
            parsed_structured = extract_json_from_llm_response(structured)
            if parsed_structured:
                return parsed_structured
            else:
                logger.error("Failed to extract JSON from structured evaluation response")
                return {"error": "Evaluation parsing failed", "confidence": 0.1}
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return {"overall_score": 5.0, "error": str(e), "confidence": 0.1}
    
    async def process_task(self, task_context: TaskContext) -> TaskResult:
        """Process architecture tasks and provide system design guidance."""
        logger.info(f"Processing architecture task: {task_context.task_id}")
        
        try:
            # Extract project details from task context
            metadata = task_context.metadata
            project_name = metadata.get("project_name", "flutter_app")
            features = metadata.get("features", [])
            task_type = task_context.description.lower()
            
            logger.info(f"Analyzing architecture for: {task_context.description}")
            logger.info(f"Project: {project_name}, Features: {features}")
            
            # Determine task type and route to appropriate handler
            if "architecture" in task_type or "design" in task_type:
                result = await self._handle_architecture_design(task_context, project_name, features)
            else:
                # Default to architecture design
                result = await self._handle_architecture_design(task_context, project_name, features)
            
            # Create task result
            task_result = TaskResult(
                task_id=task_context.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result=result,
                deliverables=result.get("deliverables", {}),
                metrics=result.get("metrics", {}),
                metadata=result.get("metadata", {}),
                errors=result.get("errors", [])
            )
            
            logger.info(f"Successfully completed architecture task: {task_context.task_id}")
            return task_result
            
        except Exception as e:
            logger.error(f"Architecture task failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TaskResult(
                task_id=task_context.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result={"error": str(e)},
                deliverables={},
                metrics={},
                metadata={},
                errors=[str(e)]
            )

    async def _handle_architecture_design(self, task_context: TaskContext, project_name: str, features: List[str]) -> Dict[str, Any]:
        """Handle architecture design and planning for Flutter project."""
        logger.info(f"Designing architecture for {project_name} with features: {features}")
        
        try:
            # Create architecture design prompt
            arch_prompt = self._create_architecture_prompt(project_name, features)
            
            # Generate architecture design using LLM
            architecture_result = await self.execute_llm_task(
                user_prompt=arch_prompt,
                context={
                    "project_name": project_name,
                    "features": features,
                    "patterns": self.flutter_best_practices
                },
                structured_output=False  # Change to False to get raw response
            )
            
            # Extract JSON from the response
            if architecture_result.get("status") == "success":
                raw_response = architecture_result.get("response", "")
                # Use the enhanced JSON extraction function
                from src.utils.json_utils import extract_json_from_llm_response
                parsed_json = extract_json_from_llm_response(raw_response)
                
                if parsed_json:
                    architecture_result = parsed_json
                else:
                    logger.error("Failed to extract JSON from architecture LLM response")
                    architecture_result = {
                        "error": "JSON extraction failed",
                        "raw_response": raw_response[:500]  # Truncated for logging
                    }
            elif architecture_result.get("error"):
                logger.error(f"Architecture LLM task failed: {architecture_result.get('error')}")
            
            # Store architecture decisions in memory only if successful
            if isinstance(architecture_result, dict) and "error" not in architecture_result:
                try:
                    await self.memory_manager.store_memory(
                        content=f"Architecture design for {project_name}: {json.dumps(architecture_result)}",
                        metadata={
                            "type": "architecture_design",
                            "project_name": project_name,
                            "features": features
                        },
                        importance=0.9,
                        long_term=True
                    )
                except Exception as memory_error:
                    logger.error(f"Failed to store architecture in memory: {memory_error}")
                    # Continue execution even if memory storage fails
            
            return {
                "status": "completed",
                "architecture_design": architecture_result,
                "deliverables": {
                    "architecture_diagram": "Project architecture and component relationships",
                    "technical_specifications": "Detailed technical specifications and patterns",
                    "dependency_list": "Required Flutter packages and dependencies"
                },
                "metadata": {
                    "recommended_patterns": architecture_result.get("recommended_patterns", []),
                    "dependencies": architecture_result.get("dependencies", []),
                    "project_structure": architecture_result.get("project_structure", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Architecture design failed: {e}")
            raise

    def _create_architecture_prompt(self, project_name: str, features: List[str]) -> str:
        """Create a comprehensive prompt for architecture design."""
        features_text = ", ".join(features) if features else "basic mobile app functionality"
        
        return f"""
You are an expert Flutter architect. Design a comprehensive, scalable architecture for "{project_name}" with the following features: {features_text}.

ARCHITECTURE REQUIREMENTS:
1. Design a clean, maintainable, and scalable Flutter app architecture
2. Recommend appropriate architectural patterns (Clean Architecture, BLoC, etc.)
3. Design proper folder structure and code organization
4. Select suitable state management solution
5. Plan navigation architecture and routing
6. Recommend necessary Flutter packages and dependencies
7. Consider platform-specific requirements (iOS, Android, Web)
8. Plan for testing, error handling, and performance optimization

FEATURES TO ANALYZE:
{self._generate_architecture_analysis(features)}

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
    "recommended_patterns": {{
        "architecture": "clean_architecture|bloc|mvc|mvvm",
        "state_management": "bloc|provider|riverpod|getx",
        "navigation": "go_router|auto_route|navigator",
        "dependency_injection": "get_it|provider|injectable"
    }},
    "project_structure": {{
        "lib/": {{
            "core/": "Core utilities, constants, and base classes",
            "features/": "Feature-based modules with their own data, domain, and presentation layers",
            "shared/": "Shared widgets, services, and utilities",
            "main.dart": "App entry point and configuration"
        }}
    }},
    "dependencies": [
        "flutter_bloc",
        "go_router", 
        "dio",
        "hive",
        "get_it"
    ],
    "dev_dependencies": [
        "flutter_test",
        "mockito",
        "bloc_test"
    ],
    "technical_specifications": {{
        "state_management": "Detailed state management approach",
        "navigation": "Navigation and routing strategy", 
        "data_layer": "Data persistence and API integration strategy",
        "testing_strategy": "Unit, widget, and integration testing approach"
    }},
    "implementation_guidelines": [
        "Follow Clean Architecture principles",
        "Implement proper error handling",
        "Use dependency injection for testability",
        "Follow Flutter best practices and conventions"
    ]
}}

Provide a comprehensive, production-ready architecture design that supports all requested features.
"""

    def _generate_architecture_analysis(self, features: List[str]) -> str:
        """Generate detailed architecture analysis for each feature."""
        if not features:
            return "- Basic mobile app: Simple navigation, basic UI components, minimal state management"
        
        feature_analysis = []
        
        for feature in features:
            if "photo" in feature.lower() or "image" in feature.lower():
                feature_analysis.append("- Photo sharing: Image processing, file upload, local caching, gallery management")
            elif "auth" in feature.lower() or "login" in feature.lower():
                feature_analysis.append("- Authentication: Secure token management, user session handling, form validation")
            elif "social" in feature.lower() or "chat" in feature.lower() or "messag" in feature.lower():
                feature_analysis.append("- Social features: Real-time communication, user management, notification system")
            elif "music" in feature.lower() or "audio" in feature.lower():
                feature_analysis.append("- Music player: Audio streaming, playlist management, background playback")
            elif "ecommerce" in feature.lower() or "shop" in feature.lower():
                feature_analysis.append("- E-commerce: Product catalog, shopping cart state, payment integration")
            elif "weather" in feature.lower():
                feature_analysis.append("- Weather app: Location services, API integration, data caching")
            else:
                feature_analysis.append(f"- {feature}: Comprehensive analysis and planning for {feature}")
        
        return "\n".join(feature_analysis)

    async def process_llm_response(self, response_text):
        """Process the LLM response with improved JSON handling."""
        try:
            parsed_json = extract_json_from_llm_response(response_text)
            if parsed_json:
                return parsed_json
            else:
                self.logger.error("Failed to parse LLM response as JSON")
                return {"error": "Failed to parse response"}
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return {"error": f"Processing error: {str(e)}"}
