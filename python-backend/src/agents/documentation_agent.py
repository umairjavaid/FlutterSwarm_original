"""
Documentation Agent for FlutterSwarm Multi-Agent System.

This agent specializes in documentation generation, maintenance,
and knowledge management for Flutter applications.
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

logger = get_logger("documentation_agent")


class DocumentationAgent(BaseAgent):
    """
    Specialized agent for Flutter application documentation and knowledge management.
    
    This agent handles:
    - API documentation generation and maintenance
    - Code documentation and comments
    - User guides and tutorials creation
    - Architecture documentation
    - Release notes and changelog generation
    - Developer onboarding documentation
    - Best practices and style guides
    - Knowledge base maintenance
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: MemoryManager,
        event_bus: EventBus
    ):
        # Override config for documentation-specific settings
        docs_config = AgentConfig(
            agent_id=config.agent_id or f"documentation_agent_{str(uuid.uuid4())[:8]}",
            agent_type="documentation",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.DOCUMENTATION
            ],
            max_concurrent_tasks=4,
            llm_model=config.llm_model or "gpt-4",
            temperature=0.4,  # Moderate temperature for creative documentation
            max_tokens=8000,
            timeout=600,
            metadata=config.metadata
        )
        
        super().__init__(docs_config, llm_client, memory_manager, event_bus)
        
        # Documentation-specific state
        self.documentation_types = [
            "api_docs", "user_guides", "tutorials", "architecture_docs",
            "release_notes", "developer_guides", "best_practices", 
            "troubleshooting", "faq", "changelog"
        ]
        
        self.documentation_formats = {
            "markdown": "Markdown documentation",
            "dartdoc": "Dart documentation comments",
            "sphinx": "Sphinx documentation system",
            "gitbook": "GitBook documentation",
            "notion": "Notion knowledge base",
            "confluence": "Atlassian Confluence"
        }
        
        self.documentation_tools = {
            "generation": ["dartdoc", "sphinx", "mkdocs", "gitbook"],
            "hosting": ["github_pages", "netlify", "vercel", "gitbook_hosting"],
            "collaboration": ["notion", "confluence", "wiki", "google_docs"],
            "automation": ["github_actions", "gitlab_ci", "documentation_ci"]
        }
        
        logger.info(f"Documentation Agent {self.agent_id} initialized")

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the documentation agent."""
        return """
You are the Documentation Agent in the FlutterSwarm multi-agent system, specializing in comprehensive documentation creation and maintenance for Flutter applications.

CORE EXPERTISE:
- Technical documentation writing and structuring
- API documentation generation using DartDoc
- User guide and tutorial creation
- Architecture and design documentation
- Developer onboarding and knowledge transfer
- Documentation automation and maintenance
- Cross-platform documentation considerations
- Documentation best practices and standards

DOCUMENTATION RESPONSIBILITIES:
1. API Documentation: Generate comprehensive API documentation from code
2. User Guides: Create user-friendly guides and tutorials
3. Architecture Documentation: Document system design and architecture
4. Developer Documentation: Create onboarding and contribution guides
5. Release Documentation: Generate release notes and changelogs
6. Process Documentation: Document development and deployment processes
7. Knowledge Management: Maintain and organize project knowledge
8. Documentation Automation: Set up automated documentation workflows

DOCUMENTATION TYPES AND STANDARDS:
- Code Documentation: Inline comments, DartDoc annotations, examples
- API Documentation: Method signatures, parameters, return values, examples
- User Documentation: Step-by-step guides, tutorials, FAQs
- Technical Documentation: Architecture diagrams, design decisions, patterns
- Process Documentation: Development workflows, deployment procedures
- Reference Documentation: Configuration guides, troubleshooting, best practices

DOCUMENTATION BEST PRACTICES:
- Clarity: Write clear, concise, and unambiguous content
- Completeness: Cover all necessary aspects thoroughly
- Accuracy: Ensure technical accuracy and up-to-date information
- Consistency: Maintain consistent style and formatting
- Accessibility: Make documentation accessible to target audience
- Searchability: Structure content for easy discovery and navigation
- Maintainability: Design for easy updates and maintenance
- Examples: Include practical examples and code snippets

FLUTTER-SPECIFIC DOCUMENTATION:
- Widget documentation with usage examples
- State management pattern documentation
- Platform-specific implementation guides
- Testing strategy and examples
- Deployment and distribution guides
- Package and plugin documentation
- Migration guides and breaking changes

DOCUMENTATION FORMATS:
- Markdown for general documentation
- DartDoc for API documentation
- Mermaid diagrams for architecture visualization
- Code snippets with syntax highlighting
- Screenshots and visual guides
- Interactive examples and demos

AUTOMATION AND TOOLING:
- Automated API documentation generation
- Documentation linting and quality checks
- Version control integration
- Continuous integration for documentation
- Documentation hosting and deployment
- Search and discovery optimization

Always create comprehensive, well-structured, and maintainable documentation that serves both current and future development needs.
"""

    async def get_capabilities(self) -> List[str]:
        """Get a list of documentation-specific capabilities."""
        return [
            "api_documentation_generation",
            "user_guide_creation",
            "tutorial_development",
            "architecture_documentation",
            "release_notes_generation",
            "developer_onboarding_docs",
            "best_practices_documentation",
            "troubleshooting_guides",
            "changelog_generation",
            "documentation_automation",
            "knowledge_base_management",
            "documentation_optimization"
        ]

    async def _execute_specialized_processing(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute documentation-specific processing logic.
        """
        try:
            task_type = task_context.task_type.value
            
            if task_type == "api_documentation":
                return await self._generate_api_docs(task_context, llm_analysis)
            elif task_type == "user_guide":
                return await self._create_user_guide(task_context, llm_analysis)
            elif task_type == "tutorial":
                return await self._create_tutorial(task_context, llm_analysis)
            elif task_type == "architecture_docs":
                return await self._document_architecture(task_context, llm_analysis)
            elif task_type == "release_notes":
                return await self._generate_release_notes(task_context, llm_analysis)
            elif task_type == "developer_guide":
                return await self._create_developer_guide(task_context, llm_analysis)
            else:
                # Generic documentation processing
                return await self._process_documentation_request(task_context, llm_analysis)
                
        except Exception as e:
            logger.error(f"Documentation processing failed: {e}")
            return {
                "error": str(e),
                "documentation_files": {},
                "documentation_notes": []
            }

    async def _generate_api_docs(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive API documentation."""
        logger.info(f"Generating API documentation for task: {task_context.task_id}")
        
        api_docs_prompt = self._create_api_docs_prompt(task_context, llm_analysis)
        
        api_documentation = await self.execute_llm_task(
            user_prompt=api_docs_prompt,
            context={
                "task": task_context.to_dict(),
                "documentation_standards": self._get_documentation_standards(),
                "dartdoc_patterns": self._get_dartdoc_patterns()
            },
            structured_output=True
        )
        
        # Store API documentation
        await self.memory_manager.store_memory(
            content=f"API documentation: {json.dumps(api_documentation)}",
            metadata={
                "type": "api_documentation",
                "apis_documented": len(api_documentation.get('api_sections', [])),
                "examples_included": len(api_documentation.get('examples', []))
            },
            correlation_id=task_context.correlation_id,
            importance=0.8,
            long_term=True
        )
        
        return {
            "api_documentation": api_documentation,
            "documentation_files": api_documentation.get("files", {}),
            "api_reference": api_documentation.get("reference", {}),
            "code_examples": api_documentation.get("examples", {}),
            "integration_guides": api_documentation.get("integration", {}),
            "migration_notes": api_documentation.get("migration", [])
        }

    async def _create_user_guide(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive user guides."""
        logger.info(f"Creating user guide for task: {task_context.task_id}")
        
        user_guide_prompt = self._create_user_guide_prompt(task_context, llm_analysis)
        
        user_guide = await self.execute_llm_task(
            user_prompt=user_guide_prompt,
            context={
                "task": task_context.to_dict(),
                "guide_patterns": self._get_user_guide_patterns()
            },
            structured_output=True
        )
        
        return {
            "user_guide": user_guide,
            "guide_sections": user_guide.get("sections", {}),
            "tutorials": user_guide.get("tutorials", {}),
            "faqs": user_guide.get("faqs", []),
            "troubleshooting": user_guide.get("troubleshooting", {}),
            "quick_start": user_guide.get("quick_start", {})
        }

    async def _create_tutorial(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create step-by-step tutorials."""
        logger.info(f"Creating tutorial for task: {task_context.task_id}")
        
        tutorial_prompt = self._create_tutorial_prompt(task_context, llm_analysis)
        
        tutorial = await self.execute_llm_task(
            user_prompt=tutorial_prompt,
            context={
                "task": task_context.to_dict(),
                "tutorial_patterns": self._get_tutorial_patterns()
            },
            structured_output=True
        )
        
        return {
            "tutorial": tutorial,
            "tutorial_steps": tutorial.get("steps", []),
            "code_examples": tutorial.get("code", {}),
            "screenshots": tutorial.get("screenshots", []),
            "prerequisites": tutorial.get("prerequisites", []),
            "learning_objectives": tutorial.get("objectives", [])
        }

    # Prompt creation methods
    def _create_api_docs_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for API documentation generation."""
        return f"""
Generate comprehensive API documentation for the following Flutter code:

CODE ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

DOCUMENTATION REQUIREMENTS:
{task_context.description}

PROJECT CONTEXT:
{json.dumps(task_context.project_context, indent=2)}

Please create complete API documentation including:

1. API REFERENCE:
   - Class and method documentation
   - Parameter descriptions and types
   - Return value documentation
   - Exception handling documentation
   - Usage examples for each API

2. DARTDOC ANNOTATIONS:
   - Proper DartDoc comment formatting
   - Parameter and return value documentation
   - Example code blocks
   - See-also references
   - Version and deprecation notes

3. CODE EXAMPLES:
   - Basic usage examples
   - Advanced usage scenarios
   - Integration examples
   - Best practices demonstrations
   - Common patterns and idioms

4. INTEGRATION GUIDES:
   - Setup and installation instructions
   - Configuration requirements
   - Platform-specific considerations
   - Dependency management
   - Troubleshooting common issues

5. REFERENCE DOCUMENTATION:
   - Complete API reference
   - Type definitions and interfaces
   - Constants and enumerations
   - Configuration options
   - Error codes and messages

6. MIGRATION AND COMPATIBILITY:
   - Version compatibility information
   - Breaking changes documentation
   - Migration guides between versions
   - Deprecation notices and alternatives
   - Upgrade procedures

Provide well-structured, comprehensive documentation following DartDoc standards and Flutter documentation best practices.
"""

    def _create_user_guide_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for user guide creation."""
        return f"""
Create comprehensive user guide for the following Flutter application:

APPLICATION ANALYSIS:
{json.dumps(llm_analysis, indent=2)}

USER GUIDE REQUIREMENTS:
{task_context.description}

Please create detailed user guide including:

1. GETTING STARTED:
   - Installation and setup instructions
   - System requirements and prerequisites
   - Quick start guide and first steps
   - Basic configuration and customization
   - Initial user onboarding

2. FEATURE DOCUMENTATION:
   - Detailed feature explanations
   - Step-by-step usage instructions
   - Screenshots and visual guides
   - Best practices and tips
   - Common use cases and scenarios

3. TUTORIALS AND WALKTHROUGHS:
   - Beginner-friendly tutorials
   - Advanced feature demonstrations
   - Real-world usage examples
   - Problem-solving scenarios
   - Hands-on exercises

4. TROUBLESHOOTING:
   - Common issues and solutions
   - Error message explanations
   - Diagnostic procedures
   - Support and help resources

5. REFERENCE MATERIALS:
   - Feature comparison tables
   - Configuration reference
   - Keyboard shortcuts and gestures
   - Glossary of terms
   - Frequently asked questions

6. PLATFORM-SPECIFIC GUIDES:
   - iOS-specific instructions
   - Android-specific procedures
   - Web platform considerations
   - Desktop platform features
   - Cross-platform compatibility

Provide user-friendly, well-organized documentation that enables users to effectively use the application.
"""

    def _create_tutorial_prompt(
        self,
        task_context: TaskContext,
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Create prompt for tutorial creation."""
        return f"""
Create step-by-step tutorial for the following Flutter development topic:

TUTORIAL REQUIREMENTS:
{json.dumps(llm_analysis, indent=2)}

TUTORIAL SCOPE:
{task_context.description}

Please create comprehensive tutorial including:

1. TUTORIAL STRUCTURE:
   - Clear learning objectives
   - Prerequisites and requirements
   - Estimated completion time
   - Skill level indication
   - Required tools and resources

2. STEP-BY-STEP INSTRUCTIONS:
   - Logical progression of steps
   - Clear and concise instructions
   - Code examples with explanations
   - Expected outcomes for each step
   - Validation and testing procedures

3. PRACTICAL EXAMPLES:
   - Real-world code implementations
   - Working project examples
   - Interactive demonstrations
   - Before and after comparisons
   - Best practices illustrations

4. VISUAL AIDS:
   - Screenshots and diagrams
   - Code syntax highlighting
   - Flow charts and process diagrams
   - UI mockups and designs
   - Video or animation references

5. TROUBLESHOOTING:
   - Common mistakes and solutions
   - Error handling and debugging
   - Performance considerations
   - Alternative approaches
   - Additional resources and references

6. EXERCISES AND PRACTICE:
   - Hands-on exercises
   - Challenge problems
   - Extension activities
   - Self-assessment questions
   - Project ideas for practice

Provide engaging, educational tutorial content that effectively teaches the target concepts and skills.
"""

    # Helper methods for documentation patterns and standards
    def _get_documentation_standards(self) -> Dict[str, Any]:
        """Get documentation standards and guidelines."""
        return {
            "dartdoc_standards": [
                "triple_slash_comments", "parameter_documentation", 
                "return_documentation", "example_blocks", "see_also_references"
            ],
            "markdown_standards": [
                "proper_headings", "code_blocks", "links_and_references", 
                "tables_and_lists", "image_formatting"
            ],
            "style_guidelines": [
                "clear_language", "consistent_terminology", 
                "active_voice", "concise_writing", "logical_structure"
            ],
            "accessibility": [
                "alt_text_for_images", "descriptive_links", 
                "proper_heading_hierarchy", "readable_formatting"
            ]
        }

    def _get_dartdoc_patterns(self) -> Dict[str, Any]:
        """Get DartDoc documentation patterns."""
        return {
            "class_documentation": "/// Brief description\n///\n/// Detailed explanation with examples",
            "method_documentation": "/// Brief method description\n///\n/// [parameter] description\n/// Returns description",
            "parameter_docs": "/// {@template parameter_name}\n/// Parameter description\n/// {@endtemplate}",
            "example_blocks": "/// Example:\n/// ```dart\n/// // Code example\n/// ```",
            "see_also": "/// See also:\n///  * [RelatedClass]\n///  * [relatedMethod]"
        }

    def _get_user_guide_patterns(self) -> Dict[str, Any]:
        """Get user guide structure patterns."""
        return {
            "getting_started": ["installation", "setup", "first_steps", "basic_usage"],
            "feature_guides": ["overview", "usage", "examples", "best_practices"],
            "troubleshooting": ["common_issues", "error_messages", "solutions", "contact_support"],
            "reference": ["api_reference", "configuration", "faq", "glossary"]
        }

    def _get_tutorial_patterns(self) -> Dict[str, Any]:
        """Get tutorial structure patterns."""
        return {
            "structure": ["objectives", "prerequisites", "steps", "validation", "conclusion"],
            "step_format": ["instruction", "code_example", "explanation", "expected_result"],
            "engagement": ["hands_on_practice", "real_world_examples", "progressive_difficulty"],
            "support": ["troubleshooting", "additional_resources", "next_steps"]
        }
