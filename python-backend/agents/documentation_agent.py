"""
Documentation Agent for Flutter Development
Handles code documentation, API docs, and documentation generation
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from core.agent_types import AgentType
import re

class DocumentationAgent(BaseAgent):
    def __init__(self, agent_id: str = "documentation_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DOCUMENTATION,
            name="Documentation Agent",
            description="Specialized agent for Flutter documentation generation and maintenance"
        )
        
        self.capabilities = [
            "api_documentation",
            "code_comments_generation",
            "readme_generation",
            "changelog_maintenance",
            "inline_documentation",
            "dartdoc_generation",
            "wiki_documentation",
            "tutorial_creation",
            "architecture_documentation",
            "deployment_guides",
            "troubleshooting_guides",
            "documentation_validation"
        ]
        
        self.documentation_types = [
            'api_reference', 'user_guide', 'developer_guide', 'tutorials',
            'architecture_docs', 'troubleshooting', 'changelog', 'readme'
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process documentation-related requests"""
        try:
            request_type = request.get('type', '')
            
            if request_type == 'generate_api_docs':
                return await self._generate_api_docs(request)
            elif request_type == 'generate_readme':
                return await self._generate_readme(request)
            elif request_type == 'add_code_comments':
                return await self._add_code_comments(request)
            elif request_type == 'create_changelog':
                return await self._create_changelog(request)
            elif request_type == 'generate_user_guide':
                return await self._generate_user_guide(request)
            elif request_type == 'create_architecture_docs':
                return await self._create_architecture_docs(request)
            elif request_type == 'validate_documentation':
                return await self._validate_documentation(request)
            elif request_type == 'setup_doc_generation':
                return await self._setup_doc_generation(request)
            else:
                return await self._analyze_documentation_needs(request)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Documentation processing failed: {str(e)}",
                'agent_id': self.agent_id
            }

    async def _generate_api_docs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive API documentation"""
        code = request.get('code', '')
        output_format = request.get('format', 'dartdoc')
        include_examples = request.get('include_examples', True)
        
        # Extract API elements
        api_elements = self._extract_api_elements(code)
        
        # Generate documentation for each element
        documented_code = self._add_api_documentation(code, api_elements, include_examples)
        
        # Generate dartdoc configuration
        dartdoc_config = self._generate_dartdoc_config()
        
        # Generate API reference structure
        api_reference = self._generate_api_reference_structure(api_elements)
        
        return {
            'success': True,
            'documented_code': documented_code,
            'dartdoc_config': dartdoc_config,
            'api_reference': api_reference,
            'api_elements_count': len(api_elements),
            'generation_instructions': self._get_api_doc_instructions(),
            'agent_id': self.agent_id
        }

    async def _generate_readme(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive README.md"""
        project_info = request.get('project_info', {})
        template_type = request.get('template', 'comprehensive')
        include_badges = request.get('include_badges', True)
        
        # Generate README content
        readme_content = self._create_readme_content(project_info, template_type, include_badges)
        
        # Generate additional documentation files
        contributing_guide = self._generate_contributing_guide()
        code_of_conduct = self._generate_code_of_conduct()
        
        return {
            'success': True,
            'readme_content': readme_content,
            'contributing_guide': contributing_guide,
            'code_of_conduct': code_of_conduct,
            'documentation_structure': self._get_documentation_structure(),
            'agent_id': self.agent_id
        }

    async def _add_code_comments(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add comprehensive code comments and documentation"""
        code = request.get('code', '')
        comment_style = request.get('style', 'dartdoc')
        detail_level = request.get('detail_level', 'medium')
        
        # Analyze code structure
        code_analysis = self._analyze_code_structure(code)
        
        # Generate comments for different code elements
        commented_code = self._add_comprehensive_comments(code, code_analysis, comment_style, detail_level)
        
        # Generate documentation comments summary
        comments_summary = self._generate_comments_summary(code_analysis)
        
        return {
            'success': True,
            'commented_code': commented_code,
            'comments_summary': comments_summary,
            'documentation_coverage': self._calculate_doc_coverage(code_analysis),
            'style_guide': self._get_comment_style_guide(comment_style),
            'agent_id': self.agent_id
        }

    async def _create_changelog(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create and maintain changelog"""
        version_info = request.get('version_info', {})
        changes = request.get('changes', [])
        format_style = request.get('format', 'keep_a_changelog')
        
        # Generate changelog entry
        changelog_entry = self._generate_changelog_entry(version_info, changes, format_style)
        
        # Generate full changelog template
        changelog_template = self._generate_changelog_template(format_style)
        
        # Generate version comparison
        version_comparison = self._generate_version_comparison(version_info, changes)
        
        return {
            'success': True,
            'changelog_entry': changelog_entry,
            'changelog_template': changelog_template,
            'version_comparison': version_comparison,
            'changelog_guidelines': self._get_changelog_guidelines(),
            'agent_id': self.agent_id
        }

    async def _generate_user_guide(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user guide documentation"""
        app_features = request.get('features', [])
        target_audience = request.get('audience', 'general')
        guide_format = request.get('format', 'markdown')
        
        # Generate guide structure
        guide_structure = self._create_user_guide_structure(app_features)
        
        # Generate feature documentation
        feature_docs = self._generate_feature_documentation(app_features, target_audience)
        
        # Generate troubleshooting section
        troubleshooting = self._generate_troubleshooting_section(app_features)
        
        # Generate FAQ section
        faq_section = self._generate_faq_section(app_features)
        
        return {
            'success': True,
            'guide_structure': guide_structure,
            'feature_documentation': feature_docs,
            'troubleshooting': troubleshooting,
            'faq_section': faq_section,
            'user_guide_template': self._get_user_guide_template(),
            'agent_id': self.agent_id
        }

    async def _create_architecture_docs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create architecture documentation"""
        project_structure = request.get('project_structure', {})
        architecture_patterns = request.get('patterns', [])
        tech_stack = request.get('tech_stack', [])
        
        # Generate architecture overview
        architecture_overview = self._generate_architecture_overview(project_structure, architecture_patterns)
        
        # Generate component diagrams
        component_diagrams = self._generate_component_diagrams(project_structure)
        
        # Generate data flow documentation
        data_flow_docs = self._generate_data_flow_documentation(architecture_patterns)
        
        # Generate deployment architecture
        deployment_docs = self._generate_deployment_documentation(tech_stack)
        
        return {
            'success': True,
            'architecture_overview': architecture_overview,
            'component_diagrams': component_diagrams,
            'data_flow_docs': data_flow_docs,
            'deployment_docs': deployment_docs,
            'architecture_decisions': self._generate_adr_template(),
            'agent_id': self.agent_id
        }

    async def _validate_documentation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documentation completeness and quality"""
        documentation_files = request.get('files', {})
        validation_rules = request.get('rules', 'standard')
        
        validation_results = {
            'completeness_score': 0,
            'quality_issues': [],
            'missing_sections': [],
            'outdated_content': [],
            'link_validation': []
        }
        
        # Check documentation completeness
        completeness_analysis = self._analyze_documentation_completeness(documentation_files)
        validation_results['completeness_score'] = completeness_analysis['score']
        validation_results['missing_sections'] = completeness_analysis['missing']
        
        # Validate content quality
        quality_issues = self._validate_content_quality(documentation_files)
        validation_results['quality_issues'] = quality_issues
        
        # Check for outdated content
        outdated_content = self._identify_outdated_content(documentation_files)
        validation_results['outdated_content'] = outdated_content
        
        # Validate links
        link_validation = self._validate_documentation_links(documentation_files)
        validation_results['link_validation'] = link_validation
        
        return {
            'success': True,
            'validation_results': validation_results,
            'improvement_suggestions': self._generate_doc_improvement_suggestions(validation_results),
            'maintenance_checklist': self._get_documentation_maintenance_checklist(),
            'agent_id': self.agent_id
        }

    async def _setup_doc_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup automated documentation generation"""
        project_type = request.get('project_type', 'flutter_package')
        hosting_platform = request.get('hosting', 'github_pages')
        automation_level = request.get('automation', 'ci_cd')
        
        # Generate dartdoc configuration
        dartdoc_setup = self._generate_dartdoc_setup(project_type)
        
        # Generate CI/CD documentation workflow
        ci_workflow = self._generate_doc_ci_workflow(hosting_platform)
        
        # Generate documentation structure
        doc_structure = self._generate_documentation_structure(project_type)
        
        # Generate automation scripts
        automation_scripts = self._generate_doc_automation_scripts(automation_level)
        
        return {
            'success': True,
            'dartdoc_setup': dartdoc_setup,
            'ci_workflow': ci_workflow,
            'doc_structure': doc_structure,
            'automation_scripts': automation_scripts,
            'setup_instructions': self._get_doc_setup_instructions(),
            'agent_id': self.agent_id
        }

    def _create_readme_content(self, project_info: Dict[str, Any], template_type: str, include_badges: bool) -> str:
        """Create comprehensive README content"""
        project_name = project_info.get('name', 'Flutter Project')
        description = project_info.get('description', 'A Flutter application')
        features = project_info.get('features', [])
        
        badges_section = ""
        if include_badges:
            badges_section = f"""
[![Build Status](https://github.com/username/{project_name.lower()}/workflows/CI/badge.svg)](https://github.com/username/{project_name.lower()}/actions)
[![Coverage](https://codecov.io/gh/username/{project_name.lower()}/branch/main/graph/badge.svg)](https://codecov.io/gh/username/{project_name.lower()})
[![Version](https://img.shields.io/pub/v/{project_name.lower()}.svg)](https://pub.dev/packages/{project_name.lower()})
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

"""
        
        features_section = ""
        if features:
            features_list = '\n'.join([f"- {feature}" for feature in features])
            features_section = f"""
## ✨ Features

{features_list}
"""
        
        return f"""# {project_name}

{badges_section}

{description}
{features_section}

## 🚀 Getting Started

### Prerequisites

- Flutter (Channel stable, 3.10.0 or higher)
- Dart SDK (3.0.0 or higher)
- Android Studio / VS Code
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/{project_name.lower()}.git
   cd {project_name.lower()}
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the app**
   ```bash
   flutter run
   ```

## 📱 Screenshots

| Home Screen | Features | Settings |
|-------------|----------|----------|
| ![Home](screenshots/home.png) | ![Features](screenshots/features.png) | ![Settings](screenshots/settings.png) |

## 🏗️ Architecture

This project follows Clean Architecture principles with the following layers:

```
lib/
├── core/           # Core functionality and constants
├── data/           # Data layer (repositories, data sources)
├── domain/         # Business logic layer
├── presentation/   # UI layer (screens, widgets, state management)
└── main.dart       # Application entry point
```

## 🧪 Testing

Run tests with coverage:
```bash
flutter test --coverage
```

Generate coverage report:
```bash
genhtml coverage/lcov.info -o coverage/html
```

## 📚 Documentation

- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Documentation](https://pub.dev/documentation/{project_name.lower()}/latest/)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🚀 Deployment

### Android
```bash
flutter build apk --release
# or
flutter build appbundle --release
```

### iOS
```bash
flutter build ios --release
```

### Web
```bash
flutter build web --release
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flutter team for the amazing framework
- Contributors who helped shape this project
- Open source packages that made this possible

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/username/{project_name.lower()}/issues)

---

Made with ❤️ by [Your Name](https://github.com/username)
"""

    def _add_api_documentation(self, code: str, api_elements: List[Dict], include_examples: bool) -> str:
        """Add comprehensive API documentation to code"""
        documented_code = code
        
        for element in api_elements:
            if element['type'] == 'class':
                doc_comment = self._generate_class_documentation(element, include_examples)
                documented_code = self._insert_documentation(documented_code, element, doc_comment)
            elif element['type'] == 'method':
                doc_comment = self._generate_method_documentation(element, include_examples)
                documented_code = self._insert_documentation(documented_code, element, doc_comment)
            elif element['type'] == 'property':
                doc_comment = self._generate_property_documentation(element)
                documented_code = self._insert_documentation(documented_code, element, doc_comment)
        
        return documented_code

    def _generate_class_documentation(self, class_info: Dict[str, Any], include_examples: bool) -> str:
        """Generate comprehensive class documentation"""
        class_name = class_info.get('name', 'Unknown')
        description = class_info.get('description', f'A class representing {class_name}')
        
        doc = f'''/// {description}
///
/// This class provides functionality for {description.lower()}.
///'''
        
        if include_examples:
            doc += f'''
/// 
/// Example usage:
/// ```dart
/// final {class_name.lower()} = {class_name}();
/// // Use the {class_name.lower()} instance
/// ```'''
        
        if class_info.get('see_also'):
            see_also = class_info['see_also']
            doc += f'''
///
/// See also:
/// * [{see_also}], which provides related functionality'''
        
        return doc

    def _generate_method_documentation(self, method_info: Dict[str, Any], include_examples: bool) -> str:
        """Generate comprehensive method documentation"""
        method_name = method_info.get('name', 'unknown')
        description = method_info.get('description', f'Performs {method_name} operation')
        parameters = method_info.get('parameters', [])
        return_type = method_info.get('return_type', 'void')
        
        doc = f'''  /// {description}
  ///'''
        
        # Add parameter documentation
        if parameters:
            doc += '\n  ///'
            for param in parameters:
                param_name = param.get('name', 'param')
                param_desc = param.get('description', f'The {param_name} parameter')
                doc += f'\n  /// * [{param_name}] - {param_desc}'
        
        # Add return documentation
        if return_type != 'void':
            doc += f'\n  ///\n  /// Returns: {return_type} - Description of return value'
        
        if include_examples:
            doc += f'''
  ///
  /// Example:
  /// ```dart
  /// final result = instance.{method_name}({', '.join([p.get('name', 'param') for p in parameters])});
  /// ```'''
        
        return doc

    def _generate_dartdoc_config(self) -> str:
        """Generate dartdoc configuration"""
        return '''
# Dartdoc configuration
name: dartdoc_options
description: Configuration for dartdoc generation

dartdoc:
  categoryOrder: ["Constructors", "Properties", "Methods"]
  linkToSource:
    root: "."
    uriTemplate: "https://github.com/username/project/blob/main/%f#L%l"
  nodoc: ["**.g.dart", "**.freezed.dart"]
  exclude: ["build/**", "test/**"]
  includeExternal: ["dart:core", "dart:async", "dart:collection"]
  
  # Custom templates
  templates:
    - "doc/templates"
    
  # Footer customization
  footer:
    - text: "Generated documentation"
      href: "https://github.com/username/project"
'''

    def _generate_changelog_entry(self, version_info: Dict[str, Any], changes: List[str], format_style: str) -> str:
        """Generate changelog entry"""
        version = version_info.get('version', '1.0.0')
        date = version_info.get('date', '2024-01-01')
        
        if format_style == 'keep_a_changelog':
            entry = f'''
## [{version}] - {date}

### Added
{self._format_changes(changes, 'added')}

### Changed
{self._format_changes(changes, 'changed')}

### Fixed
{self._format_changes(changes, 'fixed')}

### Removed
{self._format_changes(changes, 'removed')}
'''
        else:
            entry = f'''
# Version {version} ({date})

## Changes:
{chr(10).join([f"- {change}" for change in changes])}
'''
        
        return entry.strip()

    def _format_changes(self, changes: List[str], category: str) -> str:
        """Format changes by category"""
        category_changes = [change for change in changes if category.lower() in change.lower()]
        if not category_changes:
            return ""
        return '\n'.join([f"- {change}" for change in category_changes])

    def _get_documentation_structure(self) -> Dict[str, List[str]]:
        """Get recommended documentation structure"""
        return {
            'root': ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE'],
            'docs': ['user_guide.md', 'developer_guide.md', 'architecture.md', 'api_reference.md'],
            'examples': ['basic_usage.dart', 'advanced_usage.dart', 'integration_examples.dart'],
            'templates': ['issue_template.md', 'pr_template.md', 'feature_request.md']
        }

    async def _analyze_documentation_needs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze documentation requirements for the project"""
        project_type = request.get('project_type', 'app')
        target_audience = request.get('audience', ['developers'])
        complexity_level = request.get('complexity', 'medium')
        
        # Analyze documentation requirements
        doc_requirements = self._assess_documentation_requirements(project_type, target_audience, complexity_level)
        
        # Generate documentation plan
        documentation_plan = self._create_documentation_plan(doc_requirements)
        
        # Estimate effort
        effort_estimation = self._estimate_documentation_effort(doc_requirements)
        
        return {
            'success': True,
            'documentation_requirements': doc_requirements,
            'documentation_plan': documentation_plan,
            'effort_estimation': effort_estimation,
            'recommended_tools': self._get_recommended_documentation_tools(),
            'best_practices': self._get_documentation_best_practices(),
            'agent_id': self.agent_id
        }

    def _get_documentation_best_practices(self) -> List[str]:
        """Get documentation best practices"""
        return [
            "Write documentation from the user's perspective",
            "Keep documentation up-to-date with code changes",
            "Use clear, concise language and avoid jargon",
            "Include practical examples and code snippets",
            "Structure documentation logically with clear headings",
            "Use consistent formatting and style throughout",
            "Include visual aids like diagrams and screenshots",
            "Validate documentation with actual users",
            "Set up automated documentation generation",
            "Review and update documentation regularly"
        ]
