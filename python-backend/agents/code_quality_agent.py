"""
Code Quality Agent for Flutter Development
Handles code analysis, linting, formatting, and quality assurance
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from core.agent_types import AgentType
import re

class CodeQualityAgent(BaseAgent):
    def __init__(self, agent_id: str = "code_quality_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CODE_QUALITY,
            name="Code Quality Agent",
            description="Specialized agent for Flutter code quality, linting, and best practices enforcement"
        )
        
        self.capabilities = [
            "static_analysis",
            "code_formatting",
            "lint_rules_setup",
            "code_metrics_analysis",
            "performance_analysis",
            "security_analysis",
            "refactoring_suggestions",
            "code_review_automation",
            "documentation_validation",
            "dependency_analysis",
            "technical_debt_assessment",
            "best_practices_enforcement"
        ]
        
        self.quality_metrics = [
            'cyclomatic_complexity', 'code_coverage', 'technical_debt',
            'maintainability_index', 'lines_of_code', 'duplication_ratio'
        ]
        
        self.dart_best_practices = [
            'prefer_const_constructors', 'prefer_final_fields', 'avoid_print',
            'prefer_single_quotes', 'use_key_in_widget_constructors',
            'prefer_const_literals_to_create_immutables'
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process code quality-related requests"""
        try:
            request_type = request.get('type', '')
            
            if request_type == 'analyze_code_quality':
                return await self._analyze_code_quality(request)
            elif request_type == 'setup_linting':
                return await self._setup_linting(request)
            elif request_type == 'format_code':
                return await self._format_code(request)
            elif request_type == 'review_code':
                return await self._review_code(request)
            elif request_type == 'analyze_performance':
                return await self._analyze_performance(request)
            elif request_type == 'check_security':
                return await self._check_security(request)
            elif request_type == 'refactor_suggestions':
                return await self._provide_refactor_suggestions(request)
            elif request_type == 'setup_ci_quality_checks':
                return await self._setup_ci_quality_checks(request)
            else:
                return await self._comprehensive_quality_analysis(request)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Code quality processing failed: {str(e)}",
                'agent_id': self.agent_id
            }

    async def _analyze_code_quality(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive code quality analysis"""
        code = request.get('code', '')
        file_path = request.get('file_path', '')
        analysis_type = request.get('analysis_type', 'comprehensive')
        
        quality_report = {
            'overall_score': 0,
            'metrics': {},
            'issues': [],
            'suggestions': [],
            'complexity_analysis': {}
        }
        
        # Analyze different aspects
        if analysis_type in ['comprehensive', 'complexity']:
            complexity_analysis = self._analyze_complexity(code)
            quality_report['complexity_analysis'] = complexity_analysis
        
        if analysis_type in ['comprehensive', 'style']:
            style_issues = self._analyze_code_style(code)
            quality_report['issues'].extend(style_issues)
        
        if analysis_type in ['comprehensive', 'performance']:
            performance_issues = self._analyze_performance_issues(code)
            quality_report['issues'].extend(performance_issues)
        
        if analysis_type in ['comprehensive', 'maintainability']:
            maintainability_score = self._calculate_maintainability_score(code)
            quality_report['metrics']['maintainability'] = maintainability_score
        
        # Calculate overall score
        quality_report['overall_score'] = self._calculate_overall_quality_score(quality_report)
        
        # Generate improvement suggestions
        quality_report['suggestions'] = self._generate_improvement_suggestions(quality_report)
        
        return {
            'success': True,
            'quality_report': quality_report,
            'file_path': file_path,
            'analysis_timestamp': self._get_timestamp(),
            'recommendations': self._get_quality_recommendations(quality_report),
            'agent_id': self.agent_id
        }

    async def _setup_linting(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Setup comprehensive linting configuration"""
        project_type = request.get('project_type', 'flutter_app')
        strictness_level = request.get('strictness_level', 'recommended')
        custom_rules = request.get('custom_rules', [])
        
        # Generate analysis_options.yaml
        analysis_options = self._generate_analysis_options(strictness_level, custom_rules)
        
        # Generate custom lint rules
        custom_lint_rules = self._generate_custom_lint_rules(project_type)
        
        # Generate pre-commit hooks
        pre_commit_config = self._generate_pre_commit_hooks()
        
        # Generate IDE configurations
        ide_configs = self._generate_ide_configurations()
        
        return {
            'success': True,
            'analysis_options': analysis_options,
            'custom_lint_rules': custom_lint_rules,
            'pre_commit_config': pre_commit_config,
            'ide_configs': ide_configs,
            'setup_instructions': self._get_linting_setup_instructions(),
            'agent_id': self.agent_id
        }

    async def _format_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Format code according to Dart style guidelines"""
        code = request.get('code', '')
        formatting_rules = request.get('formatting_rules', 'dart_style')
        
        # Apply dart formatting rules
        formatted_code = self._apply_dart_formatting(code)
        
        # Apply custom formatting rules if specified
        if formatting_rules != 'dart_style':
            formatted_code = self._apply_custom_formatting(formatted_code, formatting_rules)
        
        # Generate formatting report
        formatting_changes = self._analyze_formatting_changes(code, formatted_code)
        
        return {
            'success': True,
            'formatted_code': formatted_code,
            'formatting_changes': formatting_changes,
            'formatting_rules_applied': formatting_rules,
            'formatting_stats': self._get_formatting_stats(formatting_changes),
            'agent_id': self.agent_id
        }

    async def _review_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated code review"""
        code = request.get('code', '')
        review_scope = request.get('scope', 'comprehensive')
        context = request.get('context', {})
        
        review_results = {
            'critical_issues': [],
            'major_issues': [],
            'minor_issues': [],
            'suggestions': [],
            'positive_feedback': []
        }
        
        # Analyze different review aspects
        if review_scope in ['comprehensive', 'architecture']:
            architecture_issues = self._review_architecture(code, context)
            review_results['major_issues'].extend(architecture_issues)
        
        if review_scope in ['comprehensive', 'security']:
            security_issues = self._review_security(code)
            review_results['critical_issues'].extend(security_issues)
        
        if review_scope in ['comprehensive', 'performance']:
            performance_issues = self._review_performance(code)
            review_results['major_issues'].extend(performance_issues)
        
        if review_scope in ['comprehensive', 'best_practices']:
            best_practice_issues = self._review_best_practices(code)
            review_results['minor_issues'].extend(best_practice_issues)
        
        # Generate positive feedback
        review_results['positive_feedback'] = self._identify_good_practices(code)
        
        # Generate overall review summary
        review_summary = self._generate_review_summary(review_results)
        
        return {
            'success': True,
            'review_results': review_results,
            'review_summary': review_summary,
            'priority_actions': self._prioritize_review_actions(review_results),
            'approval_status': self._determine_approval_status(review_results),
            'agent_id': self.agent_id
        }

    async def _analyze_performance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for performance issues"""
        code = request.get('code', '')
        widget_tree = request.get('widget_tree', {})
        
        performance_analysis = {
            'build_performance': [],
            'memory_issues': [],
            'rendering_issues': [],
            'async_issues': [],
            'optimization_suggestions': []
        }
        
        # Analyze build method performance
        build_issues = self._analyze_build_performance(code)
        performance_analysis['build_performance'] = build_issues
        
        # Analyze memory usage patterns
        memory_issues = self._analyze_memory_patterns(code)
        performance_analysis['memory_issues'] = memory_issues
        
        # Analyze rendering performance
        rendering_issues = self._analyze_rendering_performance(code)
        performance_analysis['rendering_issues'] = rendering_issues
        
        # Analyze async operations
        async_issues = self._analyze_async_performance(code)
        performance_analysis['async_issues'] = async_issues
        
        # Generate optimization suggestions
        optimizations = self._generate_performance_optimizations(performance_analysis)
        performance_analysis['optimization_suggestions'] = optimizations
        
        return {
            'success': True,
            'performance_analysis': performance_analysis,
            'performance_score': self._calculate_performance_score(performance_analysis),
            'critical_issues': self._identify_critical_performance_issues(performance_analysis),
            'optimization_priority': self._prioritize_optimizations(optimizations),
            'agent_id': self.agent_id
        }

    async def _check_security(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security analysis on Flutter code"""
        code = request.get('code', '')
        dependencies = request.get('dependencies', [])
        
        security_report = {
            'vulnerabilities': [],
            'security_warnings': [],
            'best_practice_violations': [],
            'dependency_issues': []
        }
        
        # Check for common security issues
        vulnerabilities = self._scan_security_vulnerabilities(code)
        security_report['vulnerabilities'] = vulnerabilities
        
        # Check security best practices
        best_practice_violations = self._check_security_best_practices(code)
        security_report['best_practice_violations'] = best_practice_violations
        
        # Analyze dependencies for known vulnerabilities
        dependency_issues = self._analyze_dependency_security(dependencies)
        security_report['dependency_issues'] = dependency_issues
        
        # Generate security recommendations
        security_recommendations = self._generate_security_recommendations(security_report)
        
        return {
            'success': True,
            'security_report': security_report,
            'security_score': self._calculate_security_score(security_report),
            'recommendations': security_recommendations,
            'compliance_status': self._check_security_compliance(security_report),
            'agent_id': self.agent_id
        }

    async def _provide_refactor_suggestions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Provide refactoring suggestions"""
        code = request.get('code', '')
        refactor_type = request.get('type', 'general')
        
        refactor_suggestions = []
        
        if refactor_type in ['general', 'extract_method']:
            method_extractions = self._suggest_method_extractions(code)
            refactor_suggestions.extend(method_extractions)
        
        if refactor_type in ['general', 'extract_widget']:
            widget_extractions = self._suggest_widget_extractions(code)
            refactor_suggestions.extend(widget_extractions)
        
        if refactor_type in ['general', 'simplify']:
            simplifications = self._suggest_code_simplifications(code)
            refactor_suggestions.extend(simplifications)
        
        if refactor_type in ['general', 'design_patterns']:
            pattern_suggestions = self._suggest_design_patterns(code)
            refactor_suggestions.extend(pattern_suggestions)
        
        # Generate refactored code examples
        refactored_examples = self._generate_refactored_examples(refactor_suggestions)
        
        return {
            'success': True,
            'refactor_suggestions': refactor_suggestions,
            'refactored_examples': refactored_examples,
            'impact_analysis': self._analyze_refactor_impact(refactor_suggestions),
            'priority_order': self._prioritize_refactoring(refactor_suggestions),
            'agent_id': self.agent_id
        }

    def _generate_analysis_options(self, strictness_level: str, custom_rules: List[str]) -> str:
        """Generate analysis_options.yaml configuration"""
        base_rules = {
            'basic': [
                'prefer_const_constructors',
                'prefer_final_fields',
                'avoid_print'
            ],
            'recommended': [
                'prefer_const_constructors',
                'prefer_final_fields', 
                'avoid_print',
                'prefer_single_quotes',
                'use_key_in_widget_constructors',
                'prefer_const_literals_to_create_immutables',
                'avoid_unnecessary_containers',
                'sized_box_for_whitespace'
            ],
            'strict': [
                'prefer_const_constructors',
                'prefer_final_fields',
                'avoid_print',
                'prefer_single_quotes',
                'use_key_in_widget_constructors', 
                'prefer_const_literals_to_create_immutables',
                'avoid_unnecessary_containers',
                'sized_box_for_whitespace',
                'prefer_const_declarations',
                'unnecessary_brace_in_string_interps',
                'prefer_collection_literals',
                'prefer_spread_collections'
            ]
        }
        
        rules = base_rules.get(strictness_level, base_rules['recommended'])
        rules.extend(custom_rules)
        
        rules_yaml = '\n    '.join([f"{rule}: true" for rule in rules])
        
        return f'''
# Analysis options for Flutter project
include: package:flutter_lints/flutter.yaml

analyzer:
  exclude:
    - build/**
    - lib/generated/**
    - lib/**/*.g.dart
    - lib/**/*.freezed.dart
  strong-mode:
    implicit-casts: false
    implicit-dynamic: false
  language:
    strict-casts: true
    strict-inference: true
    strict-raw-types: true

linter:
  rules:
    {rules_yaml}

# Custom rules for this project
dart_code_metrics:
  metrics:
    cyclomatic-complexity: 20
    maximum-nesting-level: 5
    number-of-parameters: 4
    source-lines-of-code: 50
  rules:
    - no-boolean-literal-compare
    - no-empty-block
    - prefer-trailing-comma
    - prefer-conditional-expressions
    - no-equal-then-else
'''

    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        # Count cyclomatic complexity
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(code)
        
        # Count nesting levels
        max_nesting = self._calculate_max_nesting_level(code)
        
        # Count lines of code
        lines_of_code = len([line for line in code.split('\n') if line.strip()])
        
        # Count number of methods/functions
        method_count = len(re.findall(r'^\s*\w+\s+\w+\s*\([^)]*\)\s*{', code, re.MULTILINE))
        
        return {
            'cyclomatic_complexity': cyclomatic_complexity,
            'max_nesting_level': max_nesting,
            'lines_of_code': lines_of_code,
            'method_count': method_count,
            'complexity_rating': self._rate_complexity(cyclomatic_complexity, max_nesting)
        }

    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        # Count decision points
        decision_keywords = ['if', 'else if', 'while', 'for', 'case', 'catch', '&&', '||', '?']
        complexity = 1  # Base complexity
        
        for keyword in decision_keywords:
            if keyword in ['&&', '||', '?']:
                complexity += code.count(keyword)
            else:
                complexity += len(re.findall(rf'\b{keyword}\b', code))
        
        return complexity

    def _analyze_code_style(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code style issues"""
        issues = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 80:
                issues.append({
                    'type': 'style',
                    'severity': 'minor',
                    'message': f'Line {i} exceeds 80 characters',
                    'line': i,
                    'suggestion': 'Consider breaking long lines'
                })
            
            # Check for double quotes (prefer single quotes)
            if '"' in line and not line.strip().startswith('//'):
                issues.append({
                    'type': 'style',
                    'severity': 'minor', 
                    'message': f'Line {i} uses double quotes, prefer single quotes',
                    'line': i,
                    'suggestion': 'Use single quotes for strings'
                })
        
        return issues

    def _generate_improvement_suggestions(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on quality report"""
        suggestions = []
        
        # Complexity-based suggestions
        if quality_report.get('complexity_analysis', {}).get('cyclomatic_complexity', 0) > 10:
            suggestions.append("Consider breaking down complex methods into smaller functions")
        
        if quality_report.get('complexity_analysis', {}).get('max_nesting_level', 0) > 4:
            suggestions.append("Reduce nesting levels by using early returns or guard clauses")
        
        # Performance suggestions
        performance_issues = [issue for issue in quality_report.get('issues', []) 
                            if issue.get('type') == 'performance']
        if performance_issues:
            suggestions.append("Address performance issues to improve app responsiveness")
        
        # Style suggestions
        style_issues = [issue for issue in quality_report.get('issues', []) 
                       if issue.get('type') == 'style']
        if len(style_issues) > 5:
            suggestions.append("Run dart format to fix style inconsistencies")
        
        return suggestions

    def _get_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Get quality improvement recommendations"""
        return [
            "Set up automated code formatting with 'dart format'",
            "Configure pre-commit hooks for code quality checks",
            "Integrate static analysis into CI/CD pipeline",
            "Regularly review and update linting rules",
            "Use code metrics to track quality improvements over time",
            "Implement peer code review process",
            "Set up automated testing for quality gates",
            "Document coding standards and best practices"
        ]

    async def _comprehensive_quality_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality analysis"""
        project_path = request.get('project_path', '')
        analysis_scope = request.get('scope', 'full')
        
        comprehensive_report = {
            'project_overview': {},
            'quality_metrics': {},
            'technical_debt': {},
            'recommendations': []
        }
        
        # Analyze project structure
        project_analysis = self._analyze_project_structure(project_path)
        comprehensive_report['project_overview'] = project_analysis
        
        # Calculate quality metrics
        quality_metrics = self._calculate_project_quality_metrics(project_path)
        comprehensive_report['quality_metrics'] = quality_metrics
        
        # Assess technical debt
        technical_debt = self._assess_technical_debt(project_path)
        comprehensive_report['technical_debt'] = technical_debt
        
        # Generate actionable recommendations
        recommendations = self._generate_comprehensive_recommendations(comprehensive_report)
        comprehensive_report['recommendations'] = recommendations
        
        return {
            'success': True,
            'comprehensive_report': comprehensive_report,
            'quality_score': self._calculate_project_quality_score(comprehensive_report),
            'improvement_roadmap': self._create_improvement_roadmap(comprehensive_report),
            'agent_id': self.agent_id
        }
