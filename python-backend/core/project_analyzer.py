"""
Flutter project analysis engine for comprehensive project understanding.
Analyzes project structure, dependencies, architecture patterns, and maturity.
"""

import asyncio
import logging
import os
import re
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .agent_types import ProjectContext, ProjectState

logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    path: str
    size: int
    lines: int
    language: str
    complexity_score: float = 0.0
    test_coverage: float = 0.0
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyInfo:
    """Information about a project dependency."""
    name: str
    version: str
    source: str  # "pub.dev", "git", "path", etc.
    category: str  # "production", "dev", "override"
    description: Optional[str] = None
    latest_version: Optional[str] = None
    is_outdated: bool = False
    security_issues: List[str] = field(default_factory=list)


@dataclass
class ArchitectureInfo:
    """Architecture pattern analysis results."""
    pattern: Optional[str] = None  # "clean", "mvvm", "mvc", "layered", etc.
    state_management: Optional[str] = None  # "bloc", "provider", "riverpod", etc.
    navigation_pattern: Optional[str] = None  # "go_router", "auto_route", "navigator", etc.
    di_pattern: Optional[str] = None  # "get_it", "injectable", "provider", etc.
    confidence_score: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Code quality metrics."""
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
    code_duplication: float = 0.0
    technical_debt_ratio: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0


class ProjectAnalysisEngine:
    """
    Comprehensive Flutter project analysis engine.
    Analyzes project structure, dependencies, architecture, and quality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Analysis patterns
        self.architecture_patterns = self._load_architecture_patterns()
        self.state_management_patterns = self._load_state_management_patterns()
        self.quality_thresholds = self._load_quality_thresholds()
        
        # File type mappings
        self.dart_file_extensions = {'.dart'}
        self.config_files = {'pubspec.yaml', 'analysis_options.yaml', 'build.yaml'}
        self.ignore_patterns = {'.git', '.dart_tool', 'build', '.flutter-plugins*'}
        
        logger.info("ProjectAnalysisEngine initialized")
    
    async def analyze_project(self, project_path: str) -> ProjectContext:
        """
        Perform comprehensive project analysis.
        
        Args:
            project_path: Path to the Flutter project
            
        Returns:
            ProjectContext with complete analysis results
        """
        try:
            project_path = Path(project_path).resolve()
            if not project_path.exists():
                raise FileNotFoundError(f"Project path not found: {project_path}")
            
            logger.info(f"Starting project analysis: {project_path}")
            
            # Initialize project context
            context = ProjectContext(
                project_path=str(project_path),
                project_name=project_path.name
            )
            
            # Run analysis tasks in parallel
            analysis_tasks = [
                self._analyze_pubspec(project_path),
                self._analyze_project_structure(project_path),
                self._analyze_dart_files(project_path),
                self._analyze_architecture(project_path),
                self._analyze_dependencies(project_path),
                self._analyze_tests(project_path),
                self._analyze_quality_metrics(project_path)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            pubspec_info = results[0] if not isinstance(results[0], Exception) else {}
            structure_info = results[1] if not isinstance(results[1], Exception) else {}
            dart_analysis = results[2] if not isinstance(results[2], Exception) else {}
            arch_info = results[3] if not isinstance(results[3], Exception) else ArchitectureInfo()
            deps_info = results[4] if not isinstance(results[4], Exception) else {}
            test_info = results[5] if not isinstance(results[5], Exception) else {}
            quality_info = results[6] if not isinstance(results[6], Exception) else QualityMetrics()
            
            # Populate context with analysis results
            await self._populate_context(
                context, pubspec_info, structure_info, dart_analysis,
                arch_info, deps_info, test_info, quality_info
            )
            
            # Determine project state
            context.project_state = await self._determine_project_state(context)
            
            context.last_analyzed = datetime.now()
            
            logger.info(f"Project analysis completed: {context.project_name}")
            
            return context
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            raise
    
    async def _analyze_pubspec(self, project_path: Path) -> Dict[str, Any]:
        """Analyze pubspec.yaml file."""
        try:
            pubspec_path = project_path / "pubspec.yaml"
            if not pubspec_path.exists():
                return {"error": "pubspec.yaml not found"}
            
            with open(pubspec_path, 'r', encoding='utf-8') as file:
                pubspec_data = yaml.safe_load(file)
            
            # Extract basic information
            info = {
                "name": pubspec_data.get("name", ""),
                "version": pubspec_data.get("version", ""),
                "description": pubspec_data.get("description", ""),
                "flutter_version": pubspec_data.get("environment", {}).get("flutter", ""),
                "dart_version": pubspec_data.get("environment", {}).get("sdk", ""),
                "dependencies": pubspec_data.get("dependencies", {}),
                "dev_dependencies": pubspec_data.get("dev_dependencies", {}),
                "dependency_overrides": pubspec_data.get("dependency_overrides", {}),
                "platforms": []
            }
            
            # Detect supported platforms
            flutter_config = pubspec_data.get("flutter", {})
            if flutter_config:
                # Check for platform-specific configurations
                if "android" in str(pubspec_data).lower():
                    info["platforms"].append("android")
                if "ios" in str(pubspec_data).lower():
                    info["platforms"].append("ios")
                if "web" in str(pubspec_data).lower():
                    info["platforms"].append("web")
                if "windows" in str(pubspec_data).lower():
                    info["platforms"].append("windows")
                if "macos" in str(pubspec_data).lower():
                    info["platforms"].append("macos")
                if "linux" in str(pubspec_data).lower():
                    info["platforms"].append("linux")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to analyze pubspec.yaml: {e}")
            return {"error": str(e)}
    
    async def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project directory structure."""
        try:
            structure_info = {
                "total_files": 0,
                "dart_files": 0,
                "test_files": 0,
                "asset_files": 0,
                "directories": [],
                "has_lib": False,
                "has_test": False,
                "has_android": False,
                "has_ios": False,
                "has_web": False,
                "has_assets": False,
                "structure_score": 0.0
            }
            
            # Walk through project directory
            for root, dirs, files in os.walk(project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.ignore_patterns)]
                
                rel_root = Path(root).relative_to(project_path)
                
                # Track important directories
                if rel_root.name == "lib":
                    structure_info["has_lib"] = True
                elif rel_root.name == "test":
                    structure_info["has_test"] = True
                elif rel_root.name == "android":
                    structure_info["has_android"] = True
                elif rel_root.name == "ios":
                    structure_info["has_ios"] = True
                elif rel_root.name == "web":
                    structure_info["has_web"] = True
                elif rel_root.name == "assets":
                    structure_info["has_assets"] = True
                
                # Count files by type
                for file in files:
                    structure_info["total_files"] += 1
                    
                    file_path = Path(root) / file
                    if file_path.suffix == ".dart":
                        if "test" in str(file_path):
                            structure_info["test_files"] += 1
                        else:
                            structure_info["dart_files"] += 1
                    elif file_path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".json"}:
                        structure_info["asset_files"] += 1
            
            # Calculate structure score
            structure_info["structure_score"] = self._calculate_structure_score(structure_info)
            
            return structure_info
            
        except Exception as e:
            logger.error(f"Failed to analyze project structure: {e}")
            return {"error": str(e)}
    
    async def _analyze_dart_files(self, project_path: Path) -> Dict[str, Any]:
        """Analyze Dart source files."""
        try:
            dart_analysis = {
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
                "files": [],
                "complexity_issues": [],
                "import_analysis": {},
                "widget_analysis": {}
            }
            
            lib_path = project_path / "lib"
            if not lib_path.exists():
                return dart_analysis
            
            # Analyze each Dart file
            for dart_file in lib_path.rglob("*.dart"):
                file_analysis = await self._analyze_dart_file(dart_file)
                dart_analysis["files"].append(file_analysis)
                
                # Aggregate metrics
                dart_analysis["total_lines"] += file_analysis.lines
                dart_analysis["code_lines"] += file_analysis.metrics.get("code_lines", 0)
                dart_analysis["comment_lines"] += file_analysis.metrics.get("comment_lines", 0)
                dart_analysis["blank_lines"] += file_analysis.metrics.get("blank_lines", 0)
                
                # Collect complexity issues
                if file_analysis.complexity_score > 10:  # High complexity threshold
                    dart_analysis["complexity_issues"].append({
                        "file": str(dart_file),
                        "score": file_analysis.complexity_score
                    })
            
            return dart_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze Dart files: {e}")
            return {"error": str(e)}
    
    async def _analyze_dart_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Dart file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            lines = content.split('\n')
            total_lines = len(lines)
            
            # Basic metrics
            code_lines = 0
            comment_lines = 0
            blank_lines = 0
            complexity_score = 0
            
            # Analyze each line
            in_multiline_comment = False
            for line in lines:
                stripped_line = line.strip()
                
                if not stripped_line:
                    blank_lines += 1
                elif stripped_line.startswith('//'):
                    comment_lines += 1
                elif '/*' in stripped_line and '*/' in stripped_line:
                    comment_lines += 1
                elif '/*' in stripped_line:
                    in_multiline_comment = True
                    comment_lines += 1
                elif '*/' in stripped_line:
                    in_multiline_comment = False
                    comment_lines += 1
                elif in_multiline_comment:
                    comment_lines += 1
                else:
                    code_lines += 1
                    
                    # Calculate complexity (simplified cyclomatic complexity)
                    complexity_keywords = ['if', 'else', 'while', 'for', 'case', 'catch', '&&', '||', '?']
                    for keyword in complexity_keywords:
                        if keyword in stripped_line:
                            complexity_score += 1
            
            # Detect file type and patterns
            language = "dart"
            
            # Detect if it's a widget file
            is_widget = 'extends StatelessWidget' in content or 'extends StatefulWidget' in content
            
            # Detect if it's a test file
            is_test = 'test(' in content or 'testWidgets(' in content
            
            return FileAnalysis(
                path=str(file_path),
                size=len(content),
                lines=total_lines,
                language=language,
                complexity_score=complexity_score,
                metrics={
                    "code_lines": code_lines,
                    "comment_lines": comment_lines,
                    "blank_lines": blank_lines,
                    "is_widget": is_widget,
                    "is_test": is_test
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return FileAnalysis(
                path=str(file_path),
                size=0,
                lines=0,
                language="unknown",
                issues=[str(e)]
            )
    
    async def _analyze_architecture(self, project_path: Path) -> ArchitectureInfo:
        """Analyze project architecture patterns."""
        try:
            arch_info = ArchitectureInfo()
            evidence = []
            
            lib_path = project_path / "lib"
            if not lib_path.exists():
                return arch_info
            
            # Check for clean architecture
            clean_arch_indicators = ["domain", "data", "presentation", "core"]
            clean_arch_score = 0
            
            for indicator in clean_arch_indicators:
                if (lib_path / indicator).exists():
                    clean_arch_score += 1
                    evidence.append(f"Found {indicator} directory")
            
            if clean_arch_score >= 3:
                arch_info.pattern = "clean"
                arch_info.confidence_score = clean_arch_score / len(clean_arch_indicators)
            
            # Check for state management patterns
            state_mgmt_patterns = {
                "bloc": ["bloc", "cubit", "flutter_bloc"],
                "provider": ["provider", "change_notifier"],
                "riverpod": ["riverpod", "provider_container"],
                "getx": ["getx", "get"],
                "redux": ["redux", "flutter_redux"]
            }
            
            # Read pubspec.yaml to check dependencies
            pubspec_path = project_path / "pubspec.yaml"
            if pubspec_path.exists():
                with open(pubspec_path, 'r') as file:
                    pubspec_content = file.read().lower()
                
                for pattern, indicators in state_mgmt_patterns.items():
                    for indicator in indicators:
                        if indicator in pubspec_content:
                            arch_info.state_management = pattern
                            evidence.append(f"Found {indicator} dependency")
                            break
                    if arch_info.state_management:
                        break
            
            # Check for navigation patterns
            nav_patterns = {
                "go_router": ["go_router"],
                "auto_route": ["auto_route"],
                "fluro": ["fluro"]
            }
            
            if pubspec_path.exists():
                with open(pubspec_path, 'r') as file:
                    pubspec_content = file.read().lower()
                
                for pattern, indicators in nav_patterns.items():
                    for indicator in indicators:
                        if indicator in pubspec_content:
                            arch_info.navigation_pattern = pattern
                            evidence.append(f"Found {indicator} navigation")
                            break
                    if arch_info.navigation_pattern:
                        break
            
            arch_info.evidence = evidence
            
            return arch_info
            
        except Exception as e:
            logger.error(f"Failed to analyze architecture: {e}")
            return ArchitectureInfo()
    
    async def _analyze_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies."""
        try:
            deps_info = {
                "total_dependencies": 0,
                "dev_dependencies": 0,
                "production_dependencies": 0,
                "outdated_dependencies": [],
                "security_issues": [],
                "dependency_categories": {}
            }
            
            pubspec_path = project_path / "pubspec.yaml"
            if not pubspec_path.exists():
                return deps_info
            
            with open(pubspec_path, 'r') as file:
                pubspec_data = yaml.safe_load(file)
            
            # Analyze production dependencies
            dependencies = pubspec_data.get("dependencies", {})
            deps_info["production_dependencies"] = len(dependencies)
            
            # Analyze dev dependencies
            dev_dependencies = pubspec_data.get("dev_dependencies", {})
            deps_info["dev_dependencies"] = len(dev_dependencies)
            
            deps_info["total_dependencies"] = deps_info["production_dependencies"] + deps_info["dev_dependencies"]
            
            # Categorize dependencies
            ui_deps = ["flutter", "cupertino_icons", "material_design_icons"]
            state_deps = ["bloc", "provider", "riverpod", "getx", "redux"]
            network_deps = ["http", "dio", "graphql"]
            storage_deps = ["shared_preferences", "sqflite", "hive"]
            
            categories = {
                "ui": 0,
                "state_management": 0,
                "networking": 0,
                "storage": 0,
                "other": 0
            }
            
            all_deps = {**dependencies, **dev_dependencies}
            for dep_name in all_deps.keys():
                if any(ui_dep in dep_name for ui_dep in ui_deps):
                    categories["ui"] += 1
                elif any(state_dep in dep_name for state_dep in state_deps):
                    categories["state_management"] += 1
                elif any(net_dep in dep_name for net_dep in network_deps):
                    categories["networking"] += 1
                elif any(storage_dep in dep_name for storage_dep in storage_deps):
                    categories["storage"] += 1
                else:
                    categories["other"] += 1
            
            deps_info["dependency_categories"] = categories
            
            return deps_info
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return {"error": str(e)}
    
    async def _analyze_tests(self, project_path: Path) -> Dict[str, Any]:
        """Analyze test files and coverage."""
        try:
            test_info = {
                "has_tests": False,
                "unit_tests": 0,
                "widget_tests": 0,
                "integration_tests": 0,
                "test_coverage": 0.0,
                "test_files": []
            }
            
            test_path = project_path / "test"
            if not test_path.exists():
                return test_info
            
            test_info["has_tests"] = True
            
            # Analyze test files
            for test_file in test_path.rglob("*.dart"):
                with open(test_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Count different types of tests
                unit_test_count = content.count('test(')
                widget_test_count = content.count('testWidgets(')
                integration_test_count = content.count('IntegrationTestWidgetsFlutterBinding')
                
                test_info["unit_tests"] += unit_test_count
                test_info["widget_tests"] += widget_test_count
                test_info["integration_tests"] += integration_test_count
                
                test_info["test_files"].append({
                    "path": str(test_file),
                    "unit_tests": unit_test_count,
                    "widget_tests": widget_test_count,
                    "integration_tests": integration_test_count
                })
            
            # Try to get test coverage (if available)
            coverage_file = project_path / "coverage" / "lcov.info"
            if coverage_file.exists():
                test_info["test_coverage"] = await self._parse_coverage_file(coverage_file)
            
            return test_info
            
        except Exception as e:
            logger.error(f"Failed to analyze tests: {e}")
            return {"error": str(e)}
    
    async def _analyze_quality_metrics(self, project_path: Path) -> QualityMetrics:
        """Analyze code quality metrics."""
        try:
            metrics = QualityMetrics()
            
            # Run dart analyze if available
            try:
                result = await asyncio.create_subprocess_exec(
                    'dart', 'analyze', str(project_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0:
                    # Parse analyzer output for quality metrics
                    output = stdout.decode('utf-8')
                    # Implementation would parse dart analyze output
                    metrics.complexity_score = 7.5  # Placeholder
                    metrics.maintainability_index = 8.0  # Placeholder
                
            except FileNotFoundError:
                logger.warning("Dart analyzer not found, skipping static analysis")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze quality metrics: {e}")
            return QualityMetrics()
    
    async def _determine_project_state(self, context: ProjectContext) -> ProjectState:
        """Determine the maturity state of the project."""
        try:
            score = 0
            
            # Scoring criteria
            if context.file_count > 0:
                score += 1
            if context.has_tests:
                score += 2
            if context.dependencies:
                score += 1
            if context.architecture_pattern:
                score += 2
            if context.line_count > 1000:
                score += 1
            if context.test_coverage > 0.5:
                score += 2
            if context.has_documentation:
                score += 1
            
            # Determine state based on score
            if score <= 2:
                return ProjectState.NEW
            elif score <= 4:
                return ProjectState.INITIALIZED
            elif score <= 6:
                return ProjectState.MID_DEVELOPMENT
            elif score <= 8:
                return ProjectState.MATURE
            else:
                return ProjectState.PRODUCTION
            
        except Exception as e:
            logger.error(f"Failed to determine project state: {e}")
            return ProjectState.NEW
    
    async def _populate_context(
        self, 
        context: ProjectContext,
        pubspec_info: Dict[str, Any],
        structure_info: Dict[str, Any],
        dart_analysis: Dict[str, Any],
        arch_info: ArchitectureInfo,
        deps_info: Dict[str, Any],
        test_info: Dict[str, Any],
        quality_info: QualityMetrics
    ):
        """Populate project context with analysis results."""
        
        # Basic project info
        if not isinstance(pubspec_info, dict) or "error" not in pubspec_info:
            context.flutter_version = pubspec_info.get("flutter_version")
            context.dart_version = pubspec_info.get("dart_version")
            context.dependencies = pubspec_info.get("dependencies", {})
            context.dev_dependencies = pubspec_info.get("dev_dependencies", {})
            context.platforms = pubspec_info.get("platforms", [])
        
        # Structure info
        if not isinstance(structure_info, dict) or "error" not in structure_info:
            context.file_count = structure_info.get("total_files", 0)
        
        # Dart analysis
        if not isinstance(dart_analysis, dict) or "error" not in dart_analysis:
            context.line_count = dart_analysis.get("total_lines", 0)
        
        # Architecture info
        context.architecture_pattern = arch_info.pattern
        context.state_management = arch_info.state_management
        
        # Test info
        if not isinstance(test_info, dict) or "error" not in test_info:
            context.has_tests = test_info.get("has_tests", False)
            context.test_coverage = test_info.get("test_coverage", 0.0)
        
        # Quality metrics
        context.code_quality_score = quality_info.complexity_score
        context.complexity_score = quality_info.complexity_score
        context.maintainability_index = quality_info.maintainability_index
        
        # Features detection
        context.features = await self._detect_features(context)
        
        # Documentation check
        context.has_documentation = await self._check_documentation(Path(context.project_path))
        
        # CI/CD check
        context.has_ci_cd = await self._check_ci_cd(Path(context.project_path))
    
    async def _detect_features(self, context: ProjectContext) -> List[str]:
        """Detect project features based on dependencies and structure."""
        features = []
        
        # Check dependencies for common features
        deps = {**context.dependencies, **context.dev_dependencies}
        
        feature_mapping = {
            "authentication": ["firebase_auth", "google_sign_in", "auth0"],
            "database": ["firebase_firestore", "sqflite", "hive", "realm"],
            "http_client": ["http", "dio", "chopper"],
            "state_management": ["bloc", "provider", "riverpod", "getx"],
            "navigation": ["go_router", "auto_route", "fluro"],
            "ui_components": ["flutter_svg", "cached_network_image", "shimmer"],
            "animations": ["lottie", "rive", "flutter_staggered_animations"],
            "localization": ["flutter_localizations", "intl", "easy_localization"],
            "testing": ["mockito", "bloc_test", "flutter_driver"],
            "analytics": ["firebase_analytics", "google_analytics"],
            "crash_reporting": ["firebase_crashlytics", "sentry"],
            "push_notifications": ["firebase_messaging", "flutter_local_notifications"]
        }
        
        for feature, indicators in feature_mapping.items():
            if any(indicator in deps for indicator in indicators):
                features.append(feature)
        
        return features
    
    async def _check_documentation(self, project_path: Path) -> bool:
        """Check if project has documentation."""
        doc_files = ["README.md", "CHANGELOG.md", "doc/", "docs/"]
        
        for doc_file in doc_files:
            if (project_path / doc_file).exists():
                return True
        
        return False
    
    async def _check_ci_cd(self, project_path: Path) -> bool:
        """Check if project has CI/CD configuration."""
        ci_files = [
            ".github/workflows/",
            ".gitlab-ci.yml",
            "azure-pipelines.yml",
            "bitbucket-pipelines.yml",
            "circle.yml",
            ".travis.yml"
        ]
        
        for ci_file in ci_files:
            if (project_path / ci_file).exists():
                return True
        
        return False
    
    async def _parse_coverage_file(self, coverage_file: Path) -> float:
        """Parse LCOV coverage file."""
        try:
            with open(coverage_file, 'r') as file:
                content = file.read()
            
            # Simple LCOV parsing
            lines_found = 0
            lines_hit = 0
            
            for line in content.split('\n'):
                if line.startswith('LF:'):
                    lines_found += int(line.split(':')[1])
                elif line.startswith('LH:'):
                    lines_hit += int(line.split(':')[1])
            
            if lines_found > 0:
                return (lines_hit / lines_found) * 100
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to parse coverage file: {e}")
            return 0.0
    
    def _calculate_structure_score(self, structure_info: Dict[str, Any]) -> float:
        """Calculate project structure quality score."""
        score = 0.0
        max_score = 10.0
        
        # Check for essential directories
        if structure_info.get("has_lib"):
            score += 2.0
        if structure_info.get("has_test"):
            score += 2.0
        if structure_info.get("has_assets"):
            score += 1.0
        
        # Check for platform support
        platform_dirs = ["has_android", "has_ios", "has_web"]
        platform_count = sum(1 for p in platform_dirs if structure_info.get(p))
        score += min(platform_count * 1.0, 3.0)
        
        # File count balance
        dart_files = structure_info.get("dart_files", 0)
        test_files = structure_info.get("test_files", 0)
        
        if dart_files > 0:
            test_ratio = test_files / dart_files
            if test_ratio > 0.5:
                score += 2.0
            elif test_ratio > 0.2:
                score += 1.0
        
        return min(score, max_score)
    
    def _load_architecture_patterns(self) -> Dict[str, Any]:
        """Load architecture pattern definitions."""
        return {
            "clean": {
                "directories": ["domain", "data", "presentation"],
                "files": ["entity", "repository", "usecase"]
            },
            "mvvm": {
                "directories": ["models", "views", "viewmodels"],
                "files": ["model", "view", "viewmodel"]
            },
            "bloc": {
                "dependencies": ["flutter_bloc", "bloc"],
                "files": ["bloc", "event", "state"]
            }
        }
    
    def _load_state_management_patterns(self) -> Dict[str, Any]:
        """Load state management pattern definitions."""
        return {
            "bloc": ["flutter_bloc", "bloc"],
            "provider": ["provider"],
            "riverpod": ["flutter_riverpod", "riverpod"],
            "getx": ["get"],
            "redux": ["flutter_redux"]
        }
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality threshold values."""
        return {
            "complexity_threshold": 10.0,
            "maintainability_threshold": 70.0,
            "coverage_threshold": 80.0,
            "duplication_threshold": 5.0
        }
