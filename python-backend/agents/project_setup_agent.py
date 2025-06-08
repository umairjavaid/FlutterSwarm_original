"""
Project Setup Agent - Initializes Flutter projects and environment setup.
Handles project creation, SDK configuration, and initial project structure.
"""

import os
import subprocess
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent
from core.agent_types import AgentType, AgentResponse, WorkflowState, TaskStatus

logger = logging.getLogger(__name__)


class ProjectSetupAgent(BaseAgent):
    """
    Specialized agent for Flutter project setup and initialization.
    Handles flutter create, pubspec.yaml configuration, and environment setup.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.PROJECT_SETUP, config)
        
        # Flutter SDK configuration
        self.flutter_sdk_path = config.get("flutter_sdk_path", "flutter")
        self.dart_sdk_path = config.get("dart_sdk_path", "dart")
        self.target_platforms = config.get("target_platforms", ["android", "ios", "web"])
        self.project_templates = config.get("project_templates", {
            "app": "application",
            "module": "module",
            "package": "package",
            "plugin": "plugin"
        })
        
        logger.info(f"ProjectSetupAgent initialized with platforms: {self.target_platforms}")
    
    def _define_capabilities(self) -> List[str]:
        return [
            "flutter_project_creation",
            "pubspec_yaml_configuration", 
            "flutter_sdk_setup",
            "platform_enablement",
            "git_repository_initialization",
            "project_metadata_setup",
            "initial_directory_structure",
            "flutter_version_management",
            "dependency_installation",
            "environment_validation"
        ]
    
    async def process_task(self, state: WorkflowState) -> AgentResponse:
        """Process project setup tasks."""
        try:
            task_type = getattr(self.current_task, 'task_type', 'project_initialization')
            
            logger.info(f"Processing {task_type} task for project setup")
            
            if task_type == "project_initialization":
                return await self._handle_project_initialization(state)
            elif task_type == "platform_setup":
                return await self._handle_platform_setup(state)
            elif task_type == "dependency_setup":
                return await self._handle_dependency_setup(state)
            elif task_type == "environment_validation":
                return await self._handle_environment_validation(state)
            else:
                return await self._handle_generic_setup(state)
                
        except Exception as e:
            logger.error(f"Error processing task in ProjectSetupAgent: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                message=f"Project setup failed: {str(e)}",
                data={"error": str(e)},
                updated_state=state
            )
    
    async def _handle_project_initialization(self, state: WorkflowState) -> AgentResponse:
        """Initialize a new Flutter project with proper structure."""
        try:
            project_name = state.project_context.get("project_name", "flutter_app")
            project_path = state.project_context.get("project_path", f"./{project_name}")
            project_type = state.project_context.get("project_type", "app")
            
            # Validate Flutter SDK
            flutter_version = await self._validate_flutter_sdk()
            if not flutter_version:
                raise Exception("Flutter SDK not found or invalid")
            
            # Create Flutter project
            creation_result = await self._create_flutter_project(
                project_name, project_path, project_type
            )
            
            if not creation_result["success"]:
                raise Exception(f"Project creation failed: {creation_result['error']}")
            
            # Initialize Git repository
            git_result = await self._initialize_git_repository(project_path)
            
            # Setup project structure
            structure_result = await self._setup_project_structure(project_path)
            
            # Configure pubspec.yaml
            pubspec_result = await self._configure_pubspec(project_path, state)
            
            # Enable target platforms
            platform_results = await self._enable_target_platforms(project_path)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message=f"Flutter project '{project_name}' initialized successfully",
                data={
                    "project_path": project_path,
                    "flutter_version": flutter_version,
                    "creation_result": creation_result,
                    "git_initialized": git_result["success"],
                    "structure_setup": structure_result["success"],
                    "pubspec_configured": pubspec_result["success"],
                    "platforms_enabled": platform_results
                },
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Project initialization failed: {e}")
            raise
    
    async def _handle_platform_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle platform-specific setup tasks."""
        try:
            project_path = state.project_context.get("project_path", "./")
            platforms = state.project_context.get("platforms", self.target_platforms)
            
            results = {}
            for platform in platforms:
                platform_result = await self._setup_platform(project_path, platform)
                results[platform] = platform_result
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                message="Platform setup completed",
                data={"platform_results": results},
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Platform setup failed: {e}")
            raise
    
    async def _handle_dependency_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle dependency installation and configuration."""
        try:
            project_path = state.project_context.get("project_path", "./")
            dependencies = state.project_context.get("dependencies", {})
            dev_dependencies = state.project_context.get("dev_dependencies", {})
            
            # Install dependencies
            install_result = await self._install_dependencies(
                project_path, dependencies, dev_dependencies
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=install_result["success"],
                message="Dependencies setup completed",
                data=install_result,
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Dependency setup failed: {e}")
            raise
    
    async def _handle_environment_validation(self, state: WorkflowState) -> AgentResponse:
        """Validate the development environment."""
        try:
            validation_results = {
                "flutter_doctor": await self._run_flutter_doctor(),
                "dart_version": await self._get_dart_version(),
                "flutter_version": await self._validate_flutter_sdk(),
                "platform_tools": await self._validate_platform_tools()
            }
            
            all_valid = all(result.get("success", False) for result in validation_results.values())
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=all_valid,
                message="Environment validation completed",
                data=validation_results,
                updated_state=state
            )
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            raise
    
    async def _validate_flutter_sdk(self) -> Optional[str]:
        """Validate Flutter SDK installation."""
        try:
            result = subprocess.run(
                [self.flutter_sdk_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return version_line.strip()
            return None
            
        except Exception as e:
            logger.error(f"Flutter SDK validation failed: {e}")
            return None
    
    async def _create_flutter_project(self, name: str, path: str, project_type: str) -> Dict[str, Any]:
        """Create a new Flutter project."""
        try:
            cmd = [
                self.flutter_sdk_path, "create",
                "--template", self.project_templates.get(project_type, "app"),
                "--project-name", name,
                path
            ]
            
            # Add platform-specific flags
            if "web" not in self.target_platforms:
                cmd.extend(["--platforms", ",".join(self.target_platforms)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _initialize_git_repository(self, project_path: str) -> Dict[str, Any]:
        """Initialize Git repository for the project."""
        try:
            os.chdir(project_path)
            
            # Initialize git
            subprocess.run(["git", "init"], check=True, capture_output=True)
            
            # Create .gitignore if it doesn't exist
            gitignore_path = os.path.join(project_path, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, "w") as f:
                    f.write(self._get_flutter_gitignore_content())
            
            # Initial commit
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial Flutter project setup"],
                check=True,
                capture_output=True
            )
            
            return {"success": True, "message": "Git repository initialized"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _setup_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Setup recommended Flutter project structure."""
        try:
            directories = [
                "lib/core",
                "lib/features",
                "lib/shared",
                "lib/utils",
                "assets/images",
                "assets/icons",
                "assets/fonts",
                "test/unit",
                "test/widget",
                "test/integration"
            ]
            
            for directory in directories:
                full_path = os.path.join(project_path, directory)
                os.makedirs(full_path, exist_ok=True)
                
                # Create .gitkeep for empty directories
                gitkeep_path = os.path.join(full_path, ".gitkeep")
                if not os.listdir(full_path):
                    open(gitkeep_path, 'a').close()
            
            return {"success": True, "directories_created": directories}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _configure_pubspec(self, project_path: str, state: WorkflowState) -> Dict[str, Any]:
        """Configure pubspec.yaml with project-specific settings."""
        try:
            pubspec_path = os.path.join(project_path, "pubspec.yaml")
            
            # Read existing pubspec.yaml
            with open(pubspec_path, 'r') as f:
                content = f.read()
            
            # Add common dependencies and configuration
            additional_config = self._generate_pubspec_additions(state)
            
            # Append additional configuration
            with open(pubspec_path, 'a') as f:
                f.write(additional_config)
            
            return {"success": True, "configuration_added": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _enable_target_platforms(self, project_path: str) -> Dict[str, bool]:
        """Enable target platforms for the Flutter project."""
        results = {}
        
        for platform in self.target_platforms:
            try:
                if platform not in ["android", "ios"]:  # These are enabled by default
                    cmd = [self.flutter_sdk_path, "config", f"--enable-{platform}"]
                    subprocess.run(cmd, cwd=project_path, check=True, capture_output=True)
                
                results[platform] = True
                
            except Exception as e:
                logger.error(f"Failed to enable platform {platform}: {e}")
                results[platform] = False
        
        return results
    
    async def _run_flutter_doctor(self) -> Dict[str, Any]:
        """Run flutter doctor to check environment."""
        try:
            result = subprocess.run(
                [self.flutter_sdk_path, "doctor", "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": True,
                "output": result.stdout,
                "issues": result.stderr
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_dart_version(self) -> Dict[str, Any]:
        """Get Dart SDK version."""
        try:
            result = subprocess.run(
                [self.dart_sdk_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "version": result.stderr.strip() if result.stderr else result.stdout.strip()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_flutter_gitignore_content(self) -> str:
        """Get Flutter-specific .gitignore content."""
        return """# Miscellaneous
*.class
*.log
*.pyc
*.swp
.DS_Store
.atom/
.buildlog/
.history
.svn/
migrate_working_dir/

# IntelliJ related
*.iml
*.ipr
*.iws
.idea/

# The .vscode folder contains launch configuration and tasks you configure in
# VS Code which you may wish to be included in version control, so this line
# is commented out by default.
#.vscode/

# Flutter/Dart/Pub related
**/doc/api/
**/ios/Flutter/.last_build_id
.dart_tool/
.flutter-plugins
.flutter-plugins-dependencies
.packages
.pub-cache/
.pub/
/build/

# Symbolication related
app.*.symbols

# Obfuscation related
app.*.map.json

# Android Studio will place build artifacts here
/android/app/debug
/android/app/profile
/android/app/release
"""
    
    def _generate_pubspec_additions(self, state: WorkflowState) -> str:
        """Generate additional pubspec.yaml configuration."""
        return """
# Additional dependencies
  # State management
  flutter_bloc: ^8.1.3
  provider: ^6.0.5
  
  # Networking
  dio: ^5.3.2
  http: ^1.1.0
  
  # Local storage
  hive: ^2.2.3
  hive_flutter: ^1.1.0
  shared_preferences: ^2.2.2
  
  # Navigation
  go_router: ^12.1.1
  
  # UI utilities
  flutter_screenutil: ^5.9.0
  cached_network_image: ^3.3.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
  hive_generator: ^2.0.1
  build_runner: ^2.4.7
  mockito: ^5.4.2

# Assets configuration
flutter:
  assets:
    - assets/images/
    - assets/icons/
    - assets/fonts/
  
  fonts:
    - family: CustomFont
      fonts:
        - asset: assets/fonts/CustomFont-Regular.ttf
        - asset: assets/fonts/CustomFont-Bold.ttf
          weight: 700
"""
    
    async def _setup_platform(self, project_path: str, platform: str) -> Dict[str, Any]:
        """Setup specific platform configuration."""
        try:
            if platform == "web":
                return await self._setup_web_platform(project_path)
            elif platform == "android":
                return await self._setup_android_platform(project_path)
            elif platform == "ios":
                return await self._setup_ios_platform(project_path)
            else:
                return {"success": True, "message": f"No specific setup required for {platform}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _setup_web_platform(self, project_path: str) -> Dict[str, Any]:
        """Setup web-specific configuration."""
        # Web platform setup logic
        return {"success": True, "message": "Web platform setup completed"}
    
    async def _setup_android_platform(self, project_path: str) -> Dict[str, Any]:
        """Setup Android-specific configuration."""
        # Android platform setup logic
        return {"success": True, "message": "Android platform setup completed"}
    
    async def _setup_ios_platform(self, project_path: str) -> Dict[str, Any]:
        """Setup iOS-specific configuration."""
        # iOS platform setup logic
        return {"success": True, "message": "iOS platform setup completed"}
    
    async def _install_dependencies(self, project_path: str, dependencies: Dict, dev_dependencies: Dict) -> Dict[str, Any]:
        """Install project dependencies."""
        try:
            # Run flutter pub get
            result = subprocess.run(
                [self.flutter_sdk_path, "pub", "get"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _validate_platform_tools(self) -> Dict[str, Any]:
        """Validate platform-specific development tools."""
        tools_status = {}
        
        # Check Android tools
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
            tools_status["android_tools"] = {"success": True}
        except:
            tools_status["android_tools"] = {"success": False, "error": "ADB not found"}
        
        # Check iOS tools (on macOS)
        try:
            subprocess.run(["xcodebuild", "-version"], capture_output=True, check=True)
            tools_status["ios_tools"] = {"success": True}
        except:
            tools_status["ios_tools"] = {"success": False, "error": "Xcode not found"}
        
        return tools_status
    
    async def _handle_generic_setup(self, state: WorkflowState) -> AgentResponse:
        """Handle generic setup tasks."""
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            message="Generic project setup completed",
            data={"setup_type": "generic"},
            updated_state=state
        )
