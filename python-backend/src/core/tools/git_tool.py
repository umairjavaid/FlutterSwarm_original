"""
Git Tool for FlutterSwarm Multi-Agent System.

Provides comprehensive Git version control operations for Flutter projects.
"""

import os
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re

from .base_tool import BaseTool, ToolCategory, ToolPermission # Added ToolCategory and ToolPermission
from ...models.tool_models import (
    ToolCapability, ToolOperation, ToolResult, ToolUsageExample,
    ToolValidation, ToolMetrics
)
from ...config import get_logger

logger = get_logger("git_tool")


class GitTool(BaseTool):
    """
    Comprehensive Git version control tool for Flutter projects.
    
    Capabilities:
    - Repository initialization and management
    - Branch creation and management
    - Commit operations with Flutter-specific conventions
    - Remote repository operations
    - Git flow and release management
    - Conflict resolution assistance
    - Repository health monitoring
    - Git hooks management
    """
    
    def __init__(self):
        """Initialize the Git tool."""
        super().__init__(
            name="git_tool",
            description="Comprehensive Git version control operations",
            version="1.0.0",
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.FILE_WRITE,
                ToolPermission.FILE_CREATE,
                ToolPermission.DIRECTORY_CREATE,
                ToolPermission.PROCESS_SPAWN
            ]
        )
        
        # Git configuration
        self.flutter_gitignore_template = """
# Miscellaneous
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
**/ios/Flutter/flutter_assets/
**/ios/Flutter/flutter_export_environment.sh
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
        
        # Flutter-specific commit message conventions
        self.commit_conventions = {
            "feat": "A new feature",
            "fix": "A bug fix",
            "docs": "Documentation only changes",
            "style": "Changes that do not affect the meaning of the code",
            "refactor": "A code change that neither fixes a bug nor adds a feature",
            "perf": "A code change that improves performance",
            "test": "Adding missing tests or correcting existing tests",
            "build": "Changes that affect the build system or external dependencies",
            "ci": "Changes to our CI configuration files and scripts",
            "chore": "Other changes that don't modify src or test files",
            "revert": "Reverts a previous commit",
            "flutter": "Flutter-specific changes (widgets, themes, etc.)",
            "dart": "Dart language specific changes",
            "android": "Android platform specific changes",
            "ios": "iOS platform specific changes",
            "web": "Web platform specific changes"
        }

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get Git tool capabilities."""
        return {
            "operations": [
                "init_repository", "clone_repository", "create_branch", "switch_branch",
                "commit_changes", "push_changes", "pull_changes", "merge_branch",
                "create_tag", "get_status", "get_history", "resolve_conflicts", "setup_hooks"
            ],
            "version_control_system": "git",
            "flutter_specific_conventions": True
        }

    async def validate_params(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for a given Git operation."""
        # Basic validation, can be expanded
        if operation == "clone_repository" and "repo_url" not in params:
            return False, "Missing 'repo_url' for clone_repository operation"
        if operation == "create_branch" and "branch_name" not in params:
            return False, "Missing 'branch_name' for create_branch operation"
        # Add more validation rules as needed
        return True, None

    async def execute(
        self, 
        operation: ToolOperation, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute a Git operation."""
        start_time = datetime.now()
        
        try:
            # Validate operation
            validation = await self.validate_operation(operation)
            if not validation.is_valid:
                return ToolResult(
                    tool_name=self.name,
                    operation=operation.operation_type,
                    success=False,
                    result=None,
                    error=f"Validation failed: {validation.error_message}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"validation_errors": validation.validation_errors}
                )
            
            # Route to specific operation handler
            if operation.operation_type == "init_repository":
                result = await self._init_repository(operation.parameters)
            elif operation.operation_type == "clone_repository":
                result = await self._clone_repository(operation.parameters)
            elif operation.operation_type == "create_branch":
                result = await self._create_branch(operation.parameters)
            elif operation.operation_type == "switch_branch":
                result = await self._switch_branch(operation.parameters)
            elif operation.operation_type == "commit_changes":
                result = await self._commit_changes(operation.parameters)
            elif operation.operation_type == "push_changes":
                result = await self._push_changes(operation.parameters)
            elif operation.operation_type == "pull_changes":
                result = await self._pull_changes(operation.parameters)
            elif operation.operation_type == "merge_branch":
                result = await self._merge_branch(operation.parameters)
            elif operation.operation_type == "create_tag":
                result = await self._create_tag(operation.parameters)
            elif operation.operation_type == "get_status":
                result = await self._get_status(operation.parameters)
            elif operation.operation_type == "get_history":
                result = await self._get_history(operation.parameters)
            elif operation.operation_type == "resolve_conflicts":
                result = await self._resolve_conflicts(operation.parameters)
            elif operation.operation_type == "setup_hooks":
                result = await self._setup_hooks(operation.parameters)
            else:
                return ToolResult(
                    tool_name=self.name,
                    operation=operation.operation_type,
                    success=False,
                    result=None,
                    error=f"Unknown operation: {operation.operation_type}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_name=self.name,
                operation=operation.operation_type,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    "git_operation": operation.operation_type,
                    "parameters_used": operation.parameters
                }
            )
            
        except Exception as e:
            logger.error(f"Git operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                operation=operation.operation_type,
                success=False,
                result=None,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def _run_git_command(
        self, 
        command: List[str], 
        cwd: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Run a Git command and return stdout, stderr, and return code."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            return (
                stdout.decode('utf-8').strip(),
                stderr.decode('utf-8').strip(),
                process.returncode
            )
        except Exception as e:
            logger.error(f"Failed to run git command {' '.join(command)}: {e}")
            return "", str(e), 1

    async def _init_repository(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new Git repository."""
        project_path = parameters.get("project_path", ".")
        with_flutter_gitignore = parameters.get("with_flutter_gitignore", True)
        initial_commit = parameters.get("initial_commit", True)
        
        # Initialize repository
        stdout, stderr, code = await self._run_git_command(
            ["git", "init"], cwd=project_path
        )
        
        if code != 0:
            raise Exception(f"Failed to initialize repository: {stderr}")
        
        # Create .gitignore for Flutter if requested
        if with_flutter_gitignore:
            gitignore_path = Path(project_path) / ".gitignore"
            gitignore_path.write_text(self.flutter_gitignore_template)
        
        # Create initial commit if requested
        if initial_commit:
            # Add all files
            await self._run_git_command(["git", "add", "."], cwd=project_path)
            
            # Create initial commit
            await self._run_git_command([
                "git", "commit", "-m", "feat: initial Flutter project setup"
            ], cwd=project_path)
        
        return {
            "repository_path": project_path,
            "initialized": True,
            "gitignore_created": with_flutter_gitignore,
            "initial_commit_created": initial_commit
        }

    async def _clone_repository(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clone a remote repository."""
        repo_url = parameters["repo_url"]
        destination = parameters.get("destination", ".")
        branch = parameters.get("branch")
        
        command = ["git", "clone"]
        if branch:
            command.extend(["-b", branch])
        command.extend([repo_url, destination])
        
        stdout, stderr, code = await self._run_git_command(command)
        
        if code != 0:
            raise Exception(f"Failed to clone repository: {stderr}")
        
        return {
            "cloned_url": repo_url,
            "destination": destination,
            "branch": branch or "default"
        }

    async def _create_branch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new branch."""
        project_path = parameters.get("project_path", ".")
        branch_name = parameters["branch_name"]
        from_branch = parameters.get("from_branch", "main")
        switch_to_branch = parameters.get("switch_to_branch", True)
        
        # Create branch
        stdout, stderr, code = await self._run_git_command([
            "git", "checkout", "-b", branch_name, from_branch
        ], cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to create branch: {stderr}")
        
        return {
            "branch_name": branch_name,
            "from_branch": from_branch,
            "switched": switch_to_branch
        }

    async def _switch_branch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to an existing branch."""
        project_path = parameters.get("project_path", ".")
        branch_name = parameters["branch_name"]
        
        stdout, stderr, code = await self._run_git_command([
            "git", "checkout", branch_name
        ], cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to switch branch: {stderr}")
        
        return {
            "switched_to": branch_name,
            "previous_branch": stdout
        }

    async def _commit_changes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Commit changes with Flutter-specific conventions."""
        project_path = parameters.get("project_path", ".")
        message = parameters["message"]
        commit_type = parameters.get("commit_type", "feat")
        files = parameters.get("files", [])
        add_all = parameters.get("add_all", True)
        
        # Validate commit type
        if commit_type not in self.commit_conventions:
            commit_type = "feat"
        
        # Format commit message according to conventions
        if not message.startswith(f"{commit_type}:"):
            formatted_message = f"{commit_type}: {message}"
        else:
            formatted_message = message
        
        # Add files
        if add_all:
            await self._run_git_command(["git", "add", "."], cwd=project_path)
        elif files:
            for file in files:
                await self._run_git_command(["git", "add", file], cwd=project_path)
        
        # Commit
        stdout, stderr, code = await self._run_git_command([
            "git", "commit", "-m", formatted_message
        ], cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to commit changes: {stderr}")
        
        return {
            "commit_message": formatted_message,
            "commit_hash": stdout.split()[-1] if stdout else "unknown",
            "files_committed": files if not add_all else "all staged files"
        }

    async def _push_changes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Push changes to remote repository."""
        project_path = parameters.get("project_path", ".")
        remote = parameters.get("remote", "origin")
        branch = parameters.get("branch")
        force = parameters.get("force", False)
        
        command = ["git", "push"]
        if force:
            command.append("--force")
        
        command.append(remote)
        if branch:
            command.append(branch)
        
        stdout, stderr, code = await self._run_git_command(command, cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to push changes: {stderr}")
        
        return {
            "pushed_to": f"{remote}/{branch or 'current branch'}",
            "output": stdout
        }

    async def _pull_changes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Pull changes from remote repository."""
        project_path = parameters.get("project_path", ".")
        remote = parameters.get("remote", "origin")
        branch = parameters.get("branch")
        
        command = ["git", "pull", remote]
        if branch:
            command.append(branch)
        
        stdout, stderr, code = await self._run_git_command(command, cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to pull changes: {stderr}")
        
        return {
            "pulled_from": f"{remote}/{branch or 'current branch'}",
            "output": stdout
        }

    async def _merge_branch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a branch into the current branch."""
        project_path = parameters.get("project_path", ".")
        branch_name = parameters["branch_name"]
        no_ff = parameters.get("no_ff", True)
        
        command = ["git", "merge"]
        if no_ff:
            command.append("--no-ff")
        command.append(branch_name)
        
        stdout, stderr, code = await self._run_git_command(command, cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to merge branch: {stderr}")
        
        return {
            "merged_branch": branch_name,
            "merge_output": stdout
        }

    async def _create_tag(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Git tag for releases."""
        project_path = parameters.get("project_path", ".")
        tag_name = parameters["tag_name"]
        message = parameters.get("message", f"Release {tag_name}")
        
        stdout, stderr, code = await self._run_git_command([
            "git", "tag", "-a", tag_name, "-m", message
        ], cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to create tag: {stderr}")
        
        return {
            "tag_name": tag_name,
            "tag_message": message
        }

    async def _get_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get Git repository status."""
        project_path = parameters.get("project_path", ".")
        
        stdout, stderr, code = await self._run_git_command([
            "git", "status", "--porcelain"
        ], cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to get status: {stderr}")
        
        # Parse status output
        changes = []
        for line in stdout.split('\n'):
            if line.strip():
                status_code = line[:2]
                file_path = line[3:]
                changes.append({
                    "file": file_path,
                    "status": self._parse_status_code(status_code)
                })
        
        return {
            "changes": changes,
            "clean": len(changes) == 0
        }

    def _parse_status_code(self, code: str) -> str:
        """Parse Git status codes."""
        status_map = {
            "??": "untracked",
            "A ": "added",
            "M ": "modified",
            "D ": "deleted",
            "R ": "renamed",
            "C ": "copied",
            " M": "modified_unstaged",
            " D": "deleted_unstaged"
        }
        return status_map.get(code, "unknown")

    async def _get_history(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get Git commit history."""
        project_path = parameters.get("project_path", ".")
        limit = parameters.get("limit", 10)
        branch = parameters.get("branch")
        
        command = ["git", "log", f"--max-count={limit}", "--oneline"]
        if branch:
            command.append(branch)
        
        stdout, stderr, code = await self._run_git_command(command, cwd=project_path)
        
        if code != 0:
            raise Exception(f"Failed to get history: {stderr}")
        
        commits = []
        for line in stdout.split('\n'):
            if line.strip():
                parts = line.split(' ', 1)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else ""
                })
        
        return {
            "commits": commits,
            "total_shown": len(commits)
        }

    async def _resolve_conflicts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assist with conflict resolution."""
        project_path = parameters.get("project_path", ".")
        strategy = parameters.get("strategy", "manual")
        
        # Get conflicted files
        stdout, stderr, code = await self._run_git_command([
            "git", "diff", "--name-only", "--diff-filter=U"
        ], cwd=project_path)
        
        conflicted_files = [f.strip() for f in stdout.split('\n') if f.strip()]
        
        return {
            "conflicted_files": conflicted_files,
            "resolution_strategy": strategy,
            "requires_manual_resolution": len(conflicted_files) > 0
        }

    async def _setup_hooks(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Git hooks for Flutter projects."""
        project_path = parameters.get("project_path", ".")
        hooks = parameters.get("hooks", ["pre-commit", "pre-push"])
        
        hooks_dir = Path(project_path) / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        created_hooks = []
        
        for hook_name in hooks:
            if hook_name == "pre-commit":
                hook_content = """#!/bin/bash
# Flutter pre-commit hook
flutter analyze
if [ $? -ne 0 ]; then
    echo "Flutter analyze failed. Please fix the issues before committing."
    exit 1
fi

dart format --set-exit-if-changed .
if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Please run 'dart format .' and try again."
    exit 1
fi
"""
            elif hook_name == "pre-push":
                hook_content = """#!/bin/bash
# Flutter pre-push hook
flutter test
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix the failing tests before pushing."
    exit 1
fi
"""
            else:
                continue
            
            hook_file = hooks_dir / hook_name
            hook_file.write_text(hook_content)
            hook_file.chmod(0o755)
            created_hooks.append(hook_name)
        
        return {
            "created_hooks": created_hooks,
            "hooks_directory": str(hooks_dir)
        }

    async def validate_operation(self, operation: ToolOperation) -> ToolValidation:
        """Validate a Git operation."""
        errors = []
        
        # Validate operation type
        valid_operations = {
            "init_repository", "clone_repository", "create_branch", "switch_branch",
            "commit_changes", "push_changes", "pull_changes", "merge_branch",
            "create_tag", "get_status", "get_history", "resolve_conflicts", "setup_hooks"
        }
        
        if operation.operation_type not in valid_operations:
            errors.append(f"Invalid operation type: {operation.operation_type}")
        
        # Validate required parameters
        required_params = {
            "clone_repository": ["repo_url"],
            "create_branch": ["branch_name"],
            "switch_branch": ["branch_name"],
            "commit_changes": ["message"],
            "merge_branch": ["branch_name"],
            "create_tag": ["tag_name"]
        }
        
        if operation.operation_type in required_params:
            for param in required_params[operation.operation_type]:
                if param not in operation.parameters:
                    errors.append(f"Missing required parameter: {param}")
        
        # Validate project path exists if specified
        project_path = operation.parameters.get("project_path", ".")
        if not Path(project_path).exists():
            errors.append(f"Project path does not exist: {project_path}")
        
        return ToolValidation(
            is_valid=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            validation_errors=errors
        )

    async def get_usage_examples(self) -> List[ToolUsageExample]:
        """Get usage examples for the Git tool."""
        return [
            ToolUsageExample(
                operation_type="init_repository",
                description="Initialize a new Flutter project repository",
                parameters={
                    "project_path": "/path/to/flutter/project",
                    "with_flutter_gitignore": True,
                    "initial_commit": True
                },
                expected_outcome="Repository initialized with Flutter .gitignore and initial commit"
            ),
            ToolUsageExample(
                operation_type="create_branch",
                description="Create a feature branch for new development",
                parameters={
                    "project_path": "/path/to/project",
                    "branch_name": "feature/login-screen",
                    "from_branch": "main"
                },
                expected_outcome="New feature branch created and switched to"
            ),
            ToolUsageExample(
                operation_type="commit_changes",
                description="Commit Flutter widget changes with proper conventions",
                parameters={
                    "project_path": "/path/to/project",
                    "message": "add login screen with validation",
                    "commit_type": "flutter",
                    "add_all": True
                },
                expected_outcome="Changes committed with Flutter-specific commit message"
            )
        ]

    async def check_health(self) -> Dict[str, Any]:
        """Check Git tool health."""
        health_status = {
            "tool_name": self.name,
            "version": self.version,
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # Check if Git is available
            stdout, stderr, code = await self._run_git_command(["git", "--version"])
            health_status["checks"]["git_available"] = {
                "status": "pass" if code == 0 else "fail",
                "version": stdout if code == 0 else None,
                "error": stderr if code != 0 else None
            }
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["git_available"] = {
                "status": "fail",
                "error": str(e)
            }
        
        return health_status

    async def learn_from_usage(self, usage_entry: ToolUsageExample) -> None:
        """Learn from a usage example to improve future performance or suggestions."""
        # Implementation depends on how the tool is expected to learn
        # This could involve updating internal models, adjusting parameters, etc.
        # Example: git_tool.learn_from_usage(usage_entry)
        pass

    async def get_tool_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the tool's operations and parameters."""
        # This would typically be loaded from a schema file or generated
        return {
            "type": "object",
            "properties": {
                "operation_type": {"type": "string", "enum": [
                    "init_repository", "clone_repository", "create_branch", 
                    "switch_branch", "commit_changes", "push_changes", 
                    "pull_changes", "merge_branch", "create_tag", 
                    "get_status", "get_history", "resolve_conflicts", 
                    "setup_hooks"
                ]},
                "parameters": {"type": "object"} 
            },
            "required": ["operation_type", "parameters"]
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Check the health of the Git tool (e.g., Git executable available)."""
        try:
            # Check if git command is available
            process = await asyncio.create_subprocess_exec(
                "git", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {"status": "healthy", "version": stdout.decode().strip()}
            else:
                return {"status": "unhealthy", "error": stderr.decode().strip()}
        except FileNotFoundError:
            return {"status": "unhealthy", "error": "Git executable not found"}
        except Exception as e:
            return {"status": "error", "details": str(e)}
