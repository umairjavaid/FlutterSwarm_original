# FlutterSwarm Tool System - Usage Guide and Examples

## Overview

The FlutterSwarm Tool System provides a comprehensive framework for AI agents to interact with development tools through structured operations. This guide covers usage patterns, integration examples, and troubleshooting.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tool Categories](#tool-categories)
3. [Usage Examples](#usage-examples)
4. [Agent Integration Patterns](#agent-integration-patterns)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Quick Start

### Basic Tool Usage

```python
from core.tools.tool_registry import ToolRegistry
from core.tools.flutter_sdk_tool import FlutterSDKTool

# Initialize the tool registry
registry = ToolRegistry.instance()
await registry.initialize(auto_discover=True)

# Get a specific tool
flutter_tool = registry.get_tool("flutter_sdk_tool")

# Check tool capabilities
capabilities = await flutter_tool.get_capabilities()
print(f"Available operations: {[op['name'] for op in capabilities.available_operations]}")

# Execute an operation
result = await flutter_tool.execute("create_project", {
    "project_name": "my_flutter_app",
    "output_directory": "/tmp/flutter_projects",
    "template": "app"
})

print(f"Result: {result.status.value}")
if result.error_message:
    print(f"Error: {result.error_message}")
```

### Agent Integration

```python
from agents.implementation_agent import ImplementationAgent
from agents.base_agent import AgentConfig

# Configure agent with tool access
config = AgentConfig(
    agent_id="dev_agent",
    agent_type="implementation",
    capabilities=["flutter_development", "file_operations"],
    llm_model="gpt-4"
)

agent = ImplementationAgent(
    config=config,
    llm_client=llm_client,
    memory_manager=memory_manager,
    event_bus=event_bus,
    tool_registry=registry
)

# Agent can now use tools through reasoning
task = "Create a Flutter app with a todo list feature"
result = await agent.process_task(task)
```

## Tool Categories

### 1. Flutter SDK Tool

**Purpose**: Direct Flutter SDK operations
**Category**: Development

**Core Operations**:
- `create_project`: Create new Flutter projects
- `build_app`: Build applications for specific platforms
- `run_app`: Run applications on devices/simulators
- `add_platform`: Add platform support to existing projects
- `analyze_code`: Analyze code quality and style
- `test_app`: Run unit, widget, and integration tests
- `pub_operations`: Manage package dependencies

**Example Usage**:

```python
# Create a new Flutter project
result = await flutter_tool.execute("create_project", {
    "project_name": "weather_app",
    "output_directory": "/workspace/projects",
    "template": "app",
    "org": "com.example",
    "description": "A weather forecasting application",
    "platforms": ["android", "ios", "web"]
})

# Build for Android
result = await flutter_tool.execute("build_app", {
    "project_path": "/workspace/projects/weather_app",
    "platform": "android",
    "build_mode": "debug",
    "target_file": "lib/main.dart"
})

# Run tests with coverage
result = await flutter_tool.execute("test_app", {
    "project_path": "/workspace/projects/weather_app",
    "test_type": "unit",
    "coverage": True,
    "reporter": "expanded"
})
```

### 2. File System Tool

**Purpose**: Flutter-aware file operations
**Category**: File System

**Core Operations**:
- `create_file`: Create files with content
- `read_file`: Read file contents
- `write_file`: Write content to files
- `delete_file`: Delete files safely
- `create_directory`: Create directory structures
- `move_file`: Move/rename files
- `copy_file`: Copy files with backup
- `backup_file`: Create file backups
- `restore_file`: Restore from backup
- `watch_directory`: Monitor directory changes

**Flutter-Specific Features**:
- Dart import analysis and optimization
- Flutter project structure awareness
- Asset optimization
- Template-based file creation
- pubspec.yaml management

**Example Usage**:

```python
# Create a new Dart widget file
result = await file_tool.execute("create_file", {
    "file_path": "/workspace/projects/weather_app/lib/widgets/weather_card.dart",
    "content": """
import 'package:flutter/material.dart';

class WeatherCard extends StatelessWidget {
  final String temperature;
  final String condition;
  
  const WeatherCard({
    Key? key,
    required this.temperature,
    required this.condition,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(temperature, style: Theme.of(context).textTheme.headlineMedium),
            Text(condition),
          ],
        ),
      ),
    );
  }
}
""",
    "backup": True,
    "optimize_imports": True
})

# Read and analyze imports
result = await file_tool.execute("analyze_imports", {
    "file_path": "/workspace/projects/weather_app/lib/main.dart",
    "suggest_optimizations": True
})
```

### 3. Process Tool

**Purpose**: System process management
**Category**: Process

**Core Operations**:
- `execute_command`: Execute shell commands
- `start_process`: Start background processes
- `stop_process`: Stop running processes
- `monitor_process`: Monitor process status
- `get_process_info`: Get process information

**Example Usage**:

```python
# Execute Flutter doctor
result = await process_tool.execute("execute_command", {
    "command": "flutter doctor -v",
    "working_directory": "/workspace/projects/weather_app",
    "capture_output": True,
    "timeout": 30
})

# Start a development server in background
result = await process_tool.execute("start_process", {
    "command": "flutter run -d chrome --web-port 8080",
    "working_directory": "/workspace/projects/weather_app",
    "background": True,
    "process_name": "weather_app_dev_server"
})
```

## Usage Examples

### Example 1: Complete Flutter App Creation

```python
async def create_complete_flutter_app():
    """Create a complete Flutter app with features."""
    registry = ToolRegistry.instance()
    
    # Get tools
    flutter_tool = registry.get_tool("flutter_sdk_tool")
    file_tool = registry.get_tool("file_system_tool")
    process_tool = registry.get_tool("process_tool")
    
    project_path = "/workspace/todo_app"
    
    # Step 1: Create Flutter project
    result = await flutter_tool.execute("create_project", {
        "project_name": "todo_app",
        "output_directory": "/workspace",
        "template": "app",
        "org": "com.example"
    })
    
    if result.status.value != "success":
        return f"Failed to create project: {result.error_message}"
    
    # Step 2: Create feature structure
    feature_dirs = [
        "lib/features/todo",
        "lib/features/todo/models",
        "lib/features/todo/screens",
        "lib/features/todo/widgets",
        "lib/features/todo/services"
    ]
    
    for dir_path in feature_dirs:
        await file_tool.execute("create_directory", {
            "directory_path": f"{project_path}/{dir_path}"
        })
    
    # Step 3: Create Todo model
    todo_model = '''
class Todo {
  final String id;
  final String title;
  final String description;
  final bool isCompleted;
  final DateTime createdAt;
  
  Todo({
    required this.id,
    required this.title,
    required this.description,
    this.isCompleted = false,
    required this.createdAt,
  });
  
  Todo copyWith({
    String? id,
    String? title,
    String? description,
    bool? isCompleted,
    DateTime? createdAt,
  }) {
    return Todo(
      id: id ?? this.id,
      title: title ?? this.title,
      description: description ?? this.description,
      isCompleted: isCompleted ?? this.isCompleted,
      createdAt: createdAt ?? this.createdAt,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'description': description,
      'isCompleted': isCompleted,
      'createdAt': createdAt.toIso8601String(),
    };
  }
  
  factory Todo.fromJson(Map<String, dynamic> json) {
    return Todo(
      id: json['id'],
      title: json['title'],
      description: json['description'],
      isCompleted: json['isCompleted'],
      createdAt: DateTime.parse(json['createdAt']),
    );
  }
}
'''
    
    await file_tool.execute("create_file", {
        "file_path": f"{project_path}/lib/features/todo/models/todo.dart",
        "content": todo_model
    })
    
    # Step 4: Create Todo service
    todo_service = '''
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/todo.dart';

class TodoService {
  static const String _todoKey = 'todos';
  
  Future<List<Todo>> getTodos() async {
    final prefs = await SharedPreferences.getInstance();
    final todoJson = prefs.getString(_todoKey);
    
    if (todoJson == null) return [];
    
    final List<dynamic> todoList = json.decode(todoJson);
    return todoList.map((json) => Todo.fromJson(json)).toList();
  }
  
  Future<void> saveTodos(List<Todo> todos) async {
    final prefs = await SharedPreferences.getInstance();
    final todoJson = json.encode(todos.map((todo) => todo.toJson()).toList());
    await prefs.setString(_todoKey, todoJson);
  }
  
  Future<void> addTodo(Todo todo) async {
    final todos = await getTodos();
    todos.add(todo);
    await saveTodos(todos);
  }
  
  Future<void> updateTodo(Todo updatedTodo) async {
    final todos = await getTodos();
    final index = todos.indexWhere((todo) => todo.id == updatedTodo.id);
    if (index != -1) {
      todos[index] = updatedTodo;
      await saveTodos(todos);
    }
  }
  
  Future<void> deleteTodo(String id) async {
    final todos = await getTodos();
    todos.removeWhere((todo) => todo.id == id);
    await saveTodos(todos);
  }
}
'''
    
    await file_tool.execute("create_file", {
        "file_path": f"{project_path}/lib/features/todo/services/todo_service.dart",
        "content": todo_service
    })
    
    # Step 5: Add dependencies to pubspec.yaml
    pubspec_result = await file_tool.execute("read_file", {
        "file_path": f"{project_path}/pubspec.yaml"
    })
    
    if pubspec_result.status.value == "success":
        pubspec_content = pubspec_result.data["content"]
        
        # Add shared_preferences dependency
        if "shared_preferences:" not in pubspec_content:
            updated_pubspec = pubspec_content.replace(
                "dependencies:\n  flutter:\n    sdk: flutter",
                "dependencies:\n  flutter:\n    sdk: flutter\n  shared_preferences: ^2.2.2\n  uuid: ^4.1.0"
            )
            
            await file_tool.execute("write_file", {
                "file_path": f"{project_path}/pubspec.yaml",
                "content": updated_pubspec
            })
    
    # Step 6: Get dependencies
    await process_tool.execute("execute_command", {
        "command": "flutter pub get",
        "working_directory": project_path
    })
    
    # Step 7: Run analysis
    analysis_result = await flutter_tool.execute("analyze_code", {
        "project_path": project_path,
        "fix": True
    })
    
    return "Todo app created successfully!"

# Usage
result = await create_complete_flutter_app()
print(result)
```

### Example 2: Agent-Driven Development

```python
class FlutterDevelopmentAgent:
    """Agent that can develop Flutter applications using tools."""
    
    def __init__(self, tool_registry, llm_client):
        self.registry = tool_registry
        self.llm = llm_client
    
    async def develop_feature(self, feature_description: str, project_path: str):
        """Develop a Flutter feature based on description."""
        
        # Step 1: Analyze feature requirements using LLM
        analysis_prompt = f"""
        Analyze this Flutter feature request and break it down into development tasks:
        
        Feature: {feature_description}
        Project: {project_path}
        
        Provide a structured plan with:
        1. Required files to create/modify
        2. Dependencies to add
        3. Implementation steps
        """
        
        analysis = await self.llm.generate([{"role": "user", "content": analysis_prompt}])
        
        # Step 2: Select appropriate tools
        flutter_tool = self.registry.get_tool("flutter_sdk_tool")
        file_tool = self.registry.get_tool("file_system_tool")
        
        # Step 3: Execute development plan
        tasks = self._parse_development_plan(analysis.content)
        
        results = []
        for task in tasks:
            if task["type"] == "create_file":
                result = await file_tool.execute("create_file", {
                    "file_path": task["path"],
                    "content": task["content"]
                })
                results.append(f"Created {task['path']}: {result.status.value}")
                
            elif task["type"] == "add_dependency":
                # Update pubspec.yaml
                result = await self._add_dependency(project_path, task["package"])
                results.append(f"Added dependency {task['package']}: {result}")
                
            elif task["type"] == "run_analysis":
                result = await flutter_tool.execute("analyze_code", {
                    "project_path": project_path
                })
                results.append(f"Code analysis: {result.status.value}")
        
        return results
    
    def _parse_development_plan(self, llm_response: str) -> List[Dict]:
        """Parse LLM response into actionable tasks."""
        # Implementation would parse the LLM response
        # This is a simplified example
        return [
            {
                "type": "create_file",
                "path": "/lib/features/new_feature.dart",
                "content": "// Feature implementation"
            }
        ]
    
    async def _add_dependency(self, project_path: str, package: str) -> str:
        """Add a dependency to pubspec.yaml."""
        file_tool = self.registry.get_tool("file_system_tool")
        
        pubspec_result = await file_tool.execute("read_file", {
            "file_path": f"{project_path}/pubspec.yaml"
        })
        
        if pubspec_result.status.value == "success":
            content = pubspec_result.data["content"]
            
            # Simple dependency addition (real implementation would be more robust)
            updated_content = content.replace(
                "dependencies:",
                f"dependencies:\n  {package}:"
            )
            
            await file_tool.execute("write_file", {
                "file_path": f"{project_path}/pubspec.yaml",
                "content": updated_content
            })
            
            return "success"
        
        return "failed"

# Usage
agent = FlutterDevelopmentAgent(registry, llm_client)
results = await agent.develop_feature(
    "Create a user profile screen with avatar upload",
    "/workspace/my_app"
)
```

## Agent Integration Patterns

### Pattern 1: Tool Selection by Capability

```python
async def select_tools_for_task(registry: ToolRegistry, task_type: str):
    """Select the best tools for a specific task type."""
    
    if task_type == "flutter_development":
        return registry.select_tools_for_workflow("flutter_project_creation", {
            "platforms": ["android", "ios"],
            "complexity": "medium"
        })
    
    elif task_type == "file_management":
        return [registry.get_tool("file_system_tool")]
    
    elif task_type == "testing":
        return [
            registry.get_tool("flutter_sdk_tool"),
            registry.get_tool("process_tool")
        ]
    
    return []
```

### Pattern 2: LLM-Guided Tool Usage

```python
async def llm_guided_development(llm_client, registry, user_request):
    """Use LLM to guide tool selection and usage."""
    
    # Step 1: LLM analyzes the request
    analysis_prompt = f"""
    Analyze this development request and recommend tools and operations:
    
    Request: {user_request}
    
    Available tools:
    - flutter_sdk_tool: Flutter SDK operations
    - file_system_tool: File operations with Flutter awareness
    - process_tool: System command execution
    
    Recommend the sequence of tool operations needed.
    """
    
    response = await llm_client.generate([{"role": "user", "content": analysis_prompt}])
    
    # Step 2: Parse LLM recommendations
    recommendations = parse_tool_recommendations(response.content)
    
    # Step 3: Execute recommended operations
    results = []
    for rec in recommendations:
        tool = registry.get_tool(rec["tool_name"])
        if tool:
            result = await tool.execute(rec["operation"], rec["parameters"])
            results.append({
                "tool": rec["tool_name"],
                "operation": rec["operation"],
                "success": result.status.value == "success",
                "result": result.data
            })
    
    return results
```

### Pattern 3: Error Recovery with Tool Fallbacks

```python
async def resilient_tool_execution(registry, operation_plan):
    """Execute operations with automatic fallback and recovery."""
    
    for operation in operation_plan:
        primary_tool = registry.get_tool(operation["primary_tool"])
        fallback_tools = [registry.get_tool(name) for name in operation.get("fallbacks", [])]
        
        # Try primary tool first
        result = await primary_tool.execute(operation["name"], operation["parameters"])
        
        if result.status.value == "success":
            continue
        
        # Try fallback tools
        for fallback_tool in fallback_tools:
            if fallback_tool:
                result = await fallback_tool.execute(operation["name"], operation["parameters"])
                if result.status.value == "success":
                    break
        
        if result.status.value != "success":
            # Attempt recovery
            await attempt_recovery(operation, result.error_message)
```

## Advanced Features

### Performance Monitoring

```python
# Monitor tool performance
registry = ToolRegistry.instance()
analytics = registry.get_performance_analytics()

print(f"Tool usage statistics:")
for tool_name, stats in analytics.items():
    print(f"  {tool_name}:")
    print(f"    Average execution time: {stats['avg_execution_time']:.2f}s")
    print(f"    Success rate: {stats['success_rate']:.1%}")
    print(f"    Total executions: {stats['total_executions']}")
```

### Health Monitoring

```python
# Check tool health
health_status = await registry.get_registry_status()

for tool_name, status in health_status["tools"].items():
    if not status["is_healthy"]:
        print(f"⚠️ {tool_name} is unhealthy: {status['error']}")
        
        # Attempt recovery
        await registry.reload_tool(tool_name)
```

### Configuration Management

```python
# Update tool configuration
await registry.update_tool_config("flutter_sdk_tool", {
    "flutter_path": "/custom/flutter/bin/flutter",
    "default_platforms": ["android", "web"],
    "enable_analytics": False
})

# Get current configuration
config = registry.get_tool_config("flutter_sdk_tool")
print(f"Flutter SDK path: {config.get('flutter_path')}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Tool Not Found

**Problem**: `registry.get_tool("tool_name")` returns `None`

**Solutions**:
```python
# Check if tool is registered
available_tools = registry.get_available_tools()
tool_names = [tool.name for tool in available_tools]
print(f"Available tools: {tool_names}")

# Manually register tool if needed
from core.tools.flutter_sdk_tool import FlutterSDKTool
flutter_tool = FlutterSDKTool()
await registry.register_tool(flutter_tool)
```

#### 2. Operation Fails with Validation Error

**Problem**: `validate_params` returns `False` with error

**Solutions**:
```python
# Check operation schema
capabilities = await tool.get_capabilities()
for op in capabilities.available_operations:
    if op["name"] == "target_operation":
        print(f"Required parameters: {op['parameters']}")
        break

# Validate before execution
is_valid, error = await tool.validate_params("operation_name", params)
if not is_valid:
    print(f"Validation error: {error}")
    # Fix parameters based on error message
```

#### 3. Flutter SDK Not Found

**Problem**: Flutter operations fail with "Flutter SDK not found"

**Solutions**:
```python
# Check Flutter SDK availability
flutter_tool = registry.get_tool("flutter_sdk_tool")
health_check = await flutter_tool.health_check()

if not health_check["flutter_available"]:
    print("Flutter SDK not found. Please install Flutter or update PATH.")
    
    # Update Flutter path manually
    await registry.update_tool_config("flutter_sdk_tool", {
        "flutter_path": "/path/to/flutter/bin/flutter"
    })
```

#### 4. Permission Errors

**Problem**: File operations fail with permission errors

**Solutions**:
```python
# Check file permissions
import os
file_path = "/target/file/path"
if not os.access(file_path, os.W_OK):
    print(f"No write permission for {file_path}")
    
# Use backup option for safer operations
result = await file_tool.execute("write_file", {
    "file_path": file_path,
    "content": content,
    "backup": True,  # Creates backup before writing
    "create_dirs": True  # Creates parent directories
})
```

#### 5. Performance Issues

**Problem**: Tool operations are slow

**Solutions**:
```python
# Monitor performance
start_time = time.time()
result = await tool.execute("operation", params)
execution_time = time.time() - start_time

if execution_time > 5.0:  # If operation takes more than 5 seconds
    print(f"Slow operation detected: {execution_time:.2f}s")
    
    # Check tool health
    health = await tool.health_check()
    print(f"Tool health: {health}")
    
    # Consider using background execution for long operations
    if hasattr(tool, 'execute_background'):
        task_id = await tool.execute_background("operation", params)
        # Check status later
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging

# Enable debug logging
logging.getLogger("tool_registry").setLevel(logging.DEBUG)
logging.getLogger("flutter_sdk_tool").setLevel(logging.DEBUG)
logging.getLogger("file_system_tool").setLevel(logging.DEBUG)

# Now all tool operations will log detailed information
result = await tool.execute("operation", params)
```

### Performance Profiling

Profile tool operations to identify bottlenecks:

```python
import cProfile
import pstats

def profile_tool_operation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute tool operation
    asyncio.run(tool.execute("operation", params))
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions

profile_tool_operation()
```

## Best Practices

### 1. Always Validate Parameters

```python
# Good: Validate before execution
is_valid, error = await tool.validate_params("operation", params)
if is_valid:
    result = await tool.execute("operation", params)
else:
    print(f"Invalid parameters: {error}")
```

### 2. Handle Errors Gracefully

```python
# Good: Comprehensive error handling
try:
    result = await tool.execute("operation", params)
    if result.status.value == "success":
        # Handle success
        process_result(result.data)
    else:
        # Handle tool-level failure
        print(f"Operation failed: {result.error_message}")
        attempt_recovery(result)
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
    log_error_for_debugging(e)
```

### 3. Use Appropriate Tools for Tasks

```python
# Good: Use specialized tools
flutter_tool = registry.get_tool("flutter_sdk_tool")  # For Flutter operations
file_tool = registry.get_tool("file_system_tool")     # For file operations
process_tool = registry.get_tool("process_tool")      # For system commands

# Avoid: Using process_tool for Flutter operations when flutter_tool exists
```

### 4. Leverage Tool Capabilities

```python
# Good: Check capabilities before using
capabilities = await tool.get_capabilities()
available_operations = [op["name"] for op in capabilities.available_operations]

if "target_operation" in available_operations:
    result = await tool.execute("target_operation", params)
else:
    print(f"Operation not supported by {tool.name}")
```

### 5. Monitor Performance

```python
# Good: Track execution times
import time

start_time = time.time()
result = await tool.execute("operation", params)
execution_time = time.time() - start_time

# Log slow operations
if execution_time > 3.0:
    logger.warning(f"Slow operation: {tool.name}.{operation} took {execution_time:.2f}s")
```

### 6. Use Background Execution for Long Operations

```python
# Good: Use background execution for builds, tests, etc.
if operation in ["build_app", "test_app", "run_app"]:
    task_id = await tool.execute_background("operation", params)
    
    # Monitor progress
    while True:
        status = await tool.get_task_status(task_id)
        if status["completed"]:
            break
        await asyncio.sleep(1)
```

### 7. Implement Retry Logic

```python
# Good: Retry with exponential backoff
async def retry_operation(tool, operation, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await tool.execute(operation, params)
            if result.status.value == "success":
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            await asyncio.sleep(wait_time)
    
    return None
```

This comprehensive guide provides everything needed to effectively use the FlutterSwarm Tool System with AI agents. The examples demonstrate real-world usage patterns, and the troubleshooting section helps resolve common issues quickly.
