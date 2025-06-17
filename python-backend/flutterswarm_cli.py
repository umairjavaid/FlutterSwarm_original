#!/usr/bin/env python3
"""
FlutterSwarm CLI Interface.

A command-line interface for interacting with the FlutterSwarm multi-agent system.
"""

import asyncio
import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.system import initialize_system, start_system, stop_system, get_system
from src.config.settings import settings
from src.config import setup_logging, get_logger

# Configure logging with enhanced logger
setup_logging()
logger = get_logger("flutterswarm_cli")


class FlutterSwarmCLI:
    """Command-line interface for FlutterSwarm."""
    
    def __init__(self):
        self.system = None
    
    async def initialize(self):
        """Initialize the FlutterSwarm system."""
        print("🚀 Initializing FlutterSwarm Multi-Agent System...")
        
        try:
            self.system = await initialize_system()
            await start_system()
            print("✅ FlutterSwarm system started successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the system."""
        if self.system:
            print("🔄 Shutting down FlutterSwarm system...")
            await stop_system()
            print("✅ System shutdown complete")
    
    def _analyze_task_requirements(self, description: str) -> Dict[str, Any]:
        """Analyze task description to extract requirements dynamically."""
        requirements = {
            "project_name": "flutter_app",
            "project_type": "app",
            "features": [],
            "complexity": "simple",
            "architectural_pattern": "basic"
        }
        
        description_lower = description.lower()
        
        # Extract project name using multiple strategies
        project_name = self._extract_project_name(description)
        if project_name:
            requirements["project_name"] = project_name
        
        # Determine project type
        if any(term in description_lower for term in ["plugin", "package", "library"]):
            requirements["project_type"] = "plugin"
        elif "web" in description_lower:
            requirements["project_type"] = "web"
        elif "desktop" in description_lower:
            requirements["project_type"] = "desktop"
        else:
            requirements["project_type"] = "app"
        
        # Extract features dynamically
        feature_keywords = {
            "music_streaming": ["music", "streaming", "audio", "playlist", "player"],
            "video_streaming": ["video", "streaming", "tiktok", "short videos"],
            "e_commerce": ["ecommerce", "e-commerce", "shopping", "store", "cart", "purchase"],
            "social_media": ["social", "media", "chat", "messaging", "friends", "follow"],
            "authentication": ["login", "auth", "signin", "signup", "user", "account"],
            "navigation": ["navigation", "routing", "drawer", "tabs", "menu", "bottom nav"],
            "data_management": ["database", "storage", "api", "network", "crud", "persistence"],
            "ui_components": ["form", "list", "grid", "card", "chart", "widgets"],
            "media_handling": ["camera", "photo", "video", "gallery", "image", "capture"],
            "location_services": ["maps", "location", "gps", "geolocation", "coordinates"],
            "notifications": ["notification", "push", "alert", "message", "fcm"],
            "payments": ["payment", "purchase", "billing", "subscription", "stripe"],
            "offline_support": ["offline", "sync", "cache", "local", "connectivity"],
            "search_functionality": ["search", "filter", "query", "find"],
            "user_profiles": ["profile", "user profile", "avatar", "bio"],
            "settings": ["settings", "preferences", "configuration", "options"],
            "dark_mode": ["dark mode", "theme", "light", "dark"],
            "real_time": ["real time", "live", "instant", "websocket"]
        }
        
        detected_features = []
        for feature, keywords in feature_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_features.append(feature)
        
        # If we detected specific app types, add comprehensive features
        if any(f in detected_features for f in ["music_streaming"]):
            detected_features.extend(["audio_player", "playlist_management", "media_controls", "search_functionality"])
        elif any(f in detected_features for f in ["video_streaming"]):
            detected_features.extend(["video_player", "user_profiles", "social_media", "search_functionality"])
        elif any(f in detected_features for f in ["e_commerce"]):
            detected_features.extend(["product_catalog", "shopping_cart", "authentication", "payments"])
        elif any(f in detected_features for f in ["social_media"]):
            detected_features.extend(["user_profiles", "messaging", "notifications", "media_handling"])
        
        # Remove duplicates and ensure we have at least basic features
        requirements["features"] = list(set(detected_features)) or ["navigation", "ui_components", "responsive_design"]
        
        # Determine complexity
        complexity_indicators = {
            "simple": ["simple", "basic", "minimal", "quick", "small"],
            "medium": ["standard", "medium", "regular", "typical"],
            "complex": ["complex", "advanced", "enterprise", "large", "comprehensive", "full-featured"]
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                requirements["complexity"] = complexity
                break
        
        # Determine architectural pattern preference
        if any(term in description_lower for term in ["clean architecture", "layered", "domain driven"]):
            requirements["architectural_pattern"] = "clean"
        elif any(term in description_lower for term in ["bloc", "cubit", "business logic"]):
            requirements["architectural_pattern"] = "bloc"
        elif "provider" in description_lower:
            requirements["architectural_pattern"] = "provider"
        elif "riverpod" in description_lower:
            requirements["architectural_pattern"] = "riverpod"
        
        return requirements
    
    def _extract_project_name(self, description: str) -> str:
        """Extract project name from description using multiple strategies."""
        import re
        
        # Strategy 1: Look for "app named X" or "called X" patterns
        name_patterns = [
            r"app named (\w+)",
            r"called (\w+)",
            r"app '([^']+)'",
            r'app "([^"]+)"',
            r"project (\w+)",
            r"build (\w+) app",
            r"create (\w+) app"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Convert to valid Dart package name
                return self._normalize_project_name(name)
        
        # Strategy 2: Extract descriptive terms and build name
        description_words = description.lower().split()
        
        # Remove common stop words
        stop_words = {
            "create", "build", "make", "develop", "a", "an", "the", "app", "application",
            "flutter", "using", "with", "for", "that", "can", "will", "should", "simple",
            "basic", "standard", "new", "mobile"
        }
        
        descriptive_words = []
        for word in description_words:
            cleaned_word = re.sub(r'[^a-zA-Z0-9]', '', word)
            if cleaned_word and cleaned_word not in stop_words and len(cleaned_word) > 2:
                descriptive_words.append(cleaned_word)
                if len(descriptive_words) >= 3:  # Limit to 3 words max
                    break
        
        if descriptive_words:
            project_name = "_".join(descriptive_words)
            return self._normalize_project_name(project_name)
        
        # Strategy 3: Generate based on detected features
        requirements = self._analyze_task_requirements(description)
        features = requirements.get("features", [])
        
        if features:
            # Use primary feature for naming
            primary_feature = features[0].replace("_", "")
            return f"{primary_feature}_app"
        
        # Fallback: Use timestamp-based name
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M")
        return f"flutter_app_{timestamp}"
    
    def _normalize_project_name(self, name: str) -> str:
        """Normalize project name to valid Dart package name format."""
        import re
        
        # Convert to lowercase and replace spaces/special chars with underscores
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        
        # Remove consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        # Ensure it starts with a letter
        if normalized and not normalized[0].isalpha():
            normalized = f"app_{normalized}"
        
        # Limit length
        if len(normalized) > 30:
            normalized = normalized[:30].rstrip('_')
        
        return normalized or "flutter_app"

    async def submit_task(self, description: str, task_type: str = "analysis", priority: str = "normal"):
        """Submit a task to the system with dynamic requirement analysis."""
        if not self.system:
            print("❌ System not initialized. Run 'init' first.")
            return
        
        print(f"📝 Analyzing task: {description[:60]}...")
        
        # Analyze requirements dynamically
        requirements = self._analyze_task_requirements(description)
        
        print(f"📊 Detected requirements:")
        print(f"  Project: {requirements['project_name']}")
        print(f"  Type: {requirements['project_type']}")
        print(f"  Complexity: {requirements['complexity']}")
        if requirements['features']:
            print(f"  Features: {', '.join(requirements['features'])}")
        print(f"  Pattern: {requirements['architectural_pattern']}")

        request = {
            "description": description,
            "task_type": task_type,
            "priority": priority,
            "project_context": {
                "project_name": requirements["project_name"],
                "project_type": requirements["project_type"],
                "features": requirements["features"],
                "complexity": requirements["complexity"],
                "architectural_pattern": requirements["architectural_pattern"]
            }
        }
        
        try:
            result = await self.system.process_request(request)
            
            print(f"\n📊 Task Result:")
            print(f"  Task ID: {result.get('task_id', 'N/A')}")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Status: {result.get('status', 'unknown')}")
            
            if result.get('error'):
                print(f"  Error: {result['error']}")
            
            if result.get('result'):
                print(f"  Result: {json.dumps(result['result'], indent=2)[:200]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Task submission failed: {e}")
            return None
    
    async def get_status(self):
        """Get system status."""
        if not self.system:
            print("❌ System not initialized. Run 'init' first.")
            return
        
        try:
            status = await self.system.get_system_status()
            
            print(f"\n📊 FlutterSwarm System Status:")
            print(f"  Initialized: {status['system']['initialized']}")
            print(f"  Running: {status['system']['running']}")
            print(f"  Agents: {status['system']['agents_count']}")
            print(f"  Memory Managers: {status['system']['memory_managers_count']}")
            
            if status['agents']:
                print(f"\n🤖 Agents:")
                for agent_id, agent_info in status['agents'].items():
                    if isinstance(agent_info, dict):
                        print(f"  - {agent_id}: {agent_info.get('status', 'unknown')}")
            
            if 'event_bus' in status and isinstance(status['event_bus'], dict):
                metrics = status['event_bus']
                print(f"\n📡 Event Bus:")
                print(f"  Total Messages: {metrics.get('total_messages', 0)}")
                print(f"  Successful Deliveries: {metrics.get('successful_deliveries', 0)}")
                print(f"  Active Topics: {metrics.get('active_topics', 0)}")
            
            return status
            
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
            return None
    
    async def list_agents(self):
        """List all agents."""
        if not self.system:
            print("❌ System not initialized. Run 'init' first.")
            return
        
        print(f"\n🤖 Registered Agents:")
        
        if not self.system.agents:
            print("  No agents registered")
            return
        
        for agent_id, agent in self.system.agents.items():
            try:
                status = await agent.get_status()
                print(f"  - {agent_id} ({status.get('agent_type', 'unknown')}): {status.get('status', 'unknown')}")
                print(f"    Capabilities: {', '.join(status.get('capabilities', []))}")
                print(f"    Active Tasks: {status.get('active_tasks', 0)}")
            except Exception as e:
                print(f"  - {agent_id}: Error getting status - {e}")
    
    async def test_system(self):
        """Run a comprehensive system test."""
        print("🧪 Running FlutterSwarm system test...")
        
        if not await self.initialize():
            return False
        
        # Test basic functionality
        tests = [
            ("System Status", self.get_status),
            ("Agent Listing", self.list_agents),
            ("Simple Task", lambda: self.submit_task(
                "Create a simple Flutter counter app",
                "code_generation",
                "normal"
            )),
            ("Analysis Task", lambda: self.submit_task(
                "Analyze the current Flutter project structure",
                "analysis",
                "high"
            ))
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running test: {test_name}")
            try:
                result = await test_func()
                if result is not None:
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        await self.shutdown()
        
        return passed == total

    async def create_project(self, project_name, project_type="app"):
        """Create a Flutter project using the agent workflow."""
        # Determine the flutter_projects directory
        flutter_projects_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flutter_projects")
        os.makedirs(flutter_projects_dir, exist_ok=True)
        
        print(f"📁 Creating Flutter app '{project_name}' in {flutter_projects_dir}...")
        
        try:
            # Analyze the project name to extract features and requirements
            requirements = self._analyze_task_requirements(project_name)
            
            # Get the supervisor directly (since _system_instance should be the supervisor)
            system = get_system()
            if not system:
                print("❌ System not properly initialized")
                return None
            
            supervisor = system  # The system instance is the supervisor itself
            
            # Create initial workflow state as a dictionary with proper task decomposition
            initial_state = {
                "workflow_id": f"create_{project_name}",
                "task_description": f"Create a Flutter app: {project_name} with features: {', '.join(requirements.get('features', []))}",
                "project_context": {
                    "project_name": project_name,
                    "project_type": project_type,
                    "output_dir": flutter_projects_dir,
                    "requirements": requirements,
                    "features": requirements.get('features', []),
                    "complexity": requirements.get('complexity', 'simple')
                },
                "messages": [],
                "available_agents": {
                    "architecture_agent": {"status": "available", "capabilities": ["architecture_analysis", "code_generation"]},
                    "implementation_agent": {"status": "available", "capabilities": ["code_generation", "file_operations"]}
                },
                "agent_assignments": {},
                "pending_tasks": [
                    {
                        "task_id": "architecture_design",
                        "description": f"Design architecture for {project_name} with features: {', '.join(requirements.get('features', []))}",
                        "agent_type": "architecture",
                        "priority": "high",
                        "estimated_duration": 30,
                        "dependencies": [],
                        "deliverables": ["architecture_diagram", "technical_specifications", "dependency_list"]
                    },
                    {
                        "task_id": "project_setup",
                        "description": f"Create Flutter project structure for {project_name}",
                        "agent_type": "implementation", 
                        "priority": "high",
                        "estimated_duration": 15,
                        "dependencies": [],
                        "deliverables": ["project_structure", "pubspec_yaml"]
                    },
                    {
                        "task_id": "feature_implementation",
                        "description": f"Implement features: {', '.join(requirements.get('features', []))} in {project_name}",
                        "agent_type": "implementation",
                        "priority": "high", 
                        "estimated_duration": 60,
                        "dependencies": ["architecture_design", "project_setup"],
                        "deliverables": ["main_dart", "feature_widgets", "state_management"]
                    }
                ],
                "active_tasks": {},
                "completed_tasks": {},
                "failed_tasks": {},
                "deliverables": {},
                "workflow_metadata": {"phase": "task_decomposition"},
                "should_continue": True,
                "execution_metrics": {}
            }
            
            # Run the workflow through the supervisor
            print("🔄 Running agent workflow to create the Flutter app...")
            result = await supervisor.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"create_{project_name}"}}
            )
            
            if result and result.get("final_result") and result["final_result"].get("status") == "completed":
                print(f"✅ Flutter project '{project_name}' created successfully!")
                return os.path.join(flutter_projects_dir, project_name)
            else:
                error_msg = result.get("error_message", "Unknown error occurred") if result else "Workflow failed"
                print(f"❌ Failed to create Flutter project: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating Flutter project: {e}")
            import traceback
            traceback.print_exc()
            return None


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="FlutterSwarm Multi-Agent System CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    subparsers.add_parser('init', help='Initialize the FlutterSwarm system')
    
    # Status command
    subparsers.add_parser('status', help='Get system status')
    
    # Agents command
    subparsers.add_parser('agents', help='List all agents')
    
    # Task command
    task_parser = subparsers.add_parser('task', help='Submit a task')
    task_parser.add_argument('description', help='Task description')
    task_parser.add_argument('--type', default='analysis', help='Task type')
    task_parser.add_argument('--priority', default='normal', help='Task priority')
    
    # Test command
    subparsers.add_parser('test', help='Run system test')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Shutdown command
    subparsers.add_parser('shutdown', help='Shutdown the system')
    
    # Create project command
    create_parser = subparsers.add_parser('create', help='Create a new Flutter project')
    create_parser.add_argument('name', help='Project name')
    create_parser.add_argument('--type', default='app', help='Project type (app, plugin, web, desktop)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = FlutterSwarmCLI()
    
    try:
        if args.command == 'init':
            await cli.initialize()
        
        elif args.command == 'status':
            if not await cli.initialize():
                return
            await cli.get_status()
            await cli.shutdown()
        
        elif args.command == 'agents':
            if not await cli.initialize():
                return
            await cli.list_agents()
            await cli.shutdown()
        
        elif args.command == 'task':
            if not await cli.initialize():
                return
            await cli.submit_task(args.description, args.type, args.priority)
            await cli.shutdown()
        
        elif args.command == 'test':
            success = await cli.test_system()
            if success:
                print("\n🎉 All tests passed! FlutterSwarm is working correctly.")
            else:
                print("\n⚠️  Some tests failed. Check the output above for details.")
        
        elif args.command == 'interactive':
            await interactive_mode(cli)
        
        elif args.command == 'shutdown':
            await cli.shutdown()
        
        elif args.command == 'create':
            if not await cli.initialize():
                return
            project_type = args.type if args.type in ["app", "plugin", "web", "desktop"] else "app"
            await cli.create_project(args.name, project_type)
            await cli.shutdown()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        await cli.shutdown()
    except Exception as e:
        print(f"\n❌ CLI Error: {e}")
        await cli.shutdown()


async def interactive_mode(cli: FlutterSwarmCLI):
    """Interactive CLI mode."""
    print("\n🎯 FlutterSwarm Interactive Mode")
    print("Type 'help' for available commands, 'quit' to exit")
    
    if not await cli.initialize():
        return
    
    while True:
        try:
            command = input("\nflutterswarm> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            elif command.lower() == 'help':
                print("""
Available commands:
  status          - Show system status
  agents          - List all agents
  task <desc>     - Submit a task
  memory <agent>  - Show agent memory stats
  help            - Show this help
  quit            - Exit interactive mode
""")
            
            elif command.lower() == 'status':
                await cli.get_status()
            
            elif command.lower() == 'agents':
                await cli.list_agents()
            
            elif command.lower().startswith('task '):
                description = command[5:].strip()
                if description:
                    await cli.submit_task(description)
                else:
                    print("Please provide a task description")
            
            elif command.lower().startswith('memory '):
                agent_id = command[7:].strip()
                if agent_id in cli.system.memory_managers:
                    stats = cli.system.memory_managers[agent_id].get_statistics()
                    print(f"Memory stats for {agent_id}: {json.dumps(stats, indent=2)}")
                else:
                    print(f"Agent {agent_id} not found")
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    await cli.shutdown()
    print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())