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
    
    async def submit_task(self, description: str, task_type: str = "analysis", priority: str = "normal"):
        """Submit a task to the system."""
        if not self.system:
            print("❌ System not initialized. Run 'init' first.")
            return
        
        print(f"📝 Submitting task: {description[:60]}...")
        
        # Dynamically determine project name and type
        # This is a simple heuristic, a more robust solution might be needed
        project_name = "new_flutter_project" # Default project name
        project_type = "app" # Default project type

        if "app" in description.lower():
            project_type = "app"
        elif "plugin" in description.lower():
            project_type = "plugin"
        elif "package" in description.lower():
            project_type = "package"
        
        # Try to extract a project name (e.g., "create ... app named X")
        name_match = re.search(r"app named (\w+)", description, re.IGNORECASE)
        if name_match:
            project_name = name_match.group(1)
        else:
            # Fallback: try to generate a name from the description
            # e.g., "create prod standard flutter music app" -> "flutter_music_app"
            name_parts = []
            for word in description.lower().split():
                if word not in ["create", "a", "an", "the", "app", "flutter", "standard", "prod"]:
                    name_parts.append(word)
                if word == "app":
                    break # stop after "app"
            if name_parts:
                project_name = "_".join(name_parts)
            if not project_name: # if still empty
                project_name = "generated_project"


        request = {
            "description": description,
            "task_type": task_type,
            "priority": priority,
            "project_context": {
                "project_name": project_name,
                "project_type": project_type
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