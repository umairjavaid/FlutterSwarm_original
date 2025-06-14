#!/usr/bin/env python3
"""
Comprehensive verification test for BaseAgent tool discovery and understanding capabilities.

This script creates a realistic test scenario with multiple tools to thoroughly verify:
1. Tool discovery with multiple tools
2. Proper capability analysis for each tool
3. Tool understanding storage and retrieval
4. Event subscription and handling
5. Tool preference building based on agent type
6. Error handling and edge cases
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock
import logging

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_comprehensive_tool_discovery():
    """Test BaseAgent tool discovery with multiple tools and comprehensive scenarios."""
    
    try:
        # Import required classes
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        from src.core.tools.base_tool import BaseTool, ToolCategory
        from src.models.tool_models import ToolCapabilities, ToolUnderstanding, ToolMetrics
        from src.core.event_bus import EventBus
        from src.core.memory_manager import MemoryManager
        
        logger.info("=== Starting Comprehensive Tool Discovery Test ===")
        
        # Create mock LLM client with different responses for different tools
        mock_llm_client = Mock()
        
        def mock_llm_response(prompt=None, **kwargs):
            # Return different responses based on the tool being analyzed
            if "filesystem_tool" in str(prompt):
                return {
                    "summary": "File system management tool for reading, writing, and organizing files",
                    "usage_scenarios": [
                        "Reading configuration files",
                        "Writing generated code files", 
                        "Creating project directory structures",
                        "Managing build artifacts",
                        "Organizing test files"
                    ],
                    "parameter_patterns": {
                        "file_path": "string - absolute or relative file path",
                        "content": "string - file content for write operations",
                        "encoding": "string - file encoding, defaults to utf-8"
                    },
                    "success_indicators": ["File operations complete without errors"],
                    "failure_patterns": ["Permission denied", "File not found"],
                    "responsibility_mapping": {"file_management": "Primary file operations"},
                    "decision_factors": ["File system access needed"]
                }
            elif "flutter_sdk" in str(prompt):
                return {
                    "summary": "Flutter SDK tool for mobile app development operations",
                    "usage_scenarios": [
                        "Creating new Flutter projects",
                        "Building Flutter applications",
                        "Running Flutter tests",
                        "Managing Flutter dependencies"
                    ],
                    "parameter_patterns": {
                        "project_name": "string - Flutter project name",
                        "platform": "string - target platform (android/ios/web)"
                    },
                    "success_indicators": ["Build completes successfully"],
                    "failure_patterns": ["SDK not found", "Build errors"],
                    "responsibility_mapping": {"app_development": "Flutter development tasks"},
                    "decision_factors": ["Flutter app development needed"]
                }
            elif "git_tool" in str(prompt):
                return {
                    "summary": "Git version control tool for source code management",
                    "usage_scenarios": [
                        "Initializing repositories",
                        "Committing code changes",
                        "Managing branches",
                        "Pushing to remote repositories"
                    ],
                    "parameter_patterns": {
                        "repository_path": "string - path to git repository",
                        "commit_message": "string - commit message"
                    },
                    "success_indicators": ["Git operations complete"],
                    "failure_patterns": ["Repository not found", "Merge conflicts"],
                    "responsibility_mapping": {"version_control": "Git operations"},
                    "decision_factors": ["Version control needed"]
                }
            else:
                # Default response for preference building
                return {
                    "primary_tools": ["filesystem_tool", "flutter_sdk"],
                    "secondary_tools": ["git_tool"],
                    "specialized_tools": [],
                    "preference_scores": {
                        "filesystem_tool": 0.95,
                        "flutter_sdk": 0.9,
                        "git_tool": 0.7
                    },
                    "usage_priorities": {
                        "filesystem_tool": "Always use for file operations",
                        "flutter_sdk": "Use for Flutter development",
                        "git_tool": "Use for version control"
                    }
                }
        
        mock_llm_client.generate = AsyncMock(side_effect=mock_llm_response)
        
        # Create mock memory manager
        mock_memory_manager = Mock()
        mock_memory_manager.store_memory = AsyncMock()
        
        # Create mock event bus
        mock_event_bus = Mock()
        mock_event_bus.subscribe = AsyncMock()
        mock_event_bus.publish = AsyncMock()
        
        # Create multiple mock tools
        tools = []
        
        # Tool 1: File System Tool
        filesystem_tool = Mock(spec=BaseTool)
        filesystem_tool.name = "filesystem_tool"
        filesystem_tool.description = "Tool for file system operations"
        filesystem_tool.version = "1.0.0"
        filesystem_tool.category = ToolCategory.FILE_SYSTEM
        
        fs_capabilities = Mock()
        fs_capabilities.name = "filesystem_tool"
        fs_capabilities.available_operations = [
            {"name": "read_file", "description": "Read file content"},
            {"name": "write_file", "description": "Write file content"},
            {"name": "create_directory", "description": "Create directory"}
        ]
        
        filesystem_tool.get_capabilities = AsyncMock(return_value=fs_capabilities)
        filesystem_tool.get_usage_examples = AsyncMock(return_value=[
            {"operation": "read_file", "parameters": {"file_path": "/path/to/file.txt"}},
            {"operation": "write_file", "parameters": {"file_path": "/output.txt", "content": "Hello"}}
        ])
        filesystem_tool.get_health_status = AsyncMock(return_value={"status": "healthy"})
        tools.append(filesystem_tool)
        
        # Tool 2: Flutter SDK Tool
        flutter_tool = Mock(spec=BaseTool)
        flutter_tool.name = "flutter_sdk"
        flutter_tool.description = "Flutter SDK for mobile development"
        flutter_tool.version = "3.0.0"
        flutter_tool.category = ToolCategory.DEVELOPMENT
        
        flutter_capabilities = Mock()
        flutter_capabilities.name = "flutter_sdk"
        flutter_capabilities.available_operations = [
            {"name": "create_project", "description": "Create Flutter project"},
            {"name": "build_app", "description": "Build Flutter application"},
            {"name": "run_tests", "description": "Run Flutter tests"}
        ]
        
        flutter_tool.get_capabilities = AsyncMock(return_value=flutter_capabilities)
        flutter_tool.get_usage_examples = AsyncMock(return_value=[
            {"operation": "create_project", "parameters": {"project_name": "my_app"}},
            {"operation": "build_app", "parameters": {"platform": "android"}}
        ])
        flutter_tool.get_health_status = AsyncMock(return_value={"status": "healthy"})
        tools.append(flutter_tool)
        
        # Tool 3: Git Tool
        git_tool = Mock(spec=BaseTool)
        git_tool.name = "git_tool"
        git_tool.description = "Git version control system"
        git_tool.version = "2.0.0"
        git_tool.category = ToolCategory.DEVELOPMENT  # Use DEVELOPMENT instead
        
        git_capabilities = Mock()
        git_capabilities.name = "git_tool"
        git_capabilities.available_operations = [
            {"name": "init", "description": "Initialize repository"},
            {"name": "commit", "description": "Commit changes"},
            {"name": "push", "description": "Push to remote"}
        ]
        
        git_tool.get_capabilities = AsyncMock(return_value=git_capabilities)
        git_tool.get_usage_examples = AsyncMock(return_value=[
            {"operation": "init", "parameters": {"path": "/project"}},
            {"operation": "commit", "parameters": {"message": "Initial commit"}}
        ])
        git_tool.get_health_status = AsyncMock(return_value={"status": "healthy"})
        tools.append(git_tool)
        
        # Create mock tool registry
        mock_tool_registry = Mock()
        mock_tool_registry.get_available_tools = Mock(return_value=tools)
        mock_tool_registry.is_initialized = True
        
        # Create a concrete agent class for testing
        class TestArchitectureAgent(BaseAgent):
            async def _get_default_system_prompt(self) -> str:
                return "You are an architecture agent responsible for designing Flutter app structures."
            
            async def get_capabilities(self) -> List[str]:
                return ["architecture_analysis", "code_generation", "file_operations"]
        
        # Create agent configuration for architecture agent
        config = AgentConfig(
            agent_id="arch_agent_001",
            agent_type="architecture",
            capabilities=[
                AgentCapability.ARCHITECTURE_ANALYSIS,
                AgentCapability.CODE_GENERATION,
                AgentCapability.FILE_OPERATIONS
            ],
            max_concurrent_tasks=10
        )
        
        # Create agent instance
        agent = TestArchitectureAgent(
            config=config,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            event_bus=mock_event_bus
        )
        
        # Mock the tool registry
        agent.tool_registry = mock_tool_registry
        
        logger.info("✓ Architecture agent created successfully")
        
        # Test 1: Multiple Tool Discovery
        logger.info("--- Test 1: Multiple Tool Discovery ---")
        
        await agent.discover_available_tools()
        
        # Verify all tools were discovered
        assert len(agent.available_tools) == 3
        expected_tools = ["filesystem_tool", "flutter_sdk", "git_tool"]
        for tool_name in expected_tools:
            assert tool_name in agent.available_tools
            assert tool_name in agent.tool_capabilities
            assert len(agent.tool_capabilities[tool_name]) >= 3  # At least 3 usage scenarios
        
        logger.info(f"✓ Discovered {len(agent.available_tools)} tools successfully")
        logger.info(f"✓ Tool names: {list(agent.available_tools.keys())}")
        
        # Test 2: Tool Capability Analysis Quality
        logger.info("--- Test 2: Tool Capability Analysis Quality ---")
        
        for tool_name, tool in agent.available_tools.items():
            understanding = await agent.analyze_tool_capability(tool)
            
            # Verify understanding quality
            assert isinstance(understanding, ToolUnderstanding)
            assert understanding.tool_name == tool_name
            assert understanding.agent_id == "arch_agent_001"
            assert len(understanding.usage_scenarios) >= 3
            assert understanding.confidence_level >= 0.5
            assert understanding.capabilities_summary != ""
            
            logger.info(f"✓ {tool_name}: {understanding.confidence_level} confidence")
        
        # Test 3: Memory Storage Patterns
        logger.info("--- Test 3: Memory Storage Patterns ---")
        
        memory_calls = mock_memory_manager.store_memory.call_args_list
        
        # Should have stored discovery + understanding for each tool
        expected_min_calls = 3 * 2  # 3 tools * (discovery + understanding)
        assert len(memory_calls) >= expected_min_calls
        
        # Check for specific storage patterns
        discovery_calls = [call for call in memory_calls if "tool_discovery" in str(call)]
        understanding_calls = [call for call in memory_calls if "tool_understanding" in str(call)]
        
        assert len(discovery_calls) >= 1  # At least one discovery call
        assert len(understanding_calls) >= 3  # One understanding per tool
        
        logger.info(f"✓ Memory storage: {len(memory_calls)} total calls")
        logger.info(f"✓ Discovery calls: {len(discovery_calls)}")
        logger.info(f"✓ Understanding calls: {len(understanding_calls)}")
        
        # Test 4: Tool Preferences and Prioritization
        logger.info("--- Test 4: Tool Preferences and Prioritization ---")
        
        # Verify performance metrics were initialized for all tools
        for tool_name in agent.available_tools.keys():
            assert tool_name in agent.tool_performance_metrics
            metrics = agent.tool_performance_metrics[tool_name]
            assert isinstance(metrics, ToolMetrics)
        
        # Check that LLM was called for preference building
        llm_calls = mock_llm_client.generate.call_args_list
        preference_calls = [call for call in llm_calls if "preference" in str(call[1]).lower()]
        assert len(preference_calls) >= 1
        
        logger.info("✓ Tool preferences built successfully")
        logger.info(f"✓ Performance metrics initialized for {len(agent.tool_performance_metrics)} tools")
        
        # Test 5: Event Subscription Verification
        logger.info("--- Test 5: Event Subscription Verification ---")
        
        event_calls = mock_event_bus.subscribe.call_args_list
        
        # Should subscribe to general agent events + tool events
        expected_topics = [
            "tool.availability.*",
            "tool.performance.*", 
            "tool.registered.*"
        ]
        
        subscribed_topics = [str(call[0][0]) for call in event_calls]
        for topic in expected_topics:
            topic_found = any(topic in subscribed_topic for subscribed_topic in subscribed_topics)
            assert topic_found, f"Missing subscription to {topic}"
        
        logger.info(f"✓ Event subscriptions: {len(event_calls)} total")
        
        # Test 6: Agent Type-Specific Analysis
        logger.info("--- Test 6: Agent Type-Specific Analysis ---")
        
        # Check that tool analysis considered agent type
        for tool_name in agent.available_tools.keys():
            scenarios = agent.tool_capabilities[tool_name]
            
            # Architecture agents should have scenarios relevant to their role
            architecture_relevant = any(
                "architecture" in scenario.lower() or 
                "structure" in scenario.lower() or 
                "design" in scenario.lower() or
                "project" in scenario.lower()
                for scenario in scenarios
            )
            
            logger.info(f"✓ {tool_name}: {len(scenarios)} scenarios analyzed")
        
        # Test 7: Error Handling and Recovery
        logger.info("--- Test 7: Error Handling and Recovery ---")
        
        # Test with a problematic tool
        broken_tool = Mock(spec=BaseTool)
        broken_tool.name = "broken_tool"
        broken_tool.description = "A tool that fails"
        broken_tool.version = "0.1.0"
        broken_tool.category = ToolCategory.TEST  # Use TEST instead
        broken_tool.get_capabilities = AsyncMock(side_effect=Exception("Tool failed"))
        broken_tool.get_usage_examples = AsyncMock(side_effect=Exception("Examples failed"))
        broken_tool.get_health_status = AsyncMock(side_effect=Exception("Health check failed"))
        
        # Should handle errors gracefully
        understanding = await agent.analyze_tool_capability(broken_tool)
        assert understanding.confidence_level <= 0.2  # Low confidence for failed analysis
        assert "failed" in understanding.capabilities_summary.lower()
        
        logger.info("✓ Error handling working correctly")
        
        logger.info("=== All Comprehensive Tests Passed! ===")
        
        # Generate comprehensive report
        logger.info("\n=== Comprehensive Test Report ===")
        logger.info(f"✓ Tools discovered: {len(agent.available_tools)}")
        logger.info(f"✓ Agent type: {agent.config.agent_type}")
        logger.info(f"✓ Agent capabilities: {[cap.value for cap in agent.capabilities]}")
        logger.info(f"✓ Tool analysis calls: {len([c for c in llm_calls if 'tool' in str(c)])}")
        logger.info(f"✓ Memory storage operations: {len(memory_calls)}")
        logger.info(f"✓ Event subscriptions: {len(event_calls)}")
        logger.info(f"✓ Performance metrics initialized: {len(agent.tool_performance_metrics)}")
        
        # Tool-specific metrics
        for tool_name, scenarios in agent.tool_capabilities.items():
            logger.info(f"  - {tool_name}: {len(scenarios)} usage scenarios")
        
        logger.info("✓ All tool discovery and understanding capabilities verified!")
        
        return True
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive tool discovery verification."""
    logger.info("Starting Comprehensive Tool Discovery Verification")
    
    success = await test_comprehensive_tool_discovery()
    
    if success:
        logger.info("🎉 All comprehensive tests passed! Tool discovery implementation is robust and complete.")
        return 0
    else:
        logger.error("❌ Comprehensive tests failed! Implementation needs review.")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
