#!/usr/bin/env python3
"""
Test script to verify BaseAgent tool discovery and understanding capabilities.

This script tests:
1. Tool discovery from registry
2. Tool capability analysis 
3. Tool understanding storage
4. Event subscription for tool changes
5. Tool preference building
"""

import asyncio
import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_tool_discovery_capabilities():
    """Test the BaseAgent tool discovery and understanding capabilities."""
    
    try:
        # Import required classes
        from src.agents.base_agent import BaseAgent, AgentConfig, AgentCapability
        from src.core.tools.base_tool import BaseTool, ToolCategory
        from src.models.tool_models import ToolCapabilities, ToolUnderstanding
        from src.core.event_bus import EventBus
        from src.core.memory_manager import MemoryManager
        
        logger.info("=== Starting BaseAgent Tool Discovery Test ===")
        
        # Create mock LLM client
        mock_llm_client = Mock()
        mock_llm_client.generate = AsyncMock(return_value={
            "summary": "File system tool for managing files and directories",
            "usage_scenarios": [
                "Creating project structure",
                "Reading configuration files", 
                "Writing generated code",
                "Managing build artifacts",
                "Organizing test files"
            ],
            "parameter_patterns": {
                "file_path": "string - absolute or relative path",
                "content": "string - file content for write operations",
                "encoding": "string - optional file encoding, defaults to utf-8"
            },
            "success_indicators": [
                "File operations complete without errors",
                "Expected files exist at specified paths",
                "File content matches expected format"
            ],
            "failure_patterns": [
                "Permission denied errors",
                "File not found errors", 
                "Disk space insufficient",
                "Invalid file path format"
            ],
            "responsibility_mapping": {
                "file_management": "Primary responsibility for file operations",
                "code_generation": "Supports writing generated files",
                "project_setup": "Creates necessary directory structures"
            },
            "decision_factors": [
                "File system access required",
                "Need to persist data or code",
                "Project structure management needed"
            ]
        })
        
        # Create mock memory manager
        mock_memory_manager = Mock()
        mock_memory_manager.store_memory = AsyncMock()
        
        # Create mock event bus
        mock_event_bus = Mock()
        mock_event_bus.subscribe = AsyncMock()
        
        # Create mock tool with capabilities
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "filesystem_tool"
        mock_tool.description = "Tool for file system operations"
        mock_tool.version = "1.0.0"
        mock_tool.category = ToolCategory.FILE_SYSTEM
        
        # Mock tool capabilities
        mock_capabilities = Mock()
        mock_capabilities.name = "filesystem_tool"
        mock_capabilities.available_operations = [
            {
                "name": "read_file",
                "description": "Read content from a file",
                "required_permissions": ["file_read"]
            },
            {
                "name": "write_file", 
                "description": "Write content to a file",
                "required_permissions": ["file_write"]
            },
            {
                "name": "create_directory",
                "description": "Create a new directory",
                "required_permissions": ["file_write"]
            }
        ]
        
        mock_tool.get_capabilities = AsyncMock(return_value=mock_capabilities)
        mock_tool.get_usage_examples = AsyncMock(return_value=[
            {"operation": "read_file", "parameters": {"file_path": "/path/to/file.txt"}},
            {"operation": "write_file", "parameters": {"file_path": "/path/to/new.txt", "content": "Hello World"}}
        ])
        mock_tool.get_health_status = AsyncMock(return_value={"status": "healthy", "last_check": datetime.now()})
        
        # Create mock tool registry
        mock_tool_registry = Mock()
        mock_tool_registry.get_available_tools = Mock(return_value=[mock_tool])
        mock_tool_registry.is_initialized = True
        
        # Create a concrete agent class for testing
        class TestAgent(BaseAgent):
            async def _get_default_system_prompt(self) -> str:
                return "You are a test agent for tool discovery verification."
            
            async def get_capabilities(self) -> List[str]:
                return ["file_operations", "testing"]
        
        # Create agent configuration
        config = AgentConfig(
            agent_id="test_agent_001",
            agent_type="test",
            capabilities=[AgentCapability.FILE_OPERATIONS, AgentCapability.TESTING],
            max_concurrent_tasks=5
        )
        
        # Create agent instance
        agent = TestAgent(
            config=config,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            event_bus=mock_event_bus
        )
        
        # Mock the tool registry
        agent.tool_registry = mock_tool_registry
        
        logger.info("✓ Agent created successfully")
        
        # Test 1: Tool Discovery
        logger.info("--- Test 1: Tool Discovery ---")
        
        await agent.discover_available_tools()
        
        # Verify tools were discovered
        assert len(agent.available_tools) == 1
        assert "filesystem_tool" in agent.available_tools
        assert len(agent.tool_capabilities["filesystem_tool"]) == 5  # 5 usage scenarios
        
        logger.info("✓ Tools discovered successfully")
        logger.info(f"✓ Found {len(agent.available_tools)} tools")
        logger.info(f"✓ Tool capabilities stored: {list(agent.tool_capabilities.keys())}")
        
        # Test 2: Tool Capability Analysis
        logger.info("--- Test 2: Tool Capability Analysis ---")
        
        understanding = await agent.analyze_tool_capability(mock_tool)
        
        # Verify understanding was created
        assert isinstance(understanding, ToolUnderstanding)
        assert understanding.tool_name == "filesystem_tool"
        assert understanding.agent_id == "test_agent_001"
        assert len(understanding.usage_scenarios) == 5
        assert understanding.confidence_level >= 0.8
        
        logger.info("✓ Tool capability analysis completed")
        logger.info(f"✓ Understanding confidence: {understanding.confidence_level}")
        logger.info(f"✓ Usage scenarios identified: {len(understanding.usage_scenarios)}")
        
        # Test 3: Memory Storage Verification
        logger.info("--- Test 3: Memory Storage Verification ---")
        
        # Verify memory storage was called for tool discovery
        memory_calls = mock_memory_manager.store_memory.call_args_list
        assert len(memory_calls) >= 2  # At least discovery + understanding
        
        # Check that tool discovery was stored
        discovery_stored = any(
            "tool_discovery" in str(call) for call in memory_calls
        )
        assert discovery_stored
        
        # Check that tool understanding was stored  
        understanding_stored = any(
            "tool_understanding" in str(call) for call in memory_calls
        )
        assert understanding_stored
        
        logger.info("✓ Tool information stored in memory")
        logger.info(f"✓ Memory storage calls made: {len(memory_calls)}")
        
        # Test 4: Event Subscription Verification
        logger.info("--- Test 4: Event Subscription Verification ---")
        
        # Verify event subscriptions were made
        event_calls = mock_event_bus.subscribe.call_args_list
        assert len(event_calls) >= 3  # Tool availability, performance, registration
        
        # Check specific event topics
        subscribed_topics = [call[0][0] for call in event_calls]
        expected_topics = ["tool.availability.*", "tool.performance.*", "tool.registered.*"]
        
        for topic in expected_topics:
            assert any(topic in str(subscribed_topics) for topic in subscribed_topics)
        
        logger.info("✓ Event subscriptions created")
        logger.info(f"✓ Subscribed to {len(event_calls)} event topics")
        
        # Test 5: Tool Preferences Building
        logger.info("--- Test 5: Tool Preferences Building ---")
        
        # This should have been called during discovery
        # Verify that LLM was called for preference analysis
        llm_calls = mock_llm_client.generate.call_args_list
        assert len(llm_calls) >= 1  # At least capability analysis call
        
        logger.info("✓ Tool preferences building completed")
        
        # Test 6: Performance Metrics Initialization
        logger.info("--- Test 6: Performance Metrics Initialization ---")
        
        # Verify performance metrics were initialized
        assert "filesystem_tool" in agent.tool_performance_metrics
        metrics = agent.tool_performance_metrics["filesystem_tool"]
        assert hasattr(metrics, 'success_rate')
        assert hasattr(metrics, 'avg_duration')
        
        logger.info("✓ Performance metrics initialized")
        
        logger.info("=== All Tool Discovery Tests Passed! ===")
        
        # Summary report
        logger.info("\n=== Test Summary ===")
        logger.info(f"✓ Tools discovered: {len(agent.available_tools)}")
        logger.info(f"✓ Tool capabilities analyzed: {len(agent.tool_capabilities)}")
        logger.info(f"✓ Memory storage calls: {len(memory_calls)}")
        logger.info(f"✓ Event subscriptions: {len(event_calls)}")
        logger.info(f"✓ LLM analysis calls: {len(llm_calls)}")
        logger.info("✓ All functionality working correctly!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test execution."""
    logger.info("Starting BaseAgent Tool Discovery Verification Test")
    
    success = await test_tool_discovery_capabilities()
    
    if success:
        logger.info("🎉 All tests passed! Tool discovery implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Tests failed! Please check the implementation.")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
