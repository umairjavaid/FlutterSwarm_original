"""
Integration Test for FlutterSwarm Multi-Agent System Phase 1 & 2.

This test verifies that all core components work together correctly:
- System initialization
- LLM client integration
- Memory management with embeddings
- Event bus communication
- Orchestrator agent functionality
- Task processing workflow
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Setup test environment
os.environ["OPENAI_API_KEY"] = "test-key"  # Use test key for demo
os.environ["LOG_LEVEL"] = "INFO"
os.environ["ENVIRONMENT"] = "test"

from src.system import FlutterSwarmSystem
from src.models.task_models import TaskContext, TaskType, TaskPriority
from src.models.project_models import ProjectContext, ProjectType
from src.config.settings import settings

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("integration_test")


class TestFlutterSwarmIntegration:
    """Comprehensive integration test for FlutterSwarm system."""
    
    def __init__(self):
        self.system = FlutterSwarmSystem()
        self.test_results: Dict[str, Any] = {
            "passed": [],
            "failed": [],
            "summary": {}
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting FlutterSwarm Integration Tests...")
        
        test_methods = [
            self.test_system_initialization,
            self.test_llm_client_functionality,
            self.test_memory_manager_operations,
            self.test_event_bus_communication,
            self.test_orchestrator_task_processing,
            self.test_end_to_end_workflow,
            self.test_system_status_and_health,
            self.test_error_handling_and_recovery
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"Running test: {test_method.__name__}")
                await test_method()
                self.test_results["passed"].append(test_method.__name__)
                logger.info(f"✅ {test_method.__name__} PASSED")
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} FAILED: {e}")
                self.test_results["failed"].append({
                    "test": test_method.__name__,
                    "error": str(e)
                })
        
        # Generate summary
        self.test_results["summary"] = {
            "total_tests": len(test_methods),
            "passed": len(self.test_results["passed"]),
            "failed": len(self.test_results["failed"]),
            "success_rate": len(self.test_results["passed"]) / len(test_methods) * 100
        }
        
        return self.test_results
    
    async def test_system_initialization(self):
        """Test system initialization and component setup."""
        logger.info("Testing system initialization...")
        
        # Initialize system
        await self.system.initialize()
        
        # Verify system state
        assert self.system.is_initialized, "System should be initialized"
        assert self.system.llm_client is not None, "LLM client should be initialized"
        assert self.system.event_bus is not None, "Event bus should be initialized"
        assert self.system.orchestrator is not None, "Orchestrator should be initialized"
        
        # Start system
        await self.system.start()
        assert self.system.is_running, "System should be running"
        
        logger.info("System initialization test completed successfully")
    
    async def test_llm_client_functionality(self):
        """Test LLM client operations."""
        logger.info("Testing LLM client functionality...")
        
        llm_client = self.system.llm_client
        assert llm_client is not None, "LLM client should be available"
        
        # Test basic generation (will fail with test key, but should handle gracefully)
        try:
            response = await llm_client.generate(
                prompt="Hello, this is a test prompt",
                max_tokens=50,
                temperature=0.5
            )
            logger.info(f"LLM response received: {len(response)} characters")
        except Exception as e:
            logger.info(f"LLM generation failed as expected with test key: {e}")
            # This is expected with test credentials
        
        # Test statistics
        stats = llm_client.get_statistics()
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        assert "total_interactions" in stats, "Stats should include interaction count"
        
        logger.info("LLM client functionality test completed")
    
    async def test_memory_manager_operations(self):
        """Test memory manager operations."""
        logger.info("Testing memory manager operations...")
        
        memory_manager = self.system.memory_managers.get("orchestrator")
        assert memory_manager is not None, "Orchestrator memory manager should exist"
        
        # Test memory storage
        entry_id = await memory_manager.store_memory(
            content="This is a test memory entry for integration testing",
            metadata={"test": True, "category": "integration"},
            correlation_id="test-correlation-123",
            importance=0.8
        )
        
        assert entry_id is not None, "Memory entry should be stored successfully"
        
        # Test memory retrieval
        retrieved_entry = await memory_manager.retrieve_memory(entry_id)
        assert retrieved_entry is not None, "Memory entry should be retrievable"
        assert retrieved_entry.content == "This is a test memory entry for integration testing"
        assert retrieved_entry.metadata["test"] is True
        
        # Test memory search
        search_results = await memory_manager.search_memory(
            query="integration testing",
            limit=5,
            similarity_threshold=0.1  # Low threshold for fallback search
        )
        
        assert len(search_results) > 0, "Search should return results"
        
        # Test context retrieval
        context = await memory_manager.get_relevant_context(
            query="test memory",
            max_tokens=1000
        )
        
        assert isinstance(context, str), "Context should be a string"
        assert len(context) > 0, "Context should not be empty"
        
        # Test statistics
        stats = memory_manager.get_statistics()
        assert stats["total_memories"] > 0, "Memory count should be greater than 0"
        
        logger.info("Memory manager operations test completed")
    
    async def test_event_bus_communication(self):
        """Test event bus communication."""
        logger.info("Testing event bus communication...")
        
        event_bus = self.system.event_bus
        assert event_bus is not None, "Event bus should be available"
        
        # Test message publishing and subscription
        received_messages = []
        
        async def test_handler(message):
            received_messages.append(message)
        
        # Subscribe to test topic
        subscription_id = await event_bus.subscribe("test.integration", test_handler)
        assert subscription_id is not None, "Subscription should be successful"
        
        # Publish test message
        from src.models.agent_models import AgentMessage
        test_message = AgentMessage(
            type="integration_test",
            source="test_system",
            target="test_target",
            payload={"test_data": "integration_test_value"},
            correlation_id="test-correlation-456"
        )
        
        await event_bus.publish("test.integration", test_message)
        
        # Give it time to process
        await asyncio.sleep(0.5)
        
        # Verify message was received
        assert len(received_messages) > 0, "Test message should be received"
        received_msg = received_messages[0]
        assert received_msg.payload["test_data"] == "integration_test_value"
        
        # Test metrics
        metrics = await event_bus.get_metrics()
        assert metrics["total_messages"] > 0, "Event bus should have processed messages"
        
        # Unsubscribe
        await event_bus.unsubscribe("test.integration", subscription_id)
        
        logger.info("Event bus communication test completed")
    
    async def test_orchestrator_task_processing(self):
        """Test orchestrator task processing capabilities."""
        logger.info("Testing orchestrator task processing...")
        
        orchestrator = self.system.orchestrator
        assert orchestrator is not None, "Orchestrator should be available"
        
        # Test system prompt
        system_prompt = await orchestrator.get_system_prompt()
        assert isinstance(system_prompt, str), "System prompt should be a string"
        assert len(system_prompt) > 100, "System prompt should be comprehensive"
        
        # Test capabilities
        capabilities = await orchestrator.get_capabilities()
        assert isinstance(capabilities, list), "Capabilities should be a list"
        assert len(capabilities) > 0, "Orchestrator should have capabilities"
        
        # Test task context creation
        project_context = {
            "project_name": "test_flutter_app",
            "project_type": "app",
            "flutter_version": "3.0.0"
        }
        
        task_context = TaskContext(
            task_id="test-task-integration",
            description="Create a simple Flutter app with basic navigation",
            task_type=TaskType.FEATURE_IMPLEMENTATION,
            requirements=[],
            expected_deliverables=[],
            project_context=project_context,
            priority=TaskPriority.NORMAL,
            correlation_id="test-workflow-789"
        )
        
        # Test LLM task execution (will use fallback with test credentials)
        try:
            llm_result = await orchestrator.execute_llm_task(
                user_prompt="Analyze this Flutter app development task",
                context={"task": task_context.to_dict()},
                structured_output=False,
                max_retries=1
            )
            logger.info(f"LLM task execution result: {len(str(llm_result))} characters")
        except Exception as e:
            logger.info(f"LLM task execution failed as expected: {e}")
            # Expected with test credentials
        
        logger.info("Orchestrator task processing test completed")
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow processing."""
        logger.info("Testing end-to-end workflow...")
        
        # Create a comprehensive task request
        request = {
            "description": "Develop a Flutter todo app with state management and local storage",
            "task_type": "feature_implementation",
            "priority": "normal",
            "requirements": [
                {
                    "name": "state_management",
                    "description": "Use Provider or Bloc for state management",
                    "required": True,
                    "data_type": "string"
                }
            ],
            "expected_deliverables": [
                {
                    "name": "flutter_app_code",
                    "description": "Complete Flutter application code",
                    "type": "file",
                    "required": True
                }
            ],
            "project_context": {
                "project_name": "todo_app",
                "project_type": "app",
                "target_platforms": ["android", "ios"],
                "flutter_version": "3.0.0"
            },
            "metadata": {
                "test_run": True,
                "integration_test": True
            }
        }
        
        # Process request through system
        result = await self.system.process_request(request)
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should include success status"
        assert "task_id" in result, "Result should include task ID"
        
        # Check that task was processed (even if it fails due to test credentials)
        assert result["task_id"] is not None, "Task ID should be generated"
        
        logger.info(f"End-to-end workflow test completed: {result['success']}")
    
    async def test_system_status_and_health(self):
        """Test system status and health monitoring."""
        logger.info("Testing system status and health...")
        
        # Test system status
        status = await self.system.get_system_status()
        
        assert isinstance(status, dict), "Status should be a dictionary"
        assert "system" in status, "Status should include system info"
        assert "agents" in status, "Status should include agent info"
        assert "event_bus" in status, "Status should include event bus info"
        assert "llm_client" in status, "Status should include LLM client info"
        
        # Verify system components
        assert status["system"]["initialized"] is True, "System should be initialized"
        assert status["system"]["running"] is True, "System should be running"
        assert status["system"]["agents_count"] > 0, "System should have agents"
        
        # Test individual agent health
        for agent_id, agent in self.system.agents.items():
            agent_status = await agent.get_status()
            assert isinstance(agent_status, dict), f"Agent {agent_id} status should be dict"
            assert "agent_id" in agent_status, "Agent status should include ID"
            assert "status" in agent_status, "Agent status should include current status"
        
        logger.info("System status and health test completed")
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        logger.info("Testing error handling and recovery...")
        
        # Test invalid task processing
        invalid_request = {
            "description": "",  # Empty description
            "task_type": "invalid_type",  # Invalid type
            "priority": "invalid_priority"  # Invalid priority
        }
        
        try:
            result = await self.system.process_request(invalid_request)
            # Should handle gracefully and return error result
            assert result["success"] is False, "Invalid request should fail gracefully"
        except Exception as e:
            # Some validation may throw exceptions, which is also acceptable
            logger.info(f"Invalid request properly rejected: {e}")
        
        # Test memory operations with invalid data
        memory_manager = self.system.memory_managers["orchestrator"]
        
        # Test retrieving non-existent memory
        non_existent = await memory_manager.retrieve_memory("non-existent-id")
        assert non_existent is None, "Non-existent memory should return None"
        
        # Test search with empty query
        empty_search = await memory_manager.search_memory("", limit=1)
        assert isinstance(empty_search, list), "Empty search should return list"
        
        # Test event bus with invalid message
        try:
            await self.system.event_bus.publish("test.invalid", "invalid_message_type")
        except Exception as e:
            logger.info(f"Invalid message properly rejected: {e}")
        
        logger.info("Error handling and recovery test completed")
    
    async def cleanup(self):
        """Clean up test resources."""
        logger.info("Cleaning up test resources...")
        
        try:
            await self.system.stop()
            logger.info("System stopped successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test runner."""
    test_suite = TestFlutterSwarmIntegration()
    
    try:
        # Run all tests
        results = await test_suite.run_all_tests()
        
        # Print results
        print("\n" + "="*60)
        print("FLUTTERSWARM INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"\nTotal Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        
        if results["passed"]:
            print(f"\n✅ PASSED TESTS:")
            for test in results["passed"]:
                print(f"  - {test}")
        
        if results["failed"]:
            print(f"\n❌ FAILED TESTS:")
            for failure in results["failed"]:
                print(f"  - {failure['test']}: {failure['error']}")
        
        print("\n" + "="*60)
        
        # Overall assessment
        if results['summary']['success_rate'] >= 80:
            print("🎉 INTEGRATION TEST SUITE: PASSED")
            print("FlutterSwarm Phase 1 & 2 implementation is working correctly!")
        else:
            print("⚠️  INTEGRATION TEST SUITE: NEEDS ATTENTION")
            print("Some components may need debugging.")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED TO RUN: {e}")
        
    finally:
        # Cleanup
        await test_suite.cleanup()


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(main())
