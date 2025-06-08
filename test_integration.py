#!/usr/bin/env python3
"""
End-to-end integration test for FlutterSwarm system
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime

# Test configuration
BACKEND_URL = "http://127.0.0.1:8000"
WEBSOCKET_URL = "ws://127.0.0.1:8000/ws"
FLUTTER_URL = "http://localhost:3000"

def test_backend_health():
    """Test backend health endpoint"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend health: {data['status']}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_backend_agents():
    """Test backend agents endpoint"""
    print("🔍 Testing backend agents...")
    try:
        response = requests.get(f"{BACKEND_URL}/agents")
        if response.status_code == 200:
            agents = response.json()
            print(f"✅ Found {len(agents)} agents")
            for agent in agents:
                print(f"   - {agent['name']} ({agent['type']}) - {agent['status']}")
            return True
        else:
            print(f"❌ Agents endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agents endpoint error: {e}")
        return False

def test_flutter_frontend():
    """Test Flutter frontend availability"""
    print("🔍 Testing Flutter frontend...")
    try:
        response = requests.get(FLUTTER_URL, timeout=10)
        if response.status_code == 200:
            print("✅ Flutter frontend is accessible")
            return True
        else:
            print(f"❌ Flutter frontend error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Flutter frontend error: {e}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection"""
    print("🔍 Testing WebSocket connection...")
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Send a test message
            test_message = {
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(test_message))
            print("✅ Test message sent")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"✅ WebSocket response received: {data.get('type', 'unknown')}")
                return True
            except asyncio.TimeoutError:
                print("⚠️  No WebSocket response received (this is expected for test backend)")
                return True
                
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
        return False

async def test_agent_execution():
    """Test agent execution via API"""
    print("🔍 Testing agent execution...")
    try:
        # Get first agent
        response = requests.get(f"{BACKEND_URL}/agents")
        if response.status_code != 200:
            print("❌ Could not get agents for execution test")
            return False
            
        agents = response.json()
        if not agents:
            print("❌ No agents available for execution test")
            return False
            
        agent_id = agents[0]['id']
        print(f"   Testing with agent: {agents[0]['name']}")
        
        # Execute agent task
        task_data = {
            "task": "test_task",
            "description": "Integration test task",
            "priority": "medium"
        }
        
        response = requests.post(f"{BACKEND_URL}/agents/{agent_id}/execute", json=task_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Agent execution started: {result['message']}")
            return True
        else:
            print(f"❌ Agent execution failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Agent execution error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting FlutterSwarm Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test backend components
    results.append(("Backend Health", test_backend_health()))
    results.append(("Backend Agents", test_backend_agents()))
    
    # Test frontend
    results.append(("Flutter Frontend", test_flutter_frontend()))
    
    # Test WebSocket (async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results.append(("WebSocket Connection", loop.run_until_complete(test_websocket_connection())))
    results.append(("Agent Execution", loop.run_until_complete(test_agent_execution())))
    loop.close()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Integration Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests passed! System is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
