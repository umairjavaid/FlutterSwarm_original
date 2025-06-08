#!/usr/bin/env python3
import asyncio
import websockets
import json
import sys

async def test_websocket():
    try:
        uri = "ws://127.0.0.1:8000/ws"
        print(f"🔌 Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Send ping message
            ping_message = {"type": "ping", "timestamp": "2025-06-08T18:00:00Z"}
            await websocket.send(json.dumps(ping_message))
            print("📤 Sent ping message")
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📥 Received: {data}")
            
            if data.get('type') == 'connection_established':
                print("✅ Connection established correctly")
                
                # Wait for agent status messages
                print("🔄 Waiting for agent status updates...")
                for _ in range(3):  # Listen for a few messages
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(response)
                        if data.get('type') == 'agent_status':
                            print(f"📊 Agent Status: {data['data']['name']} - {data['data']['status']}")
                    except asyncio.TimeoutError:
                        break
                
                print("✅ WebSocket test completed successfully")
                return True
            else:
                print("❌ Unexpected response type")
                return False
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    sys.exit(0 if result else 1)
