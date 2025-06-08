#!/usr/bin/env python3
"""
Minimal test backend for FlutterSwarm VS Code extension integration testing.
"""

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Data models
class AgentStatus(BaseModel):
    id: str
    name: str
    type: str
    status: str  # idle, running, completed, error
    currentTask: Optional[str] = None
    progress: float = 0.0
    lastUpdate: str
    metrics: Dict = {}

class WorkflowState(BaseModel):
    id: str
    status: str  # pending, running, completed, error
    currentAgent: Optional[str] = None
    progress: float = 0.0
    totalTasks: int = 0
    completedTasks: int = 0
    startTime: str
    estimatedCompletion: Optional[str] = None

class AgentMessage(BaseModel):
    id: str
    type: str  # task_request, status_update, result_delivery, state_sync, error
    agentId: str
    timestamp: str
    data: Dict
    correlationId: Optional[str] = None

# Global state
agents: Dict[str, AgentStatus] = {}
workflows: Dict[str, WorkflowState] = {}
connected_clients: List[WebSocket] = []

# Initialize mock agents
def initialize_mock_agents():
    """Initialize mock agents for testing."""
    mock_agents = [
        {"id": "orchestrator-1", "name": "Orchestrator Agent", "type": "orchestrator"},
        {"id": "architecture-1", "name": "Architecture Agent", "type": "architecture"},
        {"id": "ui-1", "name": "UI/UX Agent", "type": "ui"},
        {"id": "testing-1", "name": "Testing Agent", "type": "testing"},
        {"id": "devops-1", "name": "DevOps Agent", "type": "devops"}
    ]
    
    for agent in mock_agents:
        agents[agent["id"]] = AgentStatus(
            id=agent["id"],
            name=agent["name"],
            type=agent["type"],
            status="idle",
            currentTask=None,
            progress=0.0,
            lastUpdate=datetime.now().isoformat(),
            metrics={"tasks_completed": 0, "success_rate": 100.0}
        )

def initialize_mock_workflow():
    """Initialize mock workflow for testing."""
    workflow_id = "test-workflow-1"
    workflows[workflow_id] = WorkflowState(
        id=workflow_id,
        status="pending",
        currentAgent=None,
        progress=0.0,
        totalTasks=5,
        completedTasks=0,
        startTime=datetime.now().isoformat(),
        estimatedCompletion=None
    )

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize mock data and start background tasks."""
    initialize_mock_agents()
    initialize_mock_workflow()
    
    # Start background task
    asyncio.create_task(simulate_agent_activity())
    
    print("✅ FlutterSwarm Test Backend started successfully!")
    print("📊 Mock agents and workflows initialized")
    print("🔌 WebSocket endpoint available at /ws")
    
    yield

# FastAPI app
app = FastAPI(title="FlutterSwarm Test Backend", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API endpoints
@app.get("/")
async def root():
    return {"message": "FlutterSwarm Test Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_count": len(agents),
        "workflows_count": len(workflows),
        "connected_clients": len(connected_clients)
    }

@app.get("/agents", response_model=List[AgentStatus])
async def get_agents():
    return list(agents.values())

@app.get("/agents/{agent_id}", response_model=AgentStatus)
async def get_agent(agent_id: str):
    if agent_id not in agents:
        return {"error": "Agent not found"}, 404
    return agents[agent_id]

@app.get("/workflows", response_model=List[WorkflowState])
async def get_workflows():
    return list(workflows.values())

@app.get("/workflows/{workflow_id}", response_model=WorkflowState)
async def get_workflow(workflow_id: str):
    if workflow_id not in workflows:
        return {"error": "Workflow not found"}, 404
    return workflows[workflow_id]

@app.post("/agents/{agent_id}/execute")
async def execute_agent_task(agent_id: str, task_data: Dict):
    if agent_id not in agents:
        return {"error": "Agent not found"}, 404
    
    # Update agent status
    agents[agent_id].status = "running"
    agents[agent_id].currentTask = task_data.get("task", "Unknown task")
    agents[agent_id].progress = 0.0
    agents[agent_id].lastUpdate = datetime.now().isoformat()
    
    # Broadcast update to connected clients
    await broadcast_agent_update(agent_id)
    
    return {"message": f"Task started for agent {agent_id}", "task": task_data}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial data to client
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to FlutterSwarm backend",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send current agents status
        for agent in agents.values():
            await websocket.send_json({
                "type": "agent_status",
                "data": agent.dict(),
                "timestamp": datetime.now().isoformat()
            })
        
        # Listen for messages from client
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message.get("type") == "request_status":
                # Send all agent statuses
                for agent in agents.values():
                    await websocket.send_json({
                        "type": "agent_status",
                        "data": agent.dict(),
                        "timestamp": datetime.now().isoformat()
                    })
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def broadcast_agent_update(agent_id: str):
    """Broadcast agent update to all connected clients."""
    if agent_id not in agents:
        return
    
    message = {
        "type": "agent_status",
        "data": agents[agent_id].dict(),
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected_clients = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        connected_clients.remove(client)

# Background task to simulate agent progress
async def simulate_agent_activity():
    """Simulate agent activity for testing."""
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds
        
        for agent in agents.values():
            if agent.status == "running":
                # Simulate progress
                agent.progress = min(100.0, agent.progress + 10.0)
                agent.lastUpdate = datetime.now().isoformat()
                
                # Complete task if progress reaches 100%
                if agent.progress >= 100.0:
                    agent.status = "completed"
                    agent.currentTask = None
                    agent.metrics["tasks_completed"] += 1
                
                # Broadcast update
                await broadcast_agent_update(agent.id)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize mock data and start background tasks."""
    initialize_mock_agents()
    initialize_mock_workflow()
    
    # Start background task
    asyncio.create_task(simulate_agent_activity())
    
    print("✅ FlutterSwarm Test Backend started successfully!")
    print("📊 Mock agents and workflows initialized")
    print("🔌 WebSocket endpoint available at /ws")
    
    yield

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "test_backend:app",  # Use import string instead of app object
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
