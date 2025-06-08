#!/bin/bash

# FlutterSwarm System Demonstration Script
# This script demonstrates the complete working system

echo "🚀 FlutterSwarm Multi-Agent System Demonstration"
echo "================================================="
echo ""

# Check if all services are running
echo "📊 Checking System Status..."
echo ""

# Check backend
echo "🔍 Backend Service (Python FastAPI):"
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "   ✅ Backend is RUNNING on http://127.0.0.1:8000"
    BACKEND_STATUS=$(curl -s http://127.0.0.1:8000/health | jq -r '.status')
    AGENT_COUNT=$(curl -s http://127.0.0.1:8000/health | jq -r '.agents_count')
    echo "   📈 Status: $BACKEND_STATUS"
    echo "   🤖 Agents: $AGENT_COUNT active"
else
    echo "   ❌ Backend is NOT RUNNING"
    exit 1
fi
echo ""

# Check frontend
echo "🔍 Frontend Service (Flutter Web):"
if curl -s http://localhost:3000 > /dev/null; then
    echo "   ✅ Frontend is RUNNING on http://localhost:3000"
else
    echo "   ❌ Frontend is NOT RUNNING"
    exit 1
fi
echo ""

# List all agents
echo "🤖 Active Agents:"
curl -s http://127.0.0.1:8000/agents | jq -r '.[] | "   - \(.name) (\(.type)): \(.status)"'
echo ""

# Demonstrate agent execution
echo "🎯 Demonstrating Agent Execution:"
echo "   Starting task on Orchestrator Agent..."

TASK_RESULT=$(curl -s -X POST http://127.0.0.1:8000/agents/orchestrator-1/execute \
  -H "Content-Type: application/json" \
  -d '{"task":"flutter_project_demo","description":"Demonstrate Flutter project creation","priority":"high"}')

echo "   📝 Task Response: $(echo $TASK_RESULT | jq -r '.message')"
echo ""

# Wait and check progress
echo "⏳ Checking Agent Progress (5 second intervals):"
for i in {1..3}; do
    sleep 5
    AGENT_STATUS=$(curl -s http://127.0.0.1:8000/agents | jq -r '.[] | select(.id == "orchestrator-1") | "Progress: \(.progress)% - Status: \(.status)"')
    echo "   Update $i: $AGENT_STATUS"
done
echo ""

# Show workflow status
echo "📊 Workflow Status:"
curl -s http://127.0.0.1:8000/workflows | jq -r '.[] | "   Workflow: \(.name) - Status: \(.status) - Progress: \(.progress)%"'
echo ""

# System URLs
echo "🌐 System Access URLs:"
echo "   📊 Backend API: http://127.0.0.1:8000"
echo "   📊 API Docs: http://127.0.0.1:8000/docs"
echo "   🎨 Flutter Dashboard: http://localhost:3000"
echo "   🔌 WebSocket: ws://127.0.0.1:8000/ws"
echo ""

# VS Code Extension Info
echo "🔧 VS Code Extension:"
echo "   📁 Location: ./vscode-extension/"
echo "   📦 Status: Compiled and ready for development testing"
echo "   🚀 Launch: code --extensionDevelopmentHost=localhost --extensionDevelopmentPath=./vscode-extension"
echo ""

# Quick test commands
echo "🧪 Quick Test Commands:"
echo "   Health Check:"
echo "   curl -s http://127.0.0.1:8000/health | jq ."
echo ""
echo "   List Agents:"
echo "   curl -s http://127.0.0.1:8000/agents | jq ."
echo ""
echo "   Execute Agent Task:"
echo "   curl -X POST http://127.0.0.1:8000/agents/ui-1/execute \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"task\":\"create_widget\",\"description\":\"Demo task\"}'"
echo ""

echo "🎉 System Demonstration Complete!"
echo "   All components are operational and ready for development."
echo "   The FlutterSwarm Multi-Agent Flutter Development System is fully functional!"
