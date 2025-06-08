#!/bin/bash

# FlutterSwarm System Launcher
# Easily start all components of the Multi-Agent Flutter Development System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
    echo -e "${PURPLE}🚀 FlutterSwarm Multi-Agent System${NC}"
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check Flutter
    if ! command -v flutter &> /dev/null; then
        print_warning "Flutter not found - Flutter frontend won't be available"
    else
        print_success "Flutter found: $(flutter --version | head -1)"
    fi
    
    # Check VS Code
    if ! command -v code &> /dev/null; then
        print_warning "VS Code not found - Extension won't be installable via CLI"
    else
        print_success "VS Code found"
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found - VS Code extension compilation may fail"
    else
        print_success "Node.js found: $(node --version)"
    fi
}

# Start Python backend
start_backend() {
    print_status "Starting Python backend..."
    cd "$(dirname "$0")/python-backend"
    
    # Check if backend is already running
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        print_warning "Backend already running on port 8000"
        return 0
    fi
    
    # Start backend in background
    python3 -m uvicorn test_backend:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    
    # Wait for backend to start
    print_status "Waiting for backend to start..."
    for i in {1..10}; do
        if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
            print_success "Backend started successfully (PID: $BACKEND_PID)"
            print_status "Backend available at: http://127.0.0.1:8000"
            print_status "API docs available at: http://127.0.0.1:8000/docs"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    print_error "Backend failed to start"
    return 1
}

# Compile VS Code extension
setup_extension() {
    print_status "Setting up VS Code extension..."
    cd "$(dirname "$0")/vscode-extension"
    
    if [ ! -d "node_modules" ]; then
        print_status "Installing extension dependencies..."
        npm install
    fi
    
    print_status "Compiling extension..."
    npm run compile
    
    if [ -f "out/extension.js" ]; then
        print_success "Extension compiled successfully"
        print_status "Extension location: $(pwd)"
        print_status "To install: code --install-extension ."
    else
        print_error "Extension compilation failed"
        return 1
    fi
}

# Start Flutter frontend
start_flutter() {
    if ! command -v flutter &> /dev/null; then
        print_warning "Flutter not available - skipping frontend startup"
        return 0
    fi
    
    print_status "Starting Flutter frontend..."
    cd "$(dirname "$0")/flutter_swarm_frontend"
    
    # Get dependencies
    flutter pub get
    
    print_status "Starting Flutter web server..."
    print_warning "This will block the terminal. Use Ctrl+C to stop."
    print_status "Frontend will be available at: http://localhost:3000"
    
    flutter run -d web-server --web-port 3000
}

# Show system status
show_status() {
    print_header
    echo
    print_status "System Status Check:"
    echo
    
    # Check backend
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        BACKEND_STATUS=$(curl -s http://127.0.0.1:8000/health)
        print_success "✅ Python Backend: Running"
        echo "   └── Health: $(echo "$BACKEND_STATUS" | jq -r '.status')"
        echo "   └── Agents: $(echo "$BACKEND_STATUS" | jq -r '.agents_count')"
        echo "   └── URL: http://127.0.0.1:8000"
    else
        print_error "❌ Python Backend: Not running"
    fi
    
    # Check extension
    if [ -f "vscode-extension/out/extension.js" ]; then
        print_success "✅ VS Code Extension: Compiled"
        echo "   └── Location: $(pwd)/vscode-extension"
    else
        print_warning "⚠️  VS Code Extension: Not compiled"
    fi
    
    # Check Flutter
    if [ -f "flutter_swarm_frontend/pubspec.lock" ]; then
        print_success "✅ Flutter Frontend: Dependencies ready"
        echo "   └── Location: $(pwd)/flutter_swarm_frontend"
    else
        print_warning "⚠️  Flutter Frontend: Dependencies not installed"
    fi
    
    echo
    print_status "Available Commands:"
    echo "   └── $0 start     - Start all system components"
    echo "   └── $0 backend   - Start only the Python backend"
    echo "   └── $0 frontend  - Start only the Flutter frontend"
    echo "   └── $0 extension - Setup VS Code extension"
    echo "   └── $0 stop      - Stop all running components"
    echo "   └── $0 status    - Show system status"
}

# Stop all components
stop_system() {
    print_status "Stopping FlutterSwarm system..."
    
    # Stop backend
    if [ -f "python-backend/backend.pid" ]; then
        BACKEND_PID=$(cat python-backend/backend.pid)
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            kill "$BACKEND_PID"
            print_success "Backend stopped (PID: $BACKEND_PID)"
        fi
        rm -f python-backend/backend.pid
    fi
    
    # Kill any remaining Flutter processes
    pkill -f "flutter run" && print_success "Flutter processes stopped" || true
    
    print_success "System stopped"
}

# Main function
main() {
    case "${1:-status}" in
        "start")
            check_prerequisites
            start_backend
            setup_extension
            echo
            print_success "🎉 FlutterSwarm system is ready!"
            echo
            print_status "Next steps:"
            echo "1. Install VS Code extension: cd vscode-extension && code --install-extension ."
            echo "2. Start Flutter frontend: $0 frontend"
            echo "3. Open test workspace: code test_workspace"
            echo
            ;;
        "backend")
            check_prerequisites
            start_backend
            ;;
        "frontend")
            start_flutter
            ;;
        "extension")
            setup_extension
            ;;
        "stop")
            stop_system
            ;;
        "status")
            show_status
            ;;
        *)
            print_error "Unknown command: $1"
            show_status
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
