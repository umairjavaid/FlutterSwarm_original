"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FlutterSwarmExtension = void 0;
const vscode = __importStar(require("vscode"));
const pythonBackendService_1 = require("./services/pythonBackendService");
const flutterProjectDetector_1 = require("./services/flutterProjectDetector");
const webSocketClient_1 = require("./services/webSocketClient");
const mcpClient_1 = require("./services/mcpClient");
const agentTreeProvider_1 = require("./providers/agentTreeProvider");
const commandManager_1 = require("./commands/commandManager");
class FlutterSwarmExtension {
    constructor(context, logger) {
        this.context = context;
        this.logger = logger;
        this.disposables = [];
        this.pythonBackend = new pythonBackendService_1.PythonBackendService(logger);
        this.projectDetector = new flutterProjectDetector_1.FlutterProjectDetector(logger);
        this.webSocketClient = new webSocketClient_1.WebSocketClient(logger);
        this.mcpClient = new mcpClient_1.MCPClient(logger);
        this.agentTreeProvider = new agentTreeProvider_1.AgentTreeProvider(logger);
        this.commandManager = new commandManager_1.CommandManager(this, logger);
    }
    async initialize() {
        this.logger.info('Initializing Flutter Swarm extension...');
        // Register tree view provider
        const agentTreeView = vscode.window.createTreeView('flutter-swarm-agents', {
            treeDataProvider: this.agentTreeProvider,
            showCollapseAll: true
        });
        this.disposables.push(agentTreeView);
        // Register commands
        this.commandManager.registerCommands(this.context);
        // Set up file system watchers
        this.setupFileWatchers();
        // Detect Flutter projects in the workspace
        await this.detectAndInitializeFlutterProjects();
        // Start Python backend if auto-start is enabled
        const config = vscode.workspace.getConfiguration('flutter-swarm');
        if (config.get('autoStart', true)) {
            await this.startBackendService();
        }
        this.logger.info('Flutter Swarm extension initialization complete');
    }
    setupFileWatchers() {
        // Watch for pubspec.yaml changes
        const pubspecWatcher = vscode.workspace.createFileSystemWatcher('**/pubspec.yaml');
        pubspecWatcher.onDidCreate(this.onPubspecChange.bind(this));
        pubspecWatcher.onDidChange(this.onPubspecChange.bind(this));
        pubspecWatcher.onDidDelete(this.onPubspecDelete.bind(this));
        this.disposables.push(pubspecWatcher);
        // Watch for Dart file changes
        const dartWatcher = vscode.workspace.createFileSystemWatcher('**/*.dart');
        dartWatcher.onDidCreate(this.onDartFileChange.bind(this));
        dartWatcher.onDidChange(this.onDartFileChange.bind(this));
        dartWatcher.onDidDelete(this.onDartFileChange.bind(this));
        this.disposables.push(dartWatcher);
        // Watch for workspace folder changes
        vscode.workspace.onDidChangeWorkspaceFolders(this.onWorkspaceFoldersChange.bind(this));
    }
    async onPubspecChange(uri) {
        this.logger.info(`Pubspec file changed: ${uri.fsPath}`);
        await this.projectDetector.analyzeProject(uri.fsPath);
        this.agentTreeProvider.refresh();
    }
    async onPubspecDelete(uri) {
        this.logger.info(`Pubspec file deleted: ${uri.fsPath}`);
        // Handle project deletion
        this.agentTreeProvider.refresh();
    }
    async onDartFileChange(uri) {
        // Throttle Dart file changes to avoid excessive processing
        // Implementation would include debouncing logic
        if (this.pythonBackend.isRunning()) {
            await this.pythonBackend.notifyFileChange(uri.fsPath);
        }
    }
    async onWorkspaceFoldersChange(event) {
        for (const folder of event.added) {
            await this.projectDetector.scanWorkspaceFolder(folder);
        }
        this.agentTreeProvider.refresh();
    }
    async detectAndInitializeFlutterProjects() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return;
        }
        for (const folder of workspaceFolders) {
            await this.projectDetector.scanWorkspaceFolder(folder);
        }
        const flutterProjects = this.projectDetector.getDetectedProjects();
        if (flutterProjects.length > 0) {
            this.logger.info(`Detected ${flutterProjects.length} Flutter project(s)`);
            this.agentTreeProvider.updateProjects(flutterProjects);
        }
    }
    async startBackendService() {
        try {
            this.logger.info('Starting Python backend service...');
            await this.pythonBackend.start();
            // Initialize WebSocket connection
            const port = await this.pythonBackend.getPort();
            await this.webSocketClient.connect(`ws://localhost:${port}/ws`);
            // Initialize MCP client
            await this.mcpClient.connect(`http://localhost:${port}`);
            this.logger.info('Python backend service started successfully');
            vscode.commands.executeCommand('setContext', 'flutter-swarm.backend.running', true);
        }
        catch (error) {
            this.logger.error(`Failed to start backend service: ${error}`);
            vscode.window.showErrorMessage(`Failed to start Flutter Swarm backend: ${error}`);
        }
    }
    async stopBackendService() {
        try {
            this.logger.info('Stopping Python backend service...');
            await this.webSocketClient.disconnect();
            await this.mcpClient.disconnect();
            await this.pythonBackend.stop();
            this.logger.info('Python backend service stopped');
            vscode.commands.executeCommand('setContext', 'flutter-swarm.backend.running', false);
        }
        catch (error) {
            this.logger.error(`Error stopping backend service: ${error}`);
        }
    }
    // Getter methods for accessing services
    get backend() {
        return this.pythonBackend;
    }
    get detector() {
        return this.projectDetector;
    }
    get websocket() {
        return this.webSocketClient;
    }
    get mcp() {
        return this.mcpClient;
    }
    get treeProvider() {
        return this.agentTreeProvider;
    }
    async dispose() {
        this.logger.info('Disposing Flutter Swarm extension...');
        await this.stopBackendService();
        // Dispose all disposables
        this.disposables.forEach(disposable => disposable.dispose());
        this.disposables.length = 0;
        this.logger.info('Flutter Swarm extension disposed');
    }
}
exports.FlutterSwarmExtension = FlutterSwarmExtension;
//# sourceMappingURL=flutterSwarmExtension.js.map