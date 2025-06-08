import * as vscode from 'vscode';
import { PythonBackendService } from './services/pythonBackendService';
import { FlutterProjectDetector } from './services/flutterProjectDetector';
import { WebSocketClient } from './services/webSocketClient';
import { MCPClient } from './services/mcpClient';
import { AgentTreeProvider } from './providers/agentTreeProvider';
import { LogOutputChannel } from './utils/logOutputChannel';
import { CommandManager } from './commands/commandManager';

export class FlutterSwarmExtension implements vscode.Disposable {
    private readonly disposables: vscode.Disposable[] = [];
    private readonly pythonBackend: PythonBackendService;
    private readonly projectDetector: FlutterProjectDetector;
    private readonly webSocketClient: WebSocketClient;
    private readonly mcpClient: MCPClient;
    private readonly agentTreeProvider: AgentTreeProvider;
    private readonly commandManager: CommandManager;

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly logger: LogOutputChannel
    ) {
        this.pythonBackend = new PythonBackendService(logger);
        this.projectDetector = new FlutterProjectDetector(logger);
        this.webSocketClient = new WebSocketClient(logger);
        this.mcpClient = new MCPClient(logger);
        this.agentTreeProvider = new AgentTreeProvider(logger);
        this.commandManager = new CommandManager(this, logger);
    }

    async initialize(): Promise<void> {
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

    private setupFileWatchers(): void {
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

    private async onPubspecChange(uri: vscode.Uri): Promise<void> {
        this.logger.info(`Pubspec file changed: ${uri.fsPath}`);
        await this.projectDetector.analyzeProject(uri.fsPath);
        this.agentTreeProvider.refresh();
    }

    private async onPubspecDelete(uri: vscode.Uri): Promise<void> {
        this.logger.info(`Pubspec file deleted: ${uri.fsPath}`);
        // Handle project deletion
        this.agentTreeProvider.refresh();
    }

    private async onDartFileChange(uri: vscode.Uri): Promise<void> {
        // Throttle Dart file changes to avoid excessive processing
        // Implementation would include debouncing logic
        if (this.pythonBackend.isRunning()) {
            await this.pythonBackend.notifyFileChange(uri.fsPath);
        }
    }

    private async onWorkspaceFoldersChange(event: vscode.WorkspaceFoldersChangeEvent): Promise<void> {
        for (const folder of event.added) {
            await this.projectDetector.scanWorkspaceFolder(folder);
        }
        this.agentTreeProvider.refresh();
    }

    private async detectAndInitializeFlutterProjects(): Promise<void> {
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

    async startBackendService(): Promise<void> {
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
            
        } catch (error) {
            this.logger.error(`Failed to start backend service: ${error}`);
            vscode.window.showErrorMessage(`Failed to start Flutter Swarm backend: ${error}`);
        }
    }

    async stopBackendService(): Promise<void> {
        try {
            this.logger.info('Stopping Python backend service...');
            
            await this.webSocketClient.disconnect();
            await this.mcpClient.disconnect();
            await this.pythonBackend.stop();
            
            this.logger.info('Python backend service stopped');
            vscode.commands.executeCommand('setContext', 'flutter-swarm.backend.running', false);
            
        } catch (error) {
            this.logger.error(`Error stopping backend service: ${error}`);
        }
    }

    // Getter methods for accessing services
    get backend(): PythonBackendService {
        return this.pythonBackend;
    }

    get detector(): FlutterProjectDetector {
        return this.projectDetector;
    }

    get websocket(): WebSocketClient {
        return this.webSocketClient;
    }

    get mcp(): MCPClient {
        return this.mcpClient;
    }

    get treeProvider(): AgentTreeProvider {
        return this.agentTreeProvider;
    }

    async dispose(): Promise<void> {
        this.logger.info('Disposing Flutter Swarm extension...');
        
        await this.stopBackendService();
        
        // Dispose all disposables
        this.disposables.forEach(disposable => disposable.dispose());
        this.disposables.length = 0;
        
        this.logger.info('Flutter Swarm extension disposed');
    }
}
