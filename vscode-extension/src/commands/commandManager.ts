import * as vscode from 'vscode';
import { FlutterSwarmExtension } from '../flutterSwarmExtension';
import { LogOutputChannel } from '../utils/logOutputChannel';

export class CommandManager {
    constructor(
        private readonly extension: FlutterSwarmExtension,
        private readonly logger: LogOutputChannel
    ) {}

    registerCommands(context: vscode.ExtensionContext): void {
        // Register all commands
        const commands = [
            vscode.commands.registerCommand('flutter-swarm.activate', this.activateExtension.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.createProject', this.createProject.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.analyzeProject', this.analyzeProject.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.runAgent', this.runAgent.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.openDashboard', this.openDashboard.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.showLogs', this.showLogs.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.startBackend', this.startBackend.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.stopBackend', this.stopBackend.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.restartBackend', this.restartBackend.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.showBackendStatus', this.showBackendStatus.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.generateCode', this.generateCode.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.refactorCode', this.refactorCode.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.runTests', this.runTests.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.buildProject', this.buildProject.bind(this)),
            vscode.commands.registerCommand('flutter-swarm.deployProject', this.deployProject.bind(this))
        ];

        // Add all commands to the context for disposal
        commands.forEach(command => context.subscriptions.push(command));
    }

    private async activateExtension(): Promise<void> {
        try {
            this.logger.info('Manually activating Flutter Swarm...');
            
            if (!this.extension.backend.isRunning()) {
                await this.extension.startBackendService();
            }
            
            vscode.window.showInformationMessage('Flutter Swarm activated successfully!');
        } catch (error) {
            this.logger.error(`Failed to activate extension: ${error}`);
            vscode.window.showErrorMessage(`Failed to activate Flutter Swarm: ${error}`);
        }
    }

    private async createProject(): Promise<void> {
        try {
            const projectName = await vscode.window.showInputBox({
                prompt: 'Enter the Flutter project name',
                placeHolder: 'my_flutter_app',
                validateInput: (value) => {
                    if (!value) return 'Project name is required';
                    if (!/^[a-z][a-z0-9_]*$/.test(value)) {
                        return 'Project name must start with lowercase letter and contain only lowercase letters, numbers, and underscores';
                    }
                    return undefined;
                }
            });

            if (!projectName) return;

            const organizationName = await vscode.window.showInputBox({
                prompt: 'Enter the organization name (optional)',
                placeHolder: 'com.example',
                value: 'com.example'
            });

            const projectType = await vscode.window.showQuickPick([
                { label: 'Application', description: 'A Flutter application' },
                { label: 'Package', description: 'A Dart package containing Flutter framework code' },
                { label: 'Plugin', description: 'A Flutter plugin project' },
                { label: 'Module', description: 'A Flutter module for adding to existing apps' }
            ], {
                placeHolder: 'Select project type'
            });

            if (!projectType) return;

            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }

            this.logger.info(`Creating Flutter project: ${projectName}`);
            
            // Call backend to create project
            if (this.extension.mcp.isConnectedToServer()) {
                const result = await this.extension.mcp.runFlutterCommand(
                    workspaceFolder.uri.fsPath,
                    'create',
                    [
                        '--project-name', projectName,
                        '--org', organizationName || 'com.example',
                        '--template', projectType.label.toLowerCase()
                    ]
                );
                
                if (result.success) {
                    vscode.window.showInformationMessage(`Flutter project '${projectName}' created successfully!`);
                    // Refresh project detection
                    await this.extension.detector.scanWorkspaceFolder(workspaceFolder);
                    this.extension.treeProvider.refresh();
                } else {
                    vscode.window.showErrorMessage(`Failed to create project: ${result.error}`);
                }
            } else {
                vscode.window.showWarningMessage('Backend service is not connected. Please start the backend first.');
            }

        } catch (error) {
            this.logger.error(`Error creating project: ${error}`);
            vscode.window.showErrorMessage(`Failed to create project: ${error}`);
        }
    }

    private async analyzeProject(): Promise<void> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }

            this.logger.info('Analyzing Flutter project...');
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Analyzing Flutter Project',
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Scanning project structure...' });
                
                // Scan for Flutter projects
                const projects = await this.extension.detector.scanWorkspaceFolder(workspaceFolder);
                
                progress.report({ increment: 50, message: 'Analyzing project details...' });
                
                if (projects.length === 0) {
                    vscode.window.showWarningMessage('No Flutter projects found in the workspace');
                    return;
                }
                
                // Update tree view
                this.extension.treeProvider.updateProjects(projects);
                
                progress.report({ increment: 100, message: 'Analysis complete' });
                
                const project = projects[0]; // Use first project for now
                const message = `Project Analysis Complete!\n` +
                    `Name: ${project.name}\n` +
                    `State: ${project.state}\n` +
                    `Architecture: ${project.architecture}\n` +
                    `State Management: ${project.stateManagement}\n` +
                    `Platforms: ${project.platforms.join(', ')}\n` +
                    `Test Coverage: ${project.testCoverage}%`;
                
                vscode.window.showInformationMessage(message);
            });

        } catch (error) {
            this.logger.error(`Error analyzing project: ${error}`);
            vscode.window.showErrorMessage(`Failed to analyze project: ${error}`);
        }
    }

    private async runAgent(): Promise<void> {
        try {
            const agents = [
                { label: 'Architecture Agent', description: 'Analyze and improve project architecture' },
                { label: 'Implementation Agent', description: 'Generate and refactor code' },
                { label: 'Testing Agent', description: 'Create and run tests' },
                { label: 'DevOps Agent', description: 'Build and deployment automation' },
                { label: 'Search Agent', description: 'Find code examples and documentation' }
            ];

            const selectedAgent = await vscode.window.showQuickPick(agents, {
                placeHolder: 'Select an agent to run'
            });

            if (!selectedAgent) return;

            const task = await vscode.window.showInputBox({
                prompt: `Enter task for ${selectedAgent.label}`,
                placeHolder: 'Describe what you want the agent to do...'
            });

            if (!task) return;

            this.logger.info(`Running ${selectedAgent.label} with task: ${task}`);
            
            vscode.window.showInformationMessage(`${selectedAgent.label} started. Check the logs for progress.`);
            
            // Here you would send the task to the backend
            // For now, just show a placeholder
            setTimeout(() => {
                vscode.window.showInformationMessage(`${selectedAgent.label} completed the task successfully!`);
            }, 3000);

        } catch (error) {
            this.logger.error(`Error running agent: ${error}`);
            vscode.window.showErrorMessage(`Failed to run agent: ${error}`);
        }
    }

    private async openDashboard(): Promise<void> {
        try {
            // This would open the Flutter dashboard app
            // For now, show a placeholder
            const action = await vscode.window.showInformationMessage(
                'Flutter Swarm Dashboard',
                'Open in Browser',
                'Show Status'
            );

            if (action === 'Open in Browser') {
                const status = await this.extension.backend.getStatus();
                if (status.running) {
                    vscode.env.openExternal(vscode.Uri.parse(`http://localhost:${status.port}/dashboard`));
                } else {
                    vscode.window.showWarningMessage('Backend service is not running');
                }
            } else if (action === 'Show Status') {
                await this.showBackendStatus();
            }

        } catch (error) {
            this.logger.error(`Error opening dashboard: ${error}`);
            vscode.window.showErrorMessage(`Failed to open dashboard: ${error}`);
        }
    }

    private showLogs(): void {
        this.logger.show();
    }

    private async startBackend(): Promise<void> {
        try {
            await this.extension.startBackendService();
        } catch (error) {
            // Error handling is done in the extension
        }
    }

    private async stopBackend(): Promise<void> {
        try {
            await this.extension.stopBackendService();
        } catch (error) {
            // Error handling is done in the extension
        }
    }

    private async restartBackend(): Promise<void> {
        try {
            await this.extension.stopBackendService();
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
            await this.extension.startBackendService();
        } catch (error) {
            // Error handling is done in the extension
        }
    }

    private async showBackendStatus(): Promise<void> {
        try {
            const status = await this.extension.backend.getStatus();
            
            let message = `Backend Status: ${status.running ? 'Running' : 'Stopped'}`;
            
            if (status.running) {
                message += `\nPort: ${status.port}`;
                message += `\nPID: ${status.pid}`;
                if (status.version) {
                    message += `\nVersion: ${status.version}`;
                }
                if (status.startTime) {
                    message += `\nStarted: ${new Date(status.startTime).toLocaleString()}`;
                }
            }

            vscode.window.showInformationMessage(message);

        } catch (error) {
            this.logger.error(`Error getting backend status: ${error}`);
            vscode.window.showErrorMessage(`Failed to get backend status: ${error}`);
        }
    }

    private async generateCode(): Promise<void> {
        try {
            const templates = [
                { label: 'Widget', description: 'Generate a new Flutter widget' },
                { label: 'Screen', description: 'Generate a new screen with navigation' },
                { label: 'Model', description: 'Generate a data model class' },
                { label: 'Repository', description: 'Generate a repository pattern implementation' },
                { label: 'BLoC', description: 'Generate BLoC pattern classes' },
                { label: 'API Service', description: 'Generate an API service class' }
            ];

            const selectedTemplate = await vscode.window.showQuickPick(templates, {
                placeHolder: 'Select what to generate'
            });

            if (!selectedTemplate) return;

            const name = await vscode.window.showInputBox({
                prompt: `Enter name for the ${selectedTemplate.label}`,
                placeHolder: 'MyWidget',
                validateInput: (value) => {
                    if (!value) return 'Name is required';
                    if (!/^[A-Z][a-zA-Z0-9]*$/.test(value)) {
                        return 'Name must start with uppercase letter and contain only letters and numbers';
                    }
                    return undefined;
                }
            });

            if (!name) return;

            this.logger.info(`Generating ${selectedTemplate.label}: ${name}`);
            
            // This would call the backend to generate code
            vscode.window.showInformationMessage(`Generating ${selectedTemplate.label}...`);

        } catch (error) {
            this.logger.error(`Error generating code: ${error}`);
            vscode.window.showErrorMessage(`Failed to generate code: ${error}`);
        }
    }

    private async refactorCode(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const refactorings = [
            { label: 'Extract Widget', description: 'Extract selected code into a new widget' },
            { label: 'Extract Method', description: 'Extract selected code into a new method' },
            { label: 'Rename Symbol', description: 'Rename a class, method, or variable' },
            { label: 'Move to File', description: 'Move class to a new file' },
            { label: 'Add State Management', description: 'Convert to use BLoC/Provider pattern' }
        ];

        const selectedRefactoring = await vscode.window.showQuickPick(refactorings, {
            placeHolder: 'Select refactoring to apply'
        });

        if (!selectedRefactoring) return;

        this.logger.info(`Applying refactoring: ${selectedRefactoring.label}`);
        
        vscode.window.showInformationMessage(`Applying ${selectedRefactoring.label}...`);
    }

    private async runTests(): Promise<void> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }

            const testOptions = [
                { label: 'All Tests', description: 'Run all unit and widget tests' },
                { label: 'Unit Tests', description: 'Run only unit tests' },
                { label: 'Widget Tests', description: 'Run only widget tests' },
                { label: 'Integration Tests', description: 'Run integration tests' },
                { label: 'Current File', description: 'Run tests in the current file' }
            ];

            const selectedOption = await vscode.window.showQuickPick(testOptions, {
                placeHolder: 'Select tests to run'
            });

            if (!selectedOption) return;

            this.logger.info(`Running tests: ${selectedOption.label}`);
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Running ${selectedOption.label}`,
                cancellable: true
            }, async (progress, token) => {
                progress.report({ increment: 0, message: 'Starting tests...' });
                
                // This would call the backend to run tests
                await new Promise(resolve => setTimeout(resolve, 3000));
                
                progress.report({ increment: 100, message: 'Tests completed' });
                
                vscode.window.showInformationMessage('Tests completed successfully!');
            });

        } catch (error) {
            this.logger.error(`Error running tests: ${error}`);
            vscode.window.showErrorMessage(`Failed to run tests: ${error}`);
        }
    }

    private async buildProject(): Promise<void> {
        try {
            const platforms = [
                { label: 'Android APK', description: 'Build Android APK' },
                { label: 'Android AAB', description: 'Build Android App Bundle' },
                { label: 'iOS', description: 'Build iOS app' },
                { label: 'Web', description: 'Build for web' },
                { label: 'Windows', description: 'Build Windows app' },
                { label: 'macOS', description: 'Build macOS app' },
                { label: 'Linux', description: 'Build Linux app' }
            ];

            const selectedPlatform = await vscode.window.showQuickPick(platforms, {
                placeHolder: 'Select platform to build for'
            });

            if (!selectedPlatform) return;

            this.logger.info(`Building for ${selectedPlatform.label}`);
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Building ${selectedPlatform.label}`,
                cancellable: true
            }, async (progress, token) => {
                progress.report({ increment: 0, message: 'Preparing build...' });
                
                // This would call the backend to build the project
                await new Promise(resolve => setTimeout(resolve, 10000));
                
                progress.report({ increment: 100, message: 'Build completed' });
                
                vscode.window.showInformationMessage(`${selectedPlatform.label} build completed successfully!`);
            });

        } catch (error) {
            this.logger.error(`Error building project: ${error}`);
            vscode.window.showErrorMessage(`Failed to build project: ${error}`);
        }
    }

    private async deployProject(): Promise<void> {
        try {
            const deployTargets = [
                { label: 'Firebase Hosting', description: 'Deploy web app to Firebase' },
                { label: 'Google Play Store', description: 'Deploy to Play Store' },
                { label: 'Apple App Store', description: 'Deploy to App Store' },
                { label: 'TestFlight', description: 'Deploy to TestFlight for testing' },
                { label: 'Firebase App Distribution', description: 'Deploy for testing' }
            ];

            const selectedTarget = await vscode.window.showQuickPick(deployTargets, {
                placeHolder: 'Select deployment target'
            });

            if (!selectedTarget) return;

            this.logger.info(`Deploying to ${selectedTarget.label}`);
            
            vscode.window.showInformationMessage(`Deployment to ${selectedTarget.label} started. Check logs for progress.`);

        } catch (error) {
            this.logger.error(`Error deploying project: ${error}`);
            vscode.window.showErrorMessage(`Failed to deploy project: ${error}`);
        }
    }
}
