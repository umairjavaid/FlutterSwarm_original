import * as vscode from 'vscode';
import { FlutterSwarmExtension } from './flutterSwarmExtension';
import { PythonBackendService } from './services/pythonBackendService';
import { FlutterProjectDetector } from './services/flutterProjectDetector';
import { AgentTreeProvider } from './providers/agentTreeProvider';
import { LogOutputChannel } from './utils/logOutputChannel';

let extension: FlutterSwarmExtension | undefined;

export async function activate(context: vscode.ExtensionContext) {
    const logger = new LogOutputChannel('Flutter Swarm');
    logger.info('Flutter Swarm extension is being activated...');

    try {
        // Initialize the main extension class
        extension = new FlutterSwarmExtension(context, logger);
        
        // Register the extension with VS Code
        await extension.initialize();
        
        // Set context to indicate the extension is active
        vscode.commands.executeCommand('setContext', 'flutter-swarm.active', true);
        
        logger.info('Flutter Swarm extension activated successfully');
        
        // Show welcome message for first-time users
        const config = vscode.workspace.getConfiguration('flutter-swarm');
        const showWelcome = context.globalState.get('flutter-swarm.showWelcome', true);
        
        if (showWelcome) {
            const action = await vscode.window.showInformationMessage(
                'Welcome to Flutter Swarm! Would you like to see the getting started guide?',
                'Show Guide',
                'Maybe Later',
                "Don't Show Again"
            );
            
            if (action === 'Show Guide') {
                vscode.commands.executeCommand('flutter-swarm.openDashboard');
            } else if (action === "Don't Show Again") {
                context.globalState.update('flutter-swarm.showWelcome', false);
            }
        }
        
    } catch (error) {
        logger.error(`Failed to activate Flutter Swarm extension: ${error}`);
        vscode.window.showErrorMessage(`Flutter Swarm activation failed: ${error}`);
    }
}

export async function deactivate() {
    if (extension) {
        await extension.dispose();
        extension = undefined;
    }
    
    vscode.commands.executeCommand('setContext', 'flutter-swarm.active', false);
}
