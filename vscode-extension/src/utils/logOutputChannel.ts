import * as vscode from 'vscode';

export class LogOutputChannel implements vscode.Disposable {
    private readonly outputChannel: vscode.OutputChannel;
    private readonly logLevel: string;

    constructor(name: string) {
        this.outputChannel = vscode.window.createOutputChannel(name);
        this.logLevel = vscode.workspace.getConfiguration('flutter-swarm').get('logLevel', 'INFO');
    }

    debug(message: string): void {
        if (this.shouldLog('DEBUG')) {
            this.writeLog('DEBUG', message);
        }
    }

    info(message: string): void {
        if (this.shouldLog('INFO')) {
            this.writeLog('INFO', message);
        }
    }

    warn(message: string): void {
        if (this.shouldLog('WARNING')) {
            this.writeLog('WARNING', message);
        }
    }

    error(message: string): void {
        if (this.shouldLog('ERROR')) {
            this.writeLog('ERROR', message);
        }
    }

    show(): void {
        this.outputChannel.show();
    }

    hide(): void {
        this.outputChannel.hide();
    }

    clear(): void {
        this.outputChannel.clear();
    }

    private writeLog(level: string, message: string): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] [${level}] ${message}`;
        this.outputChannel.appendLine(logMessage);
    }

    private shouldLog(level: string): boolean {
        const levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
        const currentLevelIndex = levels.indexOf(this.logLevel);
        const messageLevelIndex = levels.indexOf(level);
        
        return messageLevelIndex >= currentLevelIndex;
    }

    dispose(): void {
        this.outputChannel.dispose();
    }
}
