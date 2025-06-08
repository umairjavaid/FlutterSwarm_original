import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';
import { LogOutputChannel } from '../utils/logOutputChannel';

export interface BackendStatus {
    running: boolean;
    port?: number;
    pid?: number;
    version?: string;
    startTime?: Date;
}

export class PythonBackendService implements vscode.Disposable {
    private backendProcess: ChildProcess | undefined;
    private port: number = 8000;
    private readonly maxStartupTime = 30000; // 30 seconds
    private readonly healthCheckInterval = 5000; // 5 seconds
    private healthCheckTimer: NodeJS.Timeout | undefined;
    private isStarting = false;

    constructor(private readonly logger: LogOutputChannel) {
        this.port = vscode.workspace.getConfiguration('flutter-swarm').get('backendPort', 8000);
    }

    async start(): Promise<void> {
        if (this.isRunning() || this.isStarting) {
            this.logger.info('Backend service is already running or starting');
            return;
        }

        this.isStarting = true;

        try {
            // Find available port
            this.port = await this.findAvailablePort(this.port);
            
            // Get Python path from configuration
            const pythonPath = vscode.workspace.getConfiguration('flutter-swarm').get('pythonPath', 'python3');
            
            // Get backend directory path
            const backendPath = this.getBackendPath();
            
            // Ensure Python environment is set up
            await this.setupPythonEnvironment(backendPath, pythonPath);
            
            // Start the backend process
            await this.startBackendProcess(backendPath, pythonPath);
            
            // Wait for backend to be ready
            await this.waitForBackendReady();
            
            // Start health check monitoring
            this.startHealthCheckMonitoring();
            
            this.logger.info(`Backend service started successfully on port ${this.port}`);
            
        } catch (error) {
            this.logger.error(`Failed to start backend service: ${error}`);
            throw error;
        } finally {
            this.isStarting = false;
        }
    }

    async stop(): Promise<void> {
        if (this.healthCheckTimer) {
            clearInterval(this.healthCheckTimer);
            this.healthCheckTimer = undefined;
        }

        if (this.backendProcess) {
            this.logger.info('Stopping backend process...');
            
            // Try graceful shutdown first
            try {
                await axios.post(`http://localhost:${this.port}/shutdown`, {}, { timeout: 5000 });
                // Give it time to shut down gracefully
                await new Promise(resolve => setTimeout(resolve, 2000));
            } catch (error) {
                this.logger.warn('Graceful shutdown request failed, forcing termination');
            }

            if (!this.backendProcess.killed) {
                this.backendProcess.kill('SIGTERM');
                
                // Force kill if still running after 5 seconds
                setTimeout(() => {
                    if (this.backendProcess && !this.backendProcess.killed) {
                        this.logger.warn('Force killing backend process');
                        this.backendProcess.kill('SIGKILL');
                    }
                }, 5000);
            }

            this.backendProcess = undefined;
        }
    }

    isRunning(): boolean {
        return this.backendProcess !== undefined && !this.backendProcess.killed;
    }

    async getPort(): Promise<number> {
        return this.port;
    }

    async getStatus(): Promise<BackendStatus> {
        if (!this.isRunning()) {
            return { running: false };
        }

        try {
            const response = await axios.get(`http://localhost:${this.port}/health`, { timeout: 5000 });
            return {
                running: true,
                port: this.port,
                pid: this.backendProcess?.pid,
                version: response.data.version,
                startTime: response.data.startTime
            };
        } catch (error) {
            return {
                running: false,
                port: this.port,
                pid: this.backendProcess?.pid
            };
        }
    }

    async notifyFileChange(filePath: string): Promise<void> {
        if (!this.isRunning()) {
            return;
        }

        try {
            await axios.post(`http://localhost:${this.port}/api/file-changed`, {
                filePath: filePath,
                timestamp: new Date().toISOString()
            }, { timeout: 5000 });
        } catch (error) {
            this.logger.warn(`Failed to notify backend of file change: ${error}`);
        }
    }

    private getBackendPath(): string {
        // Get the extension's installation directory
        const extensionPath = vscode.extensions.getExtension('flutterswarm.flutter-swarm')?.extensionPath;
        if (!extensionPath) {
            throw new Error('Could not determine extension path');
        }

        // Look for backend directory
        const backendPath = path.join(path.dirname(extensionPath), '..', 'python-backend');
        
        if (!fs.existsSync(backendPath)) {
            // Fallback to relative path for development
            const devBackendPath = path.join(__dirname, '..', '..', '..', 'python-backend');
            if (fs.existsSync(devBackendPath)) {
                return devBackendPath;
            }
            throw new Error(`Backend directory not found. Checked: ${backendPath} and ${devBackendPath}`);
        }

        return backendPath;
    }

    private async setupPythonEnvironment(backendPath: string, pythonPath: string): Promise<void> {
        const venvPath = path.join(backendPath, 'venv');
        const requirementsPath = path.join(backendPath, 'requirements.txt');

        // Check if virtual environment exists
        if (!fs.existsSync(venvPath)) {
            this.logger.info('Creating Python virtual environment...');
            await this.runCommand(pythonPath, ['-m', 'venv', 'venv'], backendPath);
        }

        // Check if requirements need to be installed
        const activateScript = process.platform === 'win32' 
            ? path.join(venvPath, 'Scripts', 'activate.bat')
            : path.join(venvPath, 'bin', 'activate');

        const pythonVenvPath = process.platform === 'win32'
            ? path.join(venvPath, 'Scripts', 'python.exe')
            : path.join(venvPath, 'bin', 'python');

        if (fs.existsSync(requirementsPath)) {
            this.logger.info('Installing Python dependencies...');
            await this.runCommand(pythonVenvPath, ['-m', 'pip', 'install', '-r', 'requirements.txt'], backendPath);
        }
    }

    private async startBackendProcess(backendPath: string, pythonPath: string): Promise<void> {
        const venvPath = path.join(backendPath, 'venv');
        const pythonVenvPath = process.platform === 'win32'
            ? path.join(venvPath, 'Scripts', 'python.exe')
            : path.join(venvPath, 'bin', 'python');

        const mainScript = path.join(backendPath, 'main.py');

        if (!fs.existsSync(mainScript)) {
            throw new Error(`Backend main script not found: ${mainScript}`);
        }

        this.logger.info(`Starting backend process: ${pythonVenvPath} ${mainScript}`);

        this.backendProcess = spawn(pythonVenvPath, [mainScript, '--port', this.port.toString()], {
            cwd: backendPath,
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONPATH: backendPath,
                FLUTTER_SWARM_PORT: this.port.toString()
            }
        });

        // Handle process output
        this.backendProcess.stdout?.on('data', (data) => {
            const output = data.toString().trim();
            if (output) {
                this.logger.info(`[Backend] ${output}`);
            }
        });

        this.backendProcess.stderr?.on('data', (data) => {
            const output = data.toString().trim();
            if (output) {
                this.logger.warn(`[Backend Error] ${output}`);
            }
        });

        this.backendProcess.on('exit', (code, signal) => {
            this.logger.info(`Backend process exited with code ${code}, signal ${signal}`);
            this.backendProcess = undefined;
            
            if (this.healthCheckTimer) {
                clearInterval(this.healthCheckTimer);
                this.healthCheckTimer = undefined;
            }
        });

        this.backendProcess.on('error', (error) => {
            this.logger.error(`Backend process error: ${error}`);
            this.backendProcess = undefined;
        });
    }

    private async waitForBackendReady(): Promise<void> {
        const startTime = Date.now();
        const checkInterval = 1000;

        while (Date.now() - startTime < this.maxStartupTime) {
            try {
                const response = await axios.get(`http://localhost:${this.port}/health`, { timeout: 2000 });
                if (response.status === 200) {
                    return;
                }
            } catch (error) {
                // Backend not ready yet, continue waiting
            }

            await new Promise(resolve => setTimeout(resolve, checkInterval));
        }

        throw new Error(`Backend failed to start within ${this.maxStartupTime / 1000} seconds`);
    }

    private startHealthCheckMonitoring(): void {
        this.healthCheckTimer = setInterval(async () => {
            try {
                await axios.get(`http://localhost:${this.port}/health`, { timeout: 5000 });
            } catch (error) {
                this.logger.warn('Backend health check failed, service may be down');
                if (this.backendProcess && !this.backendProcess.killed) {
                    this.logger.info('Attempting to restart backend service...');
                    await this.stop();
                    setTimeout(() => this.start(), 5000);
                }
            }
        }, this.healthCheckInterval);
    }

    private async findAvailablePort(startPort: number): Promise<number> {
        const net = require('net');
        
        for (let port = startPort; port < startPort + 100; port++) {
            try {
                await new Promise<void>((resolve, reject) => {
                    const server = net.createServer();
                    server.listen(port, () => {
                        server.close(() => resolve());
                    });
                    server.on('error', reject);
                });
                return port;
            } catch (error) {
                // Port is in use, try next one
            }
        }
        
        throw new Error(`No available port found in range ${startPort}-${startPort + 100}`);
    }

    private runCommand(command: string, args: string[], cwd?: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const process = spawn(command, args, { cwd, stdio: 'pipe' });
            
            let output = '';
            let errorOutput = '';
            
            process.stdout?.on('data', (data) => {
                output += data.toString();
            });
            
            process.stderr?.on('data', (data) => {
                errorOutput += data.toString();
            });
            
            process.on('close', (code) => {
                if (code === 0) {
                    if (output.trim()) {
                        this.logger.info(`[Command] ${output.trim()}`);
                    }
                    resolve();
                } else {
                    const error = errorOutput || `Command failed with exit code ${code}`;
                    this.logger.error(`[Command Error] ${error}`);
                    reject(new Error(error));
                }
            });
            
            process.on('error', (error) => {
                this.logger.error(`[Command Error] ${error}`);
                reject(error);
            });
        });
    }

    dispose(): void {
        this.stop();
    }
}
