import axios, { AxiosInstance } from 'axios';
import { LogOutputChannel } from '../utils/logOutputChannel';

export interface MCPTool {
    name: string;
    description: string;
    inputSchema: any;
}

export interface MCPServer {
    name: string;
    version: string;
    tools: MCPTool[];
    capabilities: string[];
}

export interface MCPRequest {
    method: string;
    params?: any;
    id?: string;
}

export interface MCPResponse {
    id?: string;
    result?: any;
    error?: {
        code: number;
        message: string;
        data?: any;
    };
}

export class MCPClient {
    private httpClient: AxiosInstance | undefined;
    private baseUrl: string = '';
    private isConnected = false;

    constructor(private readonly logger: LogOutputChannel) {}

    async connect(baseUrl: string): Promise<void> {
        this.baseUrl = baseUrl;
        
        this.httpClient = axios.create({
            baseURL: `${baseUrl}/mcp`,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'VS Code Flutter Swarm Extension'
            }
        });

        try {
            // Test connection and get server info
            const response = await this.httpClient.get('/servers');
            this.isConnected = true;
            this.logger.info(`MCP client connected to ${baseUrl}`);
            this.logger.info(`Available MCP servers: ${response.data.servers.map((s: any) => s.name).join(', ')}`);
        } catch (error) {
            this.logger.error(`Failed to connect to MCP server: ${error}`);
            throw error;
        }
    }

    async disconnect(): Promise<void> {
        this.httpClient = undefined;
        this.isConnected = false;
        this.logger.info('MCP client disconnected');
    }

    async listServers(): Promise<MCPServer[]> {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }

        try {
            const response = await this.httpClient.get('/servers');
            return response.data.servers;
        } catch (error) {
            this.logger.error(`Failed to list MCP servers: ${error}`);
            throw error;
        }
    }

    async listTools(serverName?: string): Promise<MCPTool[]> {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }

        try {
            const url = serverName ? `/servers/${serverName}/tools` : '/tools';
            const response = await this.httpClient.get(url);
            return response.data.tools;
        } catch (error) {
            this.logger.error(`Failed to list MCP tools: ${error}`);
            throw error;
        }
    }

    async callTool(serverName: string, toolName: string, params: any): Promise<any> {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }

        try {
            const request: MCPRequest = {
                method: 'tools/call',
                params: {
                    name: toolName,
                    arguments: params
                },
                id: this.generateRequestId()
            };

            this.logger.debug(`Calling MCP tool: ${serverName}/${toolName}`);
            
            const response = await this.httpClient.post(`/servers/${serverName}/call`, request);
            
            if (response.data.error) {
                throw new Error(`MCP tool error: ${response.data.error.message}`);
            }

            return response.data.result;
        } catch (error) {
            this.logger.error(`Failed to call MCP tool ${serverName}/${toolName}: ${error}`);
            throw error;
        }
    }

    // Flutter-specific MCP tool calls
    async getProjectStructure(projectPath: string): Promise<any> {
        return this.callTool('flutter-project', 'get_project_structure', { path: projectPath });
    }

    async analyzeProject(projectPath: string): Promise<any> {
        return this.callTool('flutter-analysis', 'analyze_project', { path: projectPath });
    }

    async getBuildStatus(projectPath: string): Promise<any> {
        return this.callTool('flutter-build', 'get_build_status', { path: projectPath });
    }

    async runFlutterCommand(projectPath: string, command: string, args: string[]): Promise<any> {
        return this.callTool('flutter-build', 'run_command', { 
            path: projectPath, 
            command: command, 
            args: args 
        });
    }

    async getTestResults(projectPath: string): Promise<any> {
        return this.callTool('flutter-testing', 'get_test_results', { path: projectPath });
    }

    async runTests(projectPath: string, testPath?: string): Promise<any> {
        return this.callTool('flutter-testing', 'run_tests', { 
            projectPath: projectPath,
            testPath: testPath 
        });
    }

    async getCodeCoverage(projectPath: string): Promise<any> {
        return this.callTool('flutter-testing', 'get_coverage', { path: projectPath });
    }

    async getLintResults(projectPath: string): Promise<any> {
        return this.callTool('flutter-analysis', 'get_lint_results', { path: projectPath });
    }

    async formatCode(projectPath: string, filePath?: string): Promise<any> {
        return this.callTool('flutter-analysis', 'format_code', { 
            projectPath: projectPath,
            filePath: filePath 
        });
    }

    async generateCode(template: string, params: any): Promise<any> {
        return this.callTool('flutter-generation', 'generate_code', { 
            template: template,
            parameters: params 
        });
    }

    async refactorCode(projectPath: string, refactoring: string, params: any): Promise<any> {
        return this.callTool('flutter-generation', 'refactor_code', { 
            projectPath: projectPath,
            refactoring: refactoring,
            parameters: params 
        });
    }

    // Generic MCP request method
    async sendRequest(serverName: string, request: MCPRequest): Promise<MCPResponse> {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }

        try {
            const response = await this.httpClient.post(`/servers/${serverName}/request`, request);
            return response.data;
        } catch (error) {
            this.logger.error(`Failed to send MCP request to ${serverName}: ${error}`);
            throw error;
        }
    }

    isConnectedToServer(): boolean {
        return this.isConnected;
    }

    private generateRequestId(): string {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}
