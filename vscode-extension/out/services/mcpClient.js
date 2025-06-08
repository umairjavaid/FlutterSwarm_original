"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MCPClient = void 0;
const axios_1 = __importDefault(require("axios"));
class MCPClient {
    constructor(logger) {
        this.logger = logger;
        this.baseUrl = '';
        this.isConnected = false;
    }
    async connect(baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = axios_1.default.create({
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
            this.logger.info(`Available MCP servers: ${response.data.servers.map((s) => s.name).join(', ')}`);
        }
        catch (error) {
            this.logger.error(`Failed to connect to MCP server: ${error}`);
            throw error;
        }
    }
    async disconnect() {
        this.httpClient = undefined;
        this.isConnected = false;
        this.logger.info('MCP client disconnected');
    }
    async listServers() {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }
        try {
            const response = await this.httpClient.get('/servers');
            return response.data.servers;
        }
        catch (error) {
            this.logger.error(`Failed to list MCP servers: ${error}`);
            throw error;
        }
    }
    async listTools(serverName) {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }
        try {
            const url = serverName ? `/servers/${serverName}/tools` : '/tools';
            const response = await this.httpClient.get(url);
            return response.data.tools;
        }
        catch (error) {
            this.logger.error(`Failed to list MCP tools: ${error}`);
            throw error;
        }
    }
    async callTool(serverName, toolName, params) {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }
        try {
            const request = {
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
        }
        catch (error) {
            this.logger.error(`Failed to call MCP tool ${serverName}/${toolName}: ${error}`);
            throw error;
        }
    }
    // Flutter-specific MCP tool calls
    async getProjectStructure(projectPath) {
        return this.callTool('flutter-project', 'get_project_structure', { path: projectPath });
    }
    async analyzeProject(projectPath) {
        return this.callTool('flutter-analysis', 'analyze_project', { path: projectPath });
    }
    async getBuildStatus(projectPath) {
        return this.callTool('flutter-build', 'get_build_status', { path: projectPath });
    }
    async runFlutterCommand(projectPath, command, args) {
        return this.callTool('flutter-build', 'run_command', {
            path: projectPath,
            command: command,
            args: args
        });
    }
    async getTestResults(projectPath) {
        return this.callTool('flutter-testing', 'get_test_results', { path: projectPath });
    }
    async runTests(projectPath, testPath) {
        return this.callTool('flutter-testing', 'run_tests', {
            projectPath: projectPath,
            testPath: testPath
        });
    }
    async getCodeCoverage(projectPath) {
        return this.callTool('flutter-testing', 'get_coverage', { path: projectPath });
    }
    async getLintResults(projectPath) {
        return this.callTool('flutter-analysis', 'get_lint_results', { path: projectPath });
    }
    async formatCode(projectPath, filePath) {
        return this.callTool('flutter-analysis', 'format_code', {
            projectPath: projectPath,
            filePath: filePath
        });
    }
    async generateCode(template, params) {
        return this.callTool('flutter-generation', 'generate_code', {
            template: template,
            parameters: params
        });
    }
    async refactorCode(projectPath, refactoring, params) {
        return this.callTool('flutter-generation', 'refactor_code', {
            projectPath: projectPath,
            refactoring: refactoring,
            parameters: params
        });
    }
    // Generic MCP request method
    async sendRequest(serverName, request) {
        if (!this.httpClient) {
            throw new Error('MCP client is not connected');
        }
        try {
            const response = await this.httpClient.post(`/servers/${serverName}/request`, request);
            return response.data;
        }
        catch (error) {
            this.logger.error(`Failed to send MCP request to ${serverName}: ${error}`);
            throw error;
        }
    }
    isConnectedToServer() {
        return this.isConnected;
    }
    generateRequestId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}
exports.MCPClient = MCPClient;
//# sourceMappingURL=mcpClient.js.map