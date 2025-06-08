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
exports.AgentTreeItem = exports.AgentTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
class AgentTreeProvider {
    constructor(logger) {
        this.logger = logger;
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.projects = [];
        this.agents = new Map();
        this.initializeDefaultAgents();
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    updateProjects(projects) {
        this.projects = projects;
        this.refresh();
    }
    updateAgentStatus(agentId, status) {
        const existing = this.agents.get(agentId);
        if (existing) {
            this.agents.set(agentId, { ...existing, ...status, lastUpdate: new Date() });
        }
        else {
            const newAgent = {
                id: agentId,
                name: agentId,
                type: 'unknown',
                status: 'idle',
                lastUpdate: new Date(),
                ...status
            };
            this.agents.set(agentId, newAgent);
        }
        this.refresh();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (!element) {
            // Root level - show main categories
            return Promise.resolve([
                new AgentTreeItem('Projects', vscode.TreeItemCollapsibleState.Expanded, 'category'),
                new AgentTreeItem('Agents', vscode.TreeItemCollapsibleState.Expanded, 'category'),
                new AgentTreeItem('Tasks', vscode.TreeItemCollapsibleState.Collapsed, 'category')
            ]);
        }
        switch (element.label) {
            case 'Projects':
                return Promise.resolve(this.getProjectItems());
            case 'Agents':
                return Promise.resolve(this.getAgentItems());
            case 'Tasks':
                return Promise.resolve(this.getTaskItems());
            default:
                if (element.contextValue === 'project') {
                    return Promise.resolve(this.getProjectDetails(element.id));
                }
                return Promise.resolve([]);
        }
    }
    getProjectItems() {
        return this.projects.map(project => {
            const item = new AgentTreeItem(project.name, vscode.TreeItemCollapsibleState.Collapsed, 'project', project.id);
            item.description = `${project.state} • ${project.platforms.join(', ')}`;
            item.tooltip = `Path: ${project.path}\nState: ${project.state}\nPlatforms: ${project.platforms.join(', ')}`;
            item.iconPath = new vscode.ThemeIcon('folder-library');
            return item;
        });
    }
    getAgentItems() {
        const agentGroups = new Map();
        // Group agents by type
        this.agents.forEach(agent => {
            const type = agent.type || 'other';
            if (!agentGroups.has(type)) {
                agentGroups.set(type, []);
            }
            agentGroups.get(type).push(agent);
        });
        const items = [];
        // Create tree items for each group
        agentGroups.forEach((agents, type) => {
            const typeItem = new AgentTreeItem(this.formatAgentTypeName(type), vscode.TreeItemCollapsibleState.Expanded, 'agent-type');
            items.push(typeItem);
            // Add individual agents
            agents.forEach(agent => {
                const agentItem = new AgentTreeItem(agent.name, vscode.TreeItemCollapsibleState.None, 'agent', agent.id);
                agentItem.description = this.getAgentStatusDescription(agent);
                agentItem.tooltip = this.getAgentTooltip(agent);
                agentItem.iconPath = this.getAgentIcon(agent);
                items.push(agentItem);
            });
        });
        return items;
    }
    getTaskItems() {
        // This would show current and recent tasks
        const runningTasks = Array.from(this.agents.values())
            .filter(agent => agent.status === 'running' && agent.currentTask)
            .map(agent => {
            const item = new AgentTreeItem(agent.currentTask, vscode.TreeItemCollapsibleState.None, 'task');
            item.description = `${agent.name} • ${agent.progress || 0}%`;
            item.iconPath = new vscode.ThemeIcon('loading~spin');
            return item;
        });
        if (runningTasks.length === 0) {
            return [new AgentTreeItem('No active tasks', vscode.TreeItemCollapsibleState.None, 'empty')];
        }
        return runningTasks;
    }
    getProjectDetails(projectId) {
        const project = this.projects.find(p => p.id === projectId);
        if (!project)
            return [];
        const details = [];
        // Basic info
        details.push(new AgentTreeItem(`State: ${project.state}`, vscode.TreeItemCollapsibleState.None, 'info'));
        details.push(new AgentTreeItem(`Platforms: ${project.platforms.join(', ')}`, vscode.TreeItemCollapsibleState.None, 'info'));
        if (project.stateManagement) {
            details.push(new AgentTreeItem(`State Management: ${project.stateManagement}`, vscode.TreeItemCollapsibleState.None, 'info'));
        }
        if (project.architecture) {
            details.push(new AgentTreeItem(`Architecture: ${project.architecture}`, vscode.TreeItemCollapsibleState.None, 'info'));
        }
        if (project.testCoverage !== undefined) {
            details.push(new AgentTreeItem(`Test Coverage: ${project.testCoverage}%`, vscode.TreeItemCollapsibleState.None, 'info'));
        }
        // Dependencies
        if (project.dependencies.length > 0) {
            const depsItem = new AgentTreeItem('Dependencies', vscode.TreeItemCollapsibleState.Collapsed, 'info');
            details.push(depsItem);
        }
        return details;
    }
    formatAgentTypeName(type) {
        const typeNames = {
            'orchestrator': 'Orchestrator',
            'architecture': 'Architecture',
            'implementation': 'Implementation',
            'testing': 'Testing',
            'devops': 'DevOps',
            'search': 'Search & Fetch',
            'navigation': 'Navigation',
            'animation': 'Animation & Effects',
            'localization': 'Localization',
            'accessibility': 'Accessibility',
            'data': 'Data Model',
            'storage': 'Local Storage',
            'api': 'API Integration',
            'repository': 'Repository',
            'business': 'Business Logic',
            'deployment': 'Deployment'
        };
        return typeNames[type] || type.charAt(0).toUpperCase() + type.slice(1);
    }
    getAgentStatusDescription(agent) {
        switch (agent.status) {
            case 'running':
                return agent.progress ? `${agent.progress}%` : 'Running...';
            case 'error':
                return 'Error';
            case 'disabled':
                return 'Disabled';
            default:
                return 'Idle';
        }
    }
    getAgentTooltip(agent) {
        let tooltip = `Agent: ${agent.name}\nType: ${agent.type}\nStatus: ${agent.status}`;
        if (agent.currentTask) {
            tooltip += `\nCurrent Task: ${agent.currentTask}`;
        }
        if (agent.progress) {
            tooltip += `\nProgress: ${agent.progress}%`;
        }
        if (agent.errorMessage) {
            tooltip += `\nError: ${agent.errorMessage}`;
        }
        tooltip += `\nLast Update: ${agent.lastUpdate.toLocaleString()}`;
        return tooltip;
    }
    getAgentIcon(agent) {
        switch (agent.status) {
            case 'running':
                return new vscode.ThemeIcon('loading~spin');
            case 'error':
                return new vscode.ThemeIcon('error');
            case 'disabled':
                return new vscode.ThemeIcon('circle-slash');
            default:
                return new vscode.ThemeIcon('circle');
        }
    }
    initializeDefaultAgents() {
        const defaultAgents = [
            { id: 'orchestrator', name: 'Orchestrator Agent', type: 'orchestrator', status: 'idle' },
            { id: 'architecture', name: 'Architecture Agent', type: 'architecture', status: 'idle' },
            { id: 'implementation', name: 'Implementation Agent', type: 'implementation', status: 'idle' },
            { id: 'testing', name: 'Testing Agent', type: 'testing', status: 'idle' },
            { id: 'devops', name: 'DevOps Agent', type: 'devops', status: 'idle' },
            { id: 'search', name: 'Search Agent', type: 'search', status: 'idle' },
            { id: 'navigation', name: 'Navigation Agent', type: 'navigation', status: 'idle' },
            { id: 'animation', name: 'Animation Agent', type: 'animation', status: 'idle' },
            { id: 'localization', name: 'Localization Agent', type: 'localization', status: 'idle' },
            { id: 'accessibility', name: 'Accessibility Agent', type: 'accessibility', status: 'idle' }
        ];
        defaultAgents.forEach(agent => {
            this.updateAgentStatus(agent.id, agent);
        });
    }
}
exports.AgentTreeProvider = AgentTreeProvider;
class AgentTreeItem extends vscode.TreeItem {
    constructor(label, collapsibleState, contextValue, id) {
        super(label, collapsibleState);
        this.label = label;
        this.collapsibleState = collapsibleState;
        this.contextValue = contextValue;
        this.id = id;
        this.id = id || label;
    }
}
exports.AgentTreeItem = AgentTreeItem;
//# sourceMappingURL=agentTreeProvider.js.map