import * as vscode from 'vscode';
import { FlutterProject } from '../services/flutterProjectDetector';
import { LogOutputChannel } from '../utils/logOutputChannel';

export interface AgentStatus {
    id: string;
    name: string;
    type: string;
    status: 'idle' | 'running' | 'error' | 'disabled';
    currentTask?: string;
    progress?: number;
    lastUpdate: Date;
    errorMessage?: string;
}

export class AgentTreeProvider implements vscode.TreeDataProvider<AgentTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentTreeItem | undefined | null | void> = new vscode.EventEmitter<AgentTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<AgentTreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private projects: FlutterProject[] = [];
    private agents: Map<string, AgentStatus> = new Map();

    constructor(private readonly logger: LogOutputChannel) {
        this.initializeDefaultAgents();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    updateProjects(projects: FlutterProject[]): void {
        this.projects = projects;
        this.refresh();
    }

    updateAgentStatus(agentId: string, status: Partial<AgentStatus>): void {
        const existing = this.agents.get(agentId);
        if (existing) {
            this.agents.set(agentId, { ...existing, ...status, lastUpdate: new Date() });
        } else {
            const newAgent: AgentStatus = {
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

    getTreeItem(element: AgentTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: AgentTreeItem): Thenable<AgentTreeItem[]> {
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
                    return Promise.resolve(this.getProjectDetails(element.id!));
                }
                return Promise.resolve([]);
        }
    }

    private getProjectItems(): AgentTreeItem[] {
        return this.projects.map(project => {
            const item = new AgentTreeItem(
                project.name,
                vscode.TreeItemCollapsibleState.Collapsed,
                'project',
                project.id
            );
            item.description = `${project.state} • ${project.platforms.join(', ')}`;
            item.tooltip = `Path: ${project.path}\nState: ${project.state}\nPlatforms: ${project.platforms.join(', ')}`;
            item.iconPath = new vscode.ThemeIcon('folder-library');
            return item;
        });
    }

    private getAgentItems(): AgentTreeItem[] {
        const agentGroups = new Map<string, AgentStatus[]>();
        
        // Group agents by type
        this.agents.forEach(agent => {
            const type = agent.type || 'other';
            if (!agentGroups.has(type)) {
                agentGroups.set(type, []);
            }
            agentGroups.get(type)!.push(agent);
        });

        const items: AgentTreeItem[] = [];
        
        // Create tree items for each group
        agentGroups.forEach((agents, type) => {
            const typeItem = new AgentTreeItem(
                this.formatAgentTypeName(type),
                vscode.TreeItemCollapsibleState.Expanded,
                'agent-type'
            );
            items.push(typeItem);
            
            // Add individual agents
            agents.forEach(agent => {
                const agentItem = new AgentTreeItem(
                    agent.name,
                    vscode.TreeItemCollapsibleState.None,
                    'agent',
                    agent.id
                );
                agentItem.description = this.getAgentStatusDescription(agent);
                agentItem.tooltip = this.getAgentTooltip(agent);
                agentItem.iconPath = this.getAgentIcon(agent);
                items.push(agentItem);
            });
        });

        return items;
    }

    private getTaskItems(): AgentTreeItem[] {
        // This would show current and recent tasks
        const runningTasks = Array.from(this.agents.values())
            .filter(agent => agent.status === 'running' && agent.currentTask)
            .map(agent => {
                const item = new AgentTreeItem(
                    agent.currentTask!,
                    vscode.TreeItemCollapsibleState.None,
                    'task'
                );
                item.description = `${agent.name} • ${agent.progress || 0}%`;
                item.iconPath = new vscode.ThemeIcon('loading~spin');
                return item;
            });

        if (runningTasks.length === 0) {
            return [new AgentTreeItem('No active tasks', vscode.TreeItemCollapsibleState.None, 'empty')];
        }

        return runningTasks;
    }

    private getProjectDetails(projectId: string): AgentTreeItem[] {
        const project = this.projects.find(p => p.id === projectId);
        if (!project) return [];

        const details: AgentTreeItem[] = [];

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

    private formatAgentTypeName(type: string): string {
        const typeNames: { [key: string]: string } = {
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

    private getAgentStatusDescription(agent: AgentStatus): string {
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

    private getAgentTooltip(agent: AgentStatus): string {
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

    private getAgentIcon(agent: AgentStatus): vscode.ThemeIcon {
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

    private initializeDefaultAgents(): void {
        const defaultAgents: Partial<AgentStatus>[] = [
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
            this.updateAgentStatus(agent.id!, agent);
        });
    }
}

export class AgentTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly contextValue: string,
        public readonly id?: string
    ) {
        super(label, collapsibleState);
        this.id = id || label;
    }
}
