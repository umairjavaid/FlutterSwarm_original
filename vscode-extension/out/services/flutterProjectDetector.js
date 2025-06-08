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
exports.FlutterProjectDetector = void 0;
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
class FlutterProjectDetector {
    constructor(logger) {
        this.logger = logger;
        this.detectedProjects = new Map();
    }
    async scanWorkspaceFolder(folder) {
        const projects = [];
        try {
            const pubspecFiles = await this.findPubspecFiles(folder.uri.fsPath);
            for (const pubspecPath of pubspecFiles) {
                try {
                    const project = await this.analyzeProject(pubspecPath);
                    if (project) {
                        projects.push(project);
                        this.detectedProjects.set(project.path, project);
                    }
                }
                catch (error) {
                    this.logger.warn(`Failed to analyze project at ${pubspecPath}: ${error}`);
                }
            }
            this.logger.info(`Detected ${projects.length} Flutter projects in ${folder.name}`);
        }
        catch (error) {
            this.logger.error(`Error scanning workspace folder ${folder.name}: ${error}`);
        }
        return projects;
    }
    async analyzeProject(pubspecPath) {
        try {
            if (!fs.existsSync(pubspecPath)) {
                return null;
            }
            const projectPath = path.dirname(pubspecPath);
            const pubspecContent = fs.readFileSync(pubspecPath, 'utf-8');
            // Parse pubspec.yaml
            const pubspec = this.parsePubspec(pubspecContent);
            if (!pubspec || !this.isFlutterProject(pubspec)) {
                return null;
            }
            // Perform detailed project analysis
            const analysis = await this.performProjectAnalysis(projectPath);
            const project = {
                id: this.generateProjectId(projectPath),
                name: pubspec.name || path.basename(projectPath),
                path: projectPath,
                pubspecPath,
                state: this.determineProjectState(analysis),
                dependencies: Object.keys(pubspec.dependencies || {}),
                devDependencies: Object.keys(pubspec.dev_dependencies || {}),
                flutterVersion: pubspec.environment?.flutter,
                dartVersion: pubspec.environment?.sdk,
                platforms: this.detectPlatforms(projectPath),
                stateManagement: analysis.stateManagement,
                architecture: analysis.architecture,
                testCoverage: this.calculateTestCoverage(analysis),
                lastAnalyzed: new Date()
            };
            this.logger.info(`Analyzed Flutter project: ${project.name} (${project.state})`);
            return project;
        }
        catch (error) {
            this.logger.error(`Error analyzing project at ${pubspecPath}: ${error}`);
            return null;
        }
    }
    getDetectedProjects() {
        return Array.from(this.detectedProjects.values());
    }
    getProject(projectPath) {
        return this.detectedProjects.get(projectPath);
    }
    async findPubspecFiles(rootPath) {
        const pubspecFiles = [];
        const scanDirectory = (dirPath, maxDepth = 3) => {
            if (maxDepth <= 0)
                return;
            try {
                const entries = fs.readdirSync(dirPath, { withFileTypes: true });
                for (const entry of entries) {
                    const fullPath = path.join(dirPath, entry.name);
                    if (entry.isFile() && entry.name === 'pubspec.yaml') {
                        pubspecFiles.push(fullPath);
                    }
                    else if (entry.isDirectory() && !this.shouldSkipDirectory(entry.name)) {
                        scanDirectory(fullPath, maxDepth - 1);
                    }
                }
            }
            catch (error) {
                // Ignore permission errors and continue scanning
            }
        };
        scanDirectory(rootPath);
        return pubspecFiles;
    }
    shouldSkipDirectory(dirName) {
        const skipDirs = [
            'node_modules',
            '.git',
            '.dart_tool',
            'build',
            '.vscode',
            '.idea',
            'coverage',
            'doc',
            'example'
        ];
        return skipDirs.includes(dirName) || dirName.startsWith('.');
    }
    parsePubspec(content) {
        try {
            // Simple YAML parser for pubspec.yaml
            const lines = content.split('\n');
            const result = {};
            let currentSection = '';
            let indentLevel = 0;
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith('#'))
                    continue;
                const leadingSpaces = line.length - line.trimStart().length;
                if (trimmed.includes(':')) {
                    const [key, value] = trimmed.split(':', 2);
                    const keyTrimmed = key.trim();
                    const valueTrimmed = value ? value.trim() : '';
                    if (leadingSpaces === 0) {
                        // Top-level key
                        if (valueTrimmed) {
                            result[keyTrimmed] = valueTrimmed;
                        }
                        else {
                            result[keyTrimmed] = {};
                            currentSection = keyTrimmed;
                        }
                    }
                    else if (currentSection && result[currentSection]) {
                        // Nested key
                        if (keyTrimmed === 'flutter' || keyTrimmed === 'sdk') {
                            if (!result.environment)
                                result.environment = {};
                            result.environment[keyTrimmed] = valueTrimmed;
                        }
                        else {
                            result[currentSection][keyTrimmed] = valueTrimmed || true;
                        }
                    }
                }
            }
            return result;
        }
        catch (error) {
            this.logger.warn(`Failed to parse pubspec.yaml: ${error}`);
            return null;
        }
    }
    isFlutterProject(pubspec) {
        return pubspec.dependencies &&
            (pubspec.dependencies.flutter || pubspec.dependencies['flutter']) !== undefined;
    }
    async performProjectAnalysis(projectPath) {
        const analysis = {
            fileCount: 0,
            dartFileCount: 0,
            testFileCount: 0,
            widgetCount: 0,
            hasTests: false,
            hasIntegrationTests: false,
            hasGoldenTests: false,
            architecture: 'Unknown',
            stateManagement: 'Unknown',
            complexityScore: 0
        };
        try {
            // Scan lib directory
            const libPath = path.join(projectPath, 'lib');
            if (fs.existsSync(libPath)) {
                await this.analyzeDirectory(libPath, analysis);
            }
            // Scan test directory
            const testPath = path.join(projectPath, 'test');
            if (fs.existsSync(testPath)) {
                analysis.hasTests = true;
                await this.analyzeTestDirectory(testPath, analysis);
            }
            // Scan integration_test directory
            const integrationTestPath = path.join(projectPath, 'integration_test');
            if (fs.existsSync(integrationTestPath)) {
                analysis.hasIntegrationTests = true;
            }
            // Detect architecture patterns
            analysis.architecture = this.detectArchitecture(projectPath);
            // Detect state management
            analysis.stateManagement = this.detectStateManagement(projectPath);
            // Calculate complexity score
            analysis.complexityScore = this.calculateComplexityScore(analysis);
        }
        catch (error) {
            this.logger.warn(`Error analyzing project structure: ${error}`);
        }
        return analysis;
    }
    async analyzeDirectory(dirPath, analysis) {
        try {
            const entries = fs.readdirSync(dirPath, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(dirPath, entry.name);
                if (entry.isFile()) {
                    analysis.fileCount++;
                    if (entry.name.endsWith('.dart')) {
                        analysis.dartFileCount++;
                        await this.analyzeDartFile(fullPath, analysis);
                    }
                }
                else if (entry.isDirectory()) {
                    await this.analyzeDirectory(fullPath, analysis);
                }
            }
        }
        catch (error) {
            // Ignore permission errors
        }
    }
    async analyzeTestDirectory(testPath, analysis) {
        try {
            const entries = fs.readdirSync(testPath, { withFileTypes: true });
            for (const entry of entries) {
                if (entry.isFile() && entry.name.endsWith('.dart')) {
                    analysis.testFileCount++;
                    // Check for golden tests
                    const filePath = path.join(testPath, entry.name);
                    const content = fs.readFileSync(filePath, 'utf-8');
                    if (content.includes('matchesGoldenFile') || content.includes('goldenFileComparator')) {
                        analysis.hasGoldenTests = true;
                    }
                }
                else if (entry.isDirectory()) {
                    await this.analyzeTestDirectory(path.join(testPath, entry.name), analysis);
                }
            }
        }
        catch (error) {
            // Ignore errors
        }
    }
    async analyzeDartFile(filePath, analysis) {
        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            // Count widgets (simple heuristic)
            const widgetMatches = content.match(/class\s+\w+\s+extends\s+(StatelessWidget|StatefulWidget|Widget)/g);
            if (widgetMatches) {
                analysis.widgetCount += widgetMatches.length;
            }
        }
        catch (error) {
            // Ignore file reading errors
        }
    }
    detectArchitecture(projectPath) {
        const libPath = path.join(projectPath, 'lib');
        if (!fs.existsSync(libPath))
            return 'Unknown';
        try {
            const entries = fs.readdirSync(libPath, { withFileTypes: true });
            const directories = entries.filter(e => e.isDirectory()).map(e => e.name);
            // Check for clean architecture patterns
            const cleanArchDirs = ['data', 'domain', 'presentation', 'core'];
            const hasCleanArch = cleanArchDirs.every(dir => directories.includes(dir));
            if (hasCleanArch) {
                return 'Clean Architecture';
            }
            // Check for MVC/MVP patterns
            if (directories.includes('models') && directories.includes('views') && directories.includes('controllers')) {
                return 'MVC';
            }
            // Check for feature-based architecture
            if (directories.includes('features') || directories.some(d => d.includes('feature'))) {
                return 'Feature-based';
            }
            // Basic structure
            if (directories.includes('screens') || directories.includes('pages')) {
                return 'Basic Structure';
            }
            return 'Flat Structure';
        }
        catch (error) {
            return 'Unknown';
        }
    }
    detectStateManagement(projectPath) {
        try {
            const pubspecPath = path.join(projectPath, 'pubspec.yaml');
            const pubspecContent = fs.readFileSync(pubspecPath, 'utf-8');
            if (pubspecContent.includes('flutter_bloc') || pubspecContent.includes('bloc:')) {
                return 'BLoC';
            }
            if (pubspecContent.includes('provider:')) {
                return 'Provider';
            }
            if (pubspecContent.includes('riverpod')) {
                return 'Riverpod';
            }
            if (pubspecContent.includes('get:') || pubspecContent.includes('getx')) {
                return 'GetX';
            }
            if (pubspecContent.includes('mobx')) {
                return 'MobX';
            }
            if (pubspecContent.includes('redux')) {
                return 'Redux';
            }
            // Check for setState usage in code
            const libPath = path.join(projectPath, 'lib');
            if (fs.existsSync(libPath)) {
                // Simple check - this could be more sophisticated
                return 'setState';
            }
            return 'Unknown';
        }
        catch (error) {
            return 'Unknown';
        }
    }
    detectPlatforms(projectPath) {
        const platforms = [];
        const platformDirs = [
            { dir: 'android', platform: 'Android' },
            { dir: 'ios', platform: 'iOS' },
            { dir: 'web', platform: 'Web' },
            { dir: 'windows', platform: 'Windows' },
            { dir: 'macos', platform: 'macOS' },
            { dir: 'linux', platform: 'Linux' }
        ];
        for (const { dir, platform } of platformDirs) {
            if (fs.existsSync(path.join(projectPath, dir))) {
                platforms.push(platform);
            }
        }
        return platforms.length > 0 ? platforms : ['Mobile'];
    }
    determineProjectState(analysis) {
        const score = analysis.complexityScore;
        if (score < 10) {
            return 'New';
        }
        else if (score < 50) {
            return 'Mid-Development';
        }
        else {
            return 'Mature';
        }
    }
    calculateComplexityScore(analysis) {
        let score = 0;
        // File count contribution
        score += Math.min(analysis.dartFileCount * 0.5, 20);
        // Widget count contribution
        score += Math.min(analysis.widgetCount * 1, 25);
        // Architecture bonus
        if (analysis.architecture === 'Clean Architecture') {
            score += 15;
        }
        else if (analysis.architecture !== 'Unknown' && analysis.architecture !== 'Flat Structure') {
            score += 10;
        }
        // State management bonus
        if (analysis.stateManagement !== 'Unknown' && analysis.stateManagement !== 'setState') {
            score += 10;
        }
        // Testing bonus
        if (analysis.hasTests) {
            score += 10;
            if (analysis.hasIntegrationTests)
                score += 5;
            if (analysis.hasGoldenTests)
                score += 5;
        }
        return Math.round(score);
    }
    calculateTestCoverage(analysis) {
        if (!analysis.hasTests || analysis.dartFileCount === 0) {
            return 0;
        }
        // Simple heuristic - actual coverage would need to be calculated by running tests
        const ratio = analysis.testFileCount / analysis.dartFileCount;
        return Math.min(Math.round(ratio * 100), 100);
    }
    generateProjectId(projectPath) {
        return Buffer.from(projectPath).toString('base64').replace(/[^a-zA-Z0-9]/g, '').substring(0, 16);
    }
}
exports.FlutterProjectDetector = FlutterProjectDetector;
//# sourceMappingURL=flutterProjectDetector.js.map