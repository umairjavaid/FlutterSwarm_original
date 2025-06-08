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
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const flutterSwarmExtension_1 = require("./flutterSwarmExtension");
const logOutputChannel_1 = require("./utils/logOutputChannel");
let extension;
async function activate(context) {
    const logger = new logOutputChannel_1.LogOutputChannel('Flutter Swarm');
    logger.info('Flutter Swarm extension is being activated...');
    try {
        // Initialize the main extension class
        extension = new flutterSwarmExtension_1.FlutterSwarmExtension(context, logger);
        // Register the extension with VS Code
        await extension.initialize();
        // Set context to indicate the extension is active
        vscode.commands.executeCommand('setContext', 'flutter-swarm.active', true);
        logger.info('Flutter Swarm extension activated successfully');
        // Show welcome message for first-time users
        const config = vscode.workspace.getConfiguration('flutter-swarm');
        const showWelcome = context.globalState.get('flutter-swarm.showWelcome', true);
        if (showWelcome) {
            const action = await vscode.window.showInformationMessage('Welcome to Flutter Swarm! Would you like to see the getting started guide?', 'Show Guide', 'Maybe Later', "Don't Show Again");
            if (action === 'Show Guide') {
                vscode.commands.executeCommand('flutter-swarm.openDashboard');
            }
            else if (action === "Don't Show Again") {
                context.globalState.update('flutter-swarm.showWelcome', false);
            }
        }
    }
    catch (error) {
        logger.error(`Failed to activate Flutter Swarm extension: ${error}`);
        vscode.window.showErrorMessage(`Flutter Swarm activation failed: ${error}`);
    }
}
exports.activate = activate;
async function deactivate() {
    if (extension) {
        await extension.dispose();
        extension = undefined;
    }
    vscode.commands.executeCommand('setContext', 'flutter-swarm.active', false);
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map