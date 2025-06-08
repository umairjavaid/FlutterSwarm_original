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
exports.LogOutputChannel = void 0;
const vscode = __importStar(require("vscode"));
class LogOutputChannel {
    constructor(name) {
        this.outputChannel = vscode.window.createOutputChannel(name);
        this.logLevel = vscode.workspace.getConfiguration('flutter-swarm').get('logLevel', 'INFO');
    }
    debug(message) {
        if (this.shouldLog('DEBUG')) {
            this.writeLog('DEBUG', message);
        }
    }
    info(message) {
        if (this.shouldLog('INFO')) {
            this.writeLog('INFO', message);
        }
    }
    warn(message) {
        if (this.shouldLog('WARNING')) {
            this.writeLog('WARNING', message);
        }
    }
    error(message) {
        if (this.shouldLog('ERROR')) {
            this.writeLog('ERROR', message);
        }
    }
    show() {
        this.outputChannel.show();
    }
    hide() {
        this.outputChannel.hide();
    }
    clear() {
        this.outputChannel.clear();
    }
    writeLog(level, message) {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] [${level}] ${message}`;
        this.outputChannel.appendLine(logMessage);
    }
    shouldLog(level) {
        const levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
        const currentLevelIndex = levels.indexOf(this.logLevel);
        const messageLevelIndex = levels.indexOf(level);
        return messageLevelIndex >= currentLevelIndex;
    }
    dispose() {
        this.outputChannel.dispose();
    }
}
exports.LogOutputChannel = LogOutputChannel;
//# sourceMappingURL=logOutputChannel.js.map