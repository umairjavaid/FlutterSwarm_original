import WebSocket from 'ws';
import { LogOutputChannel } from '../utils/logOutputChannel';

export interface AgentMessage {
    id: string;
    type: 'task_request' | 'status_update' | 'result_delivery' | 'state_sync' | 'error';
    agentId: string;
    timestamp: string;
    data: any;
    correlationId?: string;
}

export interface WebSocketEventHandler {
    onMessage(message: AgentMessage): void;
    onError(error: Error): void;
    onClose(): void;
    onOpen(): void;
}

export class WebSocketClient {
    private ws: WebSocket | undefined;
    private reconnectionAttempts = 0;
    private maxReconnectionAttempts = 5;
    private reconnectionDelay = 1000;
    private isConnecting = false;
    private messageHandlers: WebSocketEventHandler[] = [];
    private url: string = '';

    constructor(private readonly logger: LogOutputChannel) {}

    async connect(url: string): Promise<void> {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.logger.info('WebSocket is already connected');
            return;
        }

        if (this.isConnecting) {
            this.logger.info('WebSocket connection is already in progress');
            return;
        }

        this.url = url;
        this.isConnecting = true;

        try {
            await this.establishConnection();
            this.logger.info(`WebSocket connected to ${url}`);
        } catch (error) {
            this.logger.error(`Failed to connect WebSocket: ${error}`);
            throw error;
        } finally {
            this.isConnecting = false;
        }
    }

    async disconnect(): Promise<void> {
        if (this.ws) {
            this.reconnectionAttempts = this.maxReconnectionAttempts;
            this.ws.close(1000, 'Disconnected by client');
            this.ws = undefined;
        }
    }

    sendMessage(message: AgentMessage): void {
        if (!this.isConnected()) {
            throw new Error('WebSocket is not connected');
        }

        try {
            const jsonMessage = JSON.stringify(message);
            this.ws!.send(jsonMessage);
            this.logger.debug(`Sent WebSocket message: ${message.type} to ${message.agentId}`);
        } catch (error) {
            this.logger.error(`Failed to send WebSocket message: ${error}`);
            throw error;
        }
    }

    addHandler(handler: WebSocketEventHandler): void {
        this.messageHandlers.push(handler);
    }

    removeHandler(handler: WebSocketEventHandler): void {
        const index = this.messageHandlers.indexOf(handler);
        if (index > -1) {
            this.messageHandlers.splice(index, 1);
        }
    }

    isConnected(): boolean {
        return this.ws !== undefined && this.ws.readyState === WebSocket.OPEN;
    }

    getConnectionState(): number | undefined {
        return this.ws?.readyState;
    }

    private async establishConnection(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);

            const connectionTimeout = setTimeout(() => {
                if (this.ws) {
                    this.ws.close();
                }
                reject(new Error('WebSocket connection timeout'));
            }, 10000);

            const websocket = this.ws;

            websocket.on('open', () => {
                clearTimeout(connectionTimeout);
                this.reconnectionAttempts = 0;
                this.logger.info('WebSocket connection established');
                
                this.messageHandlers.forEach(handler => {
                    try {
                        handler.onOpen();
                    } catch (error) {
                        this.logger.error(`Error in WebSocket open handler: ${error}`);
                    }
                });
                
                resolve();
            });

            websocket.on('message', (data: WebSocket.Data) => {
                try {
                    const message: AgentMessage = JSON.parse(data.toString());
                    this.logger.debug(`Received WebSocket message: ${message.type} from ${message.agentId}`);
                    
                    this.messageHandlers.forEach(handler => {
                        try {
                            handler.onMessage(message);
                        } catch (error) {
                            this.logger.error(`Error in WebSocket message handler: ${error}`);
                        }
                    });
                } catch (error) {
                    this.logger.error(`Failed to parse WebSocket message: ${error}`);
                }
            });

            websocket.on('error', (error) => {
                clearTimeout(connectionTimeout);
                this.logger.error(`WebSocket error: ${error}`);
                
                this.messageHandlers.forEach(handler => {
                    try {
                        handler.onError(error);
                    } catch (handlerError) {
                        this.logger.error(`Error in WebSocket error handler: ${handlerError}`);
                    }
                });
                
                reject(error);
            });

            websocket.on('close', (code, reason) => {
                clearTimeout(connectionTimeout);
                this.logger.info(`WebSocket connection closed: ${code} - ${reason}`);
                
                this.messageHandlers.forEach(handler => {
                    try {
                        handler.onClose();
                    } catch (error) {
                        this.logger.error(`Error in WebSocket close handler: ${error}`);
                    }
                });

                if (code !== 1000 && this.reconnectionAttempts < this.maxReconnectionAttempts) {
                    this.scheduleReconnection();
                }
            });
        });
    }

    private scheduleReconnection(): void {
        this.reconnectionAttempts++;
        const delay = this.reconnectionDelay * Math.pow(2, this.reconnectionAttempts - 1);
        
        this.logger.info(`Scheduling WebSocket reconnection attempt ${this.reconnectionAttempts}/${this.maxReconnectionAttempts} in ${delay}ms`);
        
        setTimeout(async () => {
            try {
                await this.connect(this.url);
            } catch (error) {
                this.logger.error(`WebSocket reconnection attempt ${this.reconnectionAttempts} failed: ${error}`);
                
                if (this.reconnectionAttempts >= this.maxReconnectionAttempts) {
                    this.logger.error('Max WebSocket reconnection attempts reached, giving up');
                }
            }
        }, delay);
    }
}
