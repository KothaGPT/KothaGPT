import { Agent, AgentRun, ChatCompletion, EmbeddingResponse } from "./types.js";
import { KothaGPTError } from "./errors.js";

interface WsEnvelope {
  id: string | null;
  type: string;
  payload: Record<string, unknown>;
}

/**
 * JSON-over-WebSocket client for the Kotha GPT `/v1/ws` endpoint.
 * Uses the browser `WebSocket` API (also present in Node >= 22).
 */
export class KothaGPTWebSocket {
  private socket: WebSocket | null = null;
  private readonly url: string;

  constructor(baseURL = "ws://localhost:8000", apiKey?: string) {
    // Browsers cannot set custom headers on the handshake, so the API token is
    // sent as a query parameter (the server also accepts Authorization headers).
    const base = `${baseURL.replace(/\/$/, "")}/v1/ws`;
    this.url = apiKey ? `${base}?token=${encodeURIComponent(apiKey)}` : base;
  }

  connect(): Promise<KothaGPTWebSocket> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      this.socket = socket;
      socket.onopen = () => resolve(this);
      socket.onerror = () => reject(new KothaGPTError("WebSocket connection failed"));
    });
  }

  close(): void {
    this.socket?.close();
    this.socket = null;
  }

  private request(type: string, payload: Record<string, unknown>): Promise<WsEnvelope> {
    return new Promise((resolve, reject) => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
        reject(new KothaGPTError("Not connected"));
        return;
      }
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const onMessage = (event: MessageEvent) => {
        const envelope = JSON.parse(event.data as string) as WsEnvelope;
        if (envelope.id !== id) return;
        this.socket?.removeEventListener("message", onMessage);
        if (envelope.type === "error") reject(new KothaGPTError(String(envelope.payload?.error ?? "unknown error")));
        else resolve(envelope);
      };
      this.socket.addEventListener("message", onMessage);
      this.socket.send(JSON.stringify({ id, type, payload }));
    });
  }

  async chat(messages: Record<string, unknown>[]): Promise<ChatCompletion> {
    return (await this.request("chat", { messages })).payload as unknown as ChatCompletion;
  }

  async embed(input: string | string[], model = "kothagpt-embed"): Promise<EmbeddingResponse> {
    return (await this.request("embed", { input, model })).payload as unknown as EmbeddingResponse;
  }

  async agentsCreate(spec: Record<string, unknown>): Promise<Agent> {
    return (await this.request("agents.create", spec)).payload as unknown as Agent;
  }

  async agentsList(): Promise<Agent[]> {
    const envelope = await this.request("agents.list", {});
    return (envelope.payload.data as unknown as Agent[]) ?? [];
  }

  async agentsRun(agentId: string, message: string): Promise<AgentRun> {
    return (await this.request("agents.run", { agent_id: agentId, message })).payload as unknown as AgentRun;
  }
}