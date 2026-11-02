export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  model?: string;
  createdAt: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    latencyMs: number;
  };
  toolCalls?: ChatToolCall[];
  citations?: string[];
}

export interface ChatToolCall {
  id: string;
  name: string;
  arguments: string;
  result?: string;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatRequest {
  conversationId?: string;
  model: string;
  messages: ChatMessage[];
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

export type ChatStreamEvent =
  | { type: "start"; conversationId: string }
  | { type: "delta"; content: string }
  | { type: "tool_call"; toolCall: ChatToolCall }
  | { type: "citations"; citations: string[] }
  | {
      type: "usage";
      usage: { inputTokens: number; outputTokens: number; latencyMs: number };
    }
  | { type: "done" }
  | { type: "error"; message: string };
