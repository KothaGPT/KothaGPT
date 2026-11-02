import { get, post, del, postStream } from "@/lib/api/client";
import type { Paginated } from "@/types/api";
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  Conversation,
} from "@/types/chat";

export interface ChatApi {
  conversations(): Promise<Paginated<Conversation>>;
  conversation(id: string): Promise<Conversation>;
  messages(conversationId: string): Promise<ChatMessage[]>;
  createConversation(title: string, model: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  chat(request: ChatRequest): Promise<ChatMessage>;
  stream(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void>;
}

export const chatApi: ChatApi = {
  conversations: () => get<Paginated<Conversation>>("/v1/conversations"),
  conversation: (id) => get<Conversation>(`/v1/conversations/${id}`),
  messages: (conversationId) =>
    get<ChatMessage[]>(`/v1/conversations/${conversationId}/messages`),
  createConversation: (title, model) =>
    post<Conversation>("/v1/conversations", { title, model }),
  deleteConversation: (id) => del<void>(`/v1/conversations/${id}`),
  chat: (request) => post<ChatMessage>("/v1/chat", request),
  stream: async (request, onEvent, signal) => {
    const stream = await postStream(
      "/v1/chat",
      { ...request, stream: true },
      { signal },
    );
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") {
          onEvent({ type: "done" });
          continue;
        }
        try {
          const event = JSON.parse(payload) as ChatStreamEvent;
          onEvent(event);
        } catch {
          // ignore malformed frames
        }
      }
    }
    if (buffer.trim()) {
      try {
        const event = JSON.parse(buffer.trim()) as ChatStreamEvent;
        onEvent(event);
      } catch {
        // ignore trailing partial frame
      }
    }
  },
};
