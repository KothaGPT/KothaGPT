"use client";

import { create } from "zustand";
import type { ChatMessage, ChatToolCall } from "@/types/chat";

export interface ChatState {
  conversationId: string | null;
  messages: ChatMessage[];
  streaming: boolean;
  selectedModel: string;
  systemPrompt: string;
  temperature: number;
  maxTokens: number;
  error: string | null;
  activeToolCalls: ChatToolCall[];
  setConversationId: (id: string | null) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendDelta: (id: string, content: string) => void;
  clearMessages: () => void;
  setStreaming: (streaming: boolean) => void;
  setSelectedModel: (model: string) => void;
  setSystemPrompt: (prompt: string) => void;
  setTemperature: (temperature: number) => void;
  setMaxTokens: (maxTokens: number) => void;
  setError: (error: string | null) => void;
  addToolCall: (toolCall: ChatToolCall) => void;
  updateToolCallResult: (id: string, result: string) => void;
  clearToolCalls: () => void;
}

let idCounter = 0;
export function nextMessageId(): string {
  return `msg_${Date.now()}_${idCounter++}`;
}

export const useChatStore = create<ChatState>((set) => ({
  conversationId: null,
  messages: [],
  streaming: false,
  selectedModel: "kothagpt-0.1",
  systemPrompt: "",
  temperature: 0.7,
  maxTokens: 2048,
  error: null,
  activeToolCalls: [],
  setConversationId: (conversationId) => set({ conversationId }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((s) => ({ messages: [...s.messages, message] })),
  updateMessage: (id, patch) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    })),
  appendDelta: (id, content) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + content } : m,
      ),
    })),
  clearMessages: () => set({ messages: [], error: null }),
  setStreaming: (streaming) => set({ streaming }),
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSystemPrompt: (systemPrompt) => set({ systemPrompt }),
  setTemperature: (temperature) => set({ temperature }),
  setMaxTokens: (maxTokens) => set({ maxTokens }),
  setError: (error) => set({ error }),
  addToolCall: (toolCall) =>
    set((s) => ({ activeToolCalls: [...s.activeToolCalls, toolCall] })),
  updateToolCallResult: (id, result) =>
    set((s) => ({
      activeToolCalls: s.activeToolCalls.map((t) =>
        t.id === id ? { ...t, result } : t,
      ),
    })),
  clearToolCalls: () => set({ activeToolCalls: [] }),
}));
