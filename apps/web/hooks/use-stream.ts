"use client";

import * as React from "react";
import { chatApi } from "@/lib/api/chat";
import { useChatStore, nextMessageId } from "@/stores/chat-store";
import type { ChatMessage, ChatStreamEvent } from "@/types/chat";

interface UseStreamReturn {
  sendMessage: (content: string) => Promise<void>;
  stop: () => void;
  regenerate: (messageId: string) => Promise<void>;
}

function toApiMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({
    ...m,
    usage: undefined,
    toolCalls: undefined,
    citations: undefined,
  }));
}

export function useStream(): UseStreamReturn {
  const abortRef = React.useRef<AbortController | null>(null);

  const conversationId = useChatStore((s) => s.conversationId);
  const messages = useChatStore((s) => s.messages);
  const selectedModel = useChatStore((s) => s.selectedModel);
  const systemPrompt = useChatStore((s) => s.systemPrompt);
  const temperature = useChatStore((s) => s.temperature);
  const maxTokens = useChatStore((s) => s.maxTokens);

  const setStreaming = useChatStore((s) => s.setStreaming);
  const addMessage = useChatStore((s) => s.addMessage);
  const updateMessage = useChatStore((s) => s.updateMessage);
  const appendDelta = useChatStore((s) => s.appendDelta);
  const setConversationId = useChatStore((s) => s.setConversationId);
  const addToolCall = useChatStore((s) => s.addToolCall);
  const clearToolCalls = useChatStore((s) => s.clearToolCalls);
  const setError = useChatStore((s) => s.setError);

  const runStream = React.useCallback(
    async (apiMessages: ChatMessage[]) => {
      clearToolCalls();
      setError(null);
      const assistantId = nextMessageId();
      addMessage({
        id: assistantId,
        role: "assistant",
        content: "",
        model: selectedModel,
        createdAt: new Date().toISOString(),
      });
      setStreaming(true);
      abortRef.current = new AbortController();

      const onEvent = (event: ChatStreamEvent) => {
        switch (event.type) {
          case "start":
            if (event.conversationId) setConversationId(event.conversationId);
            break;
          case "delta":
            appendDelta(assistantId, event.content);
            break;
          case "tool_call":
            addToolCall(event.toolCall);
            break;
          case "citations":
            updateMessage(assistantId, { citations: event.citations });
            break;
          case "usage":
            updateMessage(assistantId, { usage: event.usage });
            break;
          case "error":
            setError(event.message);
            break;
          case "done":
            break;
        }
      };

      try {
        await chatApi.stream(
          {
            conversationId: conversationId ?? undefined,
            model: selectedModel,
            messages: apiMessages,
            systemPrompt: systemPrompt || undefined,
            temperature,
            maxTokens,
            stream: true,
          },
          onEvent,
          abortRef.current.signal,
        );
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) {
          setError(err instanceof Error ? err.message : "Stream failed");
        }
      } finally {
        setStreaming(false);
      }
    },
    [
      conversationId,
      selectedModel,
      systemPrompt,
      temperature,
      maxTokens,
      addMessage,
      appendDelta,
      setConversationId,
      setStreaming,
      setError,
      addToolCall,
      updateMessage,
      clearToolCalls,
    ],
  );

  const sendMessage = React.useCallback(
    async (content: string) => {
      const userMessage: ChatMessage = {
        id: nextMessageId(),
        role: "user",
        content,
        createdAt: new Date().toISOString(),
      };
      addMessage(userMessage);
      await runStream([...toApiMessages(messages), userMessage]);
    },
    [messages, addMessage, runStream],
  );

  const regenerate = React.useCallback(
    async (messageId: string) => {
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx < 0) return;
      updateMessage(messageId, { content: "" });
      await runStream(toApiMessages(messages.slice(0, idx)));
    },
    [messages, updateMessage, runStream],
  );

  const stop = React.useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, [setStreaming]);

  return { sendMessage, stop, regenerate };
}
