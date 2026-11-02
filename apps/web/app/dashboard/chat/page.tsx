"use client";

import * as React from "react";
import { useParams, useRouter } from "next/navigation";
import { Settings2 } from "lucide-react";
import { ConversationSidebar } from "@/components/chat/conversation-sidebar";
import { ChatComposer } from "@/components/chat/composer";
import { MessageList } from "@/components/chat/message-list";
import { ModelSelector } from "@/components/chat/model-selector";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useStream } from "@/hooks/use-stream";
import { useMessages, useCreateConversation } from "@/hooks";
import { useChatStore } from "@/stores/chat-store";

export default function ChatPage() {
  const params = useParams<{ conversationId?: string }>();
  const conversationId = params.conversationId;

  const { sendMessage, stop, regenerate } = useStream();
  const messages = useChatStore((s) => s.messages);
  const streaming = useChatStore((s) => s.streaming);
  const selectedModel = useChatStore((s) => s.selectedModel);
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const setMessages = useChatStore((s) => s.setMessages);
  const setConversationId = useChatStore((s) => s.setConversationId);
  const clearMessages = useChatStore((s) => s.clearMessages);
  const setTemperature = useChatStore((s) => s.setTemperature);
  const setMaxTokens = useChatStore((s) => s.setMaxTokens);
  const temperature = useChatStore((s) => s.temperature);
  const maxTokens = useChatStore((s) => s.maxTokens);

  const history = useMessages(conversationId ?? "");
  const router = useRouter();
  const createConversation = useCreateConversation();

  const handleNew = () => {
    createConversation.mutate(
      { title: "New chat", model: selectedModel },
      {
        onSuccess: (conversation) => {
          router.push(`/dashboard/chat/${conversation.id}`);
        },
      },
    );
  };

  React.useEffect(() => {
    setConversationId(conversationId ?? null);
    if (!conversationId) {
      clearMessages();
      return;
    }
    if (history.data) {
      setMessages(history.data);
    } else {
      clearMessages();
    }
  }, [
    conversationId,
    history.data,
    setConversationId,
    setMessages,
    clearMessages,
  ]);

  return (
    <div className="flex -m-6 h-[calc(100vh-3.5rem)]">
      <ConversationSidebar onNew={handleNew} />
      <div className="flex flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <ModelSelector value={selectedModel} onChange={setSelectedModel} />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <Settings2 className="size-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-72">
              <DropdownMenuLabel>Generation parameters</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                <span className="text-sm">Temperature</span>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={temperature}
                  onChange={(e) =>
                    setTemperature(parseFloat(e.target.value) || 0)
                  }
                  className="h-8 w-20"
                />
              </div>
              <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                <span className="text-sm">Max tokens</span>
                <Input
                  type="number"
                  min="1"
                  value={maxTokens}
                  onChange={(e) => {
                    const parsed = parseInt(e.target.value, 10);
                    setMaxTokens(
                      Number.isNaN(parsed) || parsed < 1 ? 1 : parsed,
                    );
                  }}
                  className="h-8 w-20"
                />
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4">
          <MessageList
            messages={messages}
            streaming={streaming}
            onRegenerate={(id) => {
              const assistant = messages.find((m) => m.id === id);
              void regenerate(assistant?.id ?? id);
            }}
          />
        </div>

        <ChatComposer
          onSend={(content) => void sendMessage(content)}
          onStop={stop}
          streaming={streaming}
        />
      </div>
    </div>
  );
}
