"use client";

import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, Loader2, RefreshCw, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import type { ChatMessage } from "@/types/chat";

function CodeBlock({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  const [copied, setCopied] = React.useState(false);
  const code = String(children ?? "");

  return (
    <div className="group relative my-2 overflow-hidden rounded-md border border-border">
      <div className="flex items-center justify-between border-b border-border bg-muted px-3 py-1.5">
        <span className="font-mono text-xs text-muted-foreground">
          {className?.replace("language-", "") || "code"}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-muted-foreground"
          onClick={() => {
            navigator.clipboard
              .writeText(code)
              .then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              })
              .catch(() => {});
          }}
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}
        </Button>
      </div>
      <pre className="overflow-x-auto p-3 text-sm">
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <Avatar
        className={cn(
          "mt-1 size-8",
          isUser && "bg-primary text-primary-foreground",
        )}
      >
        <AvatarFallback>
          {isUser ? <User className="size-4" /> : "AI"}
        </AvatarFallback>
      </Avatar>
      <div className={cn("max-w-[80%] space-y-2", isUser && "text-right")}>
        <div
          className={cn(
            "rounded-lg px-4 py-2 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground"
              : "border border-border bg-card",
          )}
        >
          {isUser ? (
            message.content
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                pre: ({ children }) => <>{children}</>,
                code: ({ className, children }) => {
                  const isBlock = className?.includes("language-");
                  if (isBlock) {
                    return (
                      <CodeBlock className={className}>
                        {String(children).replace(/\n$/, "")}
                      </CodeBlock>
                    );
                  }
                  return (
                    <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                      {children}
                    </code>
                  );
                },
                a: ({ children, href }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary underline"
                  >
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c, i) => (
              <span
                key={i}
                className="rounded border border-border bg-card px-2 py-0.5 text-xs text-muted-foreground"
              >
                {c}
              </span>
            ))}
          </div>
        )}

        {message.usage && (
          <p className="text-xs text-muted-foreground">
            {message.usage.inputTokens} in · {message.usage.outputTokens} out ·{" "}
            {message.usage.latencyMs}ms
          </p>
        )}
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  streaming,
  onRegenerate,
}: {
  messages: ChatMessage[];
  streaming: boolean;
  onRegenerate: (messageId: string) => void;
}) {
  const endRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold">How can I help?</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Ask in Bangla or English — models, RAG and tools at your command.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 overflow-y-auto py-4">
      {messages.map((m, i) => (
        <div key={m.id} className="space-y-2">
          <MessageBubble message={m} />
          {!streaming &&
            m.role === "assistant" &&
            i === messages.length - 1 && (
              <div className={cn("flex gap-3", "pl-11")}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground"
                  onClick={() => onRegenerate(m.id)}
                >
                  <RefreshCw className="size-3.5" />
                  Regenerate
                </Button>
              </div>
            )}
        </div>
      ))}
      {streaming && (
        <div className="flex gap-3">
          <Avatar className="mt-1 size-8">
            <AvatarFallback>AI</AvatarFallback>
          </Avatar>
          <Loader2 className="mt-2 size-4 animate-spin text-muted-foreground" />
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
