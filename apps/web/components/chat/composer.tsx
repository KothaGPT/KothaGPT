"use client";

import * as React from "react";
import { ArrowUp, Square } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export function ChatComposer({
  onSend,
  onStop,
  streaming,
}: {
  onSend: (content: string) => void;
  onStop: () => void;
  streaming: boolean;
}) {
  const [value, setValue] = React.useState("");
  const ref = React.useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const content = value.trim();
    if (!content || streaming) return;
    setValue("");
    onSend(content);
    ref.current?.focus();
  };

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  return (
    <div className="border-t border-border bg-card p-4">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <div className="relative">
          <Textarea
            ref={ref}
            value={value}
            rows={1}
            placeholder="Message Kotha GPT… (এখানে টাইপ করুন)"
            className="min-h-[44px] resize-none pr-12 py-3"
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {streaming ? (
            <Button
              size="icon"
              variant="outline"
              className="absolute bottom-2 right-2"
              onClick={onStop}
              aria-label="Stop generating"
            >
              <Square className="size-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              className="absolute bottom-2 right-2"
              onClick={submit}
              disabled={!value.trim()}
              aria-label="Send message"
            >
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
        <p className="text-center text-xs text-muted-foreground">
          Kotha GPT can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}
