"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { MessageSquare, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useConversations } from "@/hooks";

export function ConversationSidebar({ onNew }: { onNew: () => void }) {
  const params = useParams<{ conversationId?: string }>();
  const { data, isLoading } = useConversations();
  const activeId = params.conversationId;

  return (
    <div className="flex w-64 shrink-0 flex-col border-r border-border bg-card">
      <div className="p-3">
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start gap-2"
          onClick={onNew}
        >
          <Plus className="size-4" />
          New chat
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {isLoading ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : data && data.items.length > 0 ? (
          <div className="flex flex-col gap-0.5">
            {data.items.map((c) => (
              <Link
                key={c.id}
                href={`/dashboard/chat/${c.id}`}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
                  activeId === c.id && "bg-accent",
                )}
              >
                <MessageSquare className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate">{c.title}</span>
              </Link>
            ))}
          </div>
        ) : (
          <p className="px-2 text-xs text-muted-foreground">
            No conversations yet.
          </p>
        )}
      </div>
    </div>
  );
}
