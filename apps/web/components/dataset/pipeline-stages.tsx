import { cn } from "@/lib/utils";

const STAGES = [
  "upload",
  "normalize",
  "filter",
  "dedup",
  "quality",
  "tokenize",
  "shard",
  "ready",
] as const;

const STAGE_INDEX: Record<string, number> = {
  uploading: 0,
  normalizing: 1,
  filtering: 2,
  deduplicating: 3,
  quality_check: 4,
  tokenizing: 5,
  sharding: 6,
  ready: 7,
};

export function PipelineStages({ status }: { status: string }) {
  const current = STAGE_INDEX[status];
  const isReady = status === "ready";
  const inactive = current === undefined || status === "failed";

  return (
    <div className="flex w-full items-center gap-1">
      {STAGES.map((stage, i) => {
        const done = isReady || (!inactive && i < current);
        const active = !isReady && !inactive && i === current;
        return (
          <div
            key={stage}
            className="flex flex-1 flex-col items-center gap-1.5"
          >
            <div
              className={cn(
                "flex h-1.5 w-full rounded-full transition-colors",
                done || active ? "bg-primary" : "bg-muted",
              )}
            />
            <span
              className={cn(
                "text-[10px] uppercase tracking-wide",
                active
                  ? "text-primary"
                  : done
                    ? "text-muted-foreground"
                    : "text-muted-foreground/50",
              )}
            >
              {stage}
            </span>
          </div>
        );
      })}
    </div>
  );
}
