import { cn } from "@/lib/utils";

const SCORE_ORDER = [
  "bangla_qa",
  "bangla_translation",
  "bangla_summarization",
  "bangla_generation",
];

const SCORE_LABELS: Record<string, string> = {
  bangla_qa: "Bangla QA",
  bangla_translation: "Translation",
  bangla_summarization: "Summarization",
  bangla_generation: "Generation",
};

export function ScoreCard({ scores }: { scores: Record<string, number> }) {
  const tasks = SCORE_ORDER.filter((t) => t in scores);
  const avg = tasks.length
    ? tasks.reduce((sum, t) => sum + (scores[t] ?? 0), 0) / tasks.length
    : 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Average
          </p>
          <p className="text-3xl font-bold">{avg.toFixed(1)}</p>
        </div>
        <span
          className={cn(
            "rounded-md px-2 py-1 text-xs font-semibold",
            avg >= 70
              ? "bg-emerald-500/15 text-emerald-500"
              : avg >= 40
                ? "bg-amber-500/15 text-amber-500"
                : "bg-destructive/15 text-destructive",
          )}
        >
          {avg >= 70 ? "Strong" : avg >= 40 ? "Developing" : "Weak"}
        </span>
      </div>
      {tasks.map((t) => (
        <div key={t} className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-sm">
            <span>{SCORE_LABELS[t] ?? t}</span>
            <span className="font-semibold">{(scores[t] ?? 0).toFixed(1)}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${Math.min(100, scores[t] ?? 0)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
