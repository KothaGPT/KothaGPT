"use client";

import * as React from "react";
import { FlaskConical, Eye, Plus, Trash2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScoreCard } from "@/components/evaluation/score-card";
import { RunEvaluationDialog } from "@/components/evaluation/run-dialog";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useEvaluations, useDeleteEvaluation, useBenchmarks } from "@/hooks";
import type { EvaluationRun, EvalStatus } from "@/lib/api/evaluations";

const statusBadge: Record<
  EvalStatus,
  "success" | "warning" | "secondary" | "destructive"
> = {
  completed: "success",
  running: "warning",
  queued: "secondary",
  failed: "destructive",
};

export default function EvaluationsPage() {
  const [runOpen, setRunOpen] = React.useState(false);
  const [detail, setDetail] = React.useState<EvaluationRun | null>(null);
  const { data, isLoading } = useEvaluations();
  const { data: benchmarks } = useBenchmarks();
  const remove = useDeleteEvaluation();

  const runs = data?.items ?? [];
  const completed = runs.filter((r) => r.status === "completed");

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Evaluations</h1>
          <p className="text-sm text-muted-foreground">
            Benchmark models across Bangla QA, translation, summarization and
            generation tasks.
          </p>
        </div>
        <Button onClick={() => setRunOpen(true)}>
          <Plus />
          Run evaluation
        </Button>
      </div>

      {benchmarks && benchmarks.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {benchmarks.map((b) => (
            <Card key={`${b.name}-${b.version}`} className="px-4 py-2">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">{b.name}</span>
                <Badge variant="secondary">{b.version}</Badge>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {b.tasks.map((t) => `${t.id} (${t.total})`).join(" · ")}
              </p>
            </Card>
          ))}
        </div>
      )}

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : runs.length === 0 ? (
        <Card className="flex flex-col items-center justify-center gap-3 py-20 text-center">
          <FlaskConical className="size-10 text-muted-foreground" />
          <CardHeader>
            <CardTitle>No evaluations yet</CardTitle>
            <CardDescription>
              Run a benchmark to score a model across Bangla tasks.
            </CardDescription>
          </CardHeader>
          <Button onClick={() => setRunOpen(true)}>
            <Plus />
            Run evaluation
          </Button>
        </Card>
      ) : (
        <>
          {completed.length >= 2 && <ComparisonTable runs={completed} />}
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Benchmark</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Started</TableHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell className="font-medium">{r.name}</TableCell>
                    <TableCell>{r.model}</TableCell>
                    <TableCell>{r.benchmark}</TableCell>
                    <TableCell>
                      <Badge variant={statusBadge[r.status]}>{r.status}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDate(r.startedAt)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        {r.status === "completed" && (
                          <Button
                            variant="ghost"
                            size="icon"
                            aria-label={`View details for ${r.name}`}
                            onClick={() => setDetail(r)}
                          >
                            <Eye className="size-4" />
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={`Delete ${r.name}`}
                          className="text-muted-foreground hover:text-destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            remove.mutate(r.id);
                          }}
                        >
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </>
      )}

      <RunEvaluationDialog open={runOpen} onOpenChange={setRunOpen} />

      <Dialog open={!!detail} onOpenChange={(o) => !o && setDetail(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {detail?.name} · {detail?.model}
            </DialogTitle>
          </DialogHeader>
          {detail && <ScoreCard scores={detail.scores} />}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function ComparisonTable({ runs }: { runs: EvaluationRun[] }) {
  const taskKeys = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.scores))),
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model comparison</CardTitle>
        <CardDescription>
          Completed runs with scores, side by side.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Task</TableHead>
              {runs.map((r) => (
                <TableHead key={r.id}>{r.model}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {taskKeys.map((task) => (
              <TableRow key={task}>
                <TableCell className="font-medium">{task}</TableCell>
                {runs.map((r) => (
                  <TableCell key={r.id}>
                    {(r.scores[task] ?? 0).toFixed(1)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
            <TableRow className="border-t">
              <TableCell className="font-semibold">Average</TableCell>
              {runs.map((r) => {
                const vals = Object.values(r.scores);
                const avg = vals.length
                  ? vals.reduce((a, b) => a + b, 0) / vals.length
                  : 0;
                return (
                  <TableCell key={r.id} className="font-semibold">
                    {avg.toFixed(1)}
                  </TableCell>
                );
              })}
            </TableRow>
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString();
}
