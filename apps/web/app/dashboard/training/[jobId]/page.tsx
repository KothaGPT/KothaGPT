"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, Pause, Play, Square } from "lucide-react";
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
import { ProgressBar } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useTrainingJob,
  useTrainingCheckpoints,
  useTrainingControl,
} from "@/hooks";

export default function TrainingJobDetailPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const { data: job, isLoading } = useTrainingJob(jobId);
  const { data: checkpoints } = useTrainingCheckpoints(jobId);
  const control = useTrainingControl(jobId);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-8 w-40" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Training job not found
        </h1>
        <p className="text-sm text-muted-foreground">
          This job may have been removed or the link is incorrect.
        </p>
        <Button variant="outline" asChild className="w-fit">
          <Link href="/dashboard/training">Back to training</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" asChild>
            <Link href="/dashboard/training">
              <ArrowLeft className="size-4" />
            </Link>
          </Button>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              {job.name}
            </h1>
            <p className="text-sm text-muted-foreground">
              {job.model} · {job.dataset} · {job.tokenizer}
            </p>
          </div>
          <Badge variant={job.status === "running" ? "success" : "secondary"}>
            {job.status}
          </Badge>
        </div>
        <div className="flex gap-2">
          {job.status === "running" ? (
            <>
              <Button
                variant="outline"
                onClick={() => control.mutate({ action: "pause" })}
              >
                <Pause />
                Pause
              </Button>
              <Button
                variant="destructive"
                onClick={() => control.mutate({ action: "stop" })}
              >
                <Square />
                Stop
              </Button>
            </>
          ) : job.status === "pending" ? (
            <Button onClick={() => control.mutate({ action: "start" })}>
              <Play />
              Start
            </Button>
          ) : job.status === "paused" ? (
            <Button onClick={() => control.mutate({ action: "start" })}>
              <Play />
              Resume
            </Button>
          ) : null}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Progress</CardTitle>
            <CardDescription>
              step {job.step.toLocaleString()} /{" "}
              {job.totalSteps.toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ProgressBar value={job.progress} className="h-2" />
            <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Metric label="Loss" value={job.loss.toFixed(3)} />
              <Metric
                label="Learning rate"
                value={job.learningRate.toExponential(2)}
              />
              <Metric label="Tokens" value={formatTokens(job.tokens)} />
              <Metric label="GPU" value={`${job.gpuUtilization.toFixed(1)}%`} />
              <Metric label="Memory" value={`${job.memoryGB.toFixed(1)} GB`} />
              <Metric label="Precision" value={job.precision} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>GPU utilization</CardTitle>
          </CardHeader>
          <CardContent className="flex h-56 items-end gap-1.5">
            {spark(job.gpuUtilization, 12).map((h, i) => (
              <div
                key={i}
                className="flex-1 rounded-t bg-primary/30"
                style={{ height: `${h}%` }}
              />
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Checkpoints</CardTitle>
            <CardDescription>Saved model states</CardDescription>
          </CardHeader>
          <CardContent>
            {checkpoints && checkpoints.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Step</TableHead>
                    <TableHead>Loss</TableHead>
                    <TableHead>Size</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {checkpoints.map((c) => (
                    <TableRow key={c.id}>
                      <TableCell>{c.step.toLocaleString()}</TableCell>
                      <TableCell>{c.loss.toFixed(3)}</TableCell>
                      <TableCell>{c.sizeMB.toFixed(1)} MB</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <p className="text-sm text-muted-foreground">
                No checkpoints saved yet.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex h-56 flex-col gap-1 overflow-y-auto rounded-md border border-border bg-black/40 p-3 font-mono text-xs">
              <LogLine>config digest 7f2a9b → job {job.id.slice(0, 8)}</LogLine>
              <LogLine>data version bangla-wiki-v2 digest a1b2c3</LogLine>
              <LogLine>tokenizer bpe (32,000) digest 5d8e91</LogLine>
              <LogLine>
                step {job.step.toLocaleString()} loss {job.loss.toFixed(3)} lr{" "}
                {job.learningRate.toExponential(2)}
              </LogLine>
              <LogLine>checkpoint saved (best)</LogLine>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border px-3 py-2">
      <p className="text-sm font-semibold">{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
    </div>
  );
}

function LogLine({ children }: { children: React.ReactNode }) {
  return (
    <div className="whitespace-pre-wrap text-muted-foreground">{children}</div>
  );
}

function formatTokens(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  return `${n.toLocaleString()}`;
}

function spark(value: number, count: number): number[] {
  return Array.from({ length: count }, (_, i) => {
    const noise = Math.sin(i * 1.7 + value) * 10;
    const base = value * 0.6 + 20;
    return Math.max(8, Math.min(98, base + noise));
  });
}
