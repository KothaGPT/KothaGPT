"use client";

import * as React from "react";
import Link from "next/link";
import { Gauge, Plus } from "lucide-react";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ProgressBar } from "@/components/ui/progress";
import { CreateJobDialog } from "@/components/training/create-job-dialog";
import { useTrainingJobs } from "@/hooks";
import type { TrainingStatus } from "@/types/training";

const statusBadge: Record<
  TrainingStatus,
  "success" | "warning" | "secondary" | "destructive"
> = {
  running: "success",
  paused: "warning",
  pending: "secondary",
  completed: "secondary",
  failed: "destructive",
};

export default function TrainingPage() {
  const [createOpen, setCreateOpen] = React.useState(false);
  const { data, isLoading } = useTrainingJobs();
  const jobs = data?.items ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Training</h1>
          <p className="text-sm text-muted-foreground">
            Configure, run and monitor pretraining jobs.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus />
          New training job
        </Button>
      </div>

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : jobs.length === 0 ? (
        <Card className="flex flex-col items-center justify-center gap-3 py-20 text-center">
          <Gauge className="size-10 text-muted-foreground" />
          <CardHeader>
            <CardTitle>No training jobs</CardTitle>
            <CardDescription>
              Create a job to pretrain a model on one of your datasets.
            </CardDescription>
          </CardHeader>
          <Button onClick={() => setCreateOpen(true)}>
            <Plus />
            New training job
          </Button>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job) => (
            <Link
              key={job.id}
              href={`/dashboard/training/${job.id}`}
              className="block rounded-xl border border-transparent transition-colors hover:border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <Card className="flex flex-col gap-4">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle>{job.name}</CardTitle>
                      <CardDescription>
                        {job.model} · {job.dataset}
                      </CardDescription>
                    </div>
                    <Badge variant={statusBadge[job.status]}>
                      {job.status}
                    </Badge>
                  </div>
                </CardHeader>
                <div className="flex flex-col gap-3 px-6 pb-4">
                  <div>
                    <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                      <span>
                        step {job.step.toLocaleString()} /{" "}
                        {job.totalSteps.toLocaleString()}
                      </span>
                      <span>{Math.round(job.progress)}%</span>
                    </div>
                    <ProgressBar value={job.progress} />
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <Metric label="Loss" value={job.loss.toFixed(2)} />
                    <Metric
                      label="LR"
                      value={job.learningRate.toExponential(2)}
                    />
                    <Metric label="Tokens" value={formatTokens(job.tokens)} />
                    <Metric
                      label="GPU"
                      value={`${job.gpuUtilization.toFixed(0)}%`}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{job.precision}</span>
                    <span>{job.memoryGB.toFixed(1)} GB</span>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <CreateJobDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border px-2 py-1.5">
      <p className="font-semibold">{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
    </div>
  );
}

function formatTokens(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}
