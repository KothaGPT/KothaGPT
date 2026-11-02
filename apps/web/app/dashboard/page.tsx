"use client";

import Link from "next/link";
import {
  Activity,
  Cpu,
  Database,
  Gauge,
  MessagesSquare,
  Sparkles,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  useModels,
  useTrainingJobs,
  useConversations,
  useDatasets,
  useUsageSummary,
} from "@/hooks";

function StatCard({
  title,
  value,
  hint,
  icon: Icon,
}: {
  title: string;
  value: string;
  hint: string;
  icon: typeof Cpu;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="size-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const models = useModels();
  const training = useTrainingJobs();
  const conversations = useConversations();
  const datasets = useDatasets();
  const usage = useUsageSummary();
  const readyDatasets = datasets.data
    ? datasets.data.items.filter((d) => d.status === "ready").length
    : undefined;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
          <p className="text-sm text-muted-foreground">
            স্বাগতম — welcome back to your AI control center.
          </p>
        </div>
        <Button asChild>
          <Link href="/dashboard/chat">
            <MessagesSquare />
            New chat
          </Link>
        </Button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Models"
          value={models.data ? String(models.data.total) : "—"}
          hint={`${models.data?.total ?? 0} deployed · ${models.data?.items.filter((m) => m.status === "ready").length ?? 0} ready`}
          icon={Cpu}
        />
        <StatCard
          title="Tokens"
          value={usage.data ? usage.data.tokens.toLocaleString() : "—"}
          hint="consumed this month"
          icon={Sparkles}
        />
        <StatCard
          title="Requests"
          value={usage.data ? usage.data.requests.toLocaleString() : "—"}
          hint="in the last 30 days"
          icon={Activity}
        />
        <StatCard
          title="Datasets"
          value={readyDatasets !== undefined ? String(readyDatasets) : "—"}
          hint="ready for training"
          icon={Database}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Model activity</CardTitle>
            <CardDescription>Inference requests over time</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            <ActivityPlaceholder />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System health</CardTitle>
            <CardDescription>Runtime status</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <HealthRow label="API Gateway" />
            <HealthRow label="Model Runtime" />
            <HealthRow label="RAG Indexer" />
            <HealthRow
              label="Training Cluster"
              ok={training.data ? training.data.total > 0 : undefined}
            />
            <HealthRow label="Data Pipeline" />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent conversations</CardTitle>
            <CardDescription>Your latest chats</CardDescription>
          </CardHeader>
          <CardContent>
            {conversations.isLoading ? (
              <div className="flex flex-col gap-3">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : conversations.data && conversations.data.items.length > 0 ? (
              <div className="flex flex-col gap-2">
                {conversations.data.items.slice(0, 5).map((c) => (
                  <Link
                    key={c.id}
                    href={`/dashboard/chat/${c.id}`}
                    className="flex items-center justify-between rounded-md px-2 py-2 text-sm hover:bg-accent"
                  >
                    <span className="truncate">{c.title}</span>
                    <span className="text-xs text-muted-foreground">
                      {c.messageCount} msgs
                    </span>
                  </Link>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No conversations yet. Start your first chat.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training jobs</CardTitle>
            <CardDescription>Recent runs</CardDescription>
          </CardHeader>
          <CardContent>
            {training.isLoading ? (
              <div className="flex flex-col gap-3">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : training.data && training.data.items.length > 0 ? (
              <div className="flex flex-col gap-2">
                {training.data.items.slice(0, 4).map((j) => (
                  <div
                    key={j.id}
                    className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                  >
                    <div className="flex items-center gap-2">
                      <Gauge className="size-4 text-muted-foreground" />
                      <div>
                        <p className="text-sm">{j.name}</p>
                        <p className="text-xs text-muted-foreground">
                          step {j.step.toLocaleString()} · loss{" "}
                          {j.loss.toFixed(2)}
                        </p>
                      </div>
                    </div>
                    <Badge
                      variant={j.status === "running" ? "success" : "secondary"}
                    >
                      {j.status}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No training jobs yet.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function HealthRow({ label, ok }: { label: string; ok?: boolean }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="flex items-center gap-2">
        <span
          className={`size-2 rounded-full ${
            ok === undefined
              ? "bg-muted-foreground/40"
              : ok
                ? "bg-emerald-500"
                : "bg-amber-500"
          }`}
        />
        {label}
      </span>
      <span className="text-xs text-muted-foreground">
        {ok === undefined ? "Unavailable" : ok ? "Operational" : "No data"}
      </span>
    </div>
  );
}

function ActivityPlaceholder() {
  const bars = [35, 55, 40, 70, 60, 85, 50, 65, 75, 45, 90, 68];
  return (
    <div className="flex h-full items-end gap-1.5">
      {bars.map((h, i) => (
        <div
          key={i}
          className="flex-1 rounded-t bg-primary/30 transition-all hover:bg-primary/60"
          style={{ height: `${h}%` }}
        />
      ))}
    </div>
  );
}
