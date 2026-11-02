"use client";

import * as React from "react";
import { Database, Plus } from "lucide-react";
import {
  Card,
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
import { PipelineStages } from "@/components/dataset/pipeline-stages";
import { DatasetUploadDialog } from "@/components/dataset/upload-dialog";
import { useDatasets } from "@/hooks";
import type { DatasetStatus } from "@/types/dataset";

const statusBadge: Record<
  DatasetStatus,
  "success" | "warning" | "secondary" | "destructive"
> = {
  ready: "success",
  failed: "destructive",
  uploading: "warning",
  normalizing: "warning",
  filtering: "warning",
  deduplicating: "warning",
  quality_check: "warning",
  tokenizing: "warning",
  sharding: "warning",
};

function formatTokens(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

export default function DatasetsPage() {
  const [uploadOpen, setUploadOpen] = React.useState(false);
  const { data, isLoading } = useDatasets();
  const datasets = data?.items ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
          <p className="text-sm text-muted-foreground">
            Curate, version and inspect your training corpora.
          </p>
        </div>
        <Button onClick={() => setUploadOpen(true)}>
          <Plus />
          Upload dataset
        </Button>
      </div>

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : datasets.length === 0 ? (
        <Card className="flex flex-col items-center justify-center gap-3 py-20 text-center">
          <Database className="size-10 text-muted-foreground" />
          <CardHeader>
            <CardTitle>No datasets yet</CardTitle>
            <CardDescription>
              Upload a corpus and the pipeline will normalize, filter,
              deduplicate and shard it.
            </CardDescription>
          </CardHeader>
          <Button onClick={() => setUploadOpen(true)}>
            <Plus />
            Upload dataset
          </Button>
        </Card>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Languages</TableHead>
                <TableHead>Records</TableHead>
                <TableHead>Tokens</TableHead>
                <TableHead>Quality</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="w-48">Pipeline</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {datasets.map((d) => (
                <TableRow key={d.id}>
                  <TableCell>
                    <div>
                      <p className="font-medium">{d.name}</p>
                      <p className="text-xs text-muted-foreground">
                        v{d.version}
                      </p>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {d.languages.join(", ")}
                  </TableCell>
                  <TableCell>{d.records.toLocaleString()}</TableCell>
                  <TableCell>{formatTokens(d.tokens)}</TableCell>
                  <TableCell>
                    {d.qualityScore > 0
                      ? `${d.qualityScore.toFixed(1)} / 100`
                      : "—"}
                  </TableCell>
                  <TableCell>
                    <Badge variant={statusBadge[d.status]}>{d.status}</Badge>
                  </TableCell>
                  <TableCell>
                    <PipelineStages status={d.status} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      <DatasetUploadDialog open={uploadOpen} onOpenChange={setUploadOpen} />
    </div>
  );
}
