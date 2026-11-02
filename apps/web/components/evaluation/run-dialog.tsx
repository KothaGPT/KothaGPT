"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useModels, useBenchmarks, useCreateEvaluation } from "@/hooks";

const schema = z.object({
  name: z.string().min(1, "Name is required"),
  model: z.string().min(1, "Select a model"),
  benchmark: z.string().min(1, "Select a benchmark"),
});

type FormData = z.infer<typeof schema>;

export function RunEvaluationDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const models = useModels();
  const benchmarks = useBenchmarks();
  const create = useCreateEvaluation();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(schema) as never,
    defaultValues: { name: "", model: "", benchmark: "" },
  });

  const submit = async (values: FormData) => {
    await create.mutateAsync({
      name: values.name,
      model: values.model,
      benchmark: values.benchmark,
    });
    reset();
    onOpenChange(false);
  };

  const handleOpenChange = (next: boolean) => {
    if (!next && create.isPending) return;
    onOpenChange(next);
  };

  const modelOptions = models.data?.items ?? [];
  const benchmarkOptions = benchmarks.data ?? [];

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Run evaluation</DialogTitle>
          <DialogDescription>
            Run a model against a benchmark suite and compare across tasks.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(submit)} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="eval-name">Run name</Label>
            <Input
              id="eval-name"
              placeholder="bangla-core on KothaGPT 0.1"
              {...register("name")}
            />
            {errors.name && (
              <p className="text-xs text-destructive">{errors.name.message}</p>
            )}
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="eval-model">Model</Label>
            <select
              id="eval-model"
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
              {...register("model")}
            >
              <option value="">Select…</option>
              {modelOptions.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} (v{m.version})
                </option>
              ))}
            </select>
            {errors.model && (
              <p className="text-xs text-destructive">{errors.model.message}</p>
            )}
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="eval-benchmark">Benchmark</Label>
            <select
              id="eval-benchmark"
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
              {...register("benchmark")}
            >
              <option value="">Select…</option>
              {benchmarkOptions.map((b) => (
                <option key={`${b.name}-${b.version}`} value={b.name}>
                  {b.name} ({b.version})
                </option>
              ))}
            </select>
            {errors.benchmark && (
              <p className="text-xs text-destructive">
                {errors.benchmark.message}
              </p>
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={create.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={create.isPending}>
              {create.isPending ? "Queuing…" : "Run evaluation"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
