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
import { useModels, useDatasets, useCreateTrainingJob } from "@/hooks";

const int = (min: number) => z.coerce.number().int().min(min).pipe(z.number());
const num = () => z.coerce.number().positive().pipe(z.number());

const schema = z.object({
  name: z.string().min(1, "Name is required"),
  model: z.string().min(1, "Select a model"),
  dataset: z.string().min(1, "Select a dataset"),
  tokenizer: z.string().min(1, "Select a tokenizer"),
  batchSize: int(1),
  learningRate: num(),
  contextLength: int(256),
  gpuCount: int(1),
  precision: z.enum(["fp32", "bf16", "fp16"]),
  gradAccumulation: int(1),
});

type FormData = z.infer<typeof schema>;

export function CreateJobDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const models = useModels();
  const datasets = useDatasets();
  const create = useCreateTrainingJob();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormData>({
    // zod v4 coerce types the resolver input as `unknown`; numbers are validated
    // at runtime and converted by the schema's `.pipe(z.number())`.
    resolver: zodResolver(schema) as never,
    defaultValues: {
      name: "",
      model: "",
      dataset: "",
      tokenizer: "bpe",
      batchSize: 32,
      learningRate: 0.0002,
      contextLength: 4096,
      gpuCount: 1,
      precision: "bf16",
      gradAccumulation: 4,
    },
  });

  const submit = async (values: FormData) => {
    await create.mutateAsync({
      name: values.name,
      model: values.model,
      dataset: values.dataset,
      tokenizer: values.tokenizer,
      config: {
        batchSize: values.batchSize,
        learningRate: values.learningRate,
        contextLength: values.contextLength,
        gpuCount: values.gpuCount,
        precision: values.precision,
        gradAccumulation: values.gradAccumulation,
      },
    });
    reset();
    onOpenChange(false);
  };

  const tokenizers = [
    { id: "bpe", name: "BPE" },
    { id: "unigram", name: "Unigram" },
  ];
  const modelOptions = models.data?.items ?? [];
  const datasetOptions = datasets.data?.items ?? [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create training job</DialogTitle>
          <DialogDescription>
            Configure a pretraining run. Everything is reproducible — jobs are
            tagged with model, dataset and tokenizer digests.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(submit)} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="job-name">Job name</Label>
            <Input
              id="job-name"
              placeholder="BanglaLM-v0.1"
              {...register("name")}
            />
            {errors.name && (
              <p className="text-xs text-destructive">{errors.name.message}</p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="job-model">Model</Label>
              <select
                id="job-model"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                {...register("model")}
              >
                <option value="">Select…</option>
                {modelOptions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
              {errors.model && (
                <p className="text-xs text-destructive">
                  {errors.model.message}
                </p>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="job-dataset">Dataset</Label>
              <select
                id="job-dataset"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                {...register("dataset")}
              >
                <option value="">Select…</option>
                {datasetOptions.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
              {errors.dataset && (
                <p className="text-xs text-destructive">
                  {errors.dataset.message}
                </p>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="job-tokenizer">Tokenizer</Label>
            <select
              id="job-tokenizer"
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
              {...register("tokenizer")}
            >
              {tokenizers.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field
              label="Batch size"
              htmlFor="job-batch-size"
              error={errors.batchSize?.message}
            >
              <Input
                id="job-batch-size"
                type="number"
                {...register("batchSize")}
              />
            </Field>
            <Field
              label="Learning rate"
              htmlFor="job-learning-rate"
              error={errors.learningRate?.message}
            >
              <Input
                id="job-learning-rate"
                type="number"
                step="0.0001"
                {...register("learningRate")}
              />
            </Field>
            <Field
              label="Context length"
              htmlFor="job-context-length"
              error={errors.contextLength?.message}
            >
              <Input
                id="job-context-length"
                type="number"
                {...register("contextLength")}
              />
            </Field>
            <Field
              label="GPUs"
              htmlFor="job-gpu-count"
              error={errors.gpuCount?.message}
            >
              <Input
                id="job-gpu-count"
                type="number"
                {...register("gpuCount")}
              />
            </Field>
            <Field
              label="Grad accumulation"
              htmlFor="job-grad-accumulation"
              error={errors.gradAccumulation?.message}
            >
              <Input
                id="job-grad-accumulation"
                type="number"
                {...register("gradAccumulation")}
              />
            </Field>
            <Field label="Precision" htmlFor="job-precision">
              <select
                id="job-precision"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                {...register("precision")}
              >
                <option value="bf16">bf16</option>
                <option value="fp16">fp16</option>
                <option value="fp32">fp32</option>
              </select>
            </Field>
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={create.isPending}>
              {create.isPending ? "Creating…" : "Start training"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  htmlFor,
  error,
  children,
}: {
  label: string;
  htmlFor?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={htmlFor} className="text-xs text-muted-foreground">
        {label}
      </Label>
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
