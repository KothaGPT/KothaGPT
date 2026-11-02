export type TrainingStatus =
  "pending" | "running" | "paused" | "completed" | "failed";

export interface TrainingJob {
  id: string;
  name: string;
  model: string;
  dataset: string;
  tokenizer: string;
  status: TrainingStatus;
  step: number;
  totalSteps: number;
  progress: number;
  loss: number;
  learningRate: number;
  tokens: number;
  gpuUtilization: number;
  memoryGB: number;
  precision: string;
  createdAt: string;
  updatedAt: string;
}

export interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  contextLength: number;
  gpuCount: number;
  precision: "fp32" | "bf16" | "fp16";
  gradAccumulation: number;
  seed?: number;
}

export interface Checkpoint {
  id: string;
  step: number;
  loss: number;
  sizeMB: number;
  createdAt: string;
}
