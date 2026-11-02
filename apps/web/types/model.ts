export type ModelStatus =
  "ready" | "training" | "queued" | "failed" | "inactive";

export interface Model {
  id: string;
  name: string;
  version: string;
  description?: string;
  status: ModelStatus;
  parameters: number;
  contextLength: number;
  tokenizer: string;
  language: string;
  active: boolean;
  benchmark?: {
    banglaQA: number;
    reasoning: number;
    coding: number;
    safety: number;
  };
  deployment?: {
    gpu: string;
    requestsPerMin: number;
    latencyMs: number;
  };
  createdAt: string;
  updatedAt: string;
}

export interface ModelComparisonRow {
  benchmark: string;
  scores: Record<string, number>;
}
