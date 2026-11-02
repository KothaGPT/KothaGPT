import { get, post, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";

export type EvalStatus = "queued" | "running" | "completed" | "failed";

export interface BenchmarkTask {
  id: string;
  total: number;
  dev: number;
  test: number;
}

export interface BenchmarkDefinition {
  name: string;
  version: string;
  tasks: BenchmarkTask[];
}

export interface EvaluationRun {
  id: string;
  name: string;
  model: string;
  benchmark: string;
  status: EvalStatus;
  scores: Record<string, number>;
  startedAt: string;
  completedAt?: string;
}

export interface EvaluationsApi {
  list(): Promise<Paginated<EvaluationRun>>;
  get(id: string): Promise<EvaluationRun>;
  benchmarks(): Promise<BenchmarkDefinition[]>;
  create(input: {
    name: string;
    model: string;
    benchmark: string;
  }): Promise<EvaluationRun>;
  remove(id: string): Promise<void>;
}

export const evaluationsApi: EvaluationsApi = {
  list: () => get<Paginated<EvaluationRun>>("/v1/evaluations"),
  get: (id) => get<EvaluationRun>(`/v1/evaluations/${id}`),
  benchmarks: () => get<BenchmarkDefinition[]>("/v1/benchmarks"),
  create: (input) => post<EvaluationRun>("/v1/evaluations", input),
  remove: (id) => del<void>(`/v1/evaluations/${id}`),
};
