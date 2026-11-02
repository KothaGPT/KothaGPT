import { get, post, patch } from "@/lib/api/client";
import type { Paginated } from "@/types/api";
import type { Checkpoint, TrainingConfig, TrainingJob } from "@/types/training";

export interface TrainingApi {
  list(): Promise<Paginated<TrainingJob>>;
  get(id: string): Promise<TrainingJob>;
  create(input: {
    name: string;
    model: string;
    dataset: string;
    tokenizer: string;
    config: TrainingConfig;
  }): Promise<TrainingJob>;
  start(id: string): Promise<TrainingJob>;
  pause(id: string): Promise<TrainingJob>;
  stop(id: string): Promise<TrainingJob>;
  checkpoints(id: string): Promise<Checkpoint[]>;
  logs(id: string): Promise<string>;
}

export const trainingApi: TrainingApi = {
  list: () => get<Paginated<TrainingJob>>("/v1/training"),
  get: (id) => get<TrainingJob>(`/v1/training/${id}`),
  create: (input) => post<TrainingJob>("/v1/training", input),
  start: (id) => patch<TrainingJob>(`/v1/training/${id}/start`),
  pause: (id) => patch<TrainingJob>(`/v1/training/${id}/pause`),
  stop: (id) => patch<TrainingJob>(`/v1/training/${id}/stop`),
  checkpoints: (id) => get<Checkpoint[]>(`/v1/training/${id}/checkpoints`),
  logs: (id) => get<string>(`/v1/training/${id}/logs`),
};
