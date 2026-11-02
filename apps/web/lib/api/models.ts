import { get, post, patch, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";
import type { Model, ModelComparisonRow } from "@/types/model";

export interface ModelsApi {
  list(): Promise<Paginated<Model>>;
  get(id: string): Promise<Model>;
  create(input: Partial<Model>): Promise<Model>;
  setActive(id: string, active: boolean): Promise<Model>;
  compare(ids: string[]): Promise<ModelComparisonRow[]>;
  remove(id: string): Promise<void>;
}

export const modelsApi: ModelsApi = {
  list: () => get<Paginated<Model>>("/v1/models"),
  get: (id) => get<Model>(`/v1/models/${encodeURIComponent(id)}`),
  create: (input) => post<Model>("/v1/models", input),
  setActive: (id, active) =>
    patch<Model>(`/v1/models/${encodeURIComponent(id)}`, { active }),
  compare: (ids) => post<ModelComparisonRow[]>("/v1/models/compare", { ids }),
  remove: (id) => del<void>(`/v1/models/${encodeURIComponent(id)}`),
};
