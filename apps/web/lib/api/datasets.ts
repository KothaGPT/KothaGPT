import { get, post, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";
import type { Dataset, DatasetPreview } from "@/types/dataset";

export interface DatasetsApi {
  list(): Promise<Paginated<Dataset>>;
  get(id: string): Promise<Dataset>;
  upload(file: File, name: string): Promise<Dataset>;
  preview(id: string): Promise<DatasetPreview>;
  versions(id: string): Promise<Paginated<Dataset>>;
  remove(id: string): Promise<void>;
}

export const datasetsApi: DatasetsApi = {
  list: () => get<Paginated<Dataset>>("/v1/datasets"),
  get: (id) => get<Dataset>(`/v1/datasets/${id}`),
  upload: async (file, name) => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    return post<Dataset>("/v1/datasets", form);
  },
  preview: (id) => get<DatasetPreview>(`/v1/datasets/${id}/preview`),
  versions: (id) => get<Paginated<Dataset>>(`/v1/datasets/${id}/versions`),
  remove: (id) => del<void>(`/v1/datasets/${id}`),
};
