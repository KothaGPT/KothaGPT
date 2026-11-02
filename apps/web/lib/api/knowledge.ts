import { get, post, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";

export type KnowledgeBaseStatus = "indexing" | "ready" | "failed";

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  status: KnowledgeBaseStatus;
  documents: number;
  chunks: number;
  embeddingModel: string;
  vectorCollection: string;
  createdAt: string;
}

export interface KnowledgeRetrieval {
  query: string;
  results: {
    chunkId: string;
    documentId: string;
    content: string;
    score: number;
    citation?: string;
  }[];
}

export interface KnowledgeApi {
  list(): Promise<Paginated<KnowledgeBase>>;
  get(id: string): Promise<KnowledgeBase>;
  create(input: {
    name: string;
    description?: string;
    embeddingModel?: string;
    chunkSize?: number;
    chunkOverlap?: number;
  }): Promise<KnowledgeBase>;
  ingestDocuments(id: string, files: File[]): Promise<KnowledgeBase>;
  ingestUrl(id: string, url: string): Promise<KnowledgeBase>;
  retrieve(
    id: string,
    query: string,
    topK?: number,
  ): Promise<KnowledgeRetrieval>;
  remove(id: string): Promise<void>;
}

export const knowledgeApi: KnowledgeApi = {
  list: () => get<Paginated<KnowledgeBase>>("/v1/knowledge"),
  get: (id) => get<KnowledgeBase>(`/v1/knowledge/${encodeURIComponent(id)}`),
  create: (input) => post<KnowledgeBase>("/v1/knowledge", input),
  ingestDocuments: async (id, files) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return post<KnowledgeBase>(
      `/v1/knowledge/${encodeURIComponent(id)}/ingest`,
      form,
    );
  },
  ingestUrl: (id, url) =>
    post<KnowledgeBase>(`/v1/knowledge/${encodeURIComponent(id)}/ingest-url`, {
      url,
    }),
  retrieve: (id, query, topK = 5) =>
    post<KnowledgeRetrieval>(
      `/v1/knowledge/${encodeURIComponent(id)}/retrieve`,
      { query, topK },
    ),
  remove: (id) => del<void>(`/v1/knowledge/${encodeURIComponent(id)}`),
};
