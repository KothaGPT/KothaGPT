import { get, post, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";

export interface ToolDefinition {
  id: string;
  name: string;
  description: string;
  schema: Record<string, unknown>;
  enabled: boolean;
  usage: {
    runs: number;
    lastUsed?: string;
  };
}

export interface ToolsApi {
  list(): Promise<Paginated<ToolDefinition>>;
  get(id: string): Promise<ToolDefinition>;
  create(input: Partial<ToolDefinition>): Promise<ToolDefinition>;
  remove(id: string): Promise<void>;
}

export const toolsApi: ToolsApi = {
  list: () => get<Paginated<ToolDefinition>>("/v1/tools"),
  get: (id) => get<ToolDefinition>(`/v1/tools/${id}`),
  create: (input) => post<ToolDefinition>("/v1/tools", input),
  remove: (id) => del<void>(`/v1/tools/${id}`),
};
