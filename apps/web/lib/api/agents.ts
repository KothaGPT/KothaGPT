import { get, post, patch, del } from "@/lib/api/client";
import type { Paginated } from "@/types/api";
import type { Agent, AgentTrace } from "@/types/agent";

export interface AgentsApi {
  list(): Promise<Paginated<Agent>>;
  get(id: string): Promise<Agent>;
  create(input: Partial<Agent>): Promise<Agent>;
  update(id: string, input: Partial<Agent>): Promise<Agent>;
  traces(id: string): Promise<AgentTrace[]>;
  run(id: string, input: string): Promise<AgentTrace>;
  remove(id: string): Promise<void>;
}

export const agentsApi: AgentsApi = {
  list: () => get<Paginated<Agent>>("/v1/agents"),
  get: (id) => get<Agent>(`/v1/agents/${encodeURIComponent(id)}`),
  create: (input) => post<Agent>("/v1/agents", input),
  update: (id, input) =>
    patch<Agent>(`/v1/agents/${encodeURIComponent(id)}`, input),
  traces: (id) =>
    get<AgentTrace[]>(`/v1/agents/${encodeURIComponent(id)}/traces`),
  run: (id, input) =>
    post<AgentTrace>(`/v1/agents/${encodeURIComponent(id)}/run`, { input }),
  remove: (id) => del<void>(`/v1/agents/${encodeURIComponent(id)}`),
};
