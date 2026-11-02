export type AgentStatus = "active" | "inactive" | "draft";

export interface Agent {
  id: string;
  name: string;
  description?: string;
  model: string;
  status: AgentStatus;
  systemPrompt: string;
  tools: string[];
  knowledgeBaseId?: string;
  memoryEnabled: boolean;
  limits: {
    maxTokensPerRun: number;
    maxRunsPerMinute: number;
  };
  totalTokens: number;
  totalCost: number;
  createdAt: string;
  updatedAt: string;
}

export interface AgentTrace {
  id: string;
  agentId: string;
  startedAt: string;
  endedAt?: string;
  status: "completed" | "failed" | "running";
  steps: {
    type: "model" | "tool" | "memory" | "knowledge";
    detail: string;
    latencyMs: number;
  }[];
}
