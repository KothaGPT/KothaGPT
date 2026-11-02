import { authApi } from "@/lib/api/auth";
import { chatApi } from "@/lib/api/chat";
import { modelsApi } from "@/lib/api/models";
import { datasetsApi } from "@/lib/api/datasets";
import { trainingApi } from "@/lib/api/training";
import { evaluationsApi } from "@/lib/api/evaluations";
import { knowledgeApi } from "@/lib/api/knowledge";
import { agentsApi } from "@/lib/api/agents";
import { toolsApi } from "@/lib/api/tools";
import { usageApi } from "@/lib/api/usage";

export const api = {
  auth: authApi,
  chat: chatApi,
  models: modelsApi,
  datasets: datasetsApi,
  training: trainingApi,
  evaluations: evaluationsApi,
  knowledge: knowledgeApi,
  agents: agentsApi,
  tools: toolsApi,
  usage: usageApi,
};

export {
  authApi,
  chatApi,
  modelsApi,
  datasetsApi,
  trainingApi,
  evaluationsApi,
  knowledgeApi,
  agentsApi,
  toolsApi,
  usageApi,
};
