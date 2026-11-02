"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { authApi } from "@/lib/api/auth";
import { chatApi } from "@/lib/api/chat";
import { modelsApi } from "@/lib/api/models";
import { datasetsApi } from "@/lib/api/datasets";
import { trainingApi } from "@/lib/api/training";
import { knowledgeApi } from "@/lib/api/knowledge";
import { agentsApi } from "@/lib/api/agents";
import { evaluationsApi } from "@/lib/api/evaluations";
import { usageApi } from "@/lib/api/usage";
import type { TrainingJob } from "@/types/training";

export function useMe() {
  return useQuery({ queryKey: ["auth", "me"], queryFn: authApi.me });
}

export function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: chatApi.conversations,
  });
}

export function useConversation(id: string) {
  return useQuery({
    queryKey: ["conversations", id],
    queryFn: () => chatApi.conversation(id),
    enabled: !!id,
  });
}

export function useMessages(conversationId: string) {
  return useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => chatApi.messages(conversationId),
    enabled: !!conversationId,
  });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: modelsApi.list });
}

export function useModel(id: string) {
  return useQuery({
    queryKey: ["models", id],
    queryFn: () => modelsApi.get(id),
    enabled: !!id,
  });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });
}

export function useDataset(id: string) {
  return useQuery({
    queryKey: ["datasets", id],
    queryFn: () => datasetsApi.get(id),
    enabled: !!id,
  });
}

export function useDatasetVersions(id: string) {
  return useQuery({
    queryKey: ["datasets", id, "versions"],
    queryFn: () => datasetsApi.versions(id),
    enabled: !!id,
  });
}

export function useUploadDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, name }: { file: File; name: string }) =>
      datasetsApi.upload(file, name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useTrainingJobs() {
  return useQuery({ queryKey: ["training"], queryFn: trainingApi.list });
}

export function useTrainingJob(id: string) {
  return useQuery({
    queryKey: ["training", id],
    queryFn: () => trainingApi.get(id),
    enabled: !!id,
    refetchInterval: (query) => {
      const job = query.state.data as TrainingJob | undefined;
      return job && job.status === "running" ? 3000 : false;
    },
  });
}

export function useTrainingCheckpoints(id: string) {
  return useQuery({
    queryKey: ["training", id, "checkpoints"],
    queryFn: () => trainingApi.checkpoints(id),
    enabled: !!id,
  });
}

export function useCreateTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: trainingApi.create,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["training"] }),
  });
}

export function useTrainingControl(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ action }: { action: "start" | "pause" | "stop" }) =>
      action === "start"
        ? trainingApi.start(id)
        : action === "pause"
          ? trainingApi.pause(id)
          : trainingApi.stop(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["training"] });
      qc.invalidateQueries({ queryKey: ["training", id] });
    },
  });
}

export function useKnowledgeBases() {
  return useQuery({ queryKey: ["knowledge"], queryFn: knowledgeApi.list });
}

export function useAgents() {
  return useQuery({ queryKey: ["agents"], queryFn: agentsApi.list });
}

export function useEvaluations() {
  return useQuery({ queryKey: ["evaluations"], queryFn: evaluationsApi.list });
}

export function useBenchmarks() {
  return useQuery({
    queryKey: ["benchmarks"],
    queryFn: evaluationsApi.benchmarks,
  });
}

export function useUsageSummary() {
  return useQuery({
    queryKey: ["usage", "summary"],
    queryFn: () => usageApi.summary("30d"),
  });
}

export function useCreateEvaluation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: evaluationsApi.create,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["evaluations"] }),
  });
}

export function useDeleteEvaluation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: evaluationsApi.remove,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["evaluations"] }),
  });
}

export function useSetActiveModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, active }: { id: string; active: boolean }) =>
      modelsApi.setActive(id, active),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
}

export function useCreateConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ title, model }: { title: string; model: string }) =>
      chatApi.createConversation(title, model),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}
