import { get } from "@/lib/api/client";

export interface UsageMetric {
  period: string;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  latencyMs: number;
  errors: number;
}

export interface UsageSummary {
  requests: number;
  tokens: number;
  cost: number;
  errorRate: number;
  breakdown: Record<string, UsageMetric>;
}

export interface UsageApi {
  summary(range?: string): Promise<UsageSummary>;
  timeSeries(range?: string, granularity?: string): Promise<UsageMetric[]>;
  byModel(): Promise<Record<string, UsageMetric>>;
}

export const usageApi: UsageApi = {
  summary: (range = "30d") => {
    const params = new URLSearchParams({ range });
    return get<UsageSummary>(`/v1/usage/summary?${params.toString()}`);
  },
  timeSeries: (range = "30d", granularity = "day") => {
    const params = new URLSearchParams({ range, granularity });
    return get<UsageMetric[]>(`/v1/usage/timeseries?${params.toString()}`);
  },
  byModel: () => get<Record<string, UsageMetric>>("/v1/usage/by-model"),
};
