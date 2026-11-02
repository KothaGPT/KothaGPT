export type DatasetStatus =
  | "uploading"
  | "normalizing"
  | "filtering"
  | "deduplicating"
  | "quality_check"
  | "tokenizing"
  | "sharding"
  | "ready"
  | "failed";

export interface Dataset {
  id: string;
  name: string;
  version: string;
  description?: string;
  status: DatasetStatus;
  languages: string[];
  records: number;
  tokens: number;
  duplicateRatio: number;
  qualityScore: number;
  piiStatus: "clean" | "review" | "blocked";
  license: string;
  trainSplit: number;
  validationSplit: number;
  testSplit: number;
  createdAt: string;
  updatedAt: string;
}

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, unknown>[];
}
