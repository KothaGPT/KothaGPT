export interface ApiError {
  error: string;
  message?: string;
  code?: string;
  details?: unknown;
}

export interface Paginated<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

export interface ApiResult<T> {
  data: T | null;
  error: ApiError | null;
}

export type RequestOptions = {
  signal?: AbortSignal;
  headers?: Record<string, string>;
};

export type Body = string | object | FormData | undefined;
