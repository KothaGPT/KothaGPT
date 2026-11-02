import type { ApiError, ApiResult, Body, RequestOptions } from "@/types/api";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "kothagpt.access_token";
const REQUEST_TIMEOUT_MS = 30_000;

function timeoutSignal(signal?: AbortSignal): AbortSignal {
  if (
    typeof AbortSignal !== "undefined" &&
    typeof (AbortSignal as unknown as { timeout?: unknown }).timeout === "function" &&
    typeof (AbortSignal as unknown as { any?: unknown }).any === "function"
  ) {
    const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
    if (!signal) return timeout;
    return AbortSignal.any([signal, timeout]);
  }
  if (
    typeof AbortSignal !== "undefined" &&
    typeof (AbortSignal as unknown as { timeout?: unknown }).timeout === "function" &&
    !signal
  ) {
    return AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  }
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  if (signal?.aborted) {
    window.clearTimeout(timer);
    controller.abort(signal.reason);
    return controller.signal;
  }
  signal?.addEventListener(
    "abort",
    () => {
      window.clearTimeout(timer);
      controller.abort(signal.reason);
    },
    { once: true },
  );
  return controller.signal;
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

export class ApiClientError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(
  path: string,
  options: RequestOptions = {},
  init: Omit<RequestInit, "signal" | "headers"> & {
    headers?: Record<string, string>;
  } = {},
): Promise<T> {
  const headers: Record<string, string> = {
    ...options.headers,
    ...(init.headers ?? {}),
  };
  const token = getAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers,
    signal: timeoutSignal(options.signal),
  });

  const text = await response.text();
  const body = text ? safeParse(text) : null;

  if (!response.ok) {
    throw new ApiClientError(
      (body as ApiError)?.message ?? response.statusText,
      response.status,
      (body as ApiError)?.code,
      (body as ApiError)?.details,
    );
  }

  return body as T;
}

async function requestStream(
  path: string,
  options: RequestOptions = {},
  init: Omit<RequestInit, "signal" | "headers"> & {
    headers?: Record<string, string>;
  } = {},
): Promise<ReadableStream<Uint8Array>> {
  const headers: Record<string, string> = {
    ...options.headers,
    ...(init.headers ?? {}),
  };
  const token = getAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers,
    signal: timeoutSignal(options.signal),
  });

  if (!response.ok || !response.body) {
    const text = await response.text();
    const body = text ? safeParse(text) : null;
    throw new ApiClientError(
      (body as ApiError)?.message ?? response.statusText,
      response.status,
      (body as ApiError)?.code,
      (body as ApiError)?.details,
    );
  }

  return response.body;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function get<T>(path: string, options?: RequestOptions): Promise<T> {
  return request<T>(path, options, { method: "GET" });
}

export function post<T>(
  path: string,
  body?: Body,
  options?: RequestOptions,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body instanceof FormData) {
    // let the browser set the multipart boundary
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  return request<T>(path, options, {
    method: "POST",
    headers,
    body: body === undefined ? undefined : serialize(body),
  });
}

function postStream(
  path: string,
  body?: Body,
  options?: RequestOptions,
): Promise<ReadableStream<Uint8Array>> {
  const headers: Record<string, string> =
    body !== undefined && !(body instanceof FormData)
      ? { "Content-Type": "application/json" }
      : {};
  return requestStream(path, options, {
    method: "POST",
    headers,
    body: body === undefined ? undefined : serialize(body),
  });
}

export function put<T>(
  path: string,
  body?: Body,
  options?: RequestOptions,
): Promise<T> {
  const headers: Record<string, string> =
    body !== undefined && !(body instanceof FormData)
      ? { "Content-Type": "application/json" }
      : {};
  return request<T>(path, options, {
    method: "PUT",
    headers,
    body: body === undefined ? undefined : serialize(body),
  });
}

export function patch<T>(
  path: string,
  body?: Body,
  options?: RequestOptions,
): Promise<T> {
  const headers: Record<string, string> =
    body !== undefined && !(body instanceof FormData)
      ? { "Content-Type": "application/json" }
      : {};
  return request<T>(path, options, {
    method: "PATCH",
    headers,
    body: body === undefined ? undefined : serialize(body),
  });
}

export function del<T>(path: string, options?: RequestOptions): Promise<T> {
  return request<T>(path, options, { method: "DELETE" });
}

function serialize(body: Body): BodyInit | undefined {
  if (typeof body === "string" || body instanceof FormData) return body;
  return JSON.stringify(body);
}

export async function toResult<T>(promise: Promise<T>): Promise<ApiResult<T>> {
  try {
    return { data: await promise, error: null };
  } catch (err) {
    if (err instanceof ApiClientError) {
      return {
        data: null,
        error: {
          error: err.message,
          code: err.code,
          details: err.details,
        },
      };
    }
    return { data: null, error: { error: "Network error" } };
  }
}

export { BASE_URL, request, postStream };
export type { ApiError };
