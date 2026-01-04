// frontend/src/lib/api.ts

const RAW_API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const API_BASE = RAW_API_BASE.replace(/\/+$/, ""); // trim trailing slashes

// -----------------------------
// JSON-safe types (no `any`)
// -----------------------------
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

// -----------------------------
// Types (match backend responses)
// -----------------------------
export type DatasetUploadResponse = {
  dataset_id: string;
  stored_path: string;
  num_events: number;
  num_cases: number;
  columns: string[];
  detected_mapping: Record<string, string>;
  preview: Array<Record<string, JsonValue>>;
};

export type DatasetMeta = {
  dataset_id: string;
  stored_path: string;
  num_events: number;
  num_cases: number;
  columns: string[];
  detected_mapping: Record<string, string>;
  created_at: string;
};

export type RunStatus = {
  run_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at?: string;
  started_at?: string;
  finished_at?: string;
  pid?: number;
  error?: string | null;
};

export type CreateRunReq = {
  dataset_id: string;
  model_type: "gnn" | "transformer";
  task: "next_activity" | "event_time" | "remaining_time" | "unified";
  config?: Record<string, JsonValue>;
  split?: { test_size: number; val_split: number };
  explainability?: JsonValue; // e.g. "none" | null | {...}
};

export type CreateRunRes = {
  run_id: string;
  status: string; // backend returns "queued"
};

export type ArtifactsListRes = {
  run_id: string;
  artifacts: string[];
};

// -----------------------------
// Low-level helpers
// -----------------------------
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(url, init);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `API ${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`
    );
  }

  return res;
}

function encodeArtifactPath(path: string): string {
  // Keep slashes but encode each segment
  return path
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
}

// -----------------------------
// API functions
// -----------------------------
export async function health(): Promise<{ ok: boolean; service: string }> {
  const res = await apiFetch(`${API_BASE}/health`);
  return (await res.json()) as { ok: boolean; service: string };
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await apiFetch(`${API_BASE}/datasets/upload`, {
    method: "POST",
    body: form,
  });

  return (await res.json()) as DatasetUploadResponse;
}

export async function getDataset(dataset_id: string): Promise<DatasetMeta> {
  const res = await apiFetch(
    `${API_BASE}/datasets/${encodeURIComponent(dataset_id)}`
  );
  return (await res.json()) as DatasetMeta;
}

export async function createRun(body: CreateRunReq): Promise<CreateRunRes> {
  const res = await apiFetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...body,
      config: body.config ?? {},
      split: body.split ?? { test_size: 0.2, val_split: 0.5 },
    }),
  });

  return (await res.json()) as CreateRunRes;
}

export async function getRun(run_id: string): Promise<RunStatus> {
  const res = await apiFetch(`${API_BASE}/runs/${encodeURIComponent(run_id)}`);
  return (await res.json()) as RunStatus;
}

export async function pollRunUntilDone(
  run_id: string,
  opts?: { intervalMs?: number; timeoutMs?: number }
): Promise<RunStatus> {
  const intervalMs = opts?.intervalMs ?? 1000;
  const timeoutMs = opts?.timeoutMs ?? 10 * 60 * 1000;

  const start = Date.now();

  while (true) {
    const status = await getRun(run_id);
    if (status.status === "succeeded" || status.status === "failed") return status;

    if (Date.now() - start > timeoutMs) {
      throw new Error(`Polling timed out after ${timeoutMs}ms for run ${run_id}`);
    }

    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function listArtifacts(run_id: string): Promise<ArtifactsListRes> {
  const res = await apiFetch(
    `${API_BASE}/runs/${encodeURIComponent(run_id)}/artifacts`
  );
  return (await res.json()) as ArtifactsListRes;
}

export function artifactUrl(run_id: string, artifact_path: string): string {
  const rid = encodeURIComponent(run_id);
  const ap = encodeArtifactPath(artifact_path);
  return `${API_BASE}/runs/${rid}/artifacts/${ap}`;
}

export async function fetchArtifactBlob(
  run_id: string,
  artifact_path: string
): Promise<Blob> {
  const res = await apiFetch(artifactUrl(run_id, artifact_path));
  return await res.blob();
}

export async function fetchArtifactJson<T extends JsonValue = JsonValue>(
  run_id: string,
  artifact_path: string
): Promise<T> {
  const res = await apiFetch(artifactUrl(run_id, artifact_path));
  return (await res.json()) as T;
}
