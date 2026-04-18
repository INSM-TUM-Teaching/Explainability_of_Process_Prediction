import { useEffect, useState } from "react";

import {
  artifactUrl,
  artifactsZipUrl,
  getRun,
  listArtifacts,
  type RunStatus,
} from "../../lib/api";

type ResultsViewProps = {
  runId: string | null;
  onBackToPipeline: () => void;
};

function formatStatus(status: RunStatus["status"] | null): string {
  if (!status) return "-";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function formatDate(value?: string): string {
  if (!value) return "-";
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return value;
  return new Date(parsed).toLocaleString();
}

function formatArtifactName(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export default function ResultsView({
  runId,
  onBackToPipeline,
}: ResultsViewProps) {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      setStatus(null);
      setArtifacts([]);
      setError(null);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);

      try {
        const [nextStatus, nextArtifacts] = await Promise.all([
          getRun(runId),
          listArtifacts(runId),
        ]);
        if (cancelled) return;
        setStatus(nextStatus);
        setArtifacts(nextArtifacts.artifacts);
      } catch (e) {
        if (cancelled) return;
        const message = e instanceof Error ? e.message : String(e);
        setError(message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();

    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (!runId) {
    return (
      <div className="flex-1 min-w-0 bg-brand-50">
        <div className="mx-auto w-full max-w-5xl px-8 py-10">
          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-6">
            <h2 className="text-xl font-semibold text-amber-900">No completed run selected</h2>
            <p className="mt-2 text-sm text-amber-800">
              Start a pipeline run first, then open the results view.
            </p>
            <button
              type="button"
              onClick={onBackToPipeline}
              className="mt-4 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
            >
              Back to pipeline
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 min-w-0 overflow-auto bg-brand-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-8 py-8">
        <div className="flex flex-col gap-4 rounded-2xl border border-brand-100 bg-white p-6 shadow-sm md:flex-row md:items-start md:justify-between">
          <div>
            <div className="text-sm font-medium uppercase tracking-[0.18em] text-brand-500">
              Results
            </div>
            <h1 className="mt-2 text-2xl font-semibold text-brand-900">
              Pipeline run {runId}
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-600">
              Review the backend run metadata and download generated artifacts from this page.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onBackToPipeline}
              className="rounded-lg border border-brand-200 bg-white px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
            >
              Back to pipeline
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="rounded-lg border border-brand-200 bg-white px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
            >
              Refresh page
            </button>
            <a
              href={artifactsZipUrl(runId)}
              className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
            >
              Download all artifacts
            </a>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          <SummaryCard label="Run status" value={formatStatus(status?.status ?? null)} />
          <SummaryCard label="Artifacts" value={String(artifacts.length)} />
          <SummaryCard label="Started" value={formatDate(status?.started_at ?? status?.created_at)} />
          <SummaryCard label="Finished" value={formatDate(status?.finished_at)} />
        </div>

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="rounded-2xl border border-brand-100 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-brand-900">Artifacts</h2>
              <p className="mt-1 text-sm text-slate-600">
                Files exposed by the backend for this run.
              </p>
            </div>
            {loading && <span className="text-sm text-slate-500">Loading...</span>}
          </div>

          {artifacts.length === 0 ? (
            <div className="mt-4 rounded-xl border border-dashed border-brand-200 bg-brand-50 p-5 text-sm text-slate-600">
              No artifacts were returned for this run.
            </div>
          ) : (
            <div className="mt-4 overflow-hidden rounded-xl border border-brand-100">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] bg-brand-50 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-brand-700">
                <div>Artifact</div>
                <div>Action</div>
              </div>

              <ul className="divide-y divide-brand-100">
                {artifacts.map((path) => (
                  <li
                    key={path}
                    className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-4 py-3"
                  >
                    <div className="min-w-0">
                      <div className="truncate font-medium text-brand-900">
                        {formatArtifactName(path)}
                      </div>
                      <div className="truncate text-xs text-slate-500">{path}</div>
                    </div>

                    <a
                      href={artifactUrl(runId, path)}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg border border-brand-200 bg-white px-3 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50"
                    >
                      Open
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-brand-100 bg-white p-5 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-brand-500">{label}</div>
      <div className="mt-2 text-base font-semibold text-brand-900 break-words">{value}</div>
    </div>
  );
}
