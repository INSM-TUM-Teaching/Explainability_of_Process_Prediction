import { useEffect, useMemo, useState } from "react";

import Card from "../ui/card";
import BestPatternsPanel from "./BestPatternsPanel";
import { Button } from "../ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import NextActivityResults from "./NextActivityResults";

import {
  artifactUrl,
  artifactsZipUrl,
  fetchArtifactJson,
  listArtifacts,
  type JsonValue,
} from "../../lib/api";

type ResultsViewProps = {
  runId: string | null;
  uploadedFileName?: string;
  configMode?: string | null;
  onStartOver?: () => void;
  onBackToPipeline: () => void;
};

type MetricsFile = {
  run_id?: string;
  model_type?: string;
  task?: string;
  metrics?: JsonValue;
  finished_at?: string;
};

type SummaryFile = {
  status?: string;
  dataset?: { filename?: string; num_events?: number; num_cases?: number };
  metrics?: JsonValue;
  error?: string;
  request?: { task?: string; [key: string]: any };
};

function isImagePath(path: string): boolean {
  return /\.(png|jpe?g|gif|webp|svg)$/i.test(path);
}

function isJsonPath(path: string): boolean {
  return path.toLowerCase().endsWith(".json");
}

function plotGroupLabel(path: string): string {
  const norm = path.replace(/\\/g, "/");
  const file = norm.split("/").pop() ?? "";
  const dir = norm.includes("/") ? norm.split("/").slice(0, -1).join("/") : "plots";
  if (dir === "explainability" || dir.startsWith("explainability/")) {
    if (/top_matched|activity_importance|error_patterns|rpif|pattern_confidence|accuracy_by_prefix/.test(file)) {
      return "BEST: pattern explainability";
    }
    return "BEST: explainability";
  }
  if (dir === "benchmark" || dir.startsWith("benchmark/")) return "Benchmarking";
  return dir === "plots" ? "Training & predictions" : dir;
}

function groupPlotPaths(paths: string[]): Map<string, string[]> {
  const plots = paths.filter(isImagePath).sort();
  const groups = new Map<string, string[]>();
  for (const p of plots) {
    const label = plotGroupLabel(p);
    const list = groups.get(label) ?? [];
    list.push(p);
    groups.set(label, list);
  }
  return groups;
}

function formatMetricValue(value: JsonValue): string {
  if (value === null) return "-";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(6).replace(/\.?0+$/, "");
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function MetricsTable({ data }: { data: JsonValue }) {
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    return (
      <pre className="text-xs bg-brand-50 border border-brand-100 rounded-lg p-4 overflow-auto max-h-96">
        {formatMetricValue(data)}
      </pre>
    );
  }

  const entries = Object.entries(data as Record<string, JsonValue>);
  if (entries.length === 0) {
    return <p className="text-sm text-brand-600">No metrics recorded.</p>;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {entries.map(([key, value]) => (
        <div
          key={key}
          className="rounded-lg border border-brand-100 bg-brand-50/50 px-4 py-3"
        >
          <div className="text-xs text-brand-600 mb-1">{key}</div>
          <div className="text-sm font-medium text-brand-900 break-words whitespace-pre-wrap">
            {formatMetricValue(value)}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ResultsView({
  runId,
  uploadedFileName,
  configMode,
  onStartOver,
  onBackToPipeline,
}: ResultsViewProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [metricsFile, setMetricsFile] = useState<MetricsFile | null>(null);
  const [summaryFile, setSummaryFile] = useState<SummaryFile | null>(null);

  useEffect(() => {
    if (!runId) {
      setLoading(false);
      setArtifacts([]);
      setMetricsFile(null);
      setSummaryFile(null);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);

      try {
        const arts = await listArtifacts(runId);
        if (cancelled) return;
        setArtifacts(arts.artifacts);

        const paths = arts.artifacts;
        if (paths.includes("metrics.json")) {
          try {
            const m = await fetchArtifactJson<MetricsFile>(runId, "metrics.json");
            if (!cancelled) setMetricsFile(m);
          } catch {
            if (!cancelled) setMetricsFile(null);
          }
        } else {
          setMetricsFile(null);
        }

        if (paths.includes("summary.json")) {
          try {
            const s = await fetchArtifactJson<SummaryFile>(runId, "summary.json");
            if (!cancelled) setSummaryFile(s);
          } catch {
            if (!cancelled) setSummaryFile(null);
          }
        } else {
          setSummaryFile(null);
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [runId]);




  const plotGroups = useMemo(() => groupPlotPaths(artifacts), [artifacts]);

  const jsonFiles = useMemo(
    () => artifacts.filter(isJsonPath).filter((p) => p !== "metrics.json" && p !== "summary.json"),
    [artifacts]
  );

  const otherFiles = useMemo(
    () =>
      artifacts.filter(
        (p) => !isImagePath(p) && !isJsonPath(p)
      ),
    [artifacts]
  );

  const metricsToShow: JsonValue | null =
    metricsFile?.metrics ??
    summaryFile?.metrics ??
    null;

  const isBestRun =
    (metricsFile?.model_type ?? "").toLowerCase() === "best" ||
    artifacts.some((p) => p.replace(/\\/g, "/").includes("explainability/top_patterns_summary"));

  if (!runId) {
    return (
      <div className="flex-1 flex flex-col min-w-0 bg-brand-50 p-8">
        <Card title="No run selected">
          <p className="text-sm text-brand-600 mb-4">
            Complete a pipeline run from Review &amp; Run, then open results again.
          </p>
          <Button variant="outline" onClick={onBackToPipeline}>
            Back to pipeline
          </Button>
        </Card>
      </div>
    );
  }

  if (summaryFile?.request?.task === "next_activity" || summaryFile?.request?.task === "custom_activity") {
    return (
      <div className="flex-1 flex flex-col min-w-0 bg-brand-50">
        <div className="flex-1 overflow-auto min-w-0">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-8 py-8">
            <NextActivityResults 
              runId={runId} 
              summary={summaryFile} 
              uploadedFileName={uploadedFileName}
              configMode={configMode}
            />
          </div>
        </div>
        
        <div className="shrink-0 px-8 pb-6 border-t border-brand-100 bg-white">
          <div className="flex items-center justify-between pt-6">
            <button
              onClick={onStartOver}
              className="px-6 py-2 rounded-md border border-brand-200 bg-white text-brand-700 hover:bg-brand-50 hover:border-brand-300 transition"
            >
              Start over
            </button>
            <div className="flex items-center gap-3">
              <button
                onClick={onBackToPipeline}
                className="px-6 py-2 rounded-md border border-brand-200 text-brand-700 bg-white hover:bg-brand-50 hover:border-brand-300 transition"
              >
                Previous
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-brand-50">
      <div className="px-8 pt-8 pb-4 shrink-0 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-brand-900">Run results</h2>
          <p className="text-sm text-brand-600 mt-1">
            Run ID: <span className="font-mono">{runId}</span>
            {summaryFile?.dataset?.filename && (
              <> · {summaryFile.dataset.filename}</>
            )}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={onBackToPipeline}>
            Back to pipeline
          </Button>
          <Button asChild>
            <a href={artifactsZipUrl(runId)} download={`run_${runId}_artifacts.zip`}>
              Download ZIP
            </a>
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-auto min-w-0 px-8 pb-8">
        {loading && (
          <Card>
            <p className="text-sm text-brand-600">Loading artifacts...</p>
          </Card>
        )}

        {error && (
          <Card title="Could not load results">
            <p className="text-sm text-red-600 mb-4">{error}</p>
            <p className="text-sm text-brand-600">
              Ensure the backend is running on port 8000 and this run completed successfully.
            </p>
          </Card>
        )}

        {!loading && !error && (
          <Tabs defaultValue="plots" className="space-y-6">
            <TabsList className="bg-white border border-brand-100">
              <TabsTrigger value="plots">
                Plots{plotGroups.size > 0 ? ` (${artifacts.filter(isImagePath).length})` : ""}
              </TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
              <TabsTrigger value="files">All files</TabsTrigger>
            </TabsList>

            <TabsContent value="plots" className="space-y-6 mt-0">
              {isBestRun && runId && (
                <BestPatternsPanel runId={runId} artifactPaths={artifacts} />
              )}
              {plotGroups.size === 0 ? (
                <Card title="No plots yet">
                  <p className="text-sm text-brand-600">
                    Image artifacts appear here after training and explainability finish. Check
                    All files for CSVs and JSON outputs.
                  </p>
                </Card>
              ) : (
                Array.from(plotGroups.entries()).map(([group, paths]) => (
                  <Card key={group} title={group}>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {paths.map((path) => (
                        <figure key={path} className="space-y-2">
                          <figcaption className="text-xs text-brand-600 break-all">
                            {path}
                          </figcaption>
                          <a
                            href={artifactUrl(runId, path)}
                            target="_blank"
                            rel="noreferrer"
                            className="block rounded-lg border border-brand-100 overflow-hidden bg-white"
                          >
                            <img
                              src={artifactUrl(runId, path)}
                              alt={path}
                              className="w-full h-auto object-contain max-h-[480px]"
                              loading="lazy"
                            />
                          </a>
                        </figure>
                      ))}
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="metrics" className="space-y-6 mt-0">
              <Card title="Model performance">
                {(metricsFile?.model_type || metricsFile?.task) && (
                  <div className="flex flex-wrap gap-4 text-sm mb-4">
                    {metricsFile?.model_type && (
                      <span>
                        <span className="text-brand-600">Model: </span>
                        <span className="font-medium">{metricsFile.model_type}</span>
                      </span>
                    )}
                    {metricsFile?.task && (
                      <span>
                        <span className="text-brand-600">Task: </span>
                        <span className="font-medium">{metricsFile.task}</span>
                      </span>
                    )}
                    {metricsFile?.finished_at && (
                      <span>
                        <span className="text-brand-600">Finished: </span>
                        <span className="font-medium">{metricsFile.finished_at}</span>
                      </span>
                    )}
                  </div>
                )}
                {metricsToShow ? (
                  <MetricsTable data={metricsToShow} />
                ) : (
                  <p className="text-sm text-brand-600">
                    No metrics.json found for this run. Metrics may still be writing or the run
                    failed before completion.
                  </p>
                )}
              </Card>

              {summaryFile?.dataset && (
                <Card title="Dataset">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-brand-600 text-xs">Events</div>
                      <div className="font-medium">{summaryFile.dataset.num_events ?? "-"}</div>
                    </div>
                    <div>
                      <div className="text-brand-600 text-xs">Cases</div>
                      <div className="font-medium">{summaryFile.dataset.num_cases ?? "-"}</div>
                    </div>
                  </div>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="files" className="space-y-6 mt-0">
              <Card title="Artifact files">
                {artifacts.length === 0 ? (
                  <p className="text-sm text-brand-600">No artifacts listed for this run.</p>
                ) : (
                  <ul className="text-sm space-y-2">
                    {artifacts.map((path) => (
                      <li key={path}>
                        <a
                          className="text-brand-700 hover:underline break-all"
                          href={artifactUrl(runId, path)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {path}
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              {jsonFiles.length > 0 && (
                <Card title="Additional JSON">
                  <ul className="text-sm space-y-2">
                    {jsonFiles.map((path) => (
                      <li key={path}>
                        <a
                          className="text-brand-700 hover:underline break-all"
                          href={artifactUrl(runId, path)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {path}
                        </a>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              {otherFiles.length > 0 && (
                <Card title="Other downloads">
                  <ul className="text-sm space-y-2">
                    {otherFiles.map((path) => (
                      <li key={path}>
                        <a
                          className="text-brand-700 hover:underline break-all"
                          href={artifactUrl(runId, path)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {path}
                        </a>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  );
}
