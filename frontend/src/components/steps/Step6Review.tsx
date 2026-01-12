// frontend/src/components/steps/Step6Review.tsx
import type { DatasetUploadResponse, RunStatus } from "../../lib/api";
import { artifactUrl } from "../../lib/api";

function formatPredictionTask(task: string | null): string {
  if (!task) return "-";
  const s = task.toLowerCase().trim();
  if (s === "next_activity" || s.includes("next activity")) return "Next Activity Prediction";
  if (s === "event_time" || s.includes("event time") || s === "timestamp") return "Event Time Prediction";
  if (s === "remaining_time" || s.includes("remaining time")) return "Remaining Time Prediction";
  if (s === "unified") return "Unified Prediction";
  return task;
}

type Step6ReviewProps = {
  uploadedFile: File | null;
  dataset: DatasetUploadResponse | null;

  modelType: string | null;
  predictionTask: string | null;
  explainMethod: string | null;
  configMode: "default" | "custom" | null;

  pipelineStatus: "idle" | "running" | "completed";
  progress: number;

  runId: string | null;
  runStatus: RunStatus | null;
  artifacts: string[];

  error: string | null;

  onStartPipeline: () => void;
  onViewResults: () => void;
};

export default function Step6Review({
  uploadedFile,
  dataset,
  modelType,
  predictionTask,
  explainMethod,
  configMode,
  pipelineStatus,
  progress,
  runId,
  runStatus,
  artifacts,
  error,
  onStartPipeline,
  onViewResults,
}: Step6ReviewProps) {
  const datasetName = uploadedFile?.name ?? "-";
  const datasetId = dataset?.dataset_id ?? "-";
  const events = dataset?.num_events ?? null;
  const cases = dataset?.num_cases ?? null;

  const configLabel =
    configMode === "default"
      ? "Default Configuration"
      : configMode === "custom"
      ? "Custom Configuration"
      : "-";

  const backendStatus = runStatus?.status ?? (pipelineStatus === "idle" ? "-" : "queued");

  return (
    <div className="space-y-6 w-full">
      {/* SUMMARY */}
      <div className="border rounded-xl p-6 bg-white">
        <h2 className="text-lg font-semibold mb-4">Review & Run Pipeline</h2>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <SummaryItem label="Dataset" value={datasetName} />
          <SummaryItem label="Dataset ID" value={datasetId} />
          <SummaryItem
            label="Events / Cases"
            value={
              events !== null && cases !== null
                ? `${events.toLocaleString()} events, ${cases.toLocaleString()} cases`
                : "-"
            }
          />
          <SummaryItem label="Model Type" value={modelType ?? "-"} />
          <SummaryItem label="Prediction Task" value={formatPredictionTask(predictionTask)} />
          <SummaryItem label="Explainability" value={explainMethod ?? "-"} />
          <SummaryItem label="Configuration" value={configLabel} />
          <SummaryItem label="Run ID" value={runId ?? "-"} />
          <SummaryItem label="Run Status" value={backendStatus} />
        </div>

        {error && (
          <div className="mt-4 border border-red-200 bg-red-50 rounded-lg p-4 text-sm text-red-700">
            <div className="font-medium mb-1">Run failed</div>
            <div className="break-words">{error}</div>
          </div>
        )}
      </div>

      {/* READY */}
      {pipelineStatus === "idle" && (
        <div className="border rounded-xl p-6 bg-blue-50">
          <p className="text-sm text-gray-700 mb-4">
            Your pipeline is configured and ready to run.
          </p>

          <button
            className="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            onClick={onStartPipeline}
            disabled={!dataset}
            title={!dataset ? "Upload a dataset first" : undefined}
          >
            Start Pipeline Execution
          </button>
        </div>
      )}

      {/* RUNNING */}
      {pipelineStatus === "running" && (
        <div className="border rounded-xl p-6 bg-white">
          <div className="mb-3 text-sm font-medium">Pipeline Execution in Progress</div>

          <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="text-sm text-gray-600">
            {progress}% • backend status: <span className="font-medium">{backendStatus}</span>
          </div>
        </div>
      )}

      {/* COMPLETED */}
      {pipelineStatus === "completed" && (
        <div className="border rounded-xl p-6 bg-green-50">
          <div className="text-green-700 font-medium mb-3">
            Pipeline execution completed successfully.
          </div>

          {runId && artifacts.length > 0 && (
            <div className="mb-4">
              <div className="text-sm font-medium text-gray-800 mb-2">Artifacts</div>
              <ul className="text-sm list-disc ml-5 space-y-1">
                {artifacts.map((a) => (
                  <li key={a}>
                    <a
                      className="text-blue-700 hover:underline"
                      href={artifactUrl(runId, a)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {a}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            className="px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            onClick={onViewResults}
          >
            View Results
          </button>
        </div>
      )}
    </div>
  );
}

/* ----------------- Helper ----------------- */

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-gray-500 text-xs mb-1">{label}</div>
      <div className="font-medium text-gray-900 break-words">{value}</div>
    </div>
  );
}
