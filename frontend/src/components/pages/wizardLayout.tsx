import { useEffect, useMemo, useState } from "react";
import { useLocalStorage } from "../../lib/useLocalStorage";

import Sidebar from "../layout/Sidebar";

import Step1Upload from "../steps/Step1Upload";
import Step2Mapping, { type ManualMapping, type MappingMode } from "../steps/Step2Mapping";
import Step2Model from "../steps/Step2Model";
import Step3Prediction, { type PickableCategory } from "../steps/Step3Prediction";
import Step4Explainability, { type ExplainValue } from "../steps/Step4Explainability";
import Step5Config, { type ConfigMode } from "../steps/Step5Config";
import Step6Review from "../steps/Step6Review";
import ResultsView from "../results/ResultsView";

import StepProgressHeader from "../ui/StepProgressHeader";
import WizardFooter from "../ui/WizardFooter";
import ppmxLogo from "../../assets/ppmx.png";
import tumLogo from "../../assets/tum.png";

import {
  createRun,
  getRun,
  getRunLogs,
  listArtifacts,
  type DatasetUploadResponse,
  type ModelCapability,
  type RunStatus,
} from "../../lib/api";
import {
  defaultConfigFor,
  isExplainAllowed,
  useCapabilities,
  validateConfig,
  type ModelConfig,
} from "../../models/capabilities";

const TOTAL_STEPS = 8;

export type PipelineStatus = "idle" | "running" | "completed";
export type ViewMode = "wizard" | "results";

const ANSI_RE = /\x1b\[[0-9;]*m/g;

function cleanLogLine(line: string): string {
  return line.replace(ANSI_RE, "").replace(/\r/g, "").trim();
}

function isNoiseLine(line: string): boolean {
  if (!line) return true;
  if (line.includes("ms/step")) return true;
  if (line.includes("====")) return true;
  if (line.includes("ETA") || line.includes("eta")) return true;
  return false;
}

function extractEpochProgress(lines: string[]): { current: number; total: number } | null {
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = lines[i].match(/Epoch\s+(\d+)\s*\/\s*(\d+)/i);
    if (m) {
      const current = Number(m[1]);
      const total = Number(m[2]);
      if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
        return { current, total };
      }
    }
  }
  return null;
}

function estimateProgressFromLogs(lines: string[], status: RunStatus | null): number {
  if (!status) return 0;
  if (status.status === "queued") return 10;
  if (status.status === "failed") return 100;
  if (status.status === "succeeded") return 100;

  const epoch = extractEpochProgress(lines);
  let progress = 25;

  if (epoch) {
    const frac = Math.min(1, epoch.current / epoch.total);
    progress = 30 + frac * 50; // 30-80
  }

  const joined = lines.join("\n").toLowerCase();
  if (joined.includes("evaluating on test") || joined.includes("evaluating")) {
    progress = Math.max(progress, 85);
  }
  if (joined.includes("saving results") || joined.includes("results saved")) {
    progress = Math.max(progress, 92);
  }
  if (joined.includes("explainability")) {
    progress = Math.max(progress, 95);
  }

  return Math.min(99, Math.round(progress));
}

function validateManualMapping(m: ManualMapping): boolean {
  const requiredOk =
    m.case_id.trim().length > 0 && m.activity.trim().length > 0 && m.timestamp.trim().length > 0;
  if (!requiredOk) return false;

  const selected = [m.case_id, m.activity, m.timestamp, m.resource].filter(
    (v): v is string => !!v && v.trim().length > 0
  );
  return new Set(selected).size === selected.length;
}

/**
 * Merge stored custom overrides onto the model's field defaults, keeping only
 * keys the model actually declares. This keeps the config valid-shaped across
 * model switches and manifest changes without any per-model branching.
 */
function effectiveConfig(model: ModelCapability | undefined, overrides: ModelConfig): ModelConfig {
  if (!model) return {};
  const base = defaultConfigFor(model);
  const out: ModelConfig = { ...base };
  for (const f of model.config_fields) {
    if (f.key in overrides) out[f.key] = overrides[f.key];
  }
  return out;
}

export default function WizardLayout() {
  const { getModel, loading: capsLoading, error: capsError } = useCapabilities();

  const [step, setStep] = useLocalStorage("wizard_step", 0);

  /* -------------------- STEP DATA -------------------- */
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dataset, setDataset] = useLocalStorage<DatasetUploadResponse | null>("wizard_dataset", null);
  const [datasetMode, setDatasetMode] = useLocalStorage<"raw" | "preprocessed" | "skip" | null>("wizard_datasetMode", null);
  const [splitConfig, setSplitConfig] = useLocalStorage("wizard_splitConfig", { test_size: 0.1, val_split: 0.11 });

  const [mappingMode, setMappingMode] = useLocalStorage<MappingMode | null>("wizard_mappingMode", "manual");
  const [manualMapping, setManualMapping] = useLocalStorage<ManualMapping>("wizard_manualMapping", {
    case_id: "",
    activity: "",
    timestamp: "",
    resource: null,
  });

  const [modelType, setModelType] = useLocalStorage<string | null>("wizard_modelType", null);
  const [predictionTask, setPredictionTask] = useLocalStorage<string | null>("wizard_predictionTask", null);
  const [predictionCategory, setPredictionCategory] = useLocalStorage<PickableCategory | null>("wizard_predictionCategory", null);
  const [customTargetColumn, setCustomTargetColumn] = useLocalStorage<string | null>("wizard_customTargetColumn", null);
  const [explainMethod, setExplainMethod] = useLocalStorage<ExplainValue | null>("wizard_explainMethod", null);
  const [configMode, setConfigMode] = useLocalStorage<ConfigMode | null>("wizard_configMode", null);

  /* -------------------- CONFIG STATE -------------------- */
  // Single generic override map; defaults come from the selected model's manifest.
  const [configOverrides, setConfigOverrides] = useLocalStorage<ModelConfig>("wizard_config", {});

  /* -------------------- RUN STATE -------------------- */
  const [pipelineStatus, setPipelineStatus] = useLocalStorage<PipelineStatus>("wizard_pipelineStatus", "idle");
  const [progress, setProgress] = useLocalStorage("wizard_progress", 0);

  const [runId, setRunId] = useLocalStorage<string | null>("wizard_runId", null);
  const [runStatus, setRunStatus] = useLocalStorage<RunStatus | null>("wizard_runStatus", null);
  const [artifacts, setArtifacts] = useLocalStorage<string[]>("wizard_artifacts", []);
  const [runError, setRunError] = useLocalStorage<string | null>("wizard_runError", null);
  const [runLogs, setRunLogs] = useLocalStorage<string[]>("wizard_runLogs", []);

  const [viewMode, setViewMode] = useLocalStorage<ViewMode>("wizard_viewMode", "wizard");

  /* -------------------- DERIVED (from manifest) -------------------- */
  const selectedModel = getModel(modelType);
  const selectedTask = useMemo(
    () => selectedModel?.tasks.find((t) => t.id === predictionTask),
    [selectedModel, predictionTask]
  );
  const taskValid = !!selectedTask;

  const defaultConfig = useMemo<ModelConfig>(
    () => (selectedModel ? defaultConfigFor(selectedModel) : {}),
    [selectedModel]
  );
  const customConfig = useMemo<ModelConfig>(
    () => effectiveConfig(selectedModel, configOverrides),
    [selectedModel, configOverrides]
  );

  /* -------------------- NAVIGATION -------------------- */
  const nextStep = () => setStep((prev) => Math.min(prev + 1, TOTAL_STEPS - 1));
  const prevStep = () => setStep((prev) => Math.max(prev - 1, 0));

  /* -------------------- VALIDATION -------------------- */
  const isStepValid = () => {
    switch (step) {
      case 0:
        return dataset !== null && !!dataset.split_paths;
      case 1:
        if (!dataset) return false;
        if (mappingMode === null) return false;
        return validateManualMapping(manualMapping);
      case 2:
        return !!selectedModel;
      case 3: {
        if (!selectedModel) return false;
        if (configMode === null) return false;
        return configMode === "default" || validateConfig(selectedModel, customConfig);
      }
      case 4:
        if (!taskValid) return false;
        if (selectedTask?.needs_target_column) {
          if (!customTargetColumn) return false;
          if (!dataset?.column_types) return false;
          if (dataset.column_types[customTargetColumn] !== "categorical") return false;
        }
        return true;
      case 5:
        return explainMethod !== null;
      case 7:
        return true;
      default:
        return true;
    }
  };

  const completedSteps = [
    dataset !== null && !!dataset.split_paths,
    !!dataset && validateManualMapping(manualMapping),
    !!selectedModel,
    configMode !== null,
    taskValid,
    explainMethod !== null,
    pipelineStatus === "completed",
    viewMode === "results",
  ];

  /* -------------------- HANDLERS -------------------- */
  const handleUploaded = (file: File, resp: DatasetUploadResponse) => {
    setUploadedFile(file);
    setDataset(resp);
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    // clear run state
    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
    setRunLogs([]);
  };

  const handleDatasetUpdate = (resp: DatasetUploadResponse) => {
    setDataset(resp);
    setManualMapping((prev) => {
      const cols = resp.columns ?? [];
      const next = { ...prev };

      if (next.case_id && !cols.includes(next.case_id)) next.case_id = "";
      if (next.activity && !cols.includes(next.activity)) next.activity = "";
      if (next.timestamp && !cols.includes(next.timestamp)) next.timestamp = "";
      if (next.resource && !cols.includes(next.resource)) next.resource = null;

      return next;
    });
    if (customTargetColumn && !resp.columns.includes(customTargetColumn)) {
      setCustomTargetColumn(null);
      if (selectedTask?.needs_target_column) {
        setPredictionTask(null);
      }
    }
  };

  const clearUpload = () => {
    setUploadedFile(null);
    setDataset(null);
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
    setRunLogs([]);
  };

  // Selecting a model resets config to that model's defaults and drops any task /
  // explainability method the new model doesn't support — all manifest-driven.
  const handleSelectModelType = (id: string) => {
    setModelType(id);
    const nextModel = getModel(id);

    setConfigMode(null);
    setConfigOverrides(nextModel ? defaultConfigFor(nextModel) : {});

    if (predictionTask && !nextModel?.tasks.some((t) => t.id === predictionTask)) {
      setPredictionTask(null);
      setCustomTargetColumn(null);
    }
    // Drop the category if the new model has no tasks in it (e.g. switching to BEST).
    if (predictionCategory && !nextModel?.tasks.some((t) => t.category === predictionCategory)) {
      setPredictionCategory(null);
    }
    if (!isExplainAllowed(nextModel, explainMethod)) {
      setExplainMethod(null);
    }
  };

  const resetAll = () => {
    setStep(0);

    setUploadedFile(null);
    setDataset(null);
    setDatasetMode(null);
    setSplitConfig({ test_size: 0.1, val_split: 0.11 });
    setMappingMode("manual");
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setModelType(null);
    setPredictionTask(null);
    setPredictionCategory(null);
    setCustomTargetColumn(null);
    setExplainMethod(null);
    setConfigMode(null);
    setConfigOverrides({});

    setPipelineStatus("idle");
    setProgress(0);

    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);

    setViewMode("wizard");
  };

  /* -------------------- PIPELINE: create run -------------------- */
  const startPipeline = async () => {
    setRunError(null);
    setArtifacts([]);
    setRunStatus(null);
    setRunId(null);

    if (!dataset) {
      setRunError("No dataset available. Please upload a dataset first.");
      return;
    }
    if (!mappingMode) {
      setRunError("Please configure column mapping first.");
      return;
    }
    if (!selectedModel) {
      setRunError("Invalid model type. Please re-select Step 2.");
      return;
    }
    if (!taskValid || !predictionTask) {
      setRunError("Invalid prediction task. Please re-select Step 5.");
      return;
    }

    const explainToSend = isExplainAllowed(selectedModel, explainMethod) ? explainMethod : null;
    const configToSend = configMode === "custom" ? customConfig : defaultConfig;

    setPipelineStatus("running");
    setProgress(5);

    try {
      const res = await createRun({
        dataset_id: dataset.dataset_id,
        model_type: selectedModel.id,
        task: predictionTask,
        config: configToSend,
        split: splitConfig,
        explainability: explainToSend,
        target_column: selectedTask?.needs_target_column ? customTargetColumn : null,
        mapping_mode: "manual",
        column_mapping: manualMapping,
      });

      setRunId(res.run_id);

      const st = await getRun(res.run_id);
      setRunStatus(st);
      setProgress(st.status === "queued" ? 15 : 40);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRunError(msg);
      setPipelineStatus("idle");
      setProgress(0);
    }
  };

  /* -------------------- PIPELINE: poll status -------------------- */
  useEffect(() => {
    if (pipelineStatus !== "running") return;
    if (!runId) return;

    let cancelled = false;

    const tick = async () => {
      try {
        const st = await getRun(runId);
        if (cancelled) return;

        setRunStatus(st);
        let cleanedLogs: string[] = [];
        try {
          const logs = await getRunLogs(runId, 120);
          const rawLines = logs.lines ?? [];
          cleanedLogs = rawLines
            .map(cleanLogLine)
            .filter((line) => line.length > 0)
            .filter((line) => !isNoiseLine(line));
          if (!cancelled && cleanedLogs.length > 0) setRunLogs(cleanedLogs);
        } catch {
          // Keep last known logs if polling fails.
        }

        if (st.status === "succeeded" || st.status === "failed") {
          setProgress(100);
        } else {
          const nextProgress = estimateProgressFromLogs(cleanedLogs, st);
          setProgress((prev) => Math.max(prev, nextProgress));
        }

        if (st.status === "succeeded") {
          const arts = await listArtifacts(runId);
          if (cancelled) return;
          setArtifacts(arts.artifacts);
          setPipelineStatus("completed");
          setStep(7);
          setViewMode("results");
        }

        if (st.status === "failed") {
          setRunError(st.error ?? "Run failed. Check backend logs.txt for details.");
          setPipelineStatus("idle");
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setRunError(msg);
        setPipelineStatus("idle");
      }
    };

    const interval = window.setInterval(tick, 1000);
    tick();

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [pipelineStatus, runId]);

  /* -------------------- RENDER -------------------- */
  const showResults = viewMode === "results" || step === 7;

  return (
    <div className="flex h-screen flex-col">
      <div className="shrink-0 border-b border-brand-100 bg-white shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <img
            src={tumLogo}
            alt="TUM"
            className="h-8 w-auto object-contain"
          />
          <img
            src={ppmxLogo}
            alt="PPMX"
            className="h-12 w-auto object-contain"
          />
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <Sidebar currentStep={step} completedSteps={completedSteps} />

        {showResults ? (
          <ResultsView
            runId={runId}
            uploadedFileName={uploadedFile?.name}
            configMode={configMode}
            onStartOver={resetAll}
            onBackToPipeline={() => {
              setViewMode("wizard");
              setStep(6);
            }}
          />
        ) : (
          <div className="flex-1 flex flex-col min-w-0 bg-brand-50">
            <div className="px-8 pt-8 shrink-0">
              <StepProgressHeader step={step} totalSteps={TOTAL_STEPS} />
              {capsError && (
                <div className="mt-4 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                  Could not load model capabilities from the backend: {capsError}. Model,
                  task, config and explainability steps will be empty until it is reachable.
                </div>
              )}
            </div>

            <div className="flex-1 overflow-auto min-w-0">
              <div className="w-full px-8 py-6">
                {step === 0 && (
                  <Step1Upload
                    uploadedFile={uploadedFile}
                    dataset={dataset}
                    onUploaded={handleUploaded}
                    onDatasetUpdate={handleDatasetUpdate}
                    onClear={clearUpload}
                    mode={datasetMode}
                    onModeChange={setDatasetMode}
                    splitConfig={splitConfig}
                    onSplitConfigChange={setSplitConfig}
                  />
                )}

                {step === 1 && (
                  <Step2Mapping
                    dataset={dataset}
                    manualMapping={manualMapping}
                    onManualMappingChange={(patch) =>
                      setManualMapping((prev) => ({ ...prev, ...patch }))
                    }
                  />
                )}

                {step === 2 && (
                  <Step2Model modelType={modelType} onSelect={handleSelectModelType} />
                )}

                {step === 3 && (
                  <Step5Config
                    modelType={modelType}
                    mode={configMode}
                    onSelect={setConfigMode}
                    config={customConfig}
                    onConfigChange={setConfigOverrides}
                    defaultConfig={defaultConfig}
                  />
                )}

                {step === 4 && (
                  <Step3Prediction
                    modelType={modelType}
                    task={predictionTask}
                    category={predictionCategory}
                    targetColumn={customTargetColumn}
                    dataset={dataset}
                    onSelectTask={(nextTask) => {
                      setPredictionTask(nextTask);
                      const meta = selectedModel?.tasks.find((t) => t.id === nextTask);
                      if (meta && (meta.category === "classification" || meta.category === "regression")) {
                        setPredictionCategory(meta.category);
                      }
                      if (!meta?.needs_target_column) {
                        setCustomTargetColumn(null);
                      }
                    }}
                    onSelectCategory={(nextCategory) => {
                      setPredictionCategory(nextCategory);
                      setPredictionTask(null);
                      setCustomTargetColumn(null);
                    }}
                    onTargetColumnChange={setCustomTargetColumn}
                  />
                )}

                {step === 5 && (
                  <Step4Explainability
                    modelType={modelType}
                    method={explainMethod}
                    onSelect={setExplainMethod}
                  />
                )}

                {step === 6 && (
                  <Step6Review
                    uploadedFile={uploadedFile}
                    dataset={dataset}
                    modelType={modelType}
                    predictionTask={predictionTask}
                    explainMethod={explainMethod} // OK: ExplainValue is a string
                    mappingMode={mappingMode}
                    manualMapping={manualMapping}
                    configMode={configMode}
                    pipelineStatus={pipelineStatus}
                    progress={progress}
                    runId={runId}
                    runStatus={runStatus}
                    artifacts={artifacts}
                    logs={runLogs}
                    error={runError}
                    onStartPipeline={startPipeline}
                    onViewResults={() => {
                      setStep(7);
                      setViewMode("results");
                    }}
                  />
                )}
              </div>
            </div>

            <div className="shrink-0 px-8 pb-6 border-t border-brand-100 bg-white">
              <WizardFooter
                step={step}
                canContinue={pipelineStatus !== "running" && !capsLoading && isStepValid()}
                onCancel={resetAll}
                onPrevious={() => {
                  if (pipelineStatus === "running") return;
                  prevStep();
                }}
                onContinue={() => {
                  if (pipelineStatus === "running") return;
                  nextStep();
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
