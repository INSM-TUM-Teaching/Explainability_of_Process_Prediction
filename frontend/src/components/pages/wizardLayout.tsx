// frontend/src/components/pages/wizardLayout.tsx
import { useEffect, useMemo, useState } from "react";

import Sidebar from "../layout/Sidebar";

import Step1Upload from "../steps/Step1Upload";
import Step2Mapping, { type ManualMapping, type MappingMode } from "../steps/Step2Mapping";
import Step2Model from "../steps/Step2Model";
import Step3Prediction from "../steps/Step3Prediction";
import Step4Explainability, { type ExplainValue } from "../steps/Step4Explainability";
import Step5Config, {
  type ConfigMode,
  type GnnConfig,
  type TransformerConfig,
} from "../steps/Step5Config";
import Step6Review from "../steps/Step6Review";
import ResultsView from "../results/ResultsView";

import StepProgressHeader from "../ui/StepProgressHeader";
import WizardFooter from "../ui/WizardFooter";

import {
  createRun,
  getRun,
  listArtifacts,
  type DatasetUploadResponse,
  type RunStatus,
} from "../../lib/api";

const TOTAL_STEPS = 7;

export type PipelineStatus = "idle" | "running" | "completed";
export type ViewMode = "wizard" | "results";

function normalizeModelType(v: string | null): "gnn" | "transformer" | null {
  if (!v) return null;
  const s = v.toLowerCase().trim();
  if (s === "gnn" || s.includes("gnn")) return "gnn";
  if (s === "transformer" || s.includes("transformer")) return "transformer";
  return null;
}

function normalizeTask(
  v: string | null
): "next_activity" | "event_time" | "remaining_time" | "unified" | null {
  if (!v) return null;
  const s = v.toLowerCase().trim();

  if (s === "next_activity" || s.includes("next activity")) return "next_activity";
  if (s === "event_time" || s.includes("event time") || s === "timestamp") return "event_time";
  if (s === "remaining_time" || s.includes("remaining time")) return "remaining_time";
  if (s === "unified") return "unified";

  return null;
}

function isExplainAllowed(
  explain: ExplainValue | null,
  model: "gnn" | "transformer" | null
): boolean {
  if (!explain) return true;
  if (!model) return true;

  if (model === "transformer") {
    return explain === "none" || explain === "lime" || explain === "shap" || explain === "all";
  }
  return explain === "none" || explain === "gradient" || explain === "lime" || explain === "all";
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

function validateTransformerConfig(cfg: TransformerConfig): boolean {
  const positiveInts = [
    cfg.max_len,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_blocks,
    cfg.epochs,
    cfg.batch_size,
    cfg.patience,
  ].every((v) => Number.isInteger(v) && v > 0);

  const dropoutOk =
    typeof cfg.dropout_rate === "number" && cfg.dropout_rate > 0 && cfg.dropout_rate < 1;

  return positiveInts && dropoutOk;
}

function validateGnnConfig(cfg: GnnConfig): boolean {
  const positiveInts = [cfg.hidden, cfg.epochs, cfg.batch_size, cfg.patience].every(
    (v) => Number.isInteger(v) && v > 0
  );

  const dropoutOk =
    typeof cfg.dropout_rate === "number" && cfg.dropout_rate > 0 && cfg.dropout_rate < 1;

  const lrOk = typeof cfg.lr === "number" && cfg.lr > 0;

  return positiveInts && dropoutOk && lrOk;
}

export default function WizardLayout() {
  const [step, setStep] = useState(0);

  /* -------------------- STEP DATA -------------------- */
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);

  const [mappingMode, setMappingMode] = useState<MappingMode | null>(null);
  const [manualMapping, setManualMapping] = useState<ManualMapping>({
    case_id: "",
    activity: "",
    timestamp: "",
    resource: null,
  });

  const [modelType, setModelType] = useState<string | null>(null);
  const [predictionTask, setPredictionTask] = useState<string | null>(null);
  const [explainMethod, setExplainMethod] = useState<ExplainValue | null>(null);
  const [configMode, setConfigMode] = useState<ConfigMode | null>(null);

  /* -------------------- CONFIG STATE -------------------- */
  const defaultTransformerConfig = useMemo<TransformerConfig>(
    () => ({
      max_len: 16,
      d_model: 64,
      num_heads: 4,
      num_blocks: 2,
      dropout_rate: 0.1,
      epochs: 5,
      batch_size: 128,
      patience: 10,
    }),
    []
  );

  const defaultGnnConfig = useMemo<GnnConfig>(
    () => ({
      hidden: 64,
      dropout_rate: 0.1,
      lr: 4e-4,
      epochs: 5,
      batch_size: 64,
      patience: 10,
    }),
    []
  );

  const [transformerConfig, setTransformerConfig] =
    useState<TransformerConfig>(defaultTransformerConfig);
  const [gnnConfig, setGnnConfig] = useState<GnnConfig>(defaultGnnConfig);

  /* -------------------- RUN STATE -------------------- */
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>("idle");
  const [progress, setProgress] = useState(0);

  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [runError, setRunError] = useState<string | null>(null);

  const [viewMode, setViewMode] = useState<ViewMode>("wizard");

  /* -------------------- DERIVED -------------------- */
  const modelTypeNormalized = useMemo(
    () => normalizeModelType(modelType),
    [modelType]
  );

  const taskNormalized = useMemo(
    () => normalizeTask(predictionTask),
    [predictionTask]
  );

  /* -------------------- NAVIGATION -------------------- */
  const nextStep = () => setStep((prev) => Math.min(prev + 1, TOTAL_STEPS - 1));
  const prevStep = () => setStep((prev) => Math.max(prev - 1, 0));

  /* -------------------- VALIDATION -------------------- */
  const isStepValid = () => {
    switch (step) {
      case 0:
        return dataset !== null;
      case 1:
        if (!dataset) return false;
        if (mappingMode === null) return false;
        if (mappingMode === "auto") return true;
        return validateManualMapping(manualMapping);
      case 2:
        return modelTypeNormalized !== null;
      case 3: {
        if (!modelTypeNormalized) return false;
        if (configMode === null) return false;

        if (modelTypeNormalized === "transformer") {
          return configMode === "default" || validateTransformerConfig(transformerConfig);
        }
        if (modelTypeNormalized === "gnn") {
          return configMode === "default" || validateGnnConfig(gnnConfig);
        }
        return false;
      }
      case 4:
        return taskNormalized !== null;
      case 5:
        return explainMethod !== null;
      default:
        return true;
    }
  };

  const completedSteps = [
    dataset !== null,
    mappingMode !== null,
    modelTypeNormalized !== null,
    configMode !== null,
    taskNormalized !== null,
    explainMethod !== null,
    pipelineStatus === "completed",
  ];

  /* -------------------- HANDLERS -------------------- */
  const handleUploaded = (file: File, resp: DatasetUploadResponse) => {
    setUploadedFile(file);
    setDataset(resp);
    setMappingMode(null);
    setManualMapping({
      case_id: resp.columns.includes("CaseID") ? "CaseID" : "",
      activity: resp.columns.includes("Activity") ? "Activity" : "",
      timestamp: resp.columns.includes("Timestamp") ? "Timestamp" : "",
      resource: resp.columns.includes("Resource") ? "Resource" : null,
    });

    // clear run state
    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
  };

  const clearUpload = () => {
    setUploadedFile(null);
    setDataset(null);
    setMappingMode(null);
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setPipelineStatus("idle");
    setProgress(0);
    setRunId(null);
    setRunStatus(null);
    setArtifacts([]);
    setRunError(null);
  };

  // Clear explainability immediately when user changes model type (no effects)
  const handleSelectModelType = (v: string) => {
    const nextModel = normalizeModelType(v);
    setModelType(v);

    // Reset Step 5 config when model changes
    setConfigMode(null);
    setTransformerConfig(defaultTransformerConfig);
    setGnnConfig(defaultGnnConfig);

    if (!isExplainAllowed(explainMethod, nextModel)) {
      setExplainMethod(null);
    }
  };

  const resetAll = () => {
    setStep(0);

    setUploadedFile(null);
    setDataset(null);
    setMappingMode(null);
    setManualMapping({ case_id: "", activity: "", timestamp: "", resource: null });

    setModelType(null);
    setPredictionTask(null);
    setExplainMethod(null);
    setConfigMode(null);
    setTransformerConfig(defaultTransformerConfig);
    setGnnConfig(defaultGnnConfig);

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

    const mt = modelTypeNormalized;
    const task = taskNormalized;

    if (!mt) {
      setRunError("Invalid model type. Please re-select Step 3.");
      return;
    }
    if (!task) {
      setRunError("Invalid prediction task. Please re-select Step 5.");
      return;
    }

    const explainToSend = isExplainAllowed(explainMethod, mt) ? explainMethod : null;
    const configToSend =
      mt === "transformer"
        ? configMode === "custom"
          ? transformerConfig
          : defaultTransformerConfig
        : configMode === "custom"
        ? gnnConfig
        : defaultGnnConfig;

    setPipelineStatus("running");
    setProgress(5);

    try {
      const res = await createRun({
        dataset_id: dataset.dataset_id,
        model_type: mt,
        task,
        config: configToSend,
        split: { test_size: 0.2, val_split: 0.5 },
        explainability: explainToSend,
        mapping_mode: mappingMode,
        column_mapping: mappingMode === "manual" ? manualMapping : null,
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

        if (st.status === "queued") setProgress(20);
        if (st.status === "running") setProgress((p) => Math.max(p, 60));
        if (st.status === "succeeded") setProgress(100);
        if (st.status === "failed") setProgress(100);

        if (st.status === "succeeded") {
          const arts = await listArtifacts(runId);
          if (cancelled) return;
          setArtifacts(arts.artifacts);
          setPipelineStatus("completed");
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
  const showResults = viewMode === "results";

  return (
    <div className="flex h-screen">
      <Sidebar currentStep={step} completedSteps={completedSteps} />

      {showResults ? (
        <ResultsView
          runId={runId}
          onBackToPipeline={() => {
            setViewMode("wizard");
            setStep(6);
          }}
        />
      ) : (
        <div className="flex-1 flex flex-col min-w-0 bg-gray-50">
          <div className="px-8 pt-6 shrink-0">
            <StepProgressHeader step={step} totalSteps={TOTAL_STEPS} />
          </div>

          <div className="flex-1 overflow-auto min-w-0">
            <div className="w-full px-8 py-6">
              {step === 0 && (
                <Step1Upload
                  uploadedFile={uploadedFile}
                  dataset={dataset}
                  onUploaded={handleUploaded}
                  onClear={clearUpload}
                />
              )}

              {step === 1 && (
                <Step2Mapping
                  dataset={dataset}
                  mode={mappingMode}
                  manualMapping={manualMapping}
                  onModeChange={setMappingMode}
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
                  modelType={modelTypeNormalized}
                  mode={configMode}
                  onSelect={setConfigMode}
                  transformerConfig={transformerConfig}
                  onTransformerChange={setTransformerConfig}
                  defaultTransformerConfig={defaultTransformerConfig}
                  gnnConfig={gnnConfig}
                  onGnnChange={setGnnConfig}
                  defaultGnnConfig={defaultGnnConfig}
                />
              )}

              {step === 4 && (
                <Step3Prediction task={predictionTask} onSelect={setPredictionTask} />
              )}

              {step === 5 && (
                <Step4Explainability
                  modelType={modelTypeNormalized}
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
                  explainMethod={explainMethod} // OK: ExplainValue is a string union
                  mappingMode={mappingMode}
                  manualMapping={manualMapping}
                  configMode={configMode}
                  pipelineStatus={pipelineStatus}
                  progress={progress}
                  runId={runId}
                  runStatus={runStatus}
                  artifacts={artifacts}
                  error={runError}
                  onStartPipeline={startPipeline}
                  onViewResults={() => setViewMode("results")}
                />
              )}
            </div>
          </div>

          <div className="shrink-0 px-8 pb-6 border-t bg-white">
            <WizardFooter
              step={step}
              canContinue={pipelineStatus !== "running" && isStepValid()}
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
  );
}