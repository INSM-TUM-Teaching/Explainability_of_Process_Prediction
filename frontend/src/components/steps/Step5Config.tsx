import React from "react";
import { Settings } from "lucide-react";

export type ConfigMode = "default" | "custom";

// Must match backend `default_transformer_config()` keys/types in `ppm_pipeline.py`.
export type TransformerConfig = {
  max_len: number;
  d_model: number;
  num_heads: number;
  num_blocks: number;
  dropout_rate: number;
  epochs: number;
  batch_size: number;
  patience: number;
};

// Must match backend `default_gnn_config()` keys/types in `ppm_pipeline.py`.
export type GnnConfig = {
  hidden: number;
  dropout_rate: number;
  lr: number;
  epochs: number;
  batch_size: number;
  patience: number;
};

// Must match backend `default_best_config()` keys/types in `ppm_pipeline.py`.
export type BestConfig = {
  max_pattern_size_train: number;
  max_pattern_size_eval: number;
  process_stage_width_percentage: number;
  min_freq: number;
  break_buffer: number;
  filter_sequences: boolean;
  ncores: number;
};

type Step5ConfigProps = {
  modelType: "gnn" | "transformer" | "best" | null;
  mode: ConfigMode | null;
  onSelect: (mode: ConfigMode) => void;

  transformerConfig: TransformerConfig;
  onTransformerChange: (cfg: TransformerConfig) => void;
  defaultTransformerConfig: TransformerConfig;

  gnnConfig: GnnConfig;
  onGnnChange: (cfg: GnnConfig) => void;
  defaultGnnConfig: GnnConfig;

  bestConfig: BestConfig;
  onBestChange: (cfg: BestConfig) => void;
  defaultBestConfig: BestConfig;
};

export default function Step5Config({
  modelType,
  mode,
  onSelect,
  transformerConfig,
  onTransformerChange,
  defaultTransformerConfig,
  gnnConfig,
  onGnnChange,
  defaultGnnConfig,
  bestConfig,
  onBestChange,
  defaultBestConfig,
}: Step5ConfigProps) {
  if (!modelType) {
    return (
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">Select a model type in Step 2 first.</p>
      </div>
    );
  }

return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">
          {modelType === "transformer"
            ? "These options map 1:1 to the backend Transformer config."
            : modelType === "best"
            ? "These options map 1:1 to the backend BEST config."
            : "These options map 1:1 to the backend GNN config."}
        </p>
      </div>

      <ConfigCard
        title="Default Configuration"
        description="Use backend defaults."
        selected={mode === "default"}
        onClick={() => onSelect("default")}
      >
        <ParameterGrid
          modelType={modelType}
          editable={false}
          transformerConfig={transformerConfig}
          onTransformerChange={onTransformerChange}
          defaultTransformerConfig={defaultTransformerConfig}
          gnnConfig={gnnConfig}
          onGnnChange={onGnnChange}
          defaultGnnConfig={defaultGnnConfig}
          bestConfig={bestConfig}
          onBestChange={onBestChange}
          defaultBestConfig={defaultBestConfig}
        />
      </ConfigCard>

      <ConfigCard
        title="Custom Configuration"
        description="Override backend defaults."
        selected={mode === "custom"}
        onClick={() => onSelect("custom")}
      >
        <ParameterGrid
          modelType={modelType}
          editable={mode === "custom"}
          transformerConfig={transformerConfig}
          onTransformerChange={onTransformerChange}
          defaultTransformerConfig={defaultTransformerConfig}
          gnnConfig={gnnConfig}
          onGnnChange={onGnnChange}
          defaultGnnConfig={defaultGnnConfig}
          bestConfig={bestConfig}
          onBestChange={onBestChange}
          defaultBestConfig={defaultBestConfig}
        />
      </ConfigCard>
    </div>
  );
}

function ConfigCard({
  title,
  description,
  selected,
  onClick,
  children,
}: {
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      onClick={onClick}
      className={`border rounded-xl p-6 cursor-pointer transition ${
        selected ? "border-brand-500 bg-brand-50" : "border-gray-200 bg-white hover:border-gray-300"
      }`}
    >
      <div className="flex items-start gap-4">
        <Icon selected={selected} />

        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold">{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
          <div className="mt-4">{children}</div>
        </div>

        <Radio selected={selected} />
      </div>
    </div>
  );
}

function ParameterGrid({
  modelType,
  editable,
  transformerConfig,
  onTransformerChange,
  defaultTransformerConfig,
  gnnConfig,
  onGnnChange,
  defaultGnnConfig,
  bestConfig,
  onBestChange,
  defaultBestConfig,
}: {
  modelType: "gnn" | "transformer" | "best";
  editable: boolean;
  transformerConfig: TransformerConfig;
  onTransformerChange: (cfg: TransformerConfig) => void;
  defaultTransformerConfig: TransformerConfig;
  gnnConfig: GnnConfig;
  onGnnChange: (cfg: GnnConfig) => void;
  defaultGnnConfig: GnnConfig;
  bestConfig: BestConfig;
  onBestChange: (cfg: BestConfig) => void;
  defaultBestConfig: BestConfig;
}) {
  if (modelType === "transformer") {
    const cfg = editable ? transformerConfig : defaultTransformerConfig;
    const update = <K extends keyof TransformerConfig>(key: K, value: number) => {
      onTransformerChange({ ...transformerConfig, [key]: value });
    };

    return (
      <div className="grid grid-cols-2 gap-4">
        <ParameterField
          label="Max sequence length"
          value={cfg.max_len}
          placeholder="16"
          editable={editable}
          onChange={(e) => update("max_len", n(e.target.value))}
        />
        <ParameterField
          label="Model dimension"
          value={cfg.d_model}
          placeholder="64"
          editable={editable}
          onChange={(e) => update("d_model", n(e.target.value))}
        />
        <ParameterField
          label="Number of attention heads"
          value={cfg.num_heads}
          placeholder="4"
          editable={editable}
          onChange={(e) => update("num_heads", n(e.target.value))}
        />
        <ParameterField
          label="Number of transformer blocks"
          value={cfg.num_blocks}
          placeholder="2"
          editable={editable}
          onChange={(e) => update("num_blocks", n(e.target.value))}
        />
        <ParameterField
          label="Dropout rate"
          value={cfg.dropout_rate}
          placeholder="0.1"
          editable={editable}
          min="0"
          max="1"
          step="0.05"
          onChange={(e) => update("dropout_rate", n(e.target.value))}
        />
        <ParameterField
          label="Number of epochs"
          value={cfg.epochs}
          placeholder="5"
          editable={editable}
          onChange={(e) => update("epochs", n(e.target.value))}
        />
        <ParameterField
          label="Batch size"
          value={cfg.batch_size}
          placeholder="128"
          editable={editable}
          onChange={(e) => update("batch_size", n(e.target.value))}
        />
        <ParameterField
          label="Early stopping patience"
          value={cfg.patience}
          placeholder="10"
          editable={editable}
          onChange={(e) => update("patience", n(e.target.value))}
        />
      </div>
    );
  }

  if (modelType === "best") {
    const cfg = editable ? bestConfig : defaultBestConfig;
    const updateNum = <K extends keyof BestConfig>(key: K, value: number) => {
      onBestChange({ ...bestConfig, [key]: value as BestConfig[K] });
    };
    const updateBool = (key: keyof BestConfig, value: boolean) => {
      onBestChange({ ...bestConfig, [key]: value });
    };
    return (
      <div className="grid grid-cols-2 gap-4">
        <ParameterField
          label="Max pattern size (train) — odd integer"
          value={cfg.max_pattern_size_train}
          placeholder="21"
          editable={editable}
          min="3"
          step="2"
          onChange={(e) => updateNum("max_pattern_size_train", n(e.target.value))}
        />
        <ParameterField
          label="Max pattern size (eval) — odd integer ≤ train"
          value={cfg.max_pattern_size_eval}
          placeholder="21"
          editable={editable}
          min="3"
          step="2"
          onChange={(e) => updateNum("max_pattern_size_eval", n(e.target.value))}
        />
        <ParameterField
          label="Stage width % (0–1)"
          value={cfg.process_stage_width_percentage}
          placeholder="0.2"
          editable={editable}
          min="0"
          max="1"
          step="0.05"
          onChange={(e) => updateNum("process_stage_width_percentage", n(e.target.value))}
        />
        <ParameterField
          label="Min frequency"
          value={cfg.min_freq}
          placeholder="1e-14"
          editable={editable}
          min="0"
          step="any"
          onChange={(e) => updateNum("min_freq", n(e.target.value))}
        />
        <ParameterField
          label="Break buffer (RTP only, > 1)"
          value={cfg.break_buffer}
          placeholder="1.2"
          editable={editable}
          min="1"
          step="0.1"
          onChange={(e) => updateNum("break_buffer", n(e.target.value))}
        />
        <BooleanField
          label="Filter START/END tokens"
          value={cfg.filter_sequences}
          editable={editable}
          onChange={(v) => updateBool("filter_sequences", v)}
        />
        <ParameterField
          label="Parallel cores"
          value={cfg.ncores}
          placeholder="1"
          editable={editable}
          min="1"
          step="1"
          onChange={(e) => updateNum("ncores", n(e.target.value))}
        />
      </div>
    );
  }

  const cfg = editable ? gnnConfig : defaultGnnConfig;
  const update = <K extends keyof GnnConfig>(key: K, value: number) => {
    onGnnChange({ ...gnnConfig, [key]: value });
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <ParameterField
        label="Hidden channels"
        value={cfg.hidden}
        placeholder="64"
        editable={editable}
        onChange={(e) => update("hidden", n(e.target.value))}
      />
      <ParameterField
        label="Dropout rate"
        value={cfg.dropout_rate}
        placeholder="0.1"
        editable={editable}
        min="0"
        max="1"
        step="0.05"
        onChange={(e) => update("dropout_rate", n(e.target.value))}
      />
      <ParameterField
        label="Learning rate"
        value={cfg.lr}
        placeholder="0.0004"
        editable={editable}
        min="0"
        step="0.0001"
        onChange={(e) => update("lr", n(e.target.value))}
      />
      <ParameterField
        label="Number of epochs"
        value={cfg.epochs}
        placeholder="5"
        editable={editable}
        onChange={(e) => update("epochs", n(e.target.value))}
      />
      <ParameterField
        label="Batch size"
        value={cfg.batch_size}
        placeholder="64"
        editable={editable}
        onChange={(e) => update("batch_size", n(e.target.value))}
      />
      <ParameterField
        label="Early stopping patience"
        value={cfg.patience}
        placeholder="10"
        editable={editable}
        onChange={(e) => update("patience", n(e.target.value))}
      />
    </div>
  );
}

function n(v: string): number {
  return v === "" ? NaN : Number(v);
}

function BooleanField({
  label,
  value,
  editable,
  onChange,
}: {
  label: string;
  value: boolean;
  editable: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      {editable ? (
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded border-gray-300 accent-brand-600 cursor-pointer"
        />
      ) : (
        <div className="px-3 py-2 font-medium text-black">{value ? "true" : "false"}</div>
      )}
    </div>
  );
}

function ParameterField({
  label,
  value,
  placeholder,
  editable,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  placeholder: string;
  editable: boolean;
  min?: string;
  max?: string;
  step?: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      {editable ? (
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          className="w-full bg-white text-black border rounded px-3 py-2 appearance-none focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
      ) : (
        <div className="px-3 py-2 font-medium text-black">{value}</div>
      )}
    </div>
  );
}

function Icon({ selected }: { selected: boolean }) {
  return (
    <div className={`p-3 rounded-lg transition ${selected ? "bg-brand-500" : "bg-gray-100"}`}>
      <Settings className={`w-5 h-5 ${selected ? "text-white" : "text-gray-600"}`} />
    </div>
  );
}

function Radio({ selected }: { selected: boolean }) {
  return (
    <div
      className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
        selected ? "border-brand-500" : "border-gray-300"
      }`}
    >
      {selected && <div className="w-2.5 h-2.5 bg-brand-500 rounded-full" />}
    </div>
  );
}


