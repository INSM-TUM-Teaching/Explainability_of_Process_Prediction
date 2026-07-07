import React from "react";
import { Settings } from "lucide-react";
import type { ConfigFieldMeta } from "../../lib/api";
import {
  firstConstraintError,
  useCapabilities,
  type ModelConfig,
  type ConfigValue,
} from "../../models/capabilities";

export type ConfigMode = "default" | "custom";

type Step5ConfigProps = {
  modelType: string | null;
  mode: ConfigMode | null;
  onSelect: (mode: ConfigMode) => void;
  /** Editable custom-config values (already merged with defaults by the caller). */
  config: ModelConfig;
  onConfigChange: (cfg: ModelConfig) => void;
  /** Backend defaults for the selected model. */
  defaultConfig: ModelConfig;
};

export default function Step5Config({
  modelType,
  mode,
  onSelect,
  config,
  onConfigChange,
  defaultConfig,
}: Step5ConfigProps) {
  const { getModel } = useCapabilities();
  const model = getModel(modelType);

  if (!model) {
    return (
      <div className="space-y-3">
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">Select a model type in Step 2 first.</p>
      </div>
    );
  }

  const constraintError =
    mode === "custom" ? firstConstraintError(model, config) : null;

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-semibold">Model Configuration</h2>
        <p className="text-sm text-brand-600">
          {model.config_intro ?? `These options map 1:1 to the backend ${model.label} config.`}
        </p>
      </div>

      <ConfigCard
        title="Default Configuration"
        description="Use backend defaults."
        selected={mode === "default"}
        onClick={() => onSelect("default")}
      >
        <ParameterGrid
          fields={model.config_fields}
          values={defaultConfig}
          editable={false}
          onChange={() => {}}
        />
      </ConfigCard>

      <ConfigCard
        title="Custom Configuration"
        description="Override backend defaults."
        selected={mode === "custom"}
        onClick={() => onSelect("custom")}
      >
        <ParameterGrid
          fields={model.config_fields}
          values={config}
          editable={mode === "custom"}
          onChange={onConfigChange}
        />
        {constraintError && (
          <div className="mt-4 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
            {constraintError}
          </div>
        )}
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
  fields,
  values,
  editable,
  onChange,
}: {
  fields: ConfigFieldMeta[];
  values: ModelConfig;
  editable: boolean;
  onChange: (cfg: ModelConfig) => void;
}) {
  const update = (key: string, value: ConfigValue) => {
    onChange({ ...values, [key]: value });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {fields.map((field) =>
        field.kind === "boolean" ? (
          <BooleanField
            key={field.key}
            field={field}
            value={Boolean(values[field.key])}
            editable={editable}
            onChange={(v) => update(field.key, v)}
          />
        ) : (
          <ParameterField
            key={field.key}
            field={field}
            value={values[field.key] as number}
            editable={editable}
            onChange={(e) => update(field.key, n(e.target.value))}
          />
        )
      )}
    </div>
  );
}

function n(v: string): number {
  return v === "" ? NaN : Number(v);
}

function ParameterField({
  field,
  value,
  editable,
  onChange,
}: {
  field: ConfigFieldMeta;
  value: number;
  editable: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm font-medium text-brand-900">{field.label}</div>
      {field.description && (
        <p className="text-xs text-brand-600 mt-1 leading-relaxed">{field.description}</p>
      )}
      <div className="mt-3">
        {editable ? (
          <input
            type="number"
            value={Number.isFinite(value) ? value : ""}
            placeholder={field.placeholder}
            min={field.min ?? field.gt}
            max={field.max ?? field.lt}
            step={field.step}
            onChange={onChange}
            className="w-full bg-white text-black border rounded px-3 py-2 appearance-none focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
        ) : (
          <div className="px-3 py-2 font-medium text-black">{value}</div>
        )}
      </div>
    </div>
  );
}

function BooleanField({
  field,
  value,
  editable,
  onChange,
}: {
  field: ConfigFieldMeta;
  value: boolean;
  editable: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm font-medium text-brand-900">{field.label}</div>
      {field.description && (
        <p className="text-xs text-brand-600 mt-1 leading-relaxed">{field.description}</p>
      )}
      <div className="mt-3">
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
