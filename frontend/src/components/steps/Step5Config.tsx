import { Settings } from "lucide-react";

export type ConfigMode = "default" | "custom";

interface Step5ConfigProps {
  mode: ConfigMode | null;
  onSelect: (mode: ConfigMode) => void;
}

export default function Step5Config({ mode, onSelect }: Step5ConfigProps) {
  return (
    <div className="space-y-6 max-w-5xl">

      {/* DEFAULT CONFIGURATION */}
      <ConfigCard
        title="Default Configuration"
        description="Pre-configured optimal settings based on your dataset and model type. Recommended for most use cases."
        selected={mode === "default"}
        onClick={() => onSelect("default")}
      >
        <ParameterGrid editable={false} />
      </ConfigCard>

      {/* CUSTOM CONFIGURATION */}
      <ConfigCard
        title="Custom Configuration"
        description="Advanced mode for fine-tuning hyperparameters. Requires expertise in machine learning."
        selected={mode === "custom"}
        onClick={() => onSelect("custom")}
      >
        <ParameterGrid editable={mode === "custom"} />
      </ConfigCard>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* --------------------------- CARD ---------------------------------- */
/* ------------------------------------------------------------------ */

interface ConfigCardProps {
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function ConfigCard({
  title,
  description,
  selected,
  onClick,
  children,
}: ConfigCardProps) {
  return (
    <div
      onClick={onClick}
      className={`border rounded-xl p-6 cursor-pointer transition
        ${selected
          ? "border-blue-500 bg-blue-50"
          : "border-gray-200 bg-white hover:border-gray-300"
        }`}
    >
      <div className="flex items-start gap-4">
        <Icon selected={selected} />

        <div className="flex-1">
          <h3 className="text-lg font-semibold">
            {title}
          </h3>
          <p className="text-sm text-gray-600">
            {description}
          </p>

          <div className="mt-4">
            {children}
          </div>
        </div>

        <Radio selected={selected} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ----------------------- PARAMETERS GRID --------------------------- */
/* ------------------------------------------------------------------ */

function ParameterGrid({ editable }: { editable: boolean }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <ParameterField
        label="Learning Rate"
        value="0.001"
        editable={editable}
      />
      <ParameterField
        label="Batch Size"
        value="32"
        editable={editable}
      />
      <ParameterField
        label="Epochs"
        value="50"
        editable={editable}
      />
      <ParameterField
        label="Hidden Layers"
        value="3"
        editable={editable}
      />
    </div>
  );
}

function ParameterField({
  label,
  value,
  editable,
}: {
  label: string;
  value: string;
  editable: boolean;
}) {
  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="text-sm text-gray-500 mb-1">
        {label}
      </div>

      {editable ? (
        <input
          defaultValue={value}
          className="
            w-full
            bg-white
            text-black
            border
            rounded
            px-3
            py-2
            appearance-none
            focus:outline-none
            focus:ring-2
            focus:ring-blue-500
          "
        />
      ) : (
        <div className="px-3 py-2 font-medium text-black">
          {value}
        </div>
      )}
    </div>
  );
}


/* ------------------------------------------------------------------ */
/* ---------------------------- ICON --------------------------------- */
/* ------------------------------------------------------------------ */

function Icon({ selected }: { selected: boolean }) {
  return (
    <div
      className={`p-3 rounded-lg transition
        ${selected ? "bg-blue-500" : "bg-gray-100"}`}
    >
      <Settings
        className={`w-5 h-5
          ${selected ? "text-white" : "text-gray-600"}`}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ---------------------------- RADIO -------------------------------- */
/* ------------------------------------------------------------------ */

function Radio({ selected }: { selected: boolean }) {
  return (
    <div
      className={`w-5 h-5 rounded-full border-2 flex items-center justify-center
        ${selected ? "border-blue-500" : "border-gray-300"}`}
    >
      {selected && (
        <div className="w-2.5 h-2.5 bg-blue-500 rounded-full" />
      )}
    </div>
  );
}
