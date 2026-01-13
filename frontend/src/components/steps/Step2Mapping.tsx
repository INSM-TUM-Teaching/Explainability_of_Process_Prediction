import Card from "../ui/card";
import type { DatasetUploadResponse } from "../../lib/api";

export type MappingMode = "auto" | "manual";

export type ManualMapping = {
  case_id: string;
  activity: string;
  timestamp: string;
  resource: string | null;
};

type Step2MappingProps = {
  dataset: DatasetUploadResponse | null;
  mode: MappingMode | null;
  manualMapping: ManualMapping;
  onModeChange: (mode: MappingMode) => void;
  onManualMappingChange: (patch: Partial<ManualMapping>) => void;
};

function uniqueNonEmpty(values: Array<string | null>): boolean {
  const filtered = values.filter((v): v is string => !!v && v.trim().length > 0);
  return new Set(filtered).size === filtered.length;
}

export default function Step2Mapping({
  dataset,
  mode,
  manualMapping,
  onModeChange,
  onManualMappingChange,
}: Step2MappingProps) {
  const columns = dataset?.columns ?? [];
  const canShowManual = !!dataset && mode === "manual";

  const manualOk =
    manualMapping.case_id.trim().length > 0 &&
    manualMapping.activity.trim().length > 0 &&
    manualMapping.timestamp.trim().length > 0 &&
    uniqueNonEmpty([
      manualMapping.case_id,
      manualMapping.activity,
      manualMapping.timestamp,
      manualMapping.resource,
    ]);

  return (
    <div className="space-y-8 w-full">
      <div>
        <h2 className="text-2xl font-semibold">Column Mapping</h2>
        <p className="text-sm text-gray-500">
          Choose how to map your dataset columns to the required schema. Resource is optional.
        </p>
      </div>

      {!dataset ? (
        <Card>
          <div className="p-6 text-sm text-gray-700">Upload a dataset in Step 1 first.</div>
        </Card>
      ) : (
        <div className="space-y-4">
          <div
            className={[
              "border rounded-xl p-6 bg-white cursor-pointer transition",
              mode === "auto" ? "border-blue-500 ring-1 ring-blue-200" : "hover:border-gray-300",
            ].join(" ")}
            onClick={() => onModeChange("auto")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") onModeChange("auto");
            }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold">Automatic Mapping</div>
                <div className="text-sm text-gray-600 mt-1">
                  Use backend auto-detection to standardize columns to CaseID, Activity, Timestamp
                  (and Resource if present).
                </div>
              </div>
              <div
                className={[
                  "h-5 w-5 rounded-full border flex items-center justify-center mt-1",
                  mode === "auto" ? "border-blue-600" : "border-gray-300",
                ].join(" ")}
                aria-hidden="true"
              >
                {mode === "auto" && <div className="h-3 w-3 rounded-full bg-blue-600" />}
              </div>
            </div>
          </div>

          <div
            className={[
              "border rounded-xl p-6 bg-white cursor-pointer transition",
              mode === "manual" ? "border-blue-500 ring-1 ring-blue-200" : "hover:border-gray-300",
            ].join(" ")}
            onClick={() => onModeChange("manual")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") onModeChange("manual");
            }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold">Manual Mapping</div>
                <div className="text-sm text-gray-600 mt-1">
                  Select the column for each required field.
                </div>
              </div>
              <div
                className={[
                  "h-5 w-5 rounded-full border flex items-center justify-center mt-1",
                  mode === "manual" ? "border-blue-600" : "border-gray-300",
                ].join(" ")}
                aria-hidden="true"
              >
                {mode === "manual" && <div className="h-3 w-3 rounded-full bg-blue-600" />}
              </div>
            </div>
          </div>

          {canShowManual && (
            <Card title="Manual Mapping">
              <div className="space-y-4">
                <MappingSelect
                  label="Case ID column"
                  value={manualMapping.case_id}
                  columns={columns}
                  onChange={(v) => onManualMappingChange({ case_id: v })}
                />
                <MappingSelect
                  label="Activity column"
                  value={manualMapping.activity}
                  columns={columns}
                  onChange={(v) => onManualMappingChange({ activity: v })}
                />
                <MappingSelect
                  label="Timestamp column"
                  value={manualMapping.timestamp}
                  columns={columns}
                  onChange={(v) => onManualMappingChange({ timestamp: v })}
                />
                <MappingSelect
                  label="Resource column (optional)"
                  value={manualMapping.resource ?? ""}
                  columns={["", ...columns]}
                  onChange={(v) => onManualMappingChange({ resource: v.trim() ? v : null })}
                  placeholder="None"
                />

                {!manualOk && (
                  <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                    Select Case ID, Activity, and Timestamp columns (must be different). Resource is
                    optional.
                  </div>
                )}
              </div>
            </Card>
          )}

          <Card title="Detected Mapping (From Upload)">
            <div className="text-sm text-gray-700">
              {Object.keys(dataset.detected_mapping ?? {}).length === 0 ? (
                <div className="text-gray-600">No renaming was needed.</div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                  {Object.entries(dataset.detected_mapping).map(([from, to]) => (
                    <div
                      key={`${from}->${to}`}
                      className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded px-3 py-2"
                    >
                      <span className="font-mono text-gray-700">{from}</span>
                      <span className="text-gray-500 mx-2">→</span>
                      <span className="font-mono text-gray-900">{to}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

function MappingSelect({
  label,
  value,
  columns,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  columns: string[];
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
      <div className="text-sm font-medium text-gray-800">{label}</div>
      <div className="sm:col-span-2">
        <select
          className="w-full border rounded-md px-3 py-2 bg-white text-sm"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {columns.map((c) => (
            <option key={c || "__none"} value={c}>
              {c || placeholder || "Select..."}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

