// frontend/src/components/steps/Step4Explainability.tsx
import Card from "../ui/card";
import { useCapabilities } from "../../models/capabilities";

// Explainability method values are model-defined by the backend manifest
// (models/capabilities.py). The runner normalizes "none" to null.
export type ExplainValue = string;

type Step4ExplainabilityProps = {
  modelType: string | null;
  method: ExplainValue | null;
  onSelect: (method: ExplainValue) => void;
};

export default function Step4Explainability({
  modelType,
  method,
  onSelect,
}: Step4ExplainabilityProps) {
  const { models, getModel } = useCapabilities();
  const model = getModel(modelType);
  const options = model?.explain_methods ?? [];

  return (
    <div className="space-y-8 w-full">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Select Explainability Method</h2>
        <p className="text-sm text-brand-600">
          Methods are filtered automatically based on the selected model type.
        </p>
      </div>

      {!model ? (
        <Card>
          <div className="p-6 text-sm text-gray-700">
            Please select a model type in Step 2 first.
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {options.map((opt) => {
            const selected = method === opt.value;

            return (
              <div
                key={opt.value}
                className={[
                  "border rounded-xl p-6 bg-white cursor-pointer transition",
                  selected ? "border-brand-500 ring-1 ring-brand-200" : "hover:border-gray-300",
                ].join(" ")}
                onClick={() => onSelect(opt.value)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onSelect(opt.value);
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-lg font-semibold">{opt.label}</div>
                    <div className="text-sm text-gray-600 mt-1">{opt.description}</div>
                  </div>

                  <div
                    className={[
                      "h-5 w-5 rounded-full border flex items-center justify-center mt-1",
                      selected ? "border-brand-600" : "border-gray-300",
                    ].join(" ")}
                    aria-hidden="true"
                  >
                    {selected && <div className="h-3 w-3 rounded-full bg-brand-600" />}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Info: available methods per model, generated from the manifest. */}
      <Card title="Note">
        <div className="bg-brand-50 border border-brand-200 rounded-lg p-4 text-sm text-gray-700">
          <ul className="list-disc ml-5 space-y-1">
            {models.map((m) => (
              <li key={m.id}>
                {m.label}: {m.explain_methods.map((e) => e.label).join(", ")}
              </li>
            ))}
          </ul>
        </div>
      </Card>
    </div>
  );
}
