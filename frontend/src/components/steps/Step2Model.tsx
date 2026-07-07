import { useCapabilities } from "../../models/capabilities";

type Step2ModelProps = {
  modelType: string | null;
  onSelect: (type: string) => void;
};

export default function Step2Model({ modelType, onSelect }: Step2ModelProps) {
  const { models, loading, error } = useCapabilities();
  const isSelected = (type: string) => modelType === type;

  const baseCard =
    "w-full self-stretch border rounded-lg p-6 cursor-pointer transition flex items-start justify-between";
  const selectedCard = "border-brand-500 bg-brand-50 ring-1 ring-brand-400";
  const unselectedCard =
    "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50";

  return (
    <div className="space-y-8 w-full max-w-none">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Select Model Type</h2>
        <p className="text-sm text-brand-600">
          Choose the machine learning model architecture for your prediction task.
        </p>
      </div>

      {loading && (
        <p className="text-sm text-brand-600">Loading available models…</p>
      )}
      {error && (
        <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
          Could not load models from the backend: {error}
        </p>
      )}

      {models.map((model) => (
        <div
          key={model.id}
          onClick={() => onSelect(model.id)}
          className={`${baseCard} ${
            isSelected(model.id) ? selectedCard : unselectedCard
          }`}
        >
          {/* min-w-0 + break-words prevents long tokens from forcing width */}
          <div className="pr-6 min-w-0">
            <div className="font-medium text-gray-900">{model.label}</div>
            <div className="text-sm text-gray-600 mt-1 break-words">
              {model.description}
            </div>
          </div>

          {/* Radio */}
          <div
            className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
              isSelected(model.id)
                ? "border-brand-600 bg-brand-600"
                : "border-gray-300"
            }`}
          />
        </div>
      ))}
    </div>
  );
}
