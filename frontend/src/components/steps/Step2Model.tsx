type Step2ModelProps = {
  modelType: string | null;
  onSelect: (type: string) => void;
};

export default function Step2Model({ modelType, onSelect }: Step2ModelProps) {
  const isSelected = (type: string) => modelType === type;

  const baseCard =
    "w-full self-stretch border rounded-lg p-6 cursor-pointer transition flex items-start justify-between";
  const selectedCard = "border-blue-500 bg-blue-50 ring-1 ring-blue-400";
  const unselectedCard =
    "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50";

  return (
    <div className="space-y-8 w-full max-w-none">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Select Model Type</h2>
        <p className="text-sm text-gray-500">
          Choose the machine learning model architecture for your prediction task.
        </p>
      </div>

      {/* Transformer */}
      <div
        onClick={() => onSelect("transformer")}
        className={`${baseCard} ${
          isSelected("transformer") ? selectedCard : unselectedCard
        }`}
      >
        {/* min-w-0 + break-words prevents long tokens from forcing width */}
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">Transformer</div>
          <div className="text-sm text-gray-600 mt-1 break-words">
            State-of-the-art attention-based architecture, ideal for sequential data and
            complex patterns. Best for large datasets with temporal dependencies.
          </div>
        </div>

        {/* Radio */}
        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected("transformer")
              ? "border-blue-600 bg-blue-600"
              : "border-gray-300"
          }`}
        />
      </div>

      {/* GNN */}
      <div
        onClick={() => onSelect("gnn")}
        className={`${baseCard} ${isSelected("gnn") ? selectedCard : unselectedCard}`}
      >
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">GNN (Graph Neural Network)</div>
          <div className="text-sm text-gray-600 mt-1 break-words">
            Graph-based architecture, perfect for modeling relationships and dependencies
            between activities. Excels at capturing process structure.
          </div>
        </div>

        {/* Radio */}
        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected("gnn") ? "border-blue-600 bg-blue-600" : "border-gray-300"
          }`}
        />
      </div>
    </div>
  );
}
