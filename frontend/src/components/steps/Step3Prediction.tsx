type Step3PredictionProps = {
  task: string | null;
  onSelect: (task: string) => void;
};

export default function Step3Prediction({ task, onSelect }: Step3PredictionProps) {
  const isSelected = (t: string) => task === t;

  const baseCard =
    "w-full self-stretch border rounded-lg p-6 cursor-pointer transition flex items-start justify-between";
  const selectedCard = "border-blue-500 bg-blue-50 ring-1 ring-blue-400";
  const unselectedCard =
    "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50";

  return (
    <div className="space-y-8 w-full max-w-none">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Select Prediction Task</h2>
        <p className="text-sm text-gray-500">
          Choose the type of prediction you want to perform on your process data.
        </p>
      </div>

      {/* Next Activity */}
      <div
        onClick={() => onSelect("next_activity")}
        className={`${baseCard} ${
          isSelected("next_activity") ? selectedCard : unselectedCard
        }`}
      >
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">Next Activity Predictor</div>
          <div className="text-sm text-gray-600 mt-1 break-words">
            Predict the next activity in a running process instance.
          </div>
        </div>

        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected("next_activity") ? "border-blue-600 bg-blue-600" : "border-gray-300"
          }`}
        />
      </div>

      {/* Timestamp */}
      <div
        onClick={() => onSelect("timestamp")}
        className={`${baseCard} ${isSelected("timestamp") ? selectedCard : unselectedCard}`}
      >
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">Timestamp Predictor</div>
          <div className="text-sm text-gray-600 mt-1 break-words">
            Predict the timestamp for the next activity occurrence.
          </div>
        </div>

        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected("timestamp") ? "border-blue-600 bg-blue-600" : "border-gray-300"
          }`}
        />
      </div>

      {/* Remaining Time */}
      <div
        onClick={() => onSelect("remaining_time")}
        className={`${baseCard} ${
          isSelected("remaining_time") ? selectedCard : unselectedCard
        }`}
      >
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">Remaining Time Predictor</div>
          <div className="text-sm text-gray-600 mt-1 break-words">
            Estimate the time required to complete the process.
          </div>
        </div>

        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected("remaining_time")
              ? "border-blue-600 bg-blue-600"
              : "border-gray-300"
          }`}
        />
      </div>
    </div>
  );
}
