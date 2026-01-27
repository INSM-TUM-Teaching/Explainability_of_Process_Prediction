import type { DatasetUploadResponse } from "../../lib/api";

type Step3PredictionProps = {
  task: string | null;
  category: "classification" | "regression" | null;
  targetColumn: string | null;
  dataset: DatasetUploadResponse | null;
  onSelectTask: (task: string) => void;
  onSelectCategory: (category: "classification" | "regression") => void;
  onTargetColumnChange: (col: string) => void;
};

export default function Step3Prediction({
  task,
  category,
  targetColumn,
  dataset,
  onSelectTask,
  onSelectCategory,
  onTargetColumnChange,
}: Step3PredictionProps) {
  const isSelected = (t: string) => task === t;

  const baseCard =
    "w-full self-stretch border rounded-lg p-6 cursor-pointer transition flex items-start justify-between";
  const selectedCard = "border-brand-500 bg-brand-50 ring-1 ring-brand-400";
  const unselectedCard =
    "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50";

  const columns = (dataset?.columns ?? []).filter((col) => col !== "__split");
  const columnTypes = dataset?.column_types ?? {};

  const targetType = targetColumn ? columnTypes[targetColumn] : null;
  const invalidTarget = !!targetColumn && targetType !== "categorical";

  return (
    <div className="space-y-8 w-full max-w-none">
      <div>
        <h2 className="text-2xl font-semibold">Select Prediction Task</h2>
        <p className="text-sm text-brand-600">
          Choose the type of prediction you want to perform on your process data.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          onClick={() => onSelectCategory("classification")}
          className={`${baseCard} ${
            category === "classification" ? selectedCard : unselectedCard
          }`}
        >
          <div className="pr-6 min-w-0">
            <div className="font-medium text-gray-900">Classification</div>
            <div className="text-sm text-gray-600 mt-1 break-words">
              Predict a categorical outcome based on the event sequence.
            </div>
          </div>
          <div
            className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
              category === "classification" ? "border-brand-600 bg-brand-600" : "border-gray-300"
            }`}
          />
        </div>

        <div
          onClick={() => onSelectCategory("regression")}
          className={`${baseCard} ${
            category === "regression" ? selectedCard : unselectedCard
          }`}
        >
          <div className="pr-6 min-w-0">
            <div className="font-medium text-gray-900">Regression</div>
            <div className="text-sm text-gray-600 mt-1 break-words">
              Predict a continuous time-based outcome.
            </div>
          </div>
          <div
            className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
              category === "regression" ? "border-brand-600 bg-brand-600" : "border-gray-300"
            }`}
          />
        </div>
      </div>

      {category === "classification" && (
        <div className="space-y-4">
          <div
            onClick={() => onSelectTask("next_activity")}
            className={`${baseCard} ${
              isSelected("next_activity") ? selectedCard : unselectedCard
            }`}
          >
            <div className="pr-6 min-w-0">
              <div className="font-medium text-gray-900">Next Activity Prediction</div>
              <div className="text-sm text-gray-600 mt-1 break-words">
                Predict the next activity in a running process instance.
              </div>
            </div>
            <div
              className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
                isSelected("next_activity")
                  ? "border-brand-600 bg-brand-600"
                  : "border-gray-300"
              }`}
            />
          </div>

          <div
            onClick={() => onSelectTask("custom_activity")}
            className={`${baseCard} ${
              isSelected("custom_activity") ? selectedCard : unselectedCard
            }`}
          >
            <div className="pr-6 min-w-0">
              <div className="font-medium text-gray-900">Custom Prediction</div>
              <div className="text-sm text-gray-600 mt-1 break-words">
                Predict the next value of a selected categorical column.
              </div>
            </div>
            <div
              className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
                isSelected("custom_activity")
                  ? "border-brand-600 bg-brand-600"
                  : "border-gray-300"
              }`}
            />
          </div>

          {isSelected("custom_activity") && (
            <div className="border rounded-lg bg-white p-4">
              <div className="text-sm font-medium text-gray-900">Target Column</div>
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
                <div className="text-sm text-gray-700">Select categorical column</div>
                <div className="sm:col-span-2">
                  <select
                    className="w-full border rounded-md px-3 py-2 bg-white text-sm"
                    value={targetColumn ?? ""}
                    onChange={(e) => onTargetColumnChange(e.target.value)}
                  >
                    <option value="">Select column...</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col} ({columnTypes[col] ?? "unknown"})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {invalidTarget && (
                <div className="mt-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                  Invalid column selected. Please choose a categorical column.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {category === "regression" && (
        <div className="space-y-4">
          <div
            onClick={() => onSelectTask("event_time")}
            className={`${baseCard} ${isSelected("event_time") ? selectedCard : unselectedCard}`}
          >
            <div className="pr-6 min-w-0">
              <div className="font-medium text-gray-900">Event Time Prediction</div>
              <div className="text-sm text-gray-600 mt-1 break-words">
                Predict the time until the next event occurs.
              </div>
            </div>
            <div
              className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
                isSelected("event_time") ? "border-brand-600 bg-brand-600" : "border-gray-300"
              }`}
            />
          </div>

          <div
            onClick={() => onSelectTask("remaining_time")}
            className={`${baseCard} ${
              isSelected("remaining_time") ? selectedCard : unselectedCard
            }`}
          >
            <div className="pr-6 min-w-0">
              <div className="font-medium text-gray-900">Remaining Time Prediction</div>
              <div className="text-sm text-gray-600 mt-1 break-words">
                Estimate the time required to complete the process.
              </div>
            </div>
            <div
              className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
                isSelected("remaining_time")
                  ? "border-brand-600 bg-brand-600"
                  : "border-gray-300"
              }`}
            />
          </div>
        </div>
      )}
    </div>
  );
}

