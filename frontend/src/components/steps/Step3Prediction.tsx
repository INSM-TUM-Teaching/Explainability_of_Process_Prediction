import type { DatasetUploadResponse, TaskMeta } from "../../lib/api";
import { useCapabilities } from "../../models/capabilities";

// Categories the user picks between before seeing tasks (Transformer / GNN).
// "sequence" tasks (BEST) are shown as a flat list with no category picker.
export type PickableCategory = "classification" | "regression";

type Step3PredictionProps = {
  modelType: string | null;
  task: string | null;
  category: PickableCategory | null;
  targetColumn: string | null;
  dataset: DatasetUploadResponse | null;
  onSelectTask: (task: string) => void;
  onSelectCategory: (category: PickableCategory) => void;
  onTargetColumnChange: (col: string) => void;
};

const CATEGORY_META: Record<PickableCategory, { label: string; description: string }> = {
  classification: {
    label: "Classification",
    description: "Predict a categorical outcome based on the event sequence.",
  },
  regression: {
    label: "Regression",
    description: "Predict a continuous time-based outcome.",
  },
};

const PICKABLE_CATEGORIES: PickableCategory[] = ["classification", "regression"];

export default function Step3Prediction({
  modelType,
  task,
  category,
  targetColumn,
  dataset,
  onSelectTask,
  onSelectCategory,
  onTargetColumnChange,
}: Step3PredictionProps) {
  const { getModel } = useCapabilities();
  const model = getModel(modelType);
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

  const tasks = model?.tasks ?? [];
  const sequenceTasks = tasks.filter((t) => t.category === "sequence");
  const tasksByCategory = (c: PickableCategory) => tasks.filter((t) => t.category === c);
  const availableCategories = PICKABLE_CATEGORIES.filter(
    (c) => tasksByCategory(c).length > 0
  );

  // A persisted task keeps its category group visible even if `category` is null.
  const selectedTask = tasks.find((t) => t.id === task);
  const activeCategory: PickableCategory | null =
    category ??
    (selectedTask && selectedTask.category !== "sequence"
      ? (selectedTask.category as PickableCategory)
      : null);

  const renderTaskCard = (t: TaskMeta) => (
    <div key={t.id} className="space-y-4">
      <div
        onClick={() => onSelectTask(t.id)}
        className={`${baseCard} ${isSelected(t.id) ? selectedCard : unselectedCard}`}
      >
        <div className="pr-6 min-w-0">
          <div className="font-medium text-gray-900">{t.label}</div>
          <div className="text-sm text-gray-600 mt-1 break-words">{t.description}</div>
        </div>
        <div
          className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
            isSelected(t.id) ? "border-brand-600 bg-brand-600" : "border-gray-300"
          }`}
        />
      </div>

      {t.needs_target_column && isSelected(t.id) && (
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
  );

  return (
    <div className="space-y-8 w-full max-w-none">
      <div>
        <h2 className="text-2xl font-semibold">Select Prediction Task</h2>
        <p className="text-sm text-brand-600">
          Choose the type of prediction you want to perform on your process data.
        </p>
      </div>

      {!model ? (
        <div className="border rounded-lg bg-white p-6 text-sm text-gray-700">
          Please select a model type in Step 2 first.
        </div>
      ) : availableCategories.length > 0 ? (
        <>
          {/* Category picker (Classification / Regression) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {availableCategories.map((c) => (
              <div
                key={c}
                onClick={() => onSelectCategory(c)}
                className={`${baseCard} ${
                  activeCategory === c ? selectedCard : unselectedCard
                }`}
              >
                <div className="pr-6 min-w-0">
                  <div className="font-medium text-gray-900">{CATEGORY_META[c].label}</div>
                  <div className="text-sm text-gray-600 mt-1 break-words">
                    {CATEGORY_META[c].description}
                  </div>
                </div>
                <div
                  className={`h-4 w-4 rounded-full border-2 mt-1 shrink-0 ${
                    activeCategory === c ? "border-brand-600 bg-brand-600" : "border-gray-300"
                  }`}
                />
              </div>
            ))}
          </div>

          {/* Tasks for the selected category */}
          {activeCategory && (
            <div className="space-y-4">{tasksByCategory(activeCategory).map(renderTaskCard)}</div>
          )}

          {/* Any sequence tasks (rare mixed models) shown flat below */}
          {sequenceTasks.length > 0 && (
            <div className="space-y-4">{sequenceTasks.map(renderTaskCard)}</div>
          )}
        </>
      ) : (
        // Sequence-only model (BEST): flat task list, no category picker.
        <div className="space-y-4">{sequenceTasks.map(renderTaskCard)}</div>
      )}
    </div>
  );
}
