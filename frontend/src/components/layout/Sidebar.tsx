type SidebarProps = {
  currentStep: number;
  completedSteps: boolean[];
};

const STEPS = [
  "Dataset Setup",
  "Column Mapping",
  "Select Model Type",
  "Model Configuration",
  "Select Prediction Task",
  "Select Explainability Method",
  "Review & Run",
];

export default function Sidebar({
  currentStep,
  completedSteps,
}: SidebarProps) {
  return (
    <aside className="w-72 border-r bg-white p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-semibold">
          ML Pipeline Configuration
        </h1>
        <p className="text-sm text-gray-500">
          Process Mining & Prediction
        </p>
      </div>

      {/* Steps */}
      <nav className="space-y-2">
        {STEPS.map((label, index) => {
          const isActive = index === currentStep;
          const isCompleted = completedSteps[index];

          return (
            <div
              key={label}
              className={`flex items-center gap-4 rounded-lg px-4 py-3 ${
                isActive
                  ? "bg-blue-50 border-l-4 border-blue-500"
                  : ""
              }`}
            >
              {/* Step Indicator */}
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
                  isCompleted
                    ? "bg-green-500 text-white"
                    : isActive
                    ? "bg-blue-500 text-white"
                    : "border border-gray-300 text-gray-400"
                }`}
              >
                {isCompleted ? "✓" : index + 1}
              </div>

              {/* Label */}
              <span
                className={`text-sm ${
                  isActive
                    ? "font-medium text-gray-900"
                    : isCompleted
                    ? "text-gray-500"
                    : "text-gray-400"
                }`}
              >
                {label}
              </span>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
