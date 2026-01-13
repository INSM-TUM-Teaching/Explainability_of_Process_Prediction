type TopHeaderProps = {
  currentStep: number;
};

const STEP_TITLES = [
  "Upload Dataset",
  "Select Model Type",
  "Prediction Task",
  "Explainability",
  "Model Configuration",
  "Review & Run",
];

export default function TopHeader({
  currentStep,
}: TopHeaderProps) {
  return (
    <div className="mb-6">
      <p className="text-sm text-gray-500">
        Step {currentStep + 1} of {STEP_TITLES.length}
      </p>
      <h1 className="text-2xl font-semibold">
        {STEP_TITLES[currentStep]}
      </h1>
    </div>
  );
}
