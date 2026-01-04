type StepProgressHeaderProps = {
  step: number;
  totalSteps: number;
};

export default function StepProgressHeader({
  step,
  totalSteps,
}: StepProgressHeaderProps) {
  const progressPercent =
    ((step + 1) / totalSteps) * 100;

  return (
    <div className="mb-8">
      <div className="text-sm text-gray-500 mb-2">
        Step {step + 1} of {totalSteps}
      </div>

      <div className="w-full bg-gray-200 rounded h-2">
        <div
          className="bg-blue-500 h-2 rounded"
          style={{ width: `${progressPercent}%` }}
        />
      </div>
    </div>
  );
}
