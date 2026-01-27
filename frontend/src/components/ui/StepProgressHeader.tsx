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
      <div className="text-sm text-brand-600 mb-2">
        Step {step + 1} of {totalSteps}
      </div>

      <div className="w-full bg-brand-100 rounded h-2">
        <div
          className="h-2 rounded bg-brand-600"
          style={{ width: `${progressPercent}%` }}
        />
      </div>
    </div>
  );
}

