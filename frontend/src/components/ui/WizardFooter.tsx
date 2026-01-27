type WizardFooterProps = {
  step: number;
  canContinue: boolean;
  onCancel: () => void;
  onPrevious: () => void;
  onContinue: () => void;
};

export default function WizardFooter({
  step,
  canContinue,
  onCancel,
  onPrevious,
  onContinue,
}: WizardFooterProps) {
  return (
    <div className="flex items-center justify-between pt-6">
      {/* Cancel */}
      <button
        onClick={onCancel}
        className="
          px-6 py-2
          rounded-md
          border border-brand-200
          bg-white
          text-brand-700
          hover:bg-brand-50
          hover:border-brand-300
          transition
        "
      >
        Cancel
      </button>

      {/* Previous + Continue */}
      <div className="flex items-center gap-3">
        <button
          onClick={onPrevious}
          disabled={step === 0}
          className={`
            px-6 py-2
            rounded-md
            border
            transition
            ${
              step === 0
                ? "border-gray-200 text-gray-400 bg-white cursor-not-allowed"
                : "border-brand-200 text-brand-700 bg-white hover:bg-brand-50 hover:border-brand-300"
            }
          `}
        >
          Previous
        </button>

        <button
          onClick={onContinue}
          disabled={!canContinue}
          className={`
            px-6 py-2
            rounded-md
            font-medium
            transition
            ${
              canContinue
                ? "bg-brand-600 text-white hover:bg-brand-700"
                : "bg-brand-300 text-white cursor-not-allowed"
            }
          `}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

