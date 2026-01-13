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
          border border-gray-300
          bg-white
          text-gray-900
          hover:bg-gray-50
          hover:border-gray-400
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
                : "border-gray-300 text-gray-900 bg-white hover:bg-gray-50 hover:border-gray-400"
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
                ? "bg-blue-600 text-white hover:bg-blue-700"
                : "bg-blue-300 text-white cursor-not-allowed"
            }
          `}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
