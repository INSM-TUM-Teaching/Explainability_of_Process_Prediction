import type { BestConfig } from "./Step5Config";

export const BEST_CONFIG_INTRO =
  "BEST builds a bilaterally expanding subtrace tree from your event log (no neural training). " +
  "Parameters follow the official BEST framework; defaults match the BPM 2025 paper setup.";

export type BestConfigFieldKey = keyof BestConfig;

export type BestConfigFieldMeta = {
  key: BestConfigFieldKey;
  title: string;
  description: string;
  placeholder: string;
  min?: string;
  max?: string;
  step?: string;
  kind: "number" | "boolean";
};

export const BEST_CONFIG_FIELDS: BestConfigFieldMeta[] = [
  {
    key: "max_pattern_size_train",
    title: "Training pattern size (odd)",
    description:
      "Maximum subtrace pattern length while building the tree. Tree depth is (size - 1) / 2: " +
      "size 21 means up to 10 activities before and after the center activity. Larger values capture " +
      "more context but increase training time and memory.",
    placeholder: "21",
    min: "3",
    step: "2",
    kind: "number",
  },
  {
    key: "max_pattern_size_eval",
    title: "Evaluation pattern size (odd, ≤ training)",
    description:
      "Maximum pattern length when matching test prefixes to the tree during prediction. " +
      "Can be smaller than training to limit how far the matcher walks; must not exceed the training size.",
    placeholder: "21",
    min: "3",
    step: "2",
    kind: "number",
  },
  {
    key: "process_stage_width_percentage",
    title: "Process stage width (0 to 1)",
    description:
      "Controls how many process stages BEST uses. Stage width is this fraction of the longest " +
      "training trace. 0 splits into many narrow stages (one BEST model per event position); " +
      "1 uses a single stage for the whole trace. Paper experiments often use values between 0 and 1.",
    placeholder: "0.2",
    min: "0",
    max: "1",
    step: "0.05",
    kind: "number",
  },
  {
    key: "min_freq",
    title: "Minimum pattern frequency",
    description:
      "Drop subtrace patterns whose frequency in the log is below this cutoff. Values near zero " +
      "(e.g. 1e-14) keep almost all patterns; higher values prune rare patterns and speed up runs.",
    placeholder: "1e-14",
    min: "0",
    step: "any",
    kind: "number",
  },
  {
    key: "break_buffer",
    title: "Remaining-trace stop factor (> 1)",
    description:
      "Used only for remaining trace prediction (RTP). Stops extending the predicted sequence when " +
      "its length reaches break_buffer × longest training trace length. 1.2 is the value used in the BEST paper.",
    placeholder: "1.2",
    min: "1",
    step: "0.1",
    kind: "number",
  },
  {
    key: "filter_sequences",
    title: "Filter padding tokens for evaluation",
    description:
      "When enabled, removes padded dummy activities from predicted sequences before scoring RTP/NAP " +
      "metrics so evaluation matches the official BEST benchmark.",
    placeholder: "",
    kind: "boolean",
  },
  {
    key: "ncores",
    title: "Parallel cores",
    description:
      "Number of CPU cores for parallel prediction and RTP evaluation (NDLS). Use 1 on small machines; " +
      "increase on multi-core hosts to shorten remaining-trace runs.",
    placeholder: "1",
    min: "1",
    step: "1",
    kind: "number",
  },
];
