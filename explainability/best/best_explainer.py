import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})


class BESTExplainer:
    """Native pattern-based explainability for a fitted BESTPredictor.

    After BESTPredictor.predict() has been called the model exposes:
        model.choice_tracker_nap  -- {'prob': [], 'len': [], 'dist': []}
        model.choice_tracker_rtp  -- {'prob': [], 'len': [], 'dist': []}

    When a BESTRunner instance is supplied via the `runner` argument, richer
    analysis is produced using decoded predictions and actual labels.

    Outputs go to output_dir/explainability/.
    """

    def __init__(self, model, output_dir: str, task: str, runner=None):
        """
        Args:
            model:      A fitted BESTPredictor instance (after predict() was called).
            output_dir: Base artifacts directory; outputs go to output_dir/explainability/.
            task:       'nap' or 'rtp'.
            runner:     Optional BESTRunner instance for richer per-prediction analysis.
        """
        self.model = model
        self.output_dir = os.path.join(output_dir, "explainability")
        self.task = task
        self.runner = runner

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def explain(self) -> None:
        """Generate and save all explainability artefacts."""
        os.makedirs(self.output_dir, exist_ok=True)

        tracker = self._get_tracker()
        if tracker is None:
            self._write_placeholder("No tracker data available for this task.")
            return

        probs = self._clean(tracker.get("prob", []))
        distances = self._clean(tracker.get("dist", []))

        if len(probs) == 0:
            self._write_placeholder("Tracker contains no data points.")
            return

        # Always produce the RPIF distance distribution (genuinely informative)
        self._plot_distances(distances)

        # Build decoded prediction frame from runner when available
        decoded_preds, decoded_actuals, prefix_lengths = self._build_prediction_frame()

        if decoded_preds is not None:
            # Rich charts using actual predictions and labels
            self._plot_accuracy_by_prefix_length(decoded_preds, decoded_actuals, prefix_lengths)
            self._plot_confidence_by_class(probs, decoded_preds)
            self._plot_activity_distribution(decoded_preds, decoded_actuals)
            self._save_top_patterns_enriched(probs, distances, decoded_preds, decoded_actuals, prefix_lengths)
            self._write_summary_report_enriched(probs, distances, decoded_preds, decoded_actuals, prefix_lengths)
        else:
            # Fallback: basic tracker charts (pattern length histogram may be degenerate)
            lengths = self._clean(tracker.get("len", []))
            self._plot_probabilities_basic(probs)
            self._plot_lengths_basic(lengths)
            self._save_top_patterns_basic(probs, lengths, distances)
            self._write_summary_report_basic(probs, lengths, distances)

        print(f"[BEST Explainer] Artefacts written to {self.output_dir}")

    # ------------------------------------------------------------------
    # Tracker retrieval
    # ------------------------------------------------------------------

    def _get_tracker(self):
        if self.task == "nap":
            return getattr(self.model, "choice_tracker_nap", None)
        return getattr(self.model, "choice_tracker_rtp", None)

    @staticmethod
    def _clean(values: list) -> np.ndarray:
        """Remove None / NaN entries and return a float array."""
        arr = np.array([v for v in values if v is not None], dtype=float)
        return arr[np.isfinite(arr)]

    # ------------------------------------------------------------------
    # Build decoded prediction frame from runner
    # ------------------------------------------------------------------

    def _build_prediction_frame(self):
        """Decode predictions and actuals from runner if available.

        Returns:
            (decoded_preds, decoded_actuals, prefix_lengths) or (None, None, None)
            where prefix_lengths is the number of REAL events seen before each prediction.
        """
        if self.runner is None:
            return None, None, None

        runner = self.runner
        try:
            predictions = runner.predictions
            if predictions is None or len(predictions) == 0:
                return None, None, None

            if self.task == "nap":
                decoded_preds = runner._decode_nap(predictions)
                actuals_enc = getattr(runner.test_seq, "next_activities", None)
                if actuals_enc is None:
                    return None, None, None
                n = min(len(decoded_preds), len(actuals_enc))
                decoded_preds = decoded_preds[:n]
                decoded_actuals = [runner._decode_activity(a) for a in actuals_enc[:n]]
            else:
                decoded_preds = runner._decode_rtp(predictions)
                actuals_enc = getattr(runner.test_seq, "full_future_sequences", None)
                if actuals_enc is None:
                    return None, None, None
                n = min(len(decoded_preds), len(actuals_enc))
                decoded_preds = decoded_preds[:n]
                decoded_actuals = []
                for seq in actuals_enc[:n]:
                    if seq is None:
                        decoded_actuals.append(None)
                    else:
                        decoded_actuals.append(", ".join(
                            str(runner._decode_activity(idx))
                            for idx in seq if idx is not None
                        ))

            # Effective prefix length = padded prefix len - padding_size
            # padding_size = int(max_pattern_size/2)+1 (stored on BESTPredictor as _padding_size)
            padding_size = getattr(self.model, "_padding_size", 0)
            if hasattr(runner.test_seq, "relevant_prefixes") and runner.test_seq.relevant_prefixes:
                raw_lens = [len(p["prefix"]) for p in runner.test_seq.relevant_prefixes[:n]]
                prefix_lengths = [max(0, raw - padding_size) for raw in raw_lens]
            else:
                prefix_lengths = list(range(n))

            return decoded_preds, decoded_actuals, prefix_lengths

        except Exception as e:
            print(f"[BEST Explainer] Warning: could not build prediction frame: {e}")
            return None, None, None

    # ------------------------------------------------------------------
    # Rich charts (runner available)
    # ------------------------------------------------------------------

    def _plot_accuracy_by_prefix_length(self, preds, actuals, prefix_lengths):
        """Line chart: accuracy vs number of real events seen before prediction."""
        acc_by_len = defaultdict(list)
        for p, a, l in zip(preds, actuals, prefix_lengths):
            if p is not None and a is not None:
                acc_by_len[l].append(int(p == a))

        if not acc_by_len:
            return

        lengths_sorted = sorted(acc_by_len.keys())
        accuracies = [np.mean(acc_by_len[l]) for l in lengths_sorted]
        counts = [len(acc_by_len[l]) for l in lengths_sorted]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(lengths_sorted, accuracies, "o-", color="#4C72B0",
                 linewidth=2, markersize=6, label="Accuracy")
        ax1.set_xlabel("Real events in prefix (steps into trace)")
        ax1.set_ylabel("Accuracy", color="#4C72B0")
        ax1.tick_params(axis="y", labelcolor="#4C72B0")
        ax1.set_ylim(0, 1.05)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        ax2.bar(lengths_sorted, counts, alpha=0.2, color="#55A868", label="# samples")
        ax2.set_ylabel("# samples", color="#55A868")
        ax2.tick_params(axis="y", labelcolor="#55A868")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        fig.tight_layout()
        path = os.path.join(self.output_dir, "accuracy_by_prefix_length.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_confidence_by_class(self, probs: np.ndarray, decoded_preds: list) -> None:
        """Horizontal bar chart: mean pattern confidence per predicted activity class."""
        if len(probs) != len(decoded_preds):
            return

        conf_by_class = defaultdict(list)
        for p_val, cls in zip(probs, decoded_preds):
            if cls is not None:
                conf_by_class[cls].append(p_val)

        if not conf_by_class:
            return

        classes = sorted(conf_by_class.keys(),
                         key=lambda c: np.mean(conf_by_class[c]), reverse=True)
        means = [np.mean(conf_by_class[c]) for c in classes]
        stds = [np.std(conf_by_class[c]) for c in classes]

        fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.55 + 1.5)))
        y_pos = range(len(classes))
        ax.barh(list(y_pos), means, xerr=stds, color="#4C72B0", alpha=0.85,
                capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5})
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Mean pattern confidence (+/- std)")
        ax.set_xlim(0, 1.05)
        ax.axvline(probs.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"Overall mean ({probs.mean():.3f})")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, "confidence_by_predicted_class.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_activity_distribution(self, decoded_preds: list, decoded_actuals: list) -> None:
        """Side-by-side bar chart: predicted vs actual activity frequencies."""
        valid_pairs = [(p, a) for p, a in zip(decoded_preds, decoded_actuals)
                       if p is not None and a is not None]
        if not valid_pairs:
            return

        preds_clean, actuals_clean = zip(*valid_pairs)
        pred_counts = Counter(preds_clean)
        actual_counts = Counter(actuals_clean)

        all_classes = sorted(set(pred_counts) | set(actual_counts))
        actual_vals = [actual_counts.get(c, 0) for c in all_classes]
        pred_vals = [pred_counts.get(c, 0) for c in all_classes]

        x = np.arange(len(all_classes))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(10, len(all_classes) * 1.1), 6))
        ax.bar(x - width / 2, actual_vals, width, label="Actual",
               color="#55A868", alpha=0.85)
        ax.bar(x + width / 2, pred_vals, width, label="Predicted",
               color="#4C72B0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, "activity_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # RPIF distances (always generated)
    # ------------------------------------------------------------------

    def _plot_distances(self, distances: np.ndarray) -> None:
        if len(distances) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=30, color="#C44E52", edgecolor="white", alpha=0.85)
        ax.set_xlabel("RPIF distance")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "rpif_distances.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Enriched CSV + summary report (runner available)
    # ------------------------------------------------------------------

    def _save_top_patterns_enriched(
        self,
        probs: np.ndarray,
        distances: np.ndarray,
        decoded_preds: list,
        decoded_actuals: list,
        prefix_lengths: list,
    ) -> None:
        """Save per-prediction table with confidence, correctness, and prefix length."""
        n = len(probs)
        dist_arr = distances if len(distances) == n else np.full(n, np.nan)
        n_rows = min(n, len(decoded_preds), len(decoded_actuals), len(prefix_lengths))

        correct = [
            int(p == a) if p is not None and a is not None else None
            for p, a in zip(decoded_preds[:n_rows], decoded_actuals[:n_rows])
        ]

        df = pd.DataFrame({
            "prefix_length_events": prefix_lengths[:n_rows],
            "predicted_activity": decoded_preds[:n_rows],
            "actual_activity": decoded_actuals[:n_rows],
            "correct": correct,
            "pattern_confidence": probs[:n_rows],
            "rpif_distance": dist_arr[:n_rows],
        })

        path = os.path.join(self.output_dir, "top_patterns.csv")
        df.to_csv(path, index=False)

    def _write_summary_report_enriched(
        self,
        probs: np.ndarray,
        distances: np.ndarray,
        decoded_preds: list,
        decoded_actuals: list,
        prefix_lengths: list,
    ) -> None:
        """Write enriched summary with accuracy, per-class breakdown, prefix length stats."""
        n = len(decoded_preds)

        correct_flags = [
            int(p == a) for p, a in zip(decoded_preds, decoded_actuals)
            if p is not None and a is not None
        ]
        overall_acc = np.mean(correct_flags) if correct_flags else float("nan")

        # Per actual-class accuracy
        class_correct = defaultdict(list)
        for p, a in zip(decoded_preds, decoded_actuals):
            if p is not None and a is not None:
                class_correct[a].append(int(p == a))

        # Accuracy by prefix length
        len_correct = defaultdict(list)
        for p, a, l in zip(decoded_preds, decoded_actuals, prefix_lengths):
            if p is not None and a is not None:
                len_correct[l].append(int(p == a))

        lines = [
            "=" * 60,
            f"BEST Pattern-Based Explainability - {self.task.upper()} task",
            "=" * 60,
            f"Total predictions analysed : {n:,}",
            f"Overall accuracy           : {overall_acc:.4f} ({overall_acc * 100:.2f}%)",
            "",
            "Pattern confidence (probability of chosen pattern):",
            f"  Mean   : {probs.mean():.4f}",
            f"  Median : {float(np.median(probs)):.4f}",
            f"  Std    : {probs.std():.4f}",
            f"  Min    : {probs.min():.4f}",
            f"  Max    : {probs.max():.4f}",
            f"  % with confidence >= 0.99 : {(probs >= 0.99).mean() * 100:.1f}%",
            f"  % with confidence <  0.90 : {(probs < 0.90).mean() * 100:.1f}%",
        ]

        if len(distances) > 0:
            lines += [
                "",
                "RPIF distance (lower = closer match to training pattern):",
                f"  Mean   : {distances.mean():.4f}",
                f"  Median : {float(np.median(distances)):.4f}",
                f"  Std    : {distances.std():.4f}",
                f"  Min    : {distances.min():.4f}",
                f"  Max    : {distances.max():.4f}",
            ]

        lines += ["", "Accuracy by prefix length (real events seen before prediction):"]
        for l in sorted(len_correct.keys()):
            vals = len_correct[l]
            acc = np.mean(vals)
            lines.append(f"  {l:2d} events : {acc:.3f}  ({len(vals)} samples)")

        lines += ["", "Per-class accuracy (grouped by actual next activity):"]
        for cls in sorted(class_correct.keys()):
            vals = class_correct[cls]
            acc = np.mean(vals)
            lines.append(f"  {cls:<30s} : {acc:.3f}  ({len(vals)} samples)")

        lines += ["", "=" * 60]

        report_path = os.path.join(self.output_dir, "pattern_summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Fallback basic charts (no runner)
    # ------------------------------------------------------------------

    def _plot_probabilities_basic(self, probs: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(probs, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Pattern conditional probability")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "pattern_probabilities.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_lengths_basic(self, lengths: np.ndarray) -> None:
        if len(lengths) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        unique, counts = np.unique(lengths.astype(int), return_counts=True)
        ax.bar(unique, counts, color="#55A868", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Pattern length")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        fig.tight_layout()
        path = os.path.join(self.output_dir, "pattern_lengths.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _save_top_patterns_basic(
        self,
        probs: np.ndarray,
        lengths: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        n = len(probs)
        len_arr = lengths if len(lengths) == n else np.full(n, np.nan)
        dist_arr = distances if len(distances) == n else np.full(n, np.nan)

        df = pd.DataFrame({
            "probability": probs,
            "pattern_length": len_arr,
            "rpif_distance": dist_arr,
        })
        df["prob_bucket"] = (df["probability"] * 20).round() / 20
        summary = (
            df.groupby("prob_bucket")
            .agg(
                count=("probability", "size"),
                mean_probability=("probability", "mean"),
                mean_length=("pattern_length", "mean"),
                mean_distance=("rpif_distance", "mean"),
            )
            .sort_values("count", ascending=False)
            .reset_index()
        )
        path = os.path.join(self.output_dir, "top_patterns.csv")
        summary.to_csv(path, index=False)

    def _write_summary_report_basic(
        self,
        probs: np.ndarray,
        lengths: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        lines = [
            "=" * 60,
            f"BEST Pattern-Based Explainability - {self.task.upper()} task",
            "=" * 60,
            f"Total predictions analysed : {len(probs):,}",
            "",
            "Pattern probability (confidence of chosen pattern):",
            f"  Mean   : {probs.mean():.4f}",
            f"  Median : {float(np.median(probs)):.4f}",
            f"  Std    : {probs.std():.4f}",
            f"  Min    : {probs.min():.4f}",
            f"  Max    : {probs.max():.4f}",
        ]
        if len(lengths) > 0:
            lines += [
                "",
                "Pattern length (number of events in the matched pattern):",
                f"  Mean   : {lengths.mean():.2f}",
                f"  Median : {float(np.median(lengths)):.2f}",
                f"  Min    : {int(lengths.min())}",
                f"  Max    : {int(lengths.max())}",
            ]
        if len(distances) > 0:
            lines += [
                "",
                "RPIF distance (lower = closer match):",
                f"  Mean   : {distances.mean():.4f}",
                f"  Median : {float(np.median(distances)):.4f}",
                f"  Std    : {distances.std():.4f}",
            ]
        lines += ["", "=" * 60]

        report_path = os.path.join(self.output_dir, "pattern_summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Fallback placeholder
    # ------------------------------------------------------------------

    def _write_placeholder(self, message: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"BEST Pattern Analysis\n\n{message}",
                ha="center", va="center", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "pattern_probabilities.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
