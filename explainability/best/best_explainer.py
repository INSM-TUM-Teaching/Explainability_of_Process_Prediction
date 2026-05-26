import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

_BLUE  = "#4C72B0"
_GREEN = "#55A868"
_RED   = "#C44E52"
_CMAP  = mcolors.LinearSegmentedColormap.from_list("rg", [_RED, "#f0c060", _GREEN])


class BESTExplainer:

    def __init__(self, model, output_dir: str, task: str, runner=None):
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

        probs     = self._clean(tracker.get("prob", []))
        distances = self._clean(tracker.get("dist", []))
        raw_pats  = tracker.get("pattern", [])   # list[str] – empty if library not patched

        if len(probs) == 0:
            self._write_placeholder("Tracker contains no data points.")
            return

        # Decode per-prediction matched patterns (requires patched library)
        decoded_patterns = [self._decode_pattern_name(p) for p in raw_pats] if raw_pats else []

        # Build decoded prediction frame from runner
        decoded_preds, decoded_actuals, prefix_lengths = self._build_prediction_frame()

        has_patterns = len(decoded_patterns) == len(probs) and len(decoded_patterns) > 0
        has_preds    = decoded_preds is not None

        # RPIF distance distribution (always produced)
        self._plot_distances(distances)

        if has_patterns and has_preds:
            n = min(len(decoded_patterns), len(probs),
                    len(decoded_preds), len(decoded_actuals), len(prefix_lengths))
            dist_arr = distances if len(distances) == len(probs) else np.full(len(probs), np.nan)
            correct = [
                int(p == a) if p is not None and a is not None else None
                for p, a in zip(decoded_preds[:n], decoded_actuals[:n])
            ]
            self._save_matched_patterns_csv(
                decoded_patterns[:n], probs[:n], dist_arr[:n],
                decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n], correct,
            )
            self._save_top_patterns_summary(decoded_patterns[:n], correct)
            self._plot_top_matched_patterns(decoded_patterns[:n], correct)
            self._plot_activity_importance(decoded_patterns[:n], correct)
            self._plot_error_patterns(decoded_patterns[:n], correct)
            self._plot_accuracy_by_prefix_length(decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n])
            self._write_summary_report(
                probs[:n], dist_arr[:n],
                decoded_patterns[:n], correct,
                decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n],
            )

        elif has_preds:
            # Pattern names not captured – prediction-level stats only
            n = min(len(probs), len(decoded_preds), len(decoded_actuals), len(prefix_lengths))
            dist_arr = distances if len(distances) == len(probs) else np.full(len(probs), np.nan)
            correct = [
                int(p == a) if p is not None and a is not None else None
                for p, a in zip(decoded_preds[:n], decoded_actuals[:n])
            ]
            self._plot_accuracy_by_prefix_length(decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n])
            self._plot_confidence_distribution(probs)
            self._save_predictions_csv(probs[:n], dist_arr[:n], decoded_preds[:n],
                                       decoded_actuals[:n], prefix_lengths[:n], correct)
            self._write_summary_report_no_patterns(probs, dist_arr,
                                                   decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n])
        else:
            self._plot_confidence_distribution(probs)
            self._write_summary_report_minimal(probs, distances)

        print(f"[BEST Explainer] Artefacts written to {self.output_dir}")

    # ------------------------------------------------------------------
    # Tracker retrieval
    # ------------------------------------------------------------------

    def _get_tracker(self):
        if self.task == "nap":
            return getattr(self.model, "choice_tracker_nap", None)
        return getattr(self.model, "choice_tracker_rtp", None)

    @staticmethod
    def _clean(values) -> np.ndarray:
        arr = np.array([v for v in values if v is not None], dtype=float)
        return arr[np.isfinite(arr)]

    # ------------------------------------------------------------------
    # Pattern decoding
    # ------------------------------------------------------------------

    def _decode_pattern_name(self, name: str) -> str:
        """Convert comma-separated encoded-int string to 'Act A → Act B → Act C'.

        START/END padding tokens are stripped from the display.
        """
        if not name:
            return ""
        start_enc = getattr(self.model, "start_activity", None)
        end_enc   = getattr(self.model, "end_activity",   None)

        decoded = []
        for token in name.split(","):
            token = token.strip()
            try:
                idx = int(token)
            except ValueError:
                decoded.append(token)
                continue
            if (start_enc is not None and idx == start_enc) or \
               (end_enc   is not None and idx == end_enc):
                continue   # strip padding
            if self.runner is not None:
                label = self.runner._decode_activity(idx)
                decoded.append(str(label) if label is not None else f"[enc:{token}]")
            else:
                decoded.append(token)
        return " → ".join(decoded) if decoded else name

    # ------------------------------------------------------------------
    # Build decoded prediction frame from runner
    # ------------------------------------------------------------------

    def _build_prediction_frame(self):
        if self.runner is None:
            return None, None, None
        runner = self.runner
        try:
            predictions = runner.predictions
            if predictions is None or len(predictions) == 0:
                return None, None, None

            if self.task == "nap":
                decoded_preds = runner._decode_nap(predictions)
                actuals_enc   = getattr(runner.test_seq, "next_activities", None)
                if actuals_enc is None:
                    return None, None, None
                n = min(len(decoded_preds), len(actuals_enc))
                decoded_preds   = decoded_preds[:n]
                decoded_actuals = [runner._decode_activity(a) for a in actuals_enc[:n]]
            else:
                decoded_preds = runner._decode_rtp(predictions)
                actuals_enc   = getattr(runner.test_seq, "full_future_sequences", None)
                if actuals_enc is None:
                    return None, None, None
                n = min(len(decoded_preds), len(actuals_enc))
                decoded_preds   = decoded_preds[:n]
                decoded_actuals = []
                for seq in actuals_enc[:n]:
                    if seq is None:
                        decoded_actuals.append(None)
                    else:
                        decoded_actuals.append(", ".join(
                            str(runner._decode_activity(idx))
                            for idx in seq if idx is not None
                        ))

            padding_size = getattr(self.model, "_padding_size", 0)
            if hasattr(runner.test_seq, "relevant_prefixes") and runner.test_seq.relevant_prefixes:
                raw_lens       = [len(p["prefix"]) for p in runner.test_seq.relevant_prefixes[:n]]
                prefix_lengths = [max(0, raw - padding_size) for raw in raw_lens]
            else:
                prefix_lengths = list(range(n))

            return decoded_preds, decoded_actuals, prefix_lengths

        except Exception as e:
            print(f"[BEST Explainer] Warning: could not build prediction frame: {e}")
            return None, None, None

    # ------------------------------------------------------------------
    # Core explainability: matched-pattern charts
    # ------------------------------------------------------------------

    @staticmethod
    def _short_pattern(label: str, max_len: int = 72) -> str:
        text = " ".join(label.split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    def _pattern_stats(self, patterns: list, correct: list) -> dict:
        stats = defaultdict(lambda: {"count": 0, "hits": 0})
        for pat, c in zip(patterns, correct):
            if pat:
                stats[pat]["count"] += 1
                if c == 1:
                    stats[pat]["hits"] += 1
        return stats

    def _save_top_patterns_summary(self, patterns: list, correct: list) -> None:
        stats = self._pattern_stats(patterns, correct)
        if not stats:
            return

        rows = []
        for rank, (pat, s) in enumerate(
            sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)[:20], start=1
        ):
            acc = s["hits"] / s["count"] if s["count"] > 0 else 0.0
            rows.append({
                "rank": rank,
                "match_count": s["count"],
                "accuracy": round(acc, 4),
                "pattern": pat,
                "pattern_short": self._short_pattern(pat),
                "activity_count": len([a for a in pat.split("→") if a.strip()]),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "top_patterns_summary.csv"), index=False)
        with open(os.path.join(self.output_dir, "top_patterns_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    def _save_matched_patterns_csv(self, patterns, probs, distances,
                                   preds, actuals, prefix_lengths, correct):
        """One row per prediction with the actual matched pattern sequence."""
        df = pd.DataFrame({
            "prefix_length_events": prefix_lengths,
            "matched_pattern":      patterns,
            "pattern_confidence":   probs,
            "rpif_distance":        distances,
            "predicted_activity":   preds,
            "actual_activity":      actuals,
            "correct":              correct,
        })
        df.to_csv(os.path.join(self.output_dir, "matched_patterns.csv"), index=False)

    def _plot_top_matched_patterns(self, patterns: list, correct: list) -> None:
        stats = self._pattern_stats(patterns, correct)
        if not stats:
            return

        top        = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)[:15]
        full_labels = [p for p, _ in top][::-1]
        labels     = [self._short_pattern(p) for p in full_labels]
        counts     = [s["count"] for _, s in top][::-1]
        accuracies = [s["hits"] / s["count"] if s["count"] > 0 else 0 for _, s in top][::-1]
        bar_colors = [_CMAP(acc) for acc in accuracies]

        fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.6 + 2)))
        bars = ax.barh(range(len(labels)), counts, color=bar_colors,
                       edgecolor="white", alpha=0.9)
        for bar, acc, full in zip(bars, accuracies, full_labels):
            ax.text(bar.get_width() + max(counts) * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.0%}", va="center", fontsize=9, color="#333")
            if len(full) > len(self._short_pattern(full)):
                ax.text(0.01, bar.get_y() + bar.get_height() / 2, "…",
                        va="center", fontsize=8, color="#666")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Times matched across all test predictions")
        ax.set_title(
            "Top Matched Historical Patterns (top 15)\n"
            "Bar length = how often BEST used this subtrace  ·  "
            "color = share of correct predictions  ·  "
            "see top_patterns_summary table for full sequences"
        )
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "top_matched_patterns.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_activity_importance(self, patterns: list, correct: list) -> None:
        """Which activities appear most often across all matched patterns.

        Equivalent to feature importance: activities in many matched patterns
        drive most predictions. Color = accuracy when that activity is present.
        """
        act_stats = defaultdict(lambda: {"count": 0, "hits": 0, "total": 0})
        for pat, c in zip(patterns, correct):
            if not pat:
                continue
            for act in pat.split("→"):
                act = act.strip()
                if not act:
                    continue
                act_stats[act]["count"] += 1
                if c is not None:
                    act_stats[act]["total"] += 1
                    if c == 1:
                        act_stats[act]["hits"] += 1

        if not act_stats:
            return

        top        = sorted(act_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:25]
        labels     = [a for a, _ in top][::-1]
        counts     = [s["count"] for _, s in top][::-1]
        accuracies = [s["hits"] / s["total"] if s["total"] > 0 else 0 for _, s in top][::-1]
        bar_colors = [_CMAP(acc) for acc in accuracies]

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.45 + 2)))
        bars = ax.barh(range(len(labels)), counts, color=bar_colors,
                       edgecolor="white", alpha=0.9)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_width() + max(counts) * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.0%}", va="center", fontsize=9, color="#333")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Appearances across all matched patterns")
        ax.set_title(
            "Activity Importance in Matched Patterns\n"
            "Activities appearing most often in the patterns the model relies on  ·  "
            "color = prediction accuracy when this activity is present"
        )
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "activity_importance.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_error_patterns(self, patterns: list, correct: list) -> None:
        """Patterns with the highest error rate — where does the model go wrong?

        Shows up to 15 patterns by error rate (minimum 3–5 predictions each).
        These are the historical sequences that misled the model most often.
        """
        stats = defaultdict(lambda: {"count": 0, "errors": 0})
        for pat, c in zip(patterns, correct):
            if pat and c is not None:
                stats[pat]["count"] += 1
                if c == 0:
                    stats[pat]["errors"] += 1

        min_count = 5 if any(s["count"] >= 5 for s in stats.values()) else 3
        qualified = {p: s for p, s in stats.items() if s["count"] >= min_count}
        if not qualified:
            return

        error_rates = sorted(
            [(p, s["errors"] / s["count"], s["count"]) for p, s in qualified.items()],
            key=lambda x: x[1], reverse=True,
        )[:15]

        labels       = [p for p, _, _ in error_rates][::-1]
        rates        = [r for _, r, _ in error_rates][::-1]
        error_counts = [c for _, _, c in error_rates][::-1]

        fig, ax = plt.subplots(figsize=(13, max(6, len(labels) * 0.55 + 2)))
        bars = ax.barh(range(len(labels)), rates, color=_RED,
                       edgecolor="white", alpha=0.85)
        for bar, ec, rate in zip(bars, error_counts, rates):
            ax.text(min(bar.get_width() + 0.02, 1.12),
                    bar.get_y() + bar.get_height() / 2,
                    f"{ec} errors ({rate:.0%})", va="center", fontsize=9, color="#333")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Error rate (fraction of predictions that were wrong)")
        ax.set_title(
            "Patterns with Highest Error Rate\n"
            "These are the historical sequences that misled the model most often"
        )
        ax.set_xlim(0, 1.3)
        ax.axvline(0.5, color="#888", linestyle="--", linewidth=1, label="50% error")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "error_patterns.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Standard support charts
    # ------------------------------------------------------------------

    def _plot_accuracy_by_prefix_length(self, preds, actuals, prefix_lengths) -> None:
        """How accuracy changes as more events are seen before the prediction."""
        acc_by_len = defaultdict(list)
        for p, a, l in zip(preds, actuals, prefix_lengths):
            if p is not None and a is not None:
                acc_by_len[l].append(int(p == a))

        if not acc_by_len:
            return

        lengths_sorted = sorted(acc_by_len.keys())
        accuracies = [np.mean(acc_by_len[l]) for l in lengths_sorted]
        counts     = [len(acc_by_len[l])     for l in lengths_sorted]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(lengths_sorted, accuracies, "o-", color=_BLUE,
                 linewidth=2, markersize=6, label="Accuracy")
        ax1.set_xlabel("Real events seen before prediction")
        ax1.set_ylabel("Accuracy", color=_BLUE)
        ax1.tick_params(axis="y", labelcolor=_BLUE)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Prediction Accuracy by Prefix Length\n"
                      "How reliable is BEST at different points in a case?")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        ax2.bar(lengths_sorted, counts, alpha=0.2, color=_GREEN, label="# samples")
        ax2.set_ylabel("# samples", color=_GREEN)
        ax2.tick_params(axis="y", labelcolor=_GREEN)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower right")

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "accuracy_by_prefix_length.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_distances(self, distances: np.ndarray) -> None:
        if len(distances) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=30, color=_RED, edgecolor="white", alpha=0.85)
        ax.axvline(distances.mean(), color="#333", linestyle="--", linewidth=1.2,
                   label=f"Mean ({distances.mean():.3f})")
        ax.set_xlabel("RPIF distance  (lower = test case closely matches a training pattern)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of RPIF Distance Scores\n"
                     "Low distance = high confidence the matched pattern is a good fit")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "rpif_distances.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_confidence_distribution(self, probs: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(probs, bins=30, color=_BLUE, edgecolor="white", alpha=0.85)
        ax.axvline(probs.mean(), color="#333", linestyle="--", linewidth=1.2,
                   label=f"Mean ({probs.mean():.3f})")
        ax.set_xlabel("Pattern confidence (conditional probability of chosen pattern)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Pattern Confidence Scores")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pattern_confidence.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fallback CSV (no pattern names)
    # ------------------------------------------------------------------

    def _save_predictions_csv(self, probs, distances, preds, actuals, prefix_lengths, correct):
        df = pd.DataFrame({
            "prefix_length_events": prefix_lengths,
            "predicted_activity":   preds,
            "actual_activity":      actuals,
            "correct":              correct,
            "pattern_confidence":   probs,
            "rpif_distance":        distances,
        })
        df.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)

    # ------------------------------------------------------------------
    # Summary reports
    # ------------------------------------------------------------------

    def _write_summary_report(self, probs, distances, patterns, correct,
                               preds, actuals, prefix_lengths) -> None:
        n = len(probs)
        correct_flags = [c for c in correct if c is not None]
        overall_acc   = np.mean(correct_flags) if correct_flags else float("nan")

        pat_stats = defaultdict(lambda: {"count": 0, "hits": 0})
        for pat, c in zip(patterns, correct):
            if pat:
                pat_stats[pat]["count"] += 1
                if c == 1:
                    pat_stats[pat]["hits"] += 1

        top_by_count = sorted(pat_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        worst_by_err = sorted(
            [(p, (s["count"] - s["hits"]) / s["count"], s["count"])
             for p, s in pat_stats.items() if s["count"] >= 3],
            key=lambda x: x[1], reverse=True,
        )[:5]

        len_acc = defaultdict(list)
        for p, a, l in zip(preds, actuals, prefix_lengths):
            if p is not None and a is not None:
                len_acc[l].append(int(p == a))

        lines = [
            "=" * 68,
            f"BEST Pattern Explainability Report  ·  {self.task.upper()} task",
            "=" * 68,
            f"Total predictions analysed : {n:,}",
            f"Overall accuracy           : {overall_acc:.4f}  ({overall_acc * 100:.2f}%)",
            f"Unique patterns matched    : {len(pat_stats):,}",
            "",
            "Pattern confidence (probability of the matched pattern):",
            f"  Mean    : {probs.mean():.4f}",
            f"  Median  : {float(np.median(probs)):.4f}",
            f"  Std     : {probs.std():.4f}",
            f"  >= 0.99 : {(probs >= 0.99).mean() * 100:.1f}% of predictions",
            f"  <  0.90 : {(probs < 0.90).mean() * 100:.1f}% of predictions",
        ]

        d = distances[np.isfinite(distances)] if len(distances) > 0 else np.array([])
        if len(d) > 0:
            lines += [
                "",
                "RPIF distance (lower = test case closely matched a training pattern):",
                f"  Mean    : {d.mean():.4f}",
                f"  Median  : {float(np.median(d)):.4f}",
                f"  Std     : {d.std():.4f}",
            ]

        lines += ["", "Top 10 most-matched historical patterns:"]
        for pat, s in top_by_count:
            acc = s["hits"] / s["count"] if s["count"] > 0 else float("nan")
            lines.append(f"  [{s['count']:4d}x, acc={acc:.0%}]  {pat}")

        if worst_by_err:
            lines += ["", "Top 5 patterns with highest error rate (min 3 predictions):"]
            for pat, err_rate, count in worst_by_err:
                lines.append(f"  [err={err_rate:.0%}, n={count}]  {pat}")

        lines += ["", "Accuracy by prefix length (real events seen before prediction):"]
        for l in sorted(len_acc.keys()):
            vals = len_acc[l]
            lines.append(f"  {l:2d} events : {np.mean(vals):.3f}  ({len(vals)} samples)")

        lines += ["", "=" * 68]
        path = os.path.join(self.output_dir, "pattern_summary_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_summary_report_no_patterns(self, probs, distances, preds,
                                           actuals, prefix_lengths) -> None:
        correct_flags = [
            int(p == a) for p, a in zip(preds, actuals)
            if p is not None and a is not None
        ]
        overall_acc = np.mean(correct_flags) if correct_flags else float("nan")
        d = distances[np.isfinite(distances)] if len(distances) > 0 else np.array([])

        lines = [
            "=" * 68,
            f"BEST Explainability Report  ·  {self.task.upper()} task",
            "=" * 68,
            "(Pattern names unavailable — library patch not yet applied)",
            f"Total predictions analysed : {len(probs):,}",
            f"Overall accuracy           : {overall_acc:.4f}  ({overall_acc * 100:.2f}%)",
            "",
            "Pattern confidence:",
            f"  Mean   : {probs.mean():.4f}",
            f"  Median : {float(np.median(probs)):.4f}",
            f"  Std    : {probs.std():.4f}",
        ]
        if len(d) > 0:
            lines += ["", "RPIF distance:",
                      f"  Mean   : {d.mean():.4f}",
                      f"  Median : {float(np.median(d)):.4f}"]
        lines += ["", "=" * 68]
        path = os.path.join(self.output_dir, "pattern_summary_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_summary_report_minimal(self, probs, distances) -> None:
        d = distances[np.isfinite(distances)] if len(distances) > 0 else np.array([])
        lines = [
            "=" * 68,
            f"BEST Explainability Report  ·  {self.task.upper()} task",
            "=" * 68,
            f"Total predictions analysed : {len(probs):,}",
            "",
            "Pattern confidence:",
            f"  Mean   : {probs.mean():.4f}",
            f"  Median : {float(np.median(probs)):.4f}",
            f"  Std    : {probs.std():.4f}",
        ]
        if len(d) > 0:
            lines += ["", "RPIF distance:",
                      f"  Mean   : {d.mean():.4f}",
                      f"  Median : {float(np.median(d)):.4f}"]
        lines += ["", "=" * 68]
        path = os.path.join(self.output_dir, "pattern_summary_report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------------------

    def _write_placeholder(self, message: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"BEST Pattern Analysis\n\n{message}",
                ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pattern_summary.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

