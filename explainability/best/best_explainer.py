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
        df_rich, n_samples = self._build_prediction_frame()

        if df_rich is not None:
            # New artifact generation (as requested in Best Visualizaiton.docx)
            self._save_summary_json(df_rich)
            self._save_top_patterns_csv() # This is the GLOBAL dictionary
            self._save_pattern_analysis_json(df_rich)

            # Rich charts using actual predictions and labels (Legacy/Enhanced)
            self._plot_accuracy_by_prefix_length(df_rich)
            self._plot_confidence_by_class(df_rich)
            self._plot_activity_distribution(df_rich)
            # DELETED: self._save_top_patterns_enriched(df_rich, distances) 
            # This was overwriting top_patterns.csv with per-case results.
            # The per-case results are already in best_predictions.csv.
            self._write_summary_report_enriched(df_rich, distances)
        else:
            # Fallback: basic tracker charts (pattern length histogram may be degenerate)
            lengths = self._clean(tracker.get("len", []))
            self._plot_probabilities_basic(probs)
            self._plot_lengths_basic(lengths)
            self._save_top_patterns_basic(probs, lengths, distances)
            self._write_summary_report_basic(probs, lengths, distances)

        print(f"[BEST Explainer] Artefacts written to {self.output_dir}")

    # ------------------------------------------------------------------
    # New Artifact Generation (Phase 3)
    # ------------------------------------------------------------------

    def _save_summary_json(self, df: pd.DataFrame) -> None:
        """Saves summary.json with global metrics and prefix stats."""
        import json
        
        correct_flags = (df["true_next"] == df["pred_next"]).dropna()
        overall_acc = correct_flags.mean() if not correct_flags.empty else 0.0
        avg_conf = df["confidence"].dropna().mean() if not df["confidence"].dropna().empty else 0.0
        
        prefix_stats = []
        acc_by_len = defaultdict(list)
        for _, row in df.iterrows():
            if row["true_next"] is not None and row["pred_next"] is not None:
                acc_by_len[row["case_index"]].append(int(row["true_next"] == row["pred_next"]))
        
        for length in sorted(acc_by_len.keys()):
            vals = acc_by_len[length]
            prefix_stats.append({
                "prefix_length": int(length),
                "accuracy": float(np.mean(vals)),
                "sample_count": int(len(vals))
            })

        summary = {
            "task_type": self.task,
            "overall_accuracy": float(overall_acc),
            "average_confidence": float(avg_conf),
            "total_test_cases": int(len(df)),
            "prefix_stats": prefix_stats
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def _save_top_patterns_csv(self) -> None:
        """Saves top_patterns.csv as a global dictionary of patterns."""
        # We need to extract all unique patterns from the model's fitted patterns
        # self.model._unpruned_nodes contains patterns by stage
        
        all_patterns = []
        seen_patterns = {}
        
        # Each "node" in unpruned_nodes is a pattern
        for stage, nodes in getattr(self.model, "_unpruned_nodes", {}).items():
            for node_idx, node in nodes.items():
                pattern_seq = node["name"] # comma-separated string
                if not pattern_seq: continue
                
                if pattern_seq not in seen_patterns:
                    # Try to decode the sequence
                    try:
                        seq_indices = [int(idx) for idx in pattern_seq.split(",")]
                        decoded_seq = [self.runner._decode_activity(idx) for idx in seq_indices]
                        
                        # Predicted next activity for this pattern
                        # In BEST, the center activity is predicted. 
                        # Actually, _pred_for_process_stage says:
                        # pred = picked_pattern[math.floor(len(picked_pattern)/2):]
                        # So it predicts the center activity and everything after it?
                        # For NAP, it's picked_pattern[1] if len is 3?
                        # Let's use the center activity for "what it predicts" or similar logic
                        center_idx = len(seq_indices) // 2
                        predicted_next = decoded_seq[center_idx] if center_idx < len(decoded_seq) else None
                        
                        seen_patterns[pattern_seq] = {
                            "pattern_id": len(seen_patterns) + 1,
                            "sequence": json.dumps(decoded_seq),
                            "predicted_next_activity": predicted_next,
                            "global_frequency": int(node.get("freq", 0)),
                            "global_accuracy": float(node.get("prob", 0)), # 'prob' in BEST node is conditional prob
                            "avg_confidence": float(node.get("prob", 0))
                        }
                    except:
                        continue
        
        df_patterns = pd.DataFrame(list(seen_patterns.values()))
        if not df_patterns.empty:
            df_patterns = df_patterns.sort_values("global_frequency", ascending=False)
            
        path = os.path.join(self.output_dir, "top_patterns.csv")
        df_patterns.to_csv(path, index=False)

    def _save_pattern_analysis_json(self, df_rich: pd.DataFrame) -> None:
        """Saves pattern_analysis.json (The Heatmap Bridge)."""
        import json
        
        # Build a map for quick lookup of the full sequence for a case/index
        seq_map = {}
        for _, row in df_rich.iterrows():
            cid = str(row["case_id"])
            idx = int(row["case_index"])
            if cid not in seq_map:
                seq_map[cid] = {}
            seq_map[cid][idx] = json.loads(row["sequence"])

        # This requires the all_matches_tracker from BESTPredictorCustom
        tracker = getattr(self.model, "all_matches_tracker", [])
        if not tracker:
            with open(os.path.join(self.output_dir, "pattern_analysis.json"), "w") as f:
                json.dump({}, f)
            return

        pattern_id_map = {}
        id_counter = 1
        for stage, nodes in getattr(self.model, "_unpruned_nodes", {}).items():
            for node in nodes.values():
                if node["name"] and node["name"] not in pattern_id_map:
                    pattern_id_map[node["name"]] = id_counter
                    id_counter += 1

        analysis = {}
        for entry in tracker:
            case_id = f"case_{entry['case_id']}"
            # Standardize case_id for internal lookup
            raw_cid = str(entry['case_id'])
            
            # Use 1-indexing if not already ensured
            case_index = int(entry['case_index'])
            if case_index == 0: continue # Skip initial padding/start
            
            index_key = f"index_{case_index}"
            
            if case_id not in analysis:
                analysis[case_id] = {}
            
            full_seq = seq_map.get(raw_cid, {}).get(case_index, [])
            
            all_pattern_matches = []
            matches = entry.get("matches", [])
            for m in matches:
                pattern_name = m["name"]
                pid = pattern_id_map.get(pattern_name)
                if pid:
                    pattern_len = len(pattern_name.split(","))
                    # Offset logic: center of pattern is last activity in prefix
                    # Start is case_index - L//2
                    start_offset = max(0, case_index - (pattern_len // 2) - 1)
                    end_offset = case_index + (pattern_len // 2) - 1
                    
                    all_pattern_matches.append({
                        "pattern_id": pid,
                        "start_offset": int(start_offset),
                        "end_offset": int(end_offset),
                        "frequency": int(m.get("freq", 0))
                    })
            
            analysis[case_id][index_key] = {
                "full_sequence": full_seq,
                "all_pattern_matches": all_pattern_matches
            }

        path = os.path.join(self.output_dir, "pattern_analysis.json")
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2)

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
            (df, n) where df has columns [case_id, case_index, sequence, true_next, pred_next, confidence]
        """
        if self.runner is None:
            return None, 0

        runner = self.runner
        try:
            predictions = runner.predictions
            if predictions is None or len(predictions) == 0:
                return None, 0

            if self.task != "nap":
                return None, 0

            # Use the same logic as save_results in predictor.py
            prefixes = runner.test_seq.relevant_prefixes
            n = min(len(prefixes), len(predictions))
            actuals_enc = getattr(runner.test_seq, "next_activities", [None] * n)
            tracker = getattr(self.model, "choice_tracker_nap", {})
            probs = tracker.get("prob", [None] * n)
            padding_size = getattr(self.model, "_padding_size", 0)

            rows = []
            import json
            for i in range(n):
                prefix_data = prefixes[i]
                raw_seq = prefix_data["prefix"]
                real_seq_enc = raw_seq[padding_size:]
                decoded_seq = [runner._decode_activity(a) for a in real_seq_enc]
                
                rows.append({
                    "case_id": str(prefix_data["case_id"]),
                    "case_index": len(real_seq_enc),
                    "sequence": json.dumps(decoded_seq),
                    "true_next": runner._decode_activity(actuals_enc[i]),
                    "pred_next": runner._decode_activity(predictions[i]),
                    "confidence": probs[i] if i < len(probs) else None
                })
            
            return pd.DataFrame(rows), n

        except Exception as e:
            print(f"[BEST Explainer] Warning: could not build prediction frame: {e}")
            return None, 0

    # ------------------------------------------------------------------
    # Rich charts (runner available)
    # ------------------------------------------------------------------

    def _plot_accuracy_by_prefix_length(self, df):
        """Line chart: accuracy vs number of real events seen before prediction."""
        acc_by_len = defaultdict(list)
        for _, row in df.iterrows():
            if row["true_next"] is not None and row["pred_next"] is not None:
                acc_by_len[row["case_index"]].append(int(row["true_next"] == row["pred_next"]))

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
        ax1.set_title("BEST - Prediction Accuracy by Prefix Length")
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

    def _plot_confidence_by_class(self, df: pd.DataFrame) -> None:
        """Horizontal bar chart: mean pattern confidence per predicted activity class."""
        conf_by_class = defaultdict(list)
        for _, row in df.iterrows():
            if row["pred_next"] is not None and row["confidence"] is not None:
                conf_by_class[row["pred_next"]].append(row["confidence"])

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
        ax.set_title("Pattern Confidence by Predicted Activity")
        ax.set_xlim(0, 1.05)
        
        overall_mean = df["confidence"].dropna().mean()
        ax.axvline(overall_mean, color="red", linestyle="--", linewidth=1.2,
                   label=f"Overall mean ({overall_mean:.3f})")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, "confidence_by_predicted_class.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_activity_distribution(self, df: pd.DataFrame) -> None:
        """Side-by-side bar chart: predicted vs actual activity frequencies."""
        valid_df = df.dropna(subset=["pred_next", "true_next"])
        if valid_df.empty:
            return

        pred_counts = Counter(valid_df["pred_next"])
        actual_counts = Counter(valid_df["true_next"])

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
        ax.set_title("Actual vs Predicted Activity Distribution")
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
        ax.set_title("Distribution of RPIF Distance Scores")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "rpif_distances.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Enriched CSV + summary report (runner available)
    # ------------------------------------------------------------------

    def _save_top_patterns_enriched(
        self,
        df_rich: pd.DataFrame,
        distances: np.ndarray,
    ) -> None:
        """Save per-prediction table with confidence, correctness, and prefix length."""
        n = len(df_rich)
        dist_arr = distances if len(distances) == n else np.full(n, np.nan)

        correct = [
            int(p == a) if p is not None and a is not None else None
            for p, a in zip(df_rich["pred_next"], df_rich["true_next"])
        ]

        df_out = pd.DataFrame({
            "case_id": df_rich["case_id"],
            "case_index": df_rich["case_index"],
            "sequence": df_rich["sequence"],
            "predicted_activity": df_rich["pred_next"],
            "actual_activity": df_rich["true_next"],
            "correct": correct,
            "pattern_confidence": df_rich["confidence"],
            "rpif_distance": dist_arr,
        })

        path = os.path.join(self.output_dir, "top_patterns.csv")
        df_out.to_csv(path, index=False)

    def _write_summary_report_enriched(
        self,
        df_rich: pd.DataFrame,
        distances: np.ndarray,
    ) -> None:
        """Write enriched summary with accuracy, per-class breakdown, prefix length stats."""
        n = len(df_rich)

        correct_flags = [
            int(p == a) for p, a in zip(df_rich["pred_next"], df_rich["true_next"])
            if p is not None and a is not None
        ]
        overall_acc = np.mean(correct_flags) if correct_flags else float("nan")

        # Per actual-class accuracy
        class_correct = defaultdict(list)
        for p, a in zip(df_rich["pred_next"], df_rich["true_next"]):
            if p is not None and a is not None:
                class_correct[a].append(int(p == a))

        # Accuracy by prefix length
        len_correct = defaultdict(list)
        for p, a, l in zip(df_rich["pred_next"], df_rich["true_next"], df_rich["case_index"]):
            if p is not None and a is not None:
                len_correct[l].append(int(p == a))

        probs = df_rich["confidence"].dropna()

        lines = [
            "=" * 60,
            f"BEST Pattern-Based Explainability - {self.task.upper()} task",
            "=" * 60,
            f"Total predictions analysed : {n:,}",
            f"Overall accuracy           : {overall_acc:.4f} ({overall_acc * 100:.2f}%)",
            "",
            "Pattern confidence (probability of chosen pattern):",
        ]
        if not probs.empty:
            lines += [
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
        ax.set_title("Distribution of chosen-pattern probabilities")
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
        ax.set_title("Distribution of chosen-pattern lengths")
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
