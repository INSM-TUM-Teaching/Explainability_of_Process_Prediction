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
        if self.task == "rtp":
            return self._explain_rtp()

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
            pattern_id_map = self._save_top_patterns_csv() # This is the GLOBAL dictionary
            self._save_pattern_analysis_json(df_rich, pattern_id_map)

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
            "total_test_cases": int(df["case_id"].nunique()),
            "prefix_stats": prefix_stats
        }

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def _extract_patterns_from_tree(self) -> dict:
        """Helper to recursively extract all patterns from _stage_trees since _unpruned_nodes lacks 'freq'."""
        seen_patterns = {}
        
        def _walk_tree(node):
            if not node: return
            name = node.get("name")
            if name and name not in seen_patterns:
                seen_patterns[name] = node
            for child in node.get("children", []):
                _walk_tree(child)

        for stage, tree in getattr(self.model, "_stage_trees", {}).items():
            _walk_tree(tree)
            
        return seen_patterns

    def _save_top_patterns_csv(self) -> dict:
        """Saves top_patterns.csv as a global dictionary of patterns.
        
        Returns:
            dict: Mapping from pattern sequence (str) to pattern_id (int)
        """
        import json
        from collections import defaultdict
        
        tree_patterns = self._extract_patterns_from_tree()
        
        # Map internal sequences to their visible properties
        internal_to_visible = {}
        
        for pattern_seq, node in tree_patterns.items():
            if not pattern_seq: continue
            
            try:
                seq_indices = [int(idx) for idx in pattern_seq.split(",")]
                decoded_seq = [self.runner._decode_activity(idx) for idx in seq_indices]
                
                # Predicted next activity for this pattern
                center_idx = len(seq_indices) // 2
                predicted_next = decoded_seq[center_idx + 1] if (center_idx + 1) < len(decoded_seq) else None
                
                # Check if prediction is None, empty, whitespace-only, or a padding token
                if not predicted_next or not str(predicted_next).strip() or predicted_next in ["START", "END"]:
                    continue

                # Extract only the history (the left side of the pattern up to the center)
                history_seq = decoded_seq[:center_idx + 1]

                # Filter out padding tokens for UI
                filtered_seq = [act for act in history_seq if act not in ["START", "END"]]
                
                # Single-activity histories are valid, but 0-length is not
                if len(filtered_seq) < 1:
                    continue

                visible_seq_str = json.dumps(filtered_seq)
                internal_to_visible[pattern_seq] = {
                    "visible_seq": visible_seq_str,
                    "pred": predicted_next,
                    "freq": int(node.get("freq", 0)),
                    "prob": float(node.get("prob", 0))
                }
            except Exception:
                continue

        # Aggregate by visible sequence
        aggregated = {}
        for p_seq, data in internal_to_visible.items():
            vis_seq = data["visible_seq"]
            pred = data["pred"]
            freq = data["freq"]
            prob = data["prob"]
            
            if vis_seq not in aggregated:
                aggregated[vis_seq] = {
                    "preds": defaultdict(int),
                    "total_freq": 0,
                    "sum_prob": 0.0,
                    "count": 0
                }
                
            aggregated[vis_seq]["preds"][pred] += freq
            aggregated[vis_seq]["total_freq"] += freq
            aggregated[vis_seq]["sum_prob"] += prob
            aggregated[vis_seq]["count"] += 1

        final_patterns = []
        visible_to_id = {}
        pattern_id_map = {}
        
        pid_counter = 1
        for vis_seq, agg_data in aggregated.items():
            visible_to_id[vis_seq] = pid_counter
            
            # Find the most predicted output for this sequence
            best_pred = max(agg_data["preds"].items(), key=lambda x: x[1])[0] if agg_data["preds"] else None
            avg_prob = agg_data["sum_prob"] / agg_data["count"] if agg_data["count"] > 0 else 0.0
            
            final_patterns.append({
                "pattern_id": pid_counter,
                "sequence": vis_seq,
                "predicted_next_activity": best_pred,
                "global_frequency": agg_data["total_freq"],
                "global_accuracy": avg_prob,
                "avg_confidence": avg_prob
            })
            pid_counter += 1
            
        # Map internal sequences to the assigned PID so the heatmap bridge knows which ID to use
        for p_seq, data in internal_to_visible.items():
            pattern_id_map[p_seq] = visible_to_id[data["visible_seq"]]
        
        df_patterns = pd.DataFrame(final_patterns)
        if not df_patterns.empty:
            df_patterns = df_patterns.sort_values("global_frequency", ascending=False)
            
        path = os.path.join(self.output_dir, "top_patterns.csv")
        df_patterns.to_csv(path, index=False)
        
        return pattern_id_map


    def _save_pattern_analysis_json(self, df_rich: pd.DataFrame, pattern_id_map: dict) -> None:
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

        tracker = getattr(self.model, "all_matches_tracker", [])
        if not tracker:
            with open(os.path.join(self.output_dir, "pattern_analysis.json"), "w") as f:
                json.dump({}, f)
            return

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
                    # BEST patterns match with their center as the prediction.
                    # A pattern of length L has (L // 2) activities to the left of the center.
                    # These (L // 2) activities are matched against the prefix.
                    num_prefix_elements = pattern_len // 2
                    start_offset = max(0, case_index - num_prefix_elements)
                    end_offset = case_index - 1
                    
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
                
                case_index = len(real_seq_enc)
                if case_index == 0:
                    continue

                decoded_seq = [runner._decode_activity(a) for a in real_seq_enc]
                # Filter out START/END for the UI sequence
                filtered_seq = [a for a in decoded_seq if a not in ["START", "END"]]
                
                rows.append({
                    "case_id": str(prefix_data["case_id"]),
                    "case_index": len(filtered_seq), # Update case_index to reflect true length
                    "sequence": json.dumps(filtered_seq),
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

    def _plot_accuracy_by_prefix_length_rtp(self, preds, actuals, prefix_lengths) -> None:
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
        ax1.plot(lengths_sorted, accuracies, "o-", color="#4C72B0",
                 linewidth=2, markersize=6, label="Accuracy")
        ax1.set_xlabel("Real events seen before prediction")
        ax1.set_ylabel("Accuracy", color="#4C72B0")
        ax1.tick_params(axis="y", labelcolor="#4C72B0")
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Prediction Accuracy by Prefix Length\n"
                      "How reliable is BEST at different points in a case?")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        ax2.bar(lengths_sorted, counts, alpha=0.2, color="#55A868", label="# samples")
        ax2.set_ylabel("# samples", color="#55A868")
        ax2.tick_params(axis="y", labelcolor="#55A868")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower right")

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

    # --- RTP Specific Methods ---
    def _explain_rtp(self) -> None:
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
        decoded_preds, decoded_actuals, prefix_lengths = self._build_prediction_frame_rtp()

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
            self._plot_accuracy_by_prefix_length_rtp(decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n])
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
            self._plot_accuracy_by_prefix_length_rtp(decoded_preds[:n], decoded_actuals[:n], prefix_lengths[:n])
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


    def _build_prediction_frame_rtp(self):
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

    def _short_pattern(label: str, max_len: int = 72) -> str:
        text = " ".join(label.split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

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
