import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from best4ppm.models.best import BESTPredictor, Task
from best4ppm.data.sequencedata import SequenceData
from best4ppm.eval.evaluator import NAPEvaluator, RTPEvaluator

class BESTPredictorCustom(BESTPredictor):
    """Custom BESTPredictor that captures all matching patterns during prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_matches_tracker = [] # List of {'case_id': ..., 'case_index': ..., 'matches': [...]}
        self.outcome_mapping = None # Maps final activity (or trace token) to its actual outcome string
        self.prefix_to_outcome_train = None # Used for outcome mapping directly from training prefixes if needed.

    def _pred_for_process_stage(self, eval_pattern_size: int, stage: int, sequence: list[int], verbose: bool = False) -> tuple[list[int], dict]:
        # We call the original but also capture all matching patterns
        # The original _pred_for_process_stage calls extract_matching_patterns
        
        import math
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)

        # Re-implementing parts of the logic to capture ALL matching patterns
        all_applying_children = self.extract_matching_patterns(stage, sequence)
        
        # Original logic to filter single activity patterns
        all_applying_children = all_applying_children[1:]
        lens_all_children = [len(p['name'].split(',')) for p in all_applying_children]
        applying_children = [c for c, c_len in zip(all_applying_children, lens_all_children) if c_len <= eval_pattern_size]

        # Store these for pattern_analysis.json
        # Store temporarily to match with case identifiers in the predictor loop for pattern analysis
        self._last_all_matches = applying_children

        if len(applying_children) == 0:
            return [], {}

        # Rest of original logic to pick the best one
        # Select the best matching child pattern based on probabilities and lengths
        probs = [p['prob'] for p in applying_children]
        global_probs = [p['global_prob'] for p in applying_children]
        dists = [p['total_log_rpif_dist'] for p in applying_children]
        lens_children = [len(p['name'].split(',')) for p in applying_children]

        max_prob = max(probs)
        argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]
        argmax_prob_children_lens = [lens_children[idx] for idx in argmax_prob_indices]
        max_len = max(argmax_prob_children_lens)
        argmax_prob_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_prob_indices]
        argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
        min_dist_max_len = min(argmax_prob_argmax_len_dists)
        argmax_prob_argmax_len_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist_max_len and idx in argmax_prob_argmax_len_indices]

        candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmax_len_argmin_dist_indices]

        if len(argmax_prob_argmax_len_argmin_dist_indices) > 1:
            picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
        else:
            picked_child = candidate_children[0]

        picked_pattern = [int(act) for act in picked_child['name'].split(',')]
        pred = picked_pattern[math.floor(len(picked_pattern)/2):]

        return pred, {'prob':picked_child['prob'], 'len':len(picked_pattern), 'dist':picked_child['total_log_rpif_dist']}

class BESTRunner:
    """Adapter that wraps BESTPredictor + SequenceData into the pipeline interface."""

    def __init__(self, config: dict, task: str, target_column: str = "Activity"):
        """
        Args:
            config: dict with keys:
                max_pattern_size_train, max_pattern_size_eval,
                process_stage_width_percentage, min_freq,
                break_buffer, filter_sequences, ncores
            task: 'nap' (next activity), 'rtp' (remaining trace), or 'outcome'
        """
        self.config = config
        self.task = task  # 'nap' or 'rtp' or 'outcome'
        self.target_column = target_column

        self.model: BESTPredictor | None = None
        self.train_seq: SequenceData | None = None
        self.test_seq: SequenceData | None = None
        self.predictions = None
        self._act_encoder = None
        self.test_outcomes = None # Actual outcomes for test prefixes
        self.test_prefixes = None # The prefixes evaluated
        self.train_outcomes_mapping = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2,
                     prefix_lengths: list | None = None, max_len: int | None = None) -> None:
        """Build SequenceData objects and perform the train/test split.

        Args:
            df: DataFrame with standardised columns CaseID, Activity, Timestamp.
            test_size: Fraction of cases reserved for testing (default 0.2).
            prefix_lengths: Optional list of specific prefix lengths to evaluate
                            (e.g. [3, 5, 7]). If None, all prefixes are used.
        """
        self.prefix_lengths = prefix_lengths
        from utils.outcome_utils import extract_case_outcomes

        if self.task == "outcome":
            # Extract case outcomes and map to the sequence data
            df_outcomes = extract_case_outcomes(df, 'CaseID', 'Timestamp', self.target_column)
            self.train_outcomes_mapping = df_outcomes.to_dict()
            
            # Append OUTCOME event
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            new_rows = []
            for case_id, outcome in self.train_outcomes_mapping.items():
                last_event = df[df['CaseID'] == case_id].iloc[-1].copy()
                last_event['Activity'] = f"OUTCOME_{outcome}"
                last_event['Timestamp'] = last_event['Timestamp'] + pd.Timedelta(seconds=1)
                new_rows.append(last_event)
                
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df.sort_values(by=['CaseID', 'Timestamp']).reset_index(drop=True)

        seq = SequenceData(
            data=df,
            case_identifier="CaseID",
            activity_identifier="Activity",
            timestamp_identifier="Timestamp",
        )
        train_pct = 1.0 - test_size
        self.train_seq, self.test_seq = seq.train_test_split(train_pct=train_pct)

        # Apply max_len cap if specified (BEST natively uses all prefixes otherwise)
        if max_len is not None:
            # We filter relevant_prefixes for both train and test directly
            for s in [self.train_seq, self.test_seq]:
                if s and hasattr(s, 'relevant_prefixes'):
                    s.relevant_prefixes = [p for p in s.relevant_prefixes if len(p['prefix']) <= max_len]

        self._act_encoder = None

    # ------------------------------------------------------------------
    # Model fit
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Initialise and fit the BESTPredictor on the training split."""
        if self.train_seq is None or self.test_seq is None:
            raise RuntimeError("Call prepare_data() before fit().")

        max_pattern_size = self.config.get("max_pattern_size_train", 21)
        process_stage_width = self.config.get("process_stage_width_percentage", 0.2)
        min_freq = self.config.get("min_freq", 1e-14)

        self.model = BESTPredictorCustom(
            max_pattern_size=max_pattern_size,
            process_stage_width_percentage=process_stage_width,
            min_freq=min_freq,
            prune_func=None,
        )
        self.model.load_data(self.train_seq, self.test_seq)
        self._prepare_train_data()
        self._act_encoder = getattr(self.train_seq, "act_encoder", None)
        if self._act_encoder is None:
            raise RuntimeError("BEST training data was prepared without an activity encoder.")
        self.model.fit()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Run prepare_test and predict; stores results in self.predictions."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        filter_seqs = self.config.get("filter_sequences", True)
        eval_pattern_size = self.config.get("max_pattern_size_eval", 21)
        break_buffer = self.config.get("break_buffer", 1.2)
        ncores = self.config.get("ncores", 1)

        self._prepare_test_data(filter_sequences=filter_seqs)
        
        if ncores == 1 and self.task in ["nap", "rtp"]:
            from tqdm import tqdm
            self.predictions = []
            padding_size = getattr(self.model, "_padding_size", 0)
            self.model.all_matches_tracker = []

            for prefix in tqdm(self.test_seq.relevant_prefixes, desc=f"[BEST] Predicting {self.task.upper()} (Custom)"):
                prefix_sequence = prefix['prefix']
                
                # First, capture the matches for the *true* prefix sequence
                # by doing a single step prediction (works for both tasks to extract matching trees)
                self.model._predict_activity(prefix=prefix_sequence, eval_pattern_size=eval_pattern_size)
                matches = getattr(self.model, "_last_all_matches", [])

                if self.task == "nap":
                    pred_output = self.model._predict_activity(prefix=prefix_sequence,
                                                           eval_pattern_size=eval_pattern_size)
                else: # rtp
                    max_prefix_len = max([len(p['prefix']) for p in self.test_seq.relevant_prefixes])
                    break_after_seq_len = max_prefix_len * break_buffer
                    pred_output = self.model._predict_sequence(prefix=prefix_sequence, 
                                                              eval_pattern_size=eval_pattern_size, 
                                                              break_after_seq_len=break_after_seq_len)
                    
                    if filter_seqs:
                        try:
                            from best4ppm.util.sequence_utils import _filter_start_end
                            pred_output = _filter_start_end(pred_output, self.model.start_activity, self.model.end_activity)
                        except Exception:
                            pass

                self.predictions.append(pred_output)
                
                real_seq_enc = prefix_sequence[padding_size:]
                decoded_sequence = [self._decode_activity(a) for a in real_seq_enc]
                filtered_seq = [a for a in decoded_sequence if a not in ["START", "END"]]
                
                self.model.all_matches_tracker.append({
                    "case_id": str(prefix['case_id']),
                    "case_index": len(filtered_seq),
                    "matches": matches
                })
        else:
            # Fallback to original multi-core or RTP prediction (without all-matches tracking for now)
            task_arg = "rtp" if self.task == "outcome" else self.task
            self.predictions = self.model.predict(
                eval_pattern_size=eval_pattern_size,
                task=task_arg,
                break_buffer=break_buffer,
                filter_tokens=True,
                ncores=ncores,
            )

        # Apply prefix-length filtering if requested
        if self.task == "outcome" and getattr(self, "prefix_lengths", None):
            allowed = set(self.prefix_lengths)
            padding_size = getattr(self.model, "_padding_size", 0)
            filtered_preds = []
            filtered_prefixes = []
            for i, p in enumerate(self.test_seq.relevant_prefixes):
                raw_seq = p["prefix"]
                real_len = len([a for a in raw_seq[padding_size:]
                                if self._decode_activity(a) not in [None, "START", "END"]])
                if real_len in allowed:
                    filtered_preds.append(self.predictions[i])
                    filtered_prefixes.append(p)
            self.predictions = filtered_preds
            self.test_seq.relevant_prefixes = filtered_prefixes
            print(f"[BEST] Prefix-length filter applied — kept "
                  f"{len(self.predictions)} prefixes with lengths {sorted(allowed)}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Compute task-specific metrics and return them as a dict."""
        if self.predictions is None:
            raise RuntimeError("Call predict() before evaluate().")

        ncores = self.config.get("ncores", 1)

        if self.task == "nap":
            ev = NAPEvaluator(pred=self.predictions, actual=self.test_seq.next_activities)
            return {
                "nap_accuracy": float(ev.calc_accuracy_score()),
                "nap_balanced_accuracy": float(ev.calc_balanced_accuracy_score()),
                "none_share": float(ev.get_nan_share()),
            }
        elif self.task == "outcome":
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            y_true = []
            y_pred = []
            self.outcome_confidences = []  # track per-prediction confidence
            most_freq = pd.Series(list(self.train_outcomes_mapping.values())).mode()[0]

            for i, p in enumerate(self.test_seq.relevant_prefixes):
                case_id = p["case_id"]
                true_outcome = self.train_outcomes_mapping.get(case_id, None)

                pred_seq = self.predictions[i]
                pred_outcome = None
                conf = None  # no probability signal from BEST
                if pred_seq:
                    for token_idx in pred_seq:
                        token_str = self._decode_activity(token_idx)
                        if token_str and str(token_str).startswith("OUTCOME_"):
                            pred_outcome = str(token_str).replace("OUTCOME_", "")
                            break

                if pred_outcome is None:
                    pred_outcome = most_freq

                y_true.append(true_outcome)
                y_pred.append(pred_outcome)
                self.outcome_confidences.append(conf)

            self.outcome_y_true = y_true
            self.outcome_y_pred = y_pred

            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
            }
        else:  # rtp
            ev = RTPEvaluator(pred=self.predictions, actual=self.test_seq.full_future_sequences)
            return {
                "rtp_ndls": float(ev.calc_ndls(ncores=ncores)),
                "none_share": float(ev.get_nan_share()),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, output_dir: str) -> None:
        """Pickle the fitted BESTPredictor to output_dir/best_model.pkl."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[BEST] Model saved -> {model_path}")

    def save_results(self, output_dir: str) -> None:
        """Write predictions CSV with actual vs predicted columns.

        Both NAP and RTP -> best_predictions.csv  
        Columns: [case_id, case_index, sequence, true_next_activity, predicted_next_activity, confidence, correct]
        """
        import json
        os.makedirs(output_dir, exist_ok=True)

        if self.task == "outcome":
            self._save_results_outcome(output_dir)
            return
        elif self.task != "nap":
            # For RTP we keep the old format for now or implement similarly if needed
            self._save_results_rtp(output_dir)
            return

        # Pull per-prefix case ids from relevant_prefixes (same ordering as predictions)
        if hasattr(self.test_seq, "relevant_prefixes") and self.test_seq.relevant_prefixes:
            prefixes = self.test_seq.relevant_prefixes
        else:
            print("[BEST] No relevant prefixes found, cannot save results.")
            return

        n = min(len(prefixes), len(self.predictions))
        preds = self.predictions[:n]
        actuals_enc = getattr(self.test_seq, "next_activities", [None] * n)
        
        # Tracker for confidence
        tracker = getattr(self.model, "choice_tracker_nap", {})
        probs = tracker.get("prob", [None] * n)

        rows = []
        padding_size = getattr(self.model, "_padding_size", 0)

        for i in range(n):
            prefix_data = prefixes[i]
            case_id = str(prefix_data["case_id"])
            raw_sequence = prefix_data["prefix"] # This is padded internally by SequenceData
            
            # The "real" events are those after padding_size
            real_sequence_enc = raw_sequence[padding_size:]
            decoded_sequence = [self._decode_activity(a) for a in real_sequence_enc]
            filtered_seq = [a for a in decoded_sequence if a not in ["START", "END"]]
            
            case_index = len(filtered_seq) # 1-indexed step number
            
            if case_index == 0:
                continue # Skip the initial "empty" state for the UI

            true_next = self._decode_activity(actuals_enc[i])
            pred_next = self._decode_activity(preds[i])
            conf = probs[i] if i < len(probs) else None
            correct = int(true_next == pred_next) if (true_next is not None and pred_next is not None) else None

            rows.append({
                "case_id": case_id,
                "case_index": case_index,
                "sequence": json.dumps(filtered_seq),
                "true_next_activity": true_next,
                "predicted_next_activity": pred_next,
                "confidence": conf,
                "correct": correct
            })

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, "best_predictions.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[BEST] Predictions saved -> {out_path} ({len(df_out)} rows)")

    def _save_results_rtp(self, output_dir: str) -> None:
        """Fallback for RTP results."""
        if hasattr(self.test_seq, "relevant_prefixes") and self.test_seq.relevant_prefixes:
            prefixes = self.test_seq.relevant_prefixes
            case_ids = [p["case_id"] for p in prefixes]
            prefix_ids = [len(p["prefix"]) for p in prefixes]
        else:
            print("[BEST] No relevant prefixes found, cannot save results.")
            return

        n = min(len(prefixes), len(self.predictions))
        preds = self.predictions[:n]

        decoded_preds = self._decode_rtp(preds)
        actuals_enc = getattr(self.test_seq, "full_future_sequences", [None] * n)
        decoded_actuals = []
        for seq in actuals_enc[:n]:
            if seq is None:
                decoded_actuals.append(None)
            else:
                decoded_actuals.append(", ".join(
                    str(self._decode_activity(idx)) for idx in seq if idx is not None
                ))
        df_out = pd.DataFrame({
            "CaseID": case_ids,
            "prefix_length": prefix_ids,
            "actual_remaining_trace": decoded_actuals,
            "predicted_remaining_trace": decoded_preds,
        })

        out_path = os.path.join(output_dir, "best_predictions.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[BEST] RTP Predictions saved -> {out_path}")

    def _save_results_outcome(self, output_dir: str) -> None:
        rows = []
        padding_size = getattr(self.model, "_padding_size", 0)
        confidences = getattr(self, "outcome_confidences", [None] * len(self.outcome_y_true))

        for i, p in enumerate(self.test_seq.relevant_prefixes):
            case_id = p["case_id"]
            prefix_seq = p["prefix"]

            true_outcome = self.outcome_y_true[i]
            pred_outcome = self.outcome_y_pred[i]

            # Decode the real (non-padding) prefix activities
            decoded_prefix = []
            for act_enc in prefix_seq[padding_size:]:
                act = self._decode_activity(act_enc)
                if act and str(act) not in ["START", "END"] and not str(act).startswith("OUTCOME_"):
                    decoded_prefix.append(str(act))

            # BEST produces no softmax confidence — report None (not a fake 100%)
            conf = confidences[i]  # None unless a future version captures it

            rows.append({
                "case_id": str(case_id),
                "prefix_length": len(decoded_prefix),
                "prefix_sequence": ", ".join(decoded_prefix),
                "true_outcome": true_outcome,
                "predicted_outcome": pred_outcome,
                "correct": true_outcome == pred_outcome,
                # BEST uses pattern-frequency distance, not a probability score.
                # confidence_percent is left as N/A when unavailable.
                "confidence_percent": round(conf, 2) if conf is not None else "N/A",
            })

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, "outcome_predictions.csv")
        df_out.to_csv(out_path, index=False)

        total = len(df_out)
        correct = df_out["correct"].sum()
        print(f"[BEST] Outcome Predictions saved -> {out_path} ({total} rows)")
        print(f"[BEST] Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    def plot_performance(self, output_dir: str) -> None:
        """Generate a performance overview chart and save it to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        if self.predictions is None or not hasattr(self.test_seq, "next_activities"):
            print("[BEST] plot_performance: no data available, skipping.")
            return

        n = min(len(self.predictions), len(self.test_seq.next_activities or []))
        if n == 0:
            print("[BEST] plot_performance: empty predictions, skipping.")
            return

        preds_enc = self.predictions[:n]
        actuals_enc = (self.test_seq.next_activities or [])[:n]

        if self.task == "nap":
            decoded_preds = self._decode_nap(preds_enc)
            decoded_actuals = [self._decode_activity(a) for a in actuals_enc]

            # Keep only rows where both are known
            pairs = [(a, p) for a, p in zip(decoded_actuals, decoded_preds)
                     if a is not None and p is not None]
            if not pairs:
                print("[BEST] plot_performance: no decodable pairs, skipping.")
                return

            acts, preds_d = zip(*pairs)
            classes = sorted(set(acts) | set(preds_d))
            top_classes = classes[:20]  # cap at 20 for readability

            # Confusion matrix
            cm = np.zeros((len(top_classes), len(top_classes)), dtype=int)
            cls_idx = {c: i for i, c in enumerate(top_classes)}
            for a, p in zip(acts, preds_d):
                if a in cls_idx and p in cls_idx:
                    cm[cls_idx[a], cls_idx[p]] += 1

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle("BEST – Next Activity Prediction Performance", fontsize=13)

            # Left: confusion matrix
            ax = axes[0]
            im = ax.imshow(cm, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(top_classes)))
            ax.set_yticks(range(len(top_classes)))
            ax.set_xticklabels(top_classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(top_classes, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix (top classes)")
            plt.colorbar(im, ax=ax)

            # Right: per-class accuracy bar
            ax2 = axes[1]
            per_class_acc = []
            for i, cls in enumerate(top_classes):
                total = cm[i, :].sum()
                correct = cm[i, i]
                per_class_acc.append(correct / total if total > 0 else 0.0)
            ax2.barh(top_classes, per_class_acc, color="#4C72B0", alpha=0.85)
            ax2.set_xlabel("Accuracy")
            ax2.set_title("Per-class Accuracy")
            ax2.set_xlim(0, 1)
            ax2.axvline(
                sum(int(a == p) for a, p in zip(acts, preds_d)) / len(pairs),
                color="red", linestyle="--", label="Overall acc"
            )
            ax2.legend()

            plt.tight_layout()
            out_path = os.path.join(output_dir, "best_performance.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[BEST] Performance chart saved -> {out_path}")

        else:  # rtp
            # For RTP: plot actual vs predicted remaining-trace length
            actuals_enc_full = getattr(self.test_seq, "full_future_sequences", [])[:n]
            actual_lens = [len(s) if s is not None else 0 for s in actuals_enc_full]
            pred_lens = [
                len(p) if isinstance(p, (list, tuple)) else 0 for p in preds_enc
            ]
            m = min(len(actual_lens), len(pred_lens))
            if m == 0:
                return
            actual_lens = actual_lens[:m]
            pred_lens = pred_lens[:m]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(actual_lens, pred_lens, alpha=0.3, s=12, color="#4C72B0")
            max_val = max(max(actual_lens, default=1), max(pred_lens, default=1))
            ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
            ax.set_xlabel("Actual remaining trace length")
            ax.set_ylabel("Predicted remaining trace length")
            ax.set_title("BEST – Remaining Trace Prediction")
            ax.legend()
            plt.tight_layout()
            out_path = os.path.join(output_dir, "best_performance.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[BEST] Performance chart saved -> {out_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_activity_index(self, seq: SequenceData, padding_size: int) -> None:
        seq.data["activity_idx"] = seq.data.groupby(seq.case_identifier).cumcount() - padding_size

    def _prepare_train_data(self) -> None:
        if self.model is None or self.train_seq is None:
            raise RuntimeError("Training data is not loaded.")

        self.model.train_max_trace_len = self.train_seq._get_max_trace_len()
        self.train_seq.pad_columns(cols_to_pad=[self.train_seq.activity_identifier], n_pad=self.model._padding_size)
        self._set_activity_index(self.train_seq, self.model._padding_size)

        timestamp_col = self.train_seq.timestamp_identifier
        case_col = self.train_seq.case_identifier
        self.train_seq.data[timestamp_col] = self.train_seq.data.groupby(case_col)[timestamp_col].ffill()
        self.train_seq.data[timestamp_col] = self.train_seq.data.groupby(case_col)[timestamp_col].bfill()

        self.train_seq.encode_activities()
        self.train_seq.extract_traces(columns=[self.train_seq.activity_identifier])
        self.model.start_activity = self.train_seq.start_activity
        self.model.end_activity = self.train_seq.end_activity

        self.train_seq.generate_prefixes()
        self.train_seq.pick_relevant_prefixes()
        self.model.max_prefix_len = max(len(prefix["prefix"]) for prefix in self.train_seq.relevant_prefixes)

    def _prepare_test_data(self, filter_sequences: bool) -> None:
        if self.model is None or self.test_seq is None:
            raise RuntimeError("Test data is not loaded.")
        if self._act_encoder is None:
            raise RuntimeError("Training activity encoder is not available.")

        self.test_seq.pad_columns(cols_to_pad=[self.test_seq.activity_identifier], n_pad=self.model._padding_size)
        self._set_activity_index(self.test_seq, self.model._padding_size)

        timestamp_col = self.test_seq.timestamp_identifier
        case_col = self.test_seq.case_identifier
        self.test_seq.data[timestamp_col] = self.test_seq.data.groupby(case_col)[timestamp_col].ffill()
        self.test_seq.data[timestamp_col] = self.test_seq.data.groupby(case_col)[timestamp_col].bfill()

        self.test_seq.encode_activities(act_encoder=self._act_encoder)
        self.test_seq.extract_traces(columns=[self.test_seq.activity_identifier])

        self.test_seq.generate_prefixes()
        self.test_seq.pick_relevant_prefixes()
        self.test_seq.generate_full_sequences(filter_sequences=filter_sequences)
        self.test_seq.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.test_seq.generate_next_activities()

    def _decode_activity(self, idx):
        """Decode a single integer activity index to its label, or None."""
        if idx is None:
            return None
        try:
            return self._act_encoder.inverse_transform([int(idx)])[0]
        except Exception:
            return str(idx)

    def _decode_nap(self, predictions: list) -> list:
        return [self._decode_activity(p) for p in predictions]

    def _decode_rtp(self, predictions: list) -> list:
        """Decode a list of activity-index sequences to comma-separated activity strings."""
        result = []
        for seq in predictions:
            if seq is None:
                result.append(None)
            else:
                decoded = []
                for idx in seq:
                    if idx is not None:
                        act = self._decode_activity(idx)
                        decoded.append(act)
                result.append(", ".join(str(x) for x in decoded))
        return result
