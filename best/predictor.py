import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from best4ppm.models.best import BESTPredictor
from best4ppm.data.sequencedata import SequenceData
from best4ppm.eval.evaluator import NAPEvaluator, RTPEvaluator


class BESTRunner:
    """Adapter that wraps BESTPredictor + SequenceData into the pipeline interface."""

    def __init__(self, config: dict, task: str):
        """
        Args:
            config: dict with keys:
                max_pattern_size_train, max_pattern_size_eval,
                process_stage_width_percentage, min_freq,
                break_buffer, filter_sequences, ncores
            task: 'nap' (next activity) or 'rtp' (remaining trace)
        """
        self.config = config
        self.task = task  # 'nap' or 'rtp'

        self.model: BESTPredictor | None = None
        self.train_seq: SequenceData | None = None
        self.test_seq: SequenceData | None = None
        self.predictions = None
        self._act_encoder = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> None:
        """Build SequenceData objects and perform the train/test split.

        Args:
            df: DataFrame with standardised columns CaseID, Activity, Timestamp.
            test_size: Fraction of cases reserved for testing (default 0.2).
        """
        seq = SequenceData(
            data=df,
            case_identifier="CaseID",
            activity_identifier="Activity",
            timestamp_identifier="Timestamp",
        )
        train_pct = 1.0 - test_size
        self.train_seq, self.test_seq = seq.train_test_split(train_pct=train_pct)
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

        self.model = BESTPredictor(
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
        self.predictions = self.model.predict(
            eval_pattern_size=eval_pattern_size,
            task=self.task,
            break_buffer=break_buffer,
            filter_tokens=True,
            ncores=ncores,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Compute task-specific metrics and return them as a dict.

        Returns:
            For NAP: {'nap_accuracy': float, 'nap_balanced_accuracy': float, 'none_share': float}
            For RTP: {'rtp_ndls': float, 'none_share': float}
        """
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

        NAP -> best_predictions.csv  [CaseID, prefix_id, actual_next_activity, predicted_next_activity, correct]
        RTP -> best_predictions.csv  [CaseID, prefix_id, actual_remaining_trace, predicted_remaining_trace]
        """
        os.makedirs(output_dir, exist_ok=True)

        # Pull per-prefix case ids from relevant_prefixes (same ordering as predictions)
        if hasattr(self.test_seq, "relevant_prefixes") and self.test_seq.relevant_prefixes:
            prefixes = self.test_seq.relevant_prefixes
            case_ids = [p["case_id"] for p in prefixes]
            prefix_ids = [len(p["prefix"]) for p in prefixes]
        else:
            case_ids = list(range(len(self.predictions)))
            prefix_ids = case_ids

        n = min(len(case_ids), len(self.predictions))
        case_ids = case_ids[:n]
        prefix_ids = prefix_ids[:n]
        preds = self.predictions[:n]

        if self.task == "nap":
            decoded_preds = self._decode_nap(preds)
            # Actual next activities are encoded ints — decode them
            actuals_enc = getattr(self.test_seq, "next_activities", [None] * n)
            decoded_actuals = [self._decode_activity(a) for a in actuals_enc[:n]]
            correct = [
                int(p == a) if (p is not None and a is not None) else None
                for p, a in zip(decoded_preds, decoded_actuals)
            ]
            df_out = pd.DataFrame({
                "CaseID": case_ids,
                "prefix_length": prefix_ids,
                "actual_next_activity": decoded_actuals,
                "predicted_next_activity": decoded_preds,
                "correct": correct,
            })
        else:
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
        print(f"[BEST] Predictions saved -> {out_path} ({len(df_out)} rows)")

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
                decoded = [self._decode_activity(idx) for idx in seq if idx is not None]
                result.append(", ".join(str(a) for a in decoded))
        return result
