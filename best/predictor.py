import os
import pickle

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
        self._act_encoder = self.train_seq.act_encoder

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
        self.model.prepare_train()
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

        self.model.prepare_test(
            act_encoder=self._act_encoder,
            filter_sequences=filter_seqs,
        )
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
        print(f"[BEST] Model saved → {model_path}")

    def save_results(self, output_dir: str) -> None:
        """Write predictions to a CSV in output_dir.

        NAP  → predictions.csv with columns [CaseID, predicted_next_activity]
        RTP  → predictions.csv with columns [CaseID, predicted_remaining_trace]
        """
        os.makedirs(output_dir, exist_ok=True)

        # Retrieve case IDs from the test split; fall back to a numeric index
        if hasattr(self.test_seq, "data") and self.test_seq.data is not None:
            case_ids = self.test_seq.data["CaseID"].unique().tolist()
        else:
            case_ids = list(range(len(self.predictions)))

        # Align lengths (BEST may return fewer predictions than test cases)
        n = min(len(case_ids), len(self.predictions))
        case_ids = case_ids[:n]
        preds = self.predictions[:n]

        if self.task == "nap":
            decoded = self._decode_nap(preds)
            df_out = pd.DataFrame({"CaseID": case_ids, "predicted_next_activity": decoded})
        else:
            decoded = self._decode_rtp(preds)
            df_out = pd.DataFrame({"CaseID": case_ids, "predicted_remaining_trace": decoded})

        out_path = os.path.join(output_dir, "predictions.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[BEST] Predictions saved → {out_path} ({len(df_out)} rows)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
