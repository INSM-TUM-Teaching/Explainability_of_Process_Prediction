import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from ..model import build_outcome_model
from utils.outcome_utils import extract_case_outcomes


class OutcomePredictor:
    """
    Transformer-based predictor for process outcome (case-level classification).

    The outcome of a case is determined by the final value of `target_column`
    (default: 'Activity', i.e. the last activity label).  Every prefix of every
    case is labelled with the *full-case* outcome so that the model learns to
    predict the final result even from partial traces.
    """

    def __init__(
        self,
        max_len: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
    ):
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.activity_encoder = LabelEncoder()
        self.outcome_encoder: LabelEncoder | None = None
        self.model: keras.Model | None = None
        self.vocab_size: int | None = None
        self.num_outcome_classes: int | None = None
        self.history = None

        # Preserved for result saving
        self._test_case_ids: list | None = None
        self._test_prefix_lengths: list | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "Activity",
        test_size: float = 0.3,
        val_split: float = 0.5,
        max_cases: int | None = None,
        max_prefixes_per_case: int | None = None,
        **kwargs,
    ) -> dict:
        """
        Build train / val / test splits for outcome prediction.

        Returns a dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test,
          class_weights (dict, for use with model.fit)
        """
        print(f"Preparing data for Outcome Prediction (Target: {target_column})…")

        required_cols = ["CaseID", "Activity", "Timestamp"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the dataset."
            )

        # Sort chronologically once and reuse
        process_data = (
            df.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True).copy()
        )

        # ── 1. Extract the per-case outcome (last event's target value) ──────
        outcome_series = extract_case_outcomes(
            process_data, "CaseID", "Timestamp", target_column
        )

        # ── 2. Fit encoders on the full dataset ──────────────────────────────
        self.outcome_encoder = LabelEncoder()
        self.outcome_encoder.fit(outcome_series)
        self.num_outcome_classes = len(self.outcome_encoder.classes_)

        self.activity_encoder.fit(process_data["Activity"])
        # +1 for padding (0), +1 spare for safety
        self.vocab_size = len(self.activity_encoder.classes_) + 2

        print(f"  Outcome classes ({self.num_outcome_classes}): "
              f"{list(self.outcome_encoder.classes_)}")
        print(f"  Vocabulary size: {self.vocab_size}")

        # ── 3. Build prefix sequences ─────────────────────────────────────────
        split_col = "__split" if "__split" in process_data.columns else None

        def _encode(sequences, outcomes):
            """Encode activities, pad, shift; encode outcome labels."""
            X_enc = [self.activity_encoder.transform(s) for s in sequences]
            X = keras.preprocessing.sequence.pad_sequences(
                X_enc, maxlen=self.max_len, padding="pre", value=0
            )
            X = X + 1  # shift: 0 stays as pad, activities become 1-indexed
            y = self.outcome_encoder.transform(outcomes)
            return X, y

        if split_col:
            split_values = set(process_data[split_col].dropna().unique())
            if not {"train", "val", "test"}.issubset(split_values):
                raise ValueError(
                    "When '__split' column is present it must contain "
                    "'train', 'val', and 'test' values."
                )
            train_df = process_data[process_data[split_col] == "train"]
            val_df   = process_data[process_data[split_col] == "val"]
            test_df  = process_data[process_data[split_col] == "test"]

            seq_tr, out_tr, _         = self._build_prefix_sequences(train_df, outcome_series, max_cases, max_prefixes_per_case)
            seq_v,  out_v,  _         = self._build_prefix_sequences(val_df,   outcome_series, max_cases, max_prefixes_per_case)
            seq_te, out_te, meta_test = self._build_prefix_sequences(test_df,  outcome_series, max_cases, max_prefixes_per_case)

            X_train, y_train = _encode(seq_tr, out_tr)
            X_val,   y_val   = _encode(seq_v,  out_v)
            X_test,  y_test  = _encode(seq_te, out_te)
        else:
            seq_all, out_all, meta_all = self._build_prefix_sequences(
                process_data, outcome_series, max_cases, max_prefixes_per_case
            )
            X_all, y_all = _encode(seq_all, out_all)

            X_tr, X_tmp, y_tr, y_tmp, ids_tr, ids_tmp, pl_tr, pl_tmp = train_test_split(
                X_all, y_all,
                meta_all["case_ids"], meta_all["prefix_lengths"],
                test_size=test_size, random_state=42,
            )
            X_val, X_test, y_val, y_test, ids_val, ids_test, pl_val, pl_test = train_test_split(
                X_tmp, y_tmp, ids_tmp, pl_tmp,
                test_size=val_split, random_state=42,
            )
            X_train, y_train = X_tr, y_tr

            # Store test metadata for richer CSV output
            self._test_case_ids     = ids_test
            self._test_prefix_lengths = pl_test

        # ── 4. Compute class weights for imbalanced datasets ─────────────────
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes.tolist(), weights.tolist()))

        print(f"\nDataset splits:")
        print(f"  Train:      {len(X_train):,} samples")
        print(f"  Validation: {len(X_val):,} samples")
        print(f"  Test:       {len(X_test):,} samples")
        print(f"\nClass weights: {class_weights}")

        # Log class distribution
        for cls_idx, cls_name in enumerate(self.outcome_encoder.classes_):
            n = int(np.sum(y_train == cls_idx))
            print(f"  [train] class '{cls_name}': {n:,} samples "
                  f"({100 * n / len(y_train):.1f}%)")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "class_weights": class_weights,
        }

    def build_model(self) -> None:
        print("\nBuilding Outcome Prediction Model…")
        self.model = build_outcome_model(
            vocab_size=self.vocab_size,
            num_outcome_classes=self.num_outcome_classes,
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(
        self,
        data: dict,
        epochs: int = 50,
        batch_size: int = 128,
        patience: int = 10,
    ):
        print(f"\nTraining model for up to {epochs} epochs…")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        self.history = self.model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=data.get("class_weights"),
            callbacks=callbacks,
            verbose=1,
        )
        print("\nTraining completed!")
        return self.history

    def evaluate(self, data: dict) -> dict:
        """
        Evaluate on the test set and return a comprehensive metrics dict.
        Includes per-class precision/recall/F1 and a confusion-matrix summary.

        We derive labels/target_names strictly from the classes that actually
        appear in the test split — not from all classes seen during fit — so
        that classification_report and sklearn metrics never get a mismatch
        between the number of unique label indices and the number of names.
        """
        print("\nEvaluating on test set…")
        test_loss, test_accuracy = self.model.evaluate(
            data["X_test"], data["y_test"], verbose=0
        )

        y_pred_probs = self.model.predict(data["X_test"], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Only the class indices that actually appear in the test split
        # (union of true and predicted so the report covers all predicted classes too)
        present_labels = sorted(
            set(data["y_test"].tolist()) | set(y_pred.tolist())
        )
        present_names = [
            self.outcome_encoder.classes_[i] for i in present_labels
        ]

        bal_acc     = balanced_accuracy_score(data["y_test"], y_pred)
        f1_weighted = f1_score(data["y_test"], y_pred, average="weighted", labels=present_labels)
        f1_macro    = f1_score(data["y_test"], y_pred, average="macro",    labels=present_labels)

        print(f"Test accuracy:      {test_accuracy * 100:.2f}%")
        print(f"Balanced accuracy:  {bal_acc * 100:.2f}%")
        print(f"F1 (weighted):      {f1_weighted * 100:.2f}%")
        print(f"F1 (macro):         {f1_macro * 100:.2f}%")

        # Per-class report — only for classes present in the test split
        report = classification_report(
            data["y_test"],
            y_pred,
            labels=present_labels,
            target_names=present_names,
            digits=4,
        )
        print("\nClassification Report:")
        print(report)

        # Build per-class F1 dict for ALL fitted classes (0 for absent ones)
        per_class_f1 = {}
        for i, cls_name in enumerate(self.outcome_encoder.classes_):
            f1_i = f1_score(
                data["y_test"], y_pred,
                labels=[i], average="macro", zero_division=0
            )
            per_class_f1[str(cls_name)] = round(float(f1_i), 4)

        return {
            "test_loss":           float(test_loss),
            "test_accuracy":       float(test_accuracy),
            "balanced_accuracy":   float(bal_acc),
            "f1_score_weighted":   float(f1_weighted),
            "f1_score_macro":      float(f1_macro),
            "per_class_f1":        per_class_f1,
            "outcome_classes":     list(self.outcome_encoder.classes_),
        }

    def predict(self, data: dict) -> tuple:
        y_pred_probs = self.model.predict(data["X_test"], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        return y_pred, y_pred_probs

    def save_results(
        self,
        data: dict,
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray,
        output_dir: str,
    ) -> None:
        """
        Write outcome_predictions.csv with one row per test prefix.
        Columns: case_id, prefix_length, prefix_sequence,
                 true_outcome, predicted_outcome, correct,
                 confidence_percent, plus one column per class
                 showing its raw probability.
        """
        print("\nSaving outcome prediction results…")
        os.makedirs(output_dir, exist_ok=True)

        class_names = list(self.outcome_encoder.classes_)
        n = len(data["X_test"])

        rows = []
        for i in range(n):
            # Recover the activity sequence (undo padding + shift)
            seq = data["X_test"][i]
            seq = seq[seq > 0] - 1          # strip padding, undo +1 shift
            decoded_seq = self.activity_encoder.inverse_transform(seq)

            true_label = self.outcome_encoder.inverse_transform(
                [data["y_test"][i]]
            )[0]
            pred_label = self.outcome_encoder.inverse_transform([y_pred[i]])[0]
            confidence = float(y_pred_probs[i][y_pred[i]]) * 100

            # Use tracked case IDs if available, else generate a placeholder
            case_id = (
                self._test_case_ids[i]
                if self._test_case_ids is not None
                else f"test_{i}"
            )
            prefix_len = (
                self._test_prefix_lengths[i]
                if self._test_prefix_lengths is not None
                else len(decoded_seq)
            )

            row = {
                "case_id":            case_id,
                "prefix_length":      int(prefix_len),
                "prefix_sequence":    ", ".join(decoded_seq),
                "true_outcome":       true_label,
                "predicted_outcome":  pred_label,
                "correct":            true_label == pred_label,
                "confidence_percent": round(confidence, 2),
            }
            # Per-class probabilities
            for j, cls_name in enumerate(class_names):
                row[f"prob_{cls_name}"] = round(float(y_pred_probs[i][j]) * 100, 2)

            rows.append(row)

        results_df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, "outcome_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"  Predictions saved to: {output_path}")

        # ── Summary statistics ─────────────────────────────────────────────
        total = len(results_df)
        correct = results_df["correct"].sum()
        print(f"  Overall accuracy on saved results: "
              f"{correct}/{total} ({100 * correct / total:.1f}%)")
        for cls_name in class_names:
            subset = results_df[results_df["true_outcome"] == cls_name]
            if len(subset) == 0:
                continue
            cls_correct = subset["correct"].sum()
            print(f"    '{cls_name}': "
                  f"{cls_correct}/{len(subset)} "
                  f"({100 * cls_correct / len(subset):.1f}%)")

    def plot_training_history(self, output_dir: str) -> None:
        if self.history is None:
            return
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history["accuracy"],     label="Training",   linewidth=2)
        ax1.plot(self.history.history["val_accuracy"], label="Validation", linewidth=2)
        ax1.set_title("Outcome Prediction – Accuracy", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history["loss"],     label="Training",   linewidth=2)
        ax2.plot(self.history.history["val_loss"], label="Validation", linewidth=2)
        ax2.set_title("Outcome Prediction – Loss", fontsize=14)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "outcome_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Training-history plot saved to: {output_path}")

        print("\nTraining metrics:")
        print(f"  Final train accuracy: "
              f"{self.history.history['accuracy'][-1] * 100:.2f}%")
        print(f"  Final val accuracy:   "
              f"{self.history.history['val_accuracy'][-1] * 100:.2f}%")
        print(f"  Best  val accuracy:   "
              f"{max(self.history.history['val_accuracy']) * 100:.2f}%")

    def plot_confusion_matrix(self, data: dict, output_dir: str) -> None:
        """
        Save a confusion-matrix heatmap for the test-set predictions.
        Axis labels are derived from the classes actually present in the test
        split so that the tick count always matches the matrix dimensions.
        """
        os.makedirs(output_dir, exist_ok=True)

        y_pred_probs = self.model.predict(data["X_test"], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Use only labels present in true OR predicted to match matrix size
        present_labels = sorted(
            set(data["y_test"].tolist()) | set(y_pred.tolist())
        )
        present_names = [
            self.outcome_encoder.classes_[i] for i in present_labels
        ]

        cm = confusion_matrix(data["y_test"], y_pred, labels=present_labels)
        # Guard against empty rows before normalising
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division-by-zero
        cm_norm = cm.astype(float) / row_sums * 100

        # Scale figure width with number of classes so labels don't overlap
        n_cls = len(present_labels)
        fig_w = max(14, n_cls * 2.5)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, max(5, n_cls * 1.2)))

        for ax, matrix, title, fmt in [
            (axes[0], cm,      "Confusion Matrix (counts)", "d"),
            (axes[1], cm_norm, "Confusion Matrix (row %)",  ".1f"),
        ]:
            sns.heatmap(
                matrix,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=present_names,
                yticklabels=present_names,
                ax=ax,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "outcome_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved to: {output_path}")

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "outcome_transformer.keras")
        self.model.save(model_path)
        print(f"  Model saved to: {model_path}")

    def load_model(self, model_path: str) -> None:
        self.model = keras.models.load_model(model_path)
        print(f"  Model loaded from: {model_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prefix_sequences(
        self,
        data: pd.DataFrame,
        outcome_series: pd.Series,
        max_cases: int | None,
        max_prefixes_per_case: int | None,
    ) -> tuple:
        """
        For every case in *data*, produce one (prefix, outcome) pair per
        prefix length.  Cases whose outcome is not present in *outcome_series*
        (e.g. when using pre-split data) are skipped with a warning.

        Returns:
            sequences      – list of activity-name arrays
            outcomes       – list of outcome strings (one per prefix)
            metadata       – dict with 'case_ids', 'prefix_lengths', 'max_len'
        """
        sequences: list  = []
        outcomes:  list  = []
        case_ids_out:    list = []
        prefix_lengths:  list = []

        grouped  = data.groupby("CaseID")
        case_ids = sorted(grouped.groups.keys())
        if max_cases is not None:
            case_ids = case_ids[:max_cases]

        skipped = 0
        for case_id in case_ids:
            if case_id not in outcome_series.index:
                skipped += 1
                continue

            group      = grouped.get_group(case_id)
            activities = group["Activity"].values
            outcome    = outcome_series[case_id]

            limit = len(activities) + 1 if self.max_len is None else min(len(activities) + 1, self.max_len + 1)
            indices = list(range(1, limit))
            if max_prefixes_per_case and len(indices) > max_prefixes_per_case:
                step    = max(1, len(indices) // max_prefixes_per_case)
                indices = indices[::step][:max_prefixes_per_case]

            for i in indices:
                sequences.append(activities[:i])
                outcomes.append(outcome)
                case_ids_out.append(case_id)
                prefix_lengths.append(i)

        if skipped:
            print(f"  [WARN] {skipped} case(s) skipped – not found in outcome_series.")

        if not sequences:
            raise ValueError(
                "No prefix sequences were generated. "
                "Check that the data subset contains known cases."
            )

        max_len = max(len(s) for s in sequences)
        print(f"  Generated {len(sequences):,} prefix sequences "
              f"(max length: {max_len})")

        metadata = {
            "case_ids":      case_ids_out,
            "prefix_lengths": prefix_lengths,
            "max_len":       max_len,
        }
        return sequences, outcomes, metadata
