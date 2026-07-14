import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from ..model import build_outcome_model


class OutcomePredictor:

    def __init__(self, max_len=16, d_model=64, num_heads=4, num_blocks=2,
                 dropout_rate=0.1):
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.activity_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.vocab_size = None
        self.num_outcome_classes = None
        self.history = None

    def prepare_data(
        self,
        df,
        test_size=0.3,
        val_split=0.5,
        max_cases=None,
        max_prefixes_per_case=None,
        outcome_column=None,
        **kwargs
    ):
        print("Preparing data for Outcome Prediction...")

        required_cols = ['case:id', 'concept:name', 'time:timestamp']
        available_cols = [col for col in required_cols if col in df.columns]
        if len(available_cols) != len(required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {available_cols}")

        split_col = "__split" if "__split" in df.columns else None
        select_cols = required_cols + ([split_col] if split_col else [])

        if outcome_column and outcome_column in df.columns:
            select_cols.append(outcome_column)

        select_cols = list(dict.fromkeys(select_cols))
        process_data = df[select_cols].copy()
        process_data = process_data.rename(columns={
            'case:id': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
        })
        if process_data.columns.duplicated().any():
            dupes = process_data.columns[process_data.columns.duplicated()].unique().tolist()
            print(f"[WARN] Dropping duplicate columns: {dupes}")
            process_data = process_data.loc[:, ~process_data.columns.duplicated()]

        process_data['timestamp'] = pd.to_datetime(process_data['timestamp'])
        process_data = process_data.sort_values(['case_id', 'timestamp']).reset_index(drop=True)

        from utils.outcome_utils import extract_case_outcomes
        outcomes = extract_case_outcomes(process_data, outcome_column)

        sequences_all, outcome_labels_all, metadata = self._create_sequences_with_prefixes(
            process_data,
            outcomes,
            max_cases=max_cases,
            max_prefixes_per_case=max_prefixes_per_case
        )
        all_case_ids = metadata['case_ids']

        print(f"Total sequences: {len(sequences_all):,}")
        print(f"Max sequence length: {metadata['max_len']}")

        self.activity_encoder.fit(process_data['activity'])
        self.label_encoder.fit(list(outcomes.values))

        print(f"Number of outcome classes: {len(self.label_encoder.classes_)}")
        print(f"Outcome classes: {list(self.label_encoder.classes_)}")

        # Map activity -> code once; per-prefix encoder.transform calls (there can
        # be hundreds of thousands) are dominated by per-call overhead.
        act_to_idx = {c: i for i, c in enumerate(self.activity_encoder.classes_)}

        def encode_sequences(sequences, outcome_labels):
            X_encoded = [
                np.fromiter((act_to_idx[a] for a in seq), dtype=np.int64, count=len(seq))
                for seq in sequences
            ]
            y_encoded = self.label_encoder.transform(outcome_labels)
            X = keras.preprocessing.sequence.pad_sequences(
                X_encoded, maxlen=self.max_len, padding='pre', value=-1
            )
            X = X + 1
            return X, y_encoded

        if split_col:
            split_values = set(process_data[split_col].dropna().unique().tolist())
            if not {"train", "val", "test"}.issubset(split_values):
                raise ValueError("Split column must include train, val, and test values.")

            train_df = process_data[process_data[split_col] == "train"].drop(columns=[split_col])
            val_df = process_data[process_data[split_col] == "val"].drop(columns=[split_col])
            test_df = process_data[process_data[split_col] == "test"].drop(columns=[split_col])

            seq_train, out_train, meta_train = self._create_sequences_with_prefixes(
                train_df, outcomes, max_cases=max_cases, max_prefixes_per_case=max_prefixes_per_case)
            seq_val, out_val, meta_val = self._create_sequences_with_prefixes(
                val_df, outcomes, max_cases=max_cases, max_prefixes_per_case=max_prefixes_per_case)
            seq_test, out_test, meta_test = self._create_sequences_with_prefixes(
                test_df, outcomes, max_cases=max_cases, max_prefixes_per_case=max_prefixes_per_case)

            X_train, y_train = encode_sequences(seq_train, out_train)
            X_val, y_val = encode_sequences(seq_val, out_val)
            X_test, y_test = encode_sequences(seq_test, out_test)
            train_case_ids = meta_train['case_ids']
            val_case_ids = meta_val['case_ids']
            test_case_ids = meta_test['case_ids']
        else:
            X, y = encode_sequences(sequences_all, outcome_labels_all)
            all_indices = np.arange(len(X))
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_split, random_state=42
            )
            train_idx, temp_idx = train_test_split(all_indices, test_size=test_size, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=val_split, random_state=42)
            train_case_ids = [all_case_ids[i] for i in train_idx]
            val_case_ids = [all_case_ids[i] for i in val_idx]
            test_case_ids = [all_case_ids[i] for i in test_idx]

        self.vocab_size = len(self.activity_encoder.classes_) + 2
        self.num_outcome_classes = len(self.label_encoder.classes_)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"\nDataset splits:")
        print(f"Train: {len(X_train):,} samples")
        print(f"Validation: {len(X_val):,} samples")
        print(f"Test: {len(X_test):,} samples")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_case_ids': train_case_ids,
            'val_case_ids': val_case_ids,
            'test_case_ids': test_case_ids,
            'test_row_indices': list(range(len(X_test)))
        }

    def _create_sequences_with_prefixes(self, data, outcomes, max_cases=None,
                                         max_prefixes_per_case=None):
        sequences = []
        outcome_labels = []
        case_ids_list = []
        prefix_lengths = []

        grouped = data.groupby('case_id')
        case_ids = list(grouped.groups.keys())
        if max_cases is not None:
            case_ids = sorted(case_ids)[:max_cases]

        for case_id in case_ids:
            if case_id not in outcomes.index:
                continue
            outcome = outcomes[case_id]
            group = grouped.get_group(case_id)
            activities = group['activity'].values

            # For outcome prediction, we don't want to include the final activity as part of the prefix
            # if we are predicting the outcome of the case.
            max_prefix_len = len(activities) - 1 if len(activities) > 1 else 1
            prefix_indices = list(range(1, max_prefix_len + 1))
            if max_prefixes_per_case is not None and len(prefix_indices) > max_prefixes_per_case:
                step = max(1, len(prefix_indices) // max_prefixes_per_case)
                prefix_indices = prefix_indices[::step][:max_prefixes_per_case]

            for i in prefix_indices:
                prefix = activities[:i]
                sequences.append(prefix)
                outcome_labels.append(outcome)
                case_ids_list.append(case_id)
                prefix_lengths.append(len(prefix))

        max_len = max(len(seq) for seq in sequences) if sequences else 1

        metadata = {
            'case_ids': case_ids_list,
            'prefix_lengths': prefix_lengths,
            'max_len': max_len
        }
        return sequences, outcome_labels, metadata

    def build_model(self):
        print("\nBuilding Outcome Prediction Model...")

        self.model = build_outcome_model(
            vocab_size=self.vocab_size,
            num_outcome_classes=self.num_outcome_classes,
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel Summary:")
        self.model.summary()

    def train(self, data, epochs=50, batch_size=128, patience=10):
        print(f"\nTraining model for {epochs} epochs...")
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        self.history = self.model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        print("\nTraining completed!")
        return self.history

    def evaluate(self, data):
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(
            data['X_test'], data['y_test'], verbose=0
        )
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test loss: {test_loss:.4f}")
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

    def predict(self, data):
        print("\nGenerating predictions...")

        y_pred_probs = self.model.predict(data['X_test'], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        print(f"Predictions generated for {len(data['X_test']):,} test samples.")
        return y_pred, y_pred_probs

    def save_results(self, data, y_pred, y_pred_probs, output_dir):
        print("\nSaving results...")
        os.makedirs(output_dir, exist_ok=True)
        test_case_ids = data.get('test_case_ids')

        results = []
        case_counters = {}
        for i in range(len(data['X_test'])):
            seq = data['X_test'][i]
            seq = seq[seq > 0]
            seq = seq - 1
            decoded_seq = self.activity_encoder.inverse_transform(seq)

            true_decoded = self.label_encoder.inverse_transform([data['y_test'][i]])[0]
            pred_decoded = self.label_encoder.inverse_transform([y_pred[i]])[0]
            confidence = y_pred_probs[i][y_pred[i]] * 100

            c_id = test_case_ids[i] if test_case_ids is not None and i < len(test_case_ids) else None

            if c_id is not None:
                c_id = str(c_id).replace("Case ", "").replace("case ", "").strip()
                if c_id not in case_counters:
                    case_counters[c_id] = 1
                else:
                    case_counters[c_id] += 1
                c_idx = case_counters[c_id]
            else:
                c_idx = None

            results.append({
                "case_id": c_id,
                "case_index": c_idx,
                "sequence": ", ".join(decoded_seq),
                "true_outcome": str(true_decoded),
                "predicted_outcome": str(pred_decoded),
                "confidence_percent": float(round(confidence, 2))
            })

        results_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, "transformer_predictions.csv")
        results_df.to_csv(output_path, index=False)

        import json
        json_output_path = os.path.join(output_dir, "transformer_predictions.json")
        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path} and {json_output_path}")

        import pickle
        with open(os.path.join(output_dir, "transformer_artifacts.pkl"), "wb") as f:
            pickle.dump({
                "label_encoder": self.label_encoder,
                "activity_encoder": self.activity_encoder,
                "vocab_size": self.vocab_size,
                "num_outcome_classes": self.num_outcome_classes,
                "max_len": self.max_len,
            }, f)

    def plot_training_history(self, output_dir):
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        print("\nPlotting training history...")

        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history["accuracy"], label="Training Accuracy", linewidth=2)
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax1.set_title("Outcome Prediction Accuracy", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history["loss"], label="Training Loss", linewidth=2)
        ax2.plot(self.history.history["val_loss"], label="Validation Loss", linewidth=2)
        ax2.set_title("Outcome Prediction Loss", fontsize=14)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "outcome_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training history plot saved to: {output_path}")

        print("\nTraining metrics:")
        print(f"Final training accuracy: {self.history.history['accuracy'][-1] * 100:.2f}%")
        print(f"Final validation accuracy: {self.history.history['val_accuracy'][-1] * 100:.2f}%")
        print(f"Best validation accuracy: {max(self.history.history['val_accuracy']) * 100:.2f}%")

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "outcome_transformer.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
