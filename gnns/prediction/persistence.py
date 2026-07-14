"""Saving artifacts: model weights, the training-history plot, and results.

Provides :class:`PersistenceMixin` (``save_model``, ``plot_training_history``,
``save_results``). Kept apart from the training engine so I/O and plotting can
evolve without touching the learning code.
"""
import os

import matplotlib.pyplot as plt
import torch


class PersistenceMixin:

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "gnn_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def plot_training_history(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.history["train_acc"], label="Train")
        axes[0, 0].plot(self.history["val_acc"], label="Validation")
        axes[0, 0].set_title("Activity Prediction Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history["train_loss"], label="Train")
        axes[0, 1].plot(self.history["val_loss"], label="Validation")
        axes[0, 1].set_title("Total Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history["val_mae_time"])
        axes[1, 0].set_title("Event Time MAE (Validation)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history["val_mae_rem"])
        axes[1, 1].set_title("Remaining Time MAE (Validation)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MAE")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "gnn_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training history plot saved to: {output_path}")

    def save_results(self, metrics, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "gnn_results.txt")
        with open(results_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("GNN MODEL - EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Accuracy (Activity):  {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Test MAE (Event Time):     {metrics['mae_time']:.4f}\n")
            f.write(f"Test MAE (Remaining Time): {metrics['mae_rem']:.4f}\n")
            if "outcome_accuracy" in metrics:
                f.write(f"Test Accuracy (Outcome):   {metrics['outcome_accuracy']*100:.2f}%\n")
            f.write(f"Test Loss:                 {metrics['loss']:.4f}\n")
            f.write("\n" + "=" * 50 + "\n")

        print(f"Results saved to: {results_file}")

        if "predictions_df" in metrics:
            pred_file = os.path.join(output_dir, "gnn_predictions.csv")
            # Create clean case id similarly
            df = metrics["predictions_df"].copy()
            df["case_id"] = (
                df["case_id"]
                .astype(str)
                .str.replace("Case ", "", regex=False)
                .str.replace("case ", "", regex=False)
                .str.replace(" ", "_")
                .str.strip()
            )

            # Filter out target columns we didn't train on (loss weight == 0.0)
            if hasattr(self, "loss_weights"):
                if self.loss_weights[0] == 0.0:
                    df = df.drop(
                        columns=[
                            "true_next_activity",
                            "predicted_next_activity",
                            "confidence_percent",
                        ],
                        errors="ignore",
                    )
                if self.loss_weights[1] == 0.0:
                    df = df.drop(
                        columns=["actual_event_time_days", "predicted_event_time_days"],
                        errors="ignore",
                    )
                if self.loss_weights[2] == 0.0:
                    df = df.drop(
                        columns=[
                            "actual_remaining_time_days",
                            "predicted_remaining_time_days",
                        ],
                        errors="ignore",
                    )
                w_outcome = self.loss_weights[3] if len(self.loss_weights) > 3 else 0.0
                if w_outcome == 0.0:
                    df = df.drop(
                        columns=[
                            "true_outcome",
                            "predicted_outcome",
                            "outcome_confidence_percent",
                        ],
                        errors="ignore",
                    )

            df.to_csv(pred_file, index=False)
            print(f"Predictions saved to: {pred_file}")

            json_file = os.path.join(output_dir, "gnn_predictions.json")
            df.to_json(json_file, orient="records", indent=2)

            # Remove from metrics dictionary so it doesn't break JSON serialization down the line
            del metrics["predictions_df"]
