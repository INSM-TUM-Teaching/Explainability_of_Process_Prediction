"""Training / evaluation loop for the GNN predictor.

Provides :class:`TrainingMixin`: data-loader construction, the per-epoch train
step, metric evaluation, the early-stopping training loop, and the detailed
test-set evaluation that produces the predictions DataFrame. This half is the
model-agnostic "engine" and can largely be reused by other graph predictors.
"""
import os

import numpy as np
import torch
from torch_geometric.data import Batch


class TrainingMixin:

    def create_loaders(
        self, train_graphs, val_graphs, test_graphs, batch_size=64, num_workers=None
    ):
        # Centralized worker calculation
        if num_workers is None:
            max_cores = os.cpu_count() or 1
            num_workers = min(0, max_cores)

        # PyTorch DataLoader worker subprocesses are unreliable on Windows:
        # workers are spawned (not forked), so each one re-imports the whole
        # module tree (re-initializing TensorFlow) and, together with CUDA +
        # pin_memory, frequently deadlocks at loader start/teardown. The graphs
        # are already in memory, so load synchronously there.
        if os.name == "nt":
            num_workers = 0

        loader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=Batch.from_data_list,
            # Pro-tip: Add this line below to speed up data transfer to the GPU!
            pin_memory=True if torch.cuda.is_available() else False,
        )

        train_loader = torch.utils.data.DataLoader(
            train_graphs, shuffle=True, **loader_args
        )
        val_loader = torch.utils.data.DataLoader(
            val_graphs, shuffle=False, **loader_args
        )
        test_loader = torch.utils.data.DataLoader(
            test_graphs, shuffle=False, **loader_args
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, loader, log_every=200):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader, 1):
            batch = batch.to(self.device)
            act_logits, time_pred, rem_pred, outcome_logits = self.model(batch)
            loss = self.model.compute_loss(act_logits, time_pred, rem_pred, batch, outcome_logits)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if log_every and batch_idx % log_every == 0:
                print(
                    f"  [Train] batch {batch_idx}/{len(loader)} loss={loss.item():.4f}"
                )

        return total_loss / len(loader)

    def evaluate(self, loader, max_batches=None):
        self.model.eval()
        correct = 0
        total = 0
        mae_time = 0.0
        mae_rem = 0.0
        total_loss = 0.0
        outcome_correct = 0
        outcome_total = 0
        batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                act_logits, time_pred, rem_pred, outcome_logits = self.model(batch)
                loss = self.model.compute_loss(act_logits, time_pred, rem_pred, batch, outcome_logits)
                total_loss += loss.item()
                batches += 1

                y_act = batch.y_activity.view(-1)
                pred = act_logits.argmax(dim=1)
                correct += (pred == y_act).sum().item()
                total += y_act.numel()

                y_time = batch.y_timestamp.view(-1)
                y_rem = batch.y_remaining_time.view(-1)
                mae_time += torch.abs(time_pred - y_time).mean().item()
                mae_rem += torch.abs(rem_pred - y_rem).mean().item()

                if outcome_logits is not None and hasattr(batch, "y_outcome"):
                    y_out = batch.y_outcome.view(-1)
                    pred_out = outcome_logits.argmax(dim=1)
                    outcome_correct += (pred_out == y_out).sum().item()
                    outcome_total += y_out.numel()

                if max_batches is not None and batches >= max_batches:
                    break

        result = {
            "accuracy": correct / total if total else 0.0,
            "mae_time": mae_time / max(batches, 1),
            "mae_rem": mae_rem / max(batches, 1),
            "loss": total_loss / max(batches, 1),
        }
        if outcome_total > 0:
            result["outcome_accuracy"] = outcome_correct / outcome_total
        return result

    def train(
        self,
        data,
        epochs=50,
        batch_size=64,
        patience=10,
        # `num_workers` make it either 4, 8 or 10 based on PC
        # its basically the number of CPU subprocess
        num_workers=4,
        log_every=200,
        train_eval_batches=25,
    ):
        print(f"\nTraining GNN for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.lr}")

        train_loader, val_loader, _ = self.create_loaders(
            data["train"],
            data["val"],
            data["test"],
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # create_loaders forces 0 workers on Windows (spawn deadlocks); report
        # the value actually used so the log is not misleading.
        effective_workers = 0 if os.name == "nt" else num_workers
        print(f"Using {effective_workers} CPU workers for data loading")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, log_every=log_every)
            train_metrics = self.evaluate(train_loader, max_batches=train_eval_batches)
            val_metrics = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_mae_time"].append(val_metrics["mae_time"])
            self.history["val_mae_rem"].append(val_metrics["mae_rem"])

            print(
                f"Epoch {epoch:03d} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"MAE Time: {val_metrics['mae_time']:.4f} | "
                f"MAE Rem: {val_metrics['mae_rem']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print("\nTraining completed!")

    def evaluate_test(self, data, batch_size=64):
        print("\nEvaluating on test set...")

        _, _, test_loader = self.create_loaders(
            data["train"], data["val"], data["test"], batch_size=batch_size
        )

        metrics = self.evaluate(test_loader)

        self.model.eval()
        results = []
        inv_act_vocab = (
            {i: v for v, i in self.vocabs["Activity"].items()}
            if hasattr(self, "vocabs") and "Activity" in self.vocabs
            else {}
        )
        inv_outcome_vocab = (
            {i: v for v, i in self._outcome_vocab.items()}
            if hasattr(self, "_outcome_vocab")
            else {}
        )

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                act_logits, time_pred, rem_pred, outcome_logits = self.model(batch)

                probs = torch.softmax(act_logits, dim=1)
                confidences = probs.max(dim=1)[0].cpu().numpy() * 100

                y_act = batch.y_activity.view(-1).cpu().numpy()
                pred_act = act_logits.argmax(dim=1).cpu().numpy()
                y_time = batch.y_timestamp.view(-1).cpu().numpy()
                pred_time = time_pred.cpu().numpy()
                y_rem = batch.y_remaining_time.view(-1).cpu().numpy()
                pred_rem = rem_pred.cpu().numpy()

                has_outcome = outcome_logits is not None and hasattr(batch, "y_outcome")
                if has_outcome:
                    outcome_probs = torch.softmax(outcome_logits, dim=1)
                    outcome_confs = outcome_probs.max(dim=1)[0].cpu().numpy() * 100
                    y_outcome = batch.y_outcome.view(-1).cpu().numpy()
                    pred_outcome = outcome_logits.argmax(dim=1).cpu().numpy()

                case_ids = (
                    batch.case_id if hasattr(batch, "case_id") else [None] * len(y_act)
                )
                case_indices = (
                    batch.case_index
                    if hasattr(batch, "case_index")
                    else [None] * len(y_act)
                )

                act_x = batch["activity"].x.argmax(dim=-1).cpu().numpy()
                act_batch = batch["activity"].batch.cpu().numpy()

                sequences = []
                for i in range(len(y_act)):
                    graph_act_indices = act_x[act_batch == i]
                    decoded_seq = [
                        inv_act_vocab.get(int(idx), str(idx))
                        for idx in graph_act_indices
                    ]
                    sequences.append(", ".join(decoded_seq))

                for i in range(len(y_act)):
                    cid = (
                        case_ids[i].item()
                        if hasattr(case_ids[i], "item")
                        else case_ids[i]
                    )
                    cidx = (
                        case_indices[i].item()
                        if hasattr(case_indices[i], "item")
                        else case_indices[i]
                    )

                    ptr_start = int(batch["time"].ptr[i])
                    ptr_end = int(batch["time"].ptr[i+1]) - 1

                    first_ts_log = float(batch["time"].x[ptr_start][0])
                    last_ts_log = float(batch["time"].x[ptr_end][0])

                    first_ts_sec = np.expm1(first_ts_log)
                    last_ts_sec = np.expm1(last_ts_log)
                    elapsed_days = (last_ts_sec - first_ts_sec) / 86400.0

                    row = {
                        "case_id": cid,
                        "case_index": cidx,
                        "sequence": sequences[i],
                        "true_next_activity": inv_act_vocab.get(
                            int(y_act[i]), y_act[i]
                        ),
                        "predicted_next_activity": inv_act_vocab.get(
                            int(pred_act[i]), pred_act[i]
                        ),
                        "confidence_percent": round(float(confidences[i]), 2),
                        "actual_event_time_days": float(np.expm1(y_time[i]) / 86400.0),
                        "predicted_event_time_days": float(np.expm1(pred_time[i]) / 86400.0),
                        "actual_remaining_time_days": float(np.expm1(y_rem[i]) / 86400.0),
                        "predicted_remaining_time_days": float(np.expm1(pred_rem[i]) / 86400.0),
                        "current_elapsed_time_days": float(elapsed_days),
                    }

                    if has_outcome:
                        row["true_outcome"] = inv_outcome_vocab.get(int(y_outcome[i]), str(y_outcome[i]))
                        row["predicted_outcome"] = inv_outcome_vocab.get(int(pred_outcome[i]), str(pred_outcome[i]))
                        row["outcome_confidence_percent"] = round(float(outcome_confs[i]), 2)

                    results.append(row)

        import pandas as pd

        metrics["predictions_df"] = pd.DataFrame(results)

        return metrics
