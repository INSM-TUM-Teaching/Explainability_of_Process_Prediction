import os
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import Lasso
from torch_geometric.data import Batch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)


def _dir_has_png(path):
    if not os.path.isdir(path):
        return False
    for _, _, files in os.walk(path):
        if any(name.lower().endswith(".png") for name in files):
            return True
    return False


def _write_placeholder_plot(output_path, title, lines=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    body = [title] + (lines or [])
    ax.text(0.5, 0.5, "\n".join(body), ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


class GradientExplainer:
    def __init__(self, model, device, vocabularies=None):
        self.model = model
        self.device = device

        if callable(vocabularies):
            try:
                self.vocabs = vocabularies()
            except:
                self.vocabs = {}
        else:
            self.vocabs = vocabularies or {}
        if not isinstance(self.vocabs, dict):
            self.vocabs = {}

    def _get_activity_name(self, idx):
        if "Activity" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Activity"].items()}
            name = inv_vocab.get(int(idx), f"Act_{idx}")
            return name[:18] + ".." if len(name) > 18 else name
        return f"activity_{idx}"

    def _get_resource_name(self, idx):
        if "Resource" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Resource"].items()}
            name = inv_vocab.get(int(idx), f"Res_{idx}")
            name = name[:12] + ".." if len(name) > 12 else name
            return f"{name} [Resource]"
        return f"Resource_{idx} [Resource]"

    def explain_time_series_with_features(
        self, graphs, task="event_time", num_samples=50
    ):
        self.model.eval()

        time_step_data = []
        predictions = []
        true_values = []

        sample_indices = np.random.choice(
            len(graphs), min(num_samples, len(graphs)), replace=False
        )
        selected_graphs = [graphs[i] for i in sample_indices]

        print(
            f"Computing feature-level contributions for {len(selected_graphs)} graphs ({task})..."
        )

        for graph in tqdm(selected_graphs, desc=f"Analyzing ({task})"):
            graph = graph.to(self.device)
            seq_len = graph["activity"].x.shape[0]

            for key in graph.x_dict:
                if key in ["activity", "resource", "time"]:
                    graph.x_dict[key] = graph.x_dict[key].detach().clone()
                    graph.x_dict[key].requires_grad = True

            out = self.model(graph)

            if task == "event_time":
                score = out[1]
                true_val = graph.y_timestamp.item()
            elif task == "remaining_time":
                score = out[2]
                true_val = graph.y_remaining_time.item()
            else:
                continue

            self.model.zero_grad()
            score.backward()

            with torch.no_grad():
                for step in range(seq_len):
                    step_info = {"step": step}
                    contrib_sum = 0.0

                    if task in ["event_time", "remaining_time"]:
                        if (
                            "time" in graph.x_dict
                            and graph.x_dict["time"].grad is not None
                        ):
                            time_grad = graph.x_dict["time"].grad[step]
                            time_inp = graph.x_dict["time"][step]
                            contrib_sum += (time_grad * time_inp).abs().sum().item()
                            step_info["timestamp"] = time_inp.item()

                        if "activity" in graph.x_dict:
                            act_inp = graph.x_dict["activity"][step]
                            act_idx = act_inp.argmax().item()
                            step_info["activity"] = self._get_activity_name(act_idx)

                        if "resource" in graph.x_dict:
                            res_inp = graph.x_dict["resource"][step]
                            res_idx = res_inp.argmax().item()
                            step_info["resource"] = self._get_resource_name(res_idx)

                    else:
                        if (
                            "activity" in graph.x_dict
                            and graph.x_dict["activity"].grad is not None
                        ):
                            act_grad = graph.x_dict["activity"].grad[step]
                            act_inp = graph.x_dict["activity"][step]
                            contrib_sum += (act_grad * act_inp).abs().sum().item()
                            act_idx = act_inp.argmax().item()
                            step_info["activity"] = self._get_activity_name(act_idx)

                        if (
                            "resource" in graph.x_dict
                            and graph.x_dict["resource"].grad is not None
                        ):
                            res_grad = graph.x_dict["resource"].grad[step]
                            res_inp = graph.x_dict["resource"][step]
                            contrib_sum += (res_grad * res_inp).abs().sum().item()
                            res_idx = res_inp.argmax().item()
                            step_info["resource"] = self._get_resource_name(res_idx)

                        if (
                            "time" in graph.x_dict
                            and graph.x_dict["time"].grad is not None
                        ):
                            time_grad = graph.x_dict["time"].grad[step]
                            time_inp = graph.x_dict["time"][step]
                            contrib_sum += (time_grad * time_inp).abs().sum().item()
                            step_info["timestamp"] = time_inp.item()

                    step_info["contribution"] = contrib_sum
                    time_step_data.append(step_info)

                predictions.append(score.item())
                true_values.append(true_val)

        df = pd.DataFrame(time_step_data)

        step_summary = []
        for step in range(df["step"].max() + 1):
            step_data = df[df["step"] == step]

            summary = {
                "step": step,
                "avg_contribution": step_data["contribution"].mean(),
                "std_contribution": step_data["contribution"].std(),
            }

            if "activity" in step_data.columns:
                top_activity = step_data["activity"].mode()
                if len(top_activity) > 0:
                    summary["top_activity"] = top_activity.iloc[0]
                    summary["activity_freq"] = (
                        step_data["activity"] == top_activity.iloc[0]
                    ).sum() / len(step_data)
                else:
                    summary["top_activity"] = "N/A"
                    summary["activity_freq"] = 0

            if "resource" in step_data.columns:
                top_resource = step_data["resource"].mode()
                if len(top_resource) > 0:
                    summary["top_resource"] = top_resource.iloc[0]
                    summary["resource_freq"] = (
                        step_data["resource"] == top_resource.iloc[0]
                    ).sum() / len(step_data)
                else:
                    summary["top_resource"] = "N/A"
                    summary["resource_freq"] = 0

            step_summary.append(summary)

        summary_df = pd.DataFrame(step_summary)

        return {
            "summary": summary_df,
            "detailed_data": df,
            "predictions": predictions,
            "true_values": true_values,
        }

    def explain_individual_sample(self, graph, task="event_time"):
        self.model.eval()
        graph = graph.to(self.device)

        seq_len = graph["activity"].x.shape[0]

        for key in graph.x_dict:
            if key in ["activity", "resource", "time"]:
                graph.x_dict[key] = graph.x_dict[key].detach().clone()
                graph.x_dict[key].requires_grad = True

        out = self.model(graph)

        if task == "event_time":
            score = out[1]
            true_val = graph.y_timestamp.item()
        elif task == "remaining_time":
            score = out[2]
            true_val = graph.y_remaining_time.item()
        elif task in ["activity", "next_activity"]:
            predicted_class = out[0].argmax()
            score = out[0].view(-1)[predicted_class]
            true_val = graph.y_activity.item()
        else:
            return None, None, None, None

        self.model.zero_grad()
        score.backward()

        step_contributions = []
        step_info = []

        with torch.no_grad():
            for step in range(seq_len):
                contrib_sum = 0.0
                info = {"step": step}

                if task in ["event_time", "remaining_time"]:
                    if "time" in graph.x_dict and graph.x_dict["time"].grad is not None:
                        time_grad = graph.x_dict["time"].grad[step]
                        time_inp = graph.x_dict["time"][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                        info["timestamp"] = time_inp.item()

                    if "activity" in graph.x_dict:
                        act_inp = graph.x_dict["activity"][step]
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if "resource" in graph.x_dict:
                        res_inp = graph.x_dict["resource"][step]
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                elif task in ["activity", "next_activity"]:
                    if (
                        "activity" in graph.x_dict
                        and graph.x_dict["activity"].grad is not None
                    ):
                        act_grad = graph.x_dict["activity"].grad[step]
                        act_inp = graph.x_dict["activity"][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if "resource" in graph.x_dict:
                        res_inp = graph.x_dict["resource"][step]
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                    if "time" in graph.x_dict:
                        time_inp = graph.x_dict["time"][step]
                        info["timestamp"] = time_inp.item()

                else:
                    if (
                        "activity" in graph.x_dict
                        and graph.x_dict["activity"].grad is not None
                    ):
                        act_grad = graph.x_dict["activity"].grad[step]
                        act_inp = graph.x_dict["activity"][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if (
                        "resource" in graph.x_dict
                        and graph.x_dict["resource"].grad is not None
                    ):
                        res_grad = graph.x_dict["resource"].grad[step]
                        res_inp = graph.x_dict["resource"][step]
                        contrib_sum += (res_grad * res_inp).abs().sum().item()
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                    if "time" in graph.x_dict and graph.x_dict["time"].grad is not None:
                        time_grad = graph.x_dict["time"].grad[step]
                        time_inp = graph.x_dict["time"][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                        info["timestamp"] = time_inp.item()

                step_contributions.append(contrib_sum)
                step_info.append(info)

        return np.array(step_contributions), score.item(), true_val, step_info

    def plot_individual_gradient_explanation(
        self, contributions, pred, true_val, step_info, output_dir, task, sample_id
    ):
        os.makedirs(output_dir, exist_ok=True)

        if contributions is None:
            return

        num_steps = len(contributions)

        timestamps = []
        for info in step_info:
            if "timestamp" in info:
                timestamps.append(info["timestamp"])
            else:
                timestamps.append(None)

        fig, ax = plt.subplots(figsize=(18, 8))

        centered_contrib = (
            contributions  # removed np.mean(contributions) mean-centering
        )

        if task in ["event_time", "remaining_time"]:
            x_labels = []
            for i, info in enumerate(step_info):
                activity = info.get("activity", "N/A")
                if len(activity) > 15:
                    activity = activity[:13] + ".."
                    
                if timestamps[i] is not None:
                    timestamp = timestamps[i]
                    if timestamp < 1:
                        time_str = f"{timestamp*24:.1f}h"
                    else:
                        time_str = f"Day {timestamp:.1f}"
                else:
                    time_str = f"Step {info['step']}"
                
                x_labels.append(f"{activity}\n{time_str}")
        else:
            x_labels = []
            for info in step_info:
                activity = info.get("activity", f"Step_{info['step']}")
                if len(activity) > 15:
                    activity = activity[:13] + ".."
                x_labels.append(activity)

        time_steps = np.arange(num_steps)
        positive = np.maximum(centered_contrib, 0)
        negative = np.minimum(centered_contrib, 0)

        ax.bar(
            time_steps,
            positive,
            color="#2ca02c",
            alpha=1.0,
            label="Supports",
            width=0.75,
            edgecolor="#27ae60",
            linewidth=0.8,
        )
        ax.bar(
            time_steps,
            negative,
            color="#d62728",
            alpha=1.0,
            label="Contradicts",
            width=0.75,
            edgecolor="#c0392b",
            linewidth=0.8,
        )

        max_y = max(
            positive.max() if len(positive) > 0 else 0,
            abs(negative.min()) if len(negative) > 0 else 0,
        )
        offset = max_y * 0.05 if max_y > 0 else 0.01

        for i, (p, n) in enumerate(zip(positive, negative)):
            if p > 0:
                ax.text(
                    time_steps[i],
                    p + offset,
                    f"+{p:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            elif n < 0:
                ax.text(
                    time_steps[i],
                    n - offset,
                    f"{n:.3f}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=2)


        if task in ["event_time", "remaining_time"]:
            xlabel = "Event (Timestamp)"
        else:
            xlabel = "Event (activity + Timestamp)"

        ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
        ax.set_ylabel("Gradient Value (Contribution)", fontweight="bold", fontsize=14)

        task_name = task.replace("_", " ").title()

        ax.legend(loc="upper right", framealpha=0.95, fontsize=12)

        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(time_steps)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")

        max_contrib = (
            np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
        )
        y_margin = max_contrib * 0.45
        ax.set_ylim(
            negative.min() - y_margin if len(negative) > 0 else -1,
            positive.max() + y_margin if len(positive) > 0 else 1,
        )

        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f"gradient_timestep_heatmap_sample_{sample_id}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(
            f"  [✓] Gradient plot saved: gradient_timestep_heatmap_sample_{sample_id}.png"
        )

        if step_info:
            df_info = pd.DataFrame(step_info)
            import re

            m = re.search(r"case_(.+?)_idx_(.+)", str(sample_id))
            if m:
                df_info.insert(0, "case_id", m.group(1))
                df_info.insert(1, "case_index", m.group(2))

            df_info["importance"] = centered_contrib

            cols = []
            if m:
                cols = ["case_id", "case_index"]
            if "activity" in df_info.columns:
                df_info = df_info.rename(columns={"activity": "activity"})
                cols.append("activity")
            cols.append("importance")

            # Apply column selection
            df_info = df_info[cols]

            csv_path = os.path.join(
                output_dir, f"gradient_timestep_sample_{sample_id}_details.csv"
            )

            if task in ["event_time", "remaining_time"]:
                with open(csv_path, "w") as f:
                    f.write(f"# Task: {task}\n")
                    f.write(f"# Gradient values from: TIMESTAMP FEATURES ONLY\n")
                    f.write(f"# activity shown for context only\n")
                    df_info.to_csv(f, index=False)
            else:
                df_info.to_csv(csv_path, index=False)

            print(f"  [✓] Details CSV saved")

    def plot_with_readable_table(self, results, output_dir, task):
        os.makedirs(output_dir, exist_ok=True)

        summary_df = results["summary"]
        predictions = results["predictions"]
        true_values = results["true_values"]

        max_len = len(summary_df)

        if max_len > 40:
            bin_size = max(2, max_len // 35)
            print(
                f"[INFO] Long sequence ({max_len} steps) - binning into ~{max_len//bin_size} bins"
            )

            binned_data = []
            for bin_idx in range((max_len + bin_size - 1) // bin_size):
                start_step = bin_idx * bin_size
                end_step = min((bin_idx + 1) * bin_size, max_len)
                bin_df = summary_df[
                    (summary_df["step"] >= start_step) & (summary_df["step"] < end_step)
                ]

                binned_data.append(
                    {
                        "step": bin_idx,
                        "step_range": f"{start_step}-{end_step-1}",
                        "avg_contribution": bin_df["avg_contribution"].mean(),
                        "top_activity": (
                            bin_df["top_activity"].mode().iloc[0]
                            if len(bin_df["top_activity"].mode()) > 0
                            else "N/A"
                        ),
                    }
                )

            plot_df = pd.DataFrame(binned_data)
            x_values = plot_df["step"].values
            contributions = plot_df["avg_contribution"].values
            activities = plot_df["top_activity"].values
            xlabel_text = "Event sequence position (binned)"
        else:
            x_values = summary_df["step"].values
            contributions = summary_df["avg_contribution"].values
            activities = summary_df["top_activity"].values
            xlabel_text = "Event sequence position (BPI Dataset)"

        mean_val = (
            np.mean(contributions[contributions > 0])
            if np.any(contributions > 0)
            else 0
        )
        centered_contrib = contributions  # removed - mean_val mean-centering

        fig, ax = plt.subplots(figsize=(20, 8))

        positive = np.maximum(centered_contrib, 0)
        negative = np.minimum(centered_contrib, 0)

        ax.bar(
            x_values,
            positive,
            color="#2ca02c",
            alpha=1.0,
            label="Supports",
            width=0.75,
            edgecolor="darkred",
            linewidth=0.8,
        )
        ax.bar(
            x_values,
            negative,
            color="#d62728",
            alpha=1.0,
            label="Contradicts",
            width=0.75,
            edgecolor="darkblue",
            linewidth=0.8,
        )

        max_y = max(
            positive.max() if len(positive) > 0 else 0,
            abs(negative.min()) if len(negative) > 0 else 0,
        )
        offset = max_y * 0.05 if max_y > 0 else 0.01

        for i, (p, n) in enumerate(zip(positive, negative)):
            if p > 0:
                ax.text(
                    x_values[i],
                    p + offset,
                    f"+{p:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            elif n < 0:
                ax.text(
                    x_values[i],
                    n - offset,
                    f"{n:.3f}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=2)

        max_contrib = (
            np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
        )
        label_offset = max_contrib * 0.12

        threshold = (
            np.percentile(np.abs(centered_contrib), 70)
            if len(centered_contrib) > 0
            else 0
        )

        for i, (x, contrib, activity) in enumerate(
            zip(x_values, centered_contrib, activities)
        ):
            if abs(contrib) >= threshold:
                activity_str = str(activity)
                if len(activity_str) > 12:
                    activity_str = activity_str[:10] + ".."

                if contrib > 0:
                    y_pos = contrib + label_offset
                    va = "bottom"
                else:
                    y_pos = contrib - label_offset
                    va = "top"

                bbox_props = dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#FFFACD",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                )

                ax.text(
                    x,
                    y_pos,
                    activity_str,
                    ha="center",
                    va=va,
                    fontsize=10,
                    fontweight="bold",
                    rotation=45,
                    bbox=bbox_props,
                )

        num_samples = len(predictions) if len(predictions) > 0 else 0
        ax.set_xlabel(xlabel_text, fontweight="bold", fontsize=14)
        ax.set_ylabel("Feature Contribution", fontweight="bold", fontsize=14)

        task_name = task.replace("_", " ").title()

        ax.legend(loc="upper left", framealpha=0.95, fontsize=12)

        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(x_values) <= 30:
            ax.set_xticks(x_values)
            if max_len > 40:
                labels = [plot_df.iloc[i]["step_range"] for i in range(len(x_values))]
                ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
            else:
                ax.set_xticklabels([f"{int(x)}" for x in x_values], fontsize=10)
        else:
            skip = max(1, len(x_values) // 30)
            ax.set_xticks(x_values[::skip])
            ax.set_xticklabels([f"{int(x)}" for x in x_values[::skip]], fontsize=10)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"feature_level_{task}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

        print(f"[✓] SHAP-style plot saved")

        summary_df.to_csv(
            os.path.join(output_dir, f"feature_summary_{task}.csv"), index=False
        )
        print(f"[✓] CSV saved")

    def explain_global_importance(self, graphs, task="activity", num_samples=50):
        self.model.eval()
        importances = {"activity": [], "resource": []}

        sample_indices = np.random.choice(
            len(graphs), min(num_samples, len(graphs)), replace=False
        )
        selected_graphs = [graphs[i] for i in sample_indices]

        print(f"Computing gradients for {len(selected_graphs)} graphs ({task})...")

        for graph in tqdm(selected_graphs, desc=f"{task.capitalize()} Analysis"):
            graph = graph.to(self.device)
            for key in graph.x_dict:
                if key in ["activity", "resource"]:
                    graph.x_dict[key] = graph.x_dict[key].detach().clone()
                    graph.x_dict[key].requires_grad = True

            out = self.model(graph)
            if task == "activity":
                logits = out[0]
                pred_idx = logits.argmax(dim=1)
                score = logits[0, pred_idx]
            elif task == "event_time":
                score = out[1]
            elif task == "remaining_time":
                score = out[2]
            else:
                score = out[0].sum()  # Fallback

            self.model.zero_grad()
            score.backward()

            with torch.no_grad():
                for node_type in ["activity", "resource"]:
                    if (
                        node_type in graph.x_dict
                        and graph.x_dict[node_type].grad is not None
                    ):
                        grad = graph.x_dict[node_type].grad
                        inp = graph.x_dict[node_type]
                        imp = (grad * inp).abs().sum(dim=0).cpu().numpy()
                        importances[node_type].append(imp)

        global_imp = {}
        for n_type in importances:
            if len(importances[n_type]) > 0:
                global_imp[n_type] = np.mean(np.stack(importances[n_type]), axis=0)

        return global_imp

    def plot_global_importance(self, importances, output_dir, task="activity"):
        os.makedirs(output_dir, exist_ok=True)

        data = []
        for n_type, scores in importances.items():
            top_k_idx = np.argsort(scores)[-10:]
            for idx in top_k_idx:
                if scores[idx] > 0:
                    if n_type == "activity":
                        name = self._get_activity_name(idx)
                    elif n_type == "resource":
                        name = self._get_resource_name(idx)
                    elif n_type == "time":
                        name = "Elapsed Time"
                    else:
                        name = f"{n_type}_{idx}"

                    data.append({"activity": name, "importance": scores[idx]})

        df = pd.DataFrame(data).sort_values("importance", ascending=True)
        df.to_csv(os.path.join(output_dir, f"gradient_global_{task}.csv"), index=False)

        if df.empty:
            print(f"[WARNING] No gradient importances available for {task}.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        colors = {"activity": "#1f77b4", "Resource": "#ff7f0e"}

        plt.barh(
            df["activity"],
            df["importance"],
            color=[
                colors.get(t, "grey")
                for t in df["activity"].apply(
                    lambda x: "Activity" if "Act_" in x else "Resource"
                )
            ],
        )

        task_name_title = task.replace("_", " ").title()
        plt.xlabel("Mean Gradient Magnitude (Impact)", fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
        plt.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"gradient_global_{task}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

        print(f"[✓] {task_name_title} bar chart saved")


class TemporalGradientExplainer:
    """Temporal attribution explainer with timestep-level gradients."""

    def __init__(self, model, device, vocabularies=None, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler

        if callable(vocabularies):
            try:
                self.vocabs = vocabularies()
            except Exception:
                self.vocabs = {}
        else:
            self.vocabs = vocabularies or {}
        if not isinstance(self.vocabs, dict):
            self.vocabs = {}

    def _get_activity_name(self, idx):
        if "Activity" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Activity"].items()}
            name = inv_vocab.get(int(idx), f"Act_{idx}")
            return name[:18] + ".." if len(name) > 18 else name
        return f"activity_{idx}"

    def _get_resource_name(self, idx):
        if "Resource" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Resource"].items()}
            name = inv_vocab.get(int(idx), f"Res_{idx}")
            name = name[:12] + ".." if len(name) > 12 else name
            return f"{name} [Resource]"
        return f"Resource_{idx} [Resource]"

    def explain_individual_sample(self, graph, task="event_time"):
        self.model.eval()
        graph = graph.to(self.device)

        seq_len = graph["activity"].x.shape[0]

        for key in graph.x_dict:
            if key in ["activity", "resource", "time"]:
                graph.x_dict[key] = graph.x_dict[key].detach().clone()
                graph.x_dict[key].requires_grad = True

        out = self.model(graph)

        if task == "event_time":
            score = out[1]
            if score.numel() > 1:
                score = score.sum()
            true_val = graph.y_timestamp.item()
        elif task == "remaining_time":
            score = out[2]
            if score.numel() > 1:
                score = score.sum()
            true_val = graph.y_remaining_time.item()
        elif task == "activity":
            logits = out[0]
            if logits.dim() > 1:
                logits = logits[0]
            predicted_class = logits.argmax()
            score = logits[predicted_class]
            if score.numel() > 1:
                score = score.sum()
            true_val = graph.y_activity.item()
        else:
            return None, None, None, None

        self.model.zero_grad()
        score.backward()

        step_contributions = []
        step_info = []

        with torch.no_grad():
            for step in range(seq_len):
                contrib_sum = 0.0
                info = {"step": step}

                if task in ["event_time", "remaining_time"]:
                    if "time" in graph.x_dict and graph.x_dict["time"].grad is not None:
                        time_grad = graph.x_dict["time"].grad[step]
                        time_inp = graph.x_dict["time"][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                        info["timestamp"] = time_inp.item()

                    if "activity" in graph.x_dict:
                        act_inp = graph.x_dict["activity"][step]
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if "resource" in graph.x_dict:
                        res_inp = graph.x_dict["resource"][step]
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                elif task == "activity":
                    if (
                        "activity" in graph.x_dict
                        and graph.x_dict["activity"].grad is not None
                    ):
                        act_grad = graph.x_dict["activity"].grad[step]
                        act_inp = graph.x_dict["activity"][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if "resource" in graph.x_dict:
                        res_inp = graph.x_dict["resource"][step]
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                    if "time" in graph.x_dict:
                        time_inp = graph.x_dict["time"][step]
                        info["timestamp"] = time_inp.item()

                else:
                    if (
                        "activity" in graph.x_dict
                        and graph.x_dict["activity"].grad is not None
                    ):
                        act_grad = graph.x_dict["activity"].grad[step]
                        act_inp = graph.x_dict["activity"][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info["activity"] = self._get_activity_name(act_idx)

                    if (
                        "resource" in graph.x_dict
                        and graph.x_dict["resource"].grad is not None
                    ):
                        res_grad = graph.x_dict["resource"].grad[step]
                        res_inp = graph.x_dict["resource"][step]
                        contrib_sum += (res_grad * res_inp).abs().sum().item()
                        res_idx = res_inp.argmax().item()
                        info["resource"] = self._get_resource_name(res_idx)

                    if "time" in graph.x_dict and graph.x_dict["time"].grad is not None:
                        time_grad = graph.x_dict["time"].grad[step]
                        time_inp = graph.x_dict["time"][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                        info["timestamp"] = time_inp.item()

                step_contributions.append(contrib_sum)
                step_info.append(info)

        return np.array(step_contributions), score.item(), true_val, step_info

    def plot_individual_gradient_explanation(
        self, contributions, pred, true_val, step_info, output_dir, task, sample_id
    ):
        os.makedirs(output_dir, exist_ok=True)

        if contributions is None:
            return

        num_steps = len(contributions)
        timestamps = []
        for info in step_info:
            if "timestamp" in info:
                timestamps.append(info["timestamp"])
            else:
                timestamps.append(None)

        fig, ax = plt.subplots(figsize=(18, 8))

        centered_contrib = (
            contributions  # removed np.mean(contributions) mean-centering
        )

        if task in ["event_time", "remaining_time"]:
            x_labels = []
            for i, info in enumerate(step_info):
                if timestamps[i] is not None:
                    timestamp = timestamps[i]
                    if timestamp < 1:
                        x_labels.append(f"{timestamp*24:.1f}h")
                    else:
                        x_labels.append(f"Day {timestamp:.1f}")
                else:
                    x_labels.append(f"Step {info['step']}")
        else:
            x_labels = []
            for info in step_info:
                activity = info.get("activity", f"Step_{info['step']}")
                if len(activity) > 15:
                    activity = activity[:13] + ".."
                x_labels.append(activity)

        time_steps = np.arange(num_steps)
        positive = np.maximum(centered_contrib, 0)
        negative = np.minimum(centered_contrib, 0)

        ax.bar(
            time_steps,
            positive,
            color="#2ca02c",
            alpha=1.0,
            label="Supports",
            width=0.75,
            edgecolor="#27ae60",
            linewidth=0.8,
        )
        ax.bar(
            time_steps,
            negative,
            color="#d62728",
            alpha=1.0,
            label="Contradicts",
            width=0.75,
            edgecolor="#c0392b",
            linewidth=0.8,
        )

        max_y = max(
            positive.max() if len(positive) > 0 else 0,
            abs(negative.min()) if len(negative) > 0 else 0,
        )
        offset = max_y * 0.05 if max_y > 0 else 0.01

        for i, (p, n) in enumerate(zip(positive, negative)):
            if p > 0:
                ax.text(
                    time_steps[i],
                    p + offset,
                    f"+{p:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            elif n < 0:
                ax.text(
                    time_steps[i],
                    n - offset,
                    f"{n:.3f}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=2)

        if task in ["event_time", "remaining_time"]:
            max_contrib = (
                np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
            )
            label_offset = max_contrib * 0.15

            for i in range(num_steps):
                info = step_info[i]
                activity = info.get("activity", "N/A")
                if len(activity) > 12:
                    activity = activity[:10] + ".."

                contrib_val = centered_contrib[i]
                if contrib_val > 0:
                    y_pos = contrib_val + label_offset
                    va = "bottom"
                else:
                    y_pos = contrib_val - label_offset
                    va = "top"

                bbox_props = dict(
                    boxstyle="round,pad=0.35",
                    facecolor="lightyellow",
                    edgecolor="gray",
                    linewidth=0.6,
                    alpha=0.85,
                )

                ax.text(
                    time_steps[i],
                    y_pos,
                    activity,
                    ha="center",
                    va=va,
                    fontsize=8,
                    bbox=bbox_props,
                    rotation=0,
                )

        if task in ["event_time", "remaining_time"]:
            xlabel = "Event (Timestamp)"
        else:
            xlabel = "Event (activity + Timestamp)"

        ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
        ax.set_ylabel("Gradient Value (Contribution)", fontweight="bold", fontsize=14)

        ax.legend(loc="upper right", framealpha=0.95, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(time_steps)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")

        max_contrib = (
            np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
        )
        y_margin = max_contrib * 0.45
        ax.set_ylim(
            negative.min() - y_margin if len(negative) > 0 else -1,
            positive.max() + y_margin if len(positive) > 0 else 1,
        )

        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f"gradient_timestep_heatmap_sample_{sample_id}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(
            f"[OK] Gradient plot saved: gradient_timestep_heatmap_sample_{sample_id}.png"
        )

        if step_info:
            df_info = pd.DataFrame(step_info)
            import re

            m = re.search(r"case_(.+?)_idx_(.+)", str(sample_id))
            if m:
                df_info.insert(0, "case_id", m.group(1))
                df_info.insert(1, "case_index", m.group(2))

            df_info["gradient_value"] = centered_contrib

            if task in ["event_time", "remaining_time"]:
                cols = ["step"]
                if m:
                    cols = ["case_id", "case_index"] + cols
                if "timestamp" in df_info.columns:
                    cols.append("timestamp")
                if "activity" in df_info.columns:
                    cols.append("activity")
                cols.append("gradient_value")
                df_info = df_info[cols]

                csv_path = os.path.join(
                    output_dir, f"gradient_timestep_sample_{sample_id}_details.csv"
                )
                with open(csv_path, "w") as f:
                    f.write(f"# Task: {task}\n")
                    f.write("# Gradient values from: TIMESTAMP FEATURES ONLY\n")
                    f.write("# activity shown for context only\n")
                    df_info.to_csv(f, index=False)
            else:
                df_info.to_csv(
                    os.path.join(
                        output_dir, f"gradient_timestep_sample_{sample_id}_details.csv"
                    ),
                    index=False,
                )

    def generate_temporal_plots(
        self, graphs, output_dir, task="activity", num_samples=10, y_true=None
    ):
        os.makedirs(output_dir, exist_ok=True)

        if not graphs:
            print("[WARNING] Temporal gradient plot skipped: no graphs available.")
            return

        sample_count = min(num_samples, len(graphs))
        sample_indices = np.random.choice(len(graphs), sample_count, replace=False)
        print(
            f"Generating temporal gradient plots for {len(sample_indices)} samples ({task})..."
        )

        created = 0
        for i, idx in enumerate(sample_indices):
            graph = graphs[idx]
            try:
                contrib, pred, true_val, step_info = self.explain_individual_sample(
                    graph, task
                )
                if contrib is None:
                    print(f"[WARNING] Sample {idx} skipped (no contribution).")
                    continue
                self.plot_individual_gradient_explanation(
                    contrib, pred, true_val, step_info, output_dir, task, i
                )
                created += 1
            except Exception as e:
                print(f"[WARNING] Temporal plot failed for sample {idx}: {e}")

        if created == 0:
            print(f"[WARNING] No temporal plots generated for {task}.")


class GraphLIMEExplainer:
    def __init__(self, model, device, vocabularies=None, top_k=10, hsic_lambda=0.01):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.hsic_lambda = hsic_lambda

        if callable(vocabularies):
            try:
                self.vocabs = vocabularies()
            except:
                self.vocabs = {}
        else:
            self.vocabs = vocabularies or {}
        if not isinstance(self.vocabs, dict):
            self.vocabs = {}

    def _get_activity_name(self, idx):
        if "Activity" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Activity"].items()}
            return inv_vocab.get(int(idx), f"activity_{idx}")
        return f"activity_{idx}"

    def _get_resource_name(self, idx):
        if "Resource" in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs["Resource"].items()}
            res_name = inv_vocab.get(int(idx), f"Resource_{idx}")
            return f"{res_name} [Resource]"
        return f"Resource_{idx} [Resource]"

    def _get_feature_name(self, node_type, feature_idx):
        if node_type == "activity" and "Activity" in self.vocabs:
            vocab = self.vocabs["Activity"]
            inv_vocab = {v: k for k, v in vocab.items()}
            return inv_vocab.get(int(feature_idx), f"activity_{feature_idx}")
        elif node_type == "resource" and "Resource" in self.vocabs:
            vocab = self.vocabs["Resource"]
            inv_vocab = {v: k for k, v in vocab.items()}
            res_name = inv_vocab.get(int(feature_idx), f"Resource_{feature_idx}")
            return f"{res_name} [Resource]"
        elif node_type == "resource":
            return f"resource_{feature_idx} [Resource]"
        elif node_type == "time":
            return "Elapsed Time"
        return f"{node_type}_{feature_idx}"

    def _aggregate_features(self, explanation):
        activity_stats = {}
        for item in explanation:
            name = item["activity"]
            importance = item["importance"]

            if name not in activity_stats:
                activity_stats[name] = {"importance": 0.0, "count": 0}
            activity_stats[name]["importance"] += importance
            activity_stats[name]["count"] += 1

        aggregated = []
        for name, stats in activity_stats.items():
            label = f"{name} (x{stats['count']})" if stats["count"] > 1 else name
            aggregated.append({"activity": label, "importance": stats["importance"]})

        return sorted(aggregated, key=lambda x: abs(x["importance"]), reverse=False)

    def _median_sigma(self, distances):
        flat = distances.ravel()
        non_zero = flat[flat > 0]
        if non_zero.size == 0:
            return 1.0
        return float(np.median(non_zero))

    def _center_and_normalize(self, k_mat):
        n = k_mat.shape[0]
        h = np.eye(n) - np.ones((n, n)) / n
        centered = h @ k_mat @ h
        norm = np.linalg.norm(centered, ord="fro")
        if norm > 0:
            return centered / norm
        return centered

    def _hsic_lasso(self, x_mat, y_vec):
        n, d = x_mat.shape
        if n < 2 or d == 0:
            return None

        y = y_vec.reshape(-1, 1)
        y_dist = y - y.T
        sigma_y = self._median_sigma(np.abs(y_dist))
        # Prevent division by zero
        if sigma_y == 0:
            sigma_y = 1e-5

        l_mat = np.exp(-(y_dist**2) / (2 * sigma_y**2))
        l_bar = self._center_and_normalize(l_mat)

        # Vectorized feature kernel computation
        # X_dist: (d, n, n)
        X_trans = x_mat.T
        X_dist = np.abs(X_trans[:, :, np.newaxis] - X_trans[:, np.newaxis, :])

        # Calculate sigma for each feature
        sigmas = np.array([self._median_sigma(X_dist[i]) for i in range(d)])
        sigmas[sigmas == 0] = 1e-5
        sigmas = sigmas.reshape(-1, 1, 1)

        # K_mats: (d, n, n)
        k_mats = np.exp(-(X_dist**2) / (2 * sigmas**2))

        # Center and normalize (batched)
        h = np.eye(n) - np.ones((n, n)) / n
        centered = h @ k_mats @ h
        norms = np.linalg.norm(centered, axis=(1, 2), ord="fro")
        norms[norms == 0] = 1.0
        normalized = centered / norms[:, np.newaxis, np.newaxis]

        # x_feat: (n*n, d)
        x_feat = normalized.reshape(d, -1).T
        y_target = l_bar.reshape(-1)

        alpha = self.hsic_lambda
        best_coef = None
        best_non_zero = -1

        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import Lasso

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            # Try progressively smaller alphas if coefficients are too sparse
            for _ in range(5):
                # Faster Lasso with fewer iterations
                model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
                model.fit(x_feat, y_target)
                
                num_non_zero = np.count_nonzero(model.coef_)
                if num_non_zero > best_non_zero:
                    best_non_zero = num_non_zero
                    best_coef = model.coef_.copy()
                    
                # Stop if we found a reasonable number of non-zero features
                if num_non_zero >= 3:
                    return model.coef_
                    
                alpha *= 0.1

        return best_coef if best_coef is not None else model.coef_

    def explain_local(self, graph, task="activity", num_perturbations=100):
        self.model.eval()
        graph = graph.to(self.device)

        predicted_class = None
        true_val = None
        with torch.no_grad():
            out = self.model(graph)
            if task == "event_time":
                base_score = out[1].item()
                true_val = graph.y_timestamp.item()
            elif task == "remaining_time":
                base_score = out[2].item()
                true_val = graph.y_remaining_time.item()
            else:
                logits = out[0].view(-1)
                probs = torch.softmax(logits, dim=0)
                predicted_class = int(torch.argmax(probs).item())
                base_score = float(probs[predicted_class].item())
                true_val = float(graph.y_activity.view(-1)[0].item())

        activity_features = (
            graph["activity"].x.shape[1] if "activity" in graph.x_dict else 0
        )
        resource_features = (
            graph["resource"].x.shape[1] if "resource" in graph.x_dict else 0
        )

        active_activity = (
            torch.where(graph["activity"].x.sum(dim=0) > 0)[0].cpu().numpy()
            if activity_features > 0
            else np.array([])
        )
        active_resource = (
            torch.where(graph["resource"].x.sum(dim=0) > 0)[0].cpu().numpy()
            if resource_features > 0
            else np.array([])
        )

        active_features = [("activity", idx) for idx in active_activity] + [
            ("resource", idx) for idx in active_resource
        ]

        # FIX: Cap features to explain for speed
        if len(active_features) > 25:
            active_features = active_features[:25]
            # Adjust active_activity and active_resource to match the capped active_features
            active_activity = [idx for t, idx in active_features if t == "activity"]
            active_resource = [idx for t, idx in active_features if t == "resource"]

        if not active_features:
            return [], base_score, true_val, [], predicted_class

        X_perturb = []
        perturbed_graphs = []  # NEW: List to hold our graphs

        for _ in range(num_perturbations):
            mask_activity = np.ones(activity_features, dtype=np.float32)
            mask_resource = np.ones(resource_features, dtype=np.float32)

            if len(active_activity) > 0:
                subset = np.random.choice(
                    active_activity,
                    size=int(max(1, len(active_activity) * 0.3)),
                    replace=False,
                )
                mask_activity[subset] = 0.0

            if len(active_resource) > 0:
                subset = np.random.choice(
                    active_resource,
                    size=int(max(1, len(active_resource) * 0.3)),
                    replace=False,
                )
                mask_resource[subset] = 0.0

            mask_activity_active = (
                mask_activity[active_activity]
                if len(active_activity) > 0
                else np.array([])
            )
            mask_resource_active = (
                mask_resource[active_resource]
                if len(active_resource) > 0
                else np.array([])
            )
            combined_mask = np.concatenate([mask_activity_active, mask_resource_active])
            X_perturb.append(combined_mask)

            from torch_geometric.data import HeteroData

            masked_graph = HeteroData()
            for key in graph.x_dict:
                masked_graph[key].x = graph[key].x.detach().clone()
            for key in graph.edge_types:
                masked_graph[key].edge_index = graph[key].edge_index

            if activity_features > 0:
                act_mask_tensor = torch.tensor(
                    mask_activity, device=self.device, dtype=torch.float32
                ).view(1, -1)
                masked_graph["activity"].x = (
                    masked_graph["activity"].x * act_mask_tensor
                )
            if resource_features > 0 and "resource" in masked_graph.x_dict:
                res_mask_tensor = torch.tensor(
                    mask_resource, device=self.device, dtype=torch.float32
                ).view(1, -1)
                masked_graph["resource"].x = (
                    masked_graph["resource"].x * res_mask_tensor
                )

            perturbed_graphs.append(masked_graph)

        # Batch inference
        from torch_geometric.data import Batch

        batched_graphs = Batch.from_data_list(perturbed_graphs).to(self.device)

        y_perturb = []
        with torch.no_grad():
            out_p = self.model(batched_graphs)
            if task == "event_time":
                y_perturb = out_p[1].cpu().numpy().flatten()
            elif task == "remaining_time":
                y_perturb = out_p[2].cpu().numpy().flatten()
            else:
                logits = out_p[0]  # (B, num_classes)
                probs = torch.softmax(logits, dim=1)
                cls = (
                    predicted_class
                    if predicted_class is not None
                    else int(torch.argmax(probs, dim=1)[0].item())
                )
                y_perturb = probs[:, cls].cpu().numpy()

        X_perturb = np.array(X_perturb)
        # y_perturb is already a numpy array from the batched inference step

        coefs = self._hsic_lasso(X_perturb, y_perturb)
        if coefs is None or np.allclose(coefs, 0):
            weights = []
            y_std = np.std(y_perturb)
            for k in range(X_perturb.shape[1]):
                x_std = np.std(X_perturb[:, k])
                if x_std == 0 or y_std == 0:
                    weights.append(0.0)
                else:
                    corr = np.corrcoef(X_perturb[:, k], y_perturb)[0, 1]
                    weights.append(float(np.nan_to_num(corr, nan=0.0)))
            coefs = np.array(weights)

        explanation = []
        for coef, (node_type, feat_idx) in zip(coefs, active_features):
            explanation.append(
                {
                    "activity": self._get_feature_name(node_type, feat_idx),
                    "importance": float(coef),
                }
            )

        aggregated_explanation = self._aggregate_features(explanation)
        if self.top_k and len(aggregated_explanation) > self.top_k:
            aggregated_explanation = sorted(
                aggregated_explanation, key=lambda x: abs(x["importance"]), reverse=True
            )[: self.top_k]
            aggregated_explanation = sorted(
                aggregated_explanation,
                key=lambda x: abs(x["importance"]),
                reverse=False,
            )

        step_info = [{"feature": row["activity"]} for row in aggregated_explanation]
        return aggregated_explanation, base_score, true_val, step_info, predicted_class

    def plot_local(
        self,
        explanation,
        base_score,
        output_dir,
        task,
        sample_id,
        true_val=None,
        predicted_class=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if not explanation:
            print(
                f"[DEBUG GraphLIME] plot_local received empty explanation for sample {sample_id}, task {task}."
            )
            return

        df = pd.DataFrame(explanation)
        import re

        m = re.search(r"case_(.+?)_idx_(.+)", str(sample_id))
        if m:
            df.insert(0, "case_id", m.group(1))
            df.insert(1, "case_index", m.group(2))

        # Ensure sorting by absolute importance, then remove temporary column
        df["_abs_importance"] = df["importance"].abs()
        df = df.sort_values("_abs_importance", ascending=True)
        df = df.drop(columns=["_abs_importance"])

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.45)))
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in df["importance"]]
        y_pos = np.arange(len(df))
        bars = ax.barh(
            y_pos,
            df["importance"].values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

        for i, bar in enumerate(bars):
            val = df["importance"].values[i]
            offset = 0.01 if val >= 0 else -0.01
            ha = "left" if val >= 0 else "right"
            prefix = "+" if val > 0 else ""
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{prefix}{val:.3f}",
                ha=ha,
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["activity"].tolist(), fontsize=9)
        ax.set_xlabel(
            "Feature Contribution (GraphLIME)", fontweight="bold", fontsize=12
        )

        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
        ax.grid(True, alpha=0.3, axis="x", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f"graphlime_sample_{sample_id}_{task}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        csv_path = os.path.join(output_dir, f"graphlime_sample_{sample_id}_{task}.csv")
        df.to_csv(csv_path, index=False)

    def plot_local_explanation(
        self,
        explanation,
        base_score,
        true_val,
        step_info,
        output_dir,
        task,
        sample_id,
        predicted_class,
    ):
        # step_info is kept for API compatibility with existing caller.
        self.plot_local(
            explanation=explanation,
            base_score=base_score,
            output_dir=output_dir,
            task=task,
            sample_id=sample_id,
            true_val=true_val,
            predicted_class=predicted_class,
        )

    def explain_global_importance(self, graphs, task="activity", num_samples=50):
        import numpy as np
        import collections
        from tqdm import tqdm

        sample_count = min(num_samples, len(graphs))
        sample_indices = np.random.choice(len(graphs), sample_count, replace=False)

        print(f"Computing GraphLIME for {sample_count} graphs ({task})...")

        global_importances = collections.defaultdict(list)
        for idx in tqdm(sample_indices, desc=f"GraphLIME ({task})"):
            graph = graphs[idx]
            try:
                # Use 50 perturbations for global summary as requested
                explanation, _, _, _, _ = self.explain_local(
                    graph, task=task, num_perturbations=50
                )
                for item in explanation:
                    name = item["activity"]
                    imp = item["importance"]
                    global_importances[name].append(abs(imp))
            except Exception as e:
                pass

        # Average the scores
        avg_importances = {}
        for name, scores in global_importances.items():
            if len(scores) > 0:
                avg_importances[name] = np.mean(scores)

        return avg_importances

    def plot_global_importance(self, importances, output_dir, task="activity"):
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(output_dir, exist_ok=True)

        if not importances:
            print(f"[WARNING] No GraphLIME global importances available for {task}.")
            return

        data = [{"activity": k, "importance": v} for k, v in importances.items()]
        df = pd.DataFrame(data).sort_values("importance", ascending=False).head(15)

        df.to_csv(os.path.join(output_dir, f"graphlime_global_{task}.csv"), index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="importance",
            y="activity",
            hue="activity",
            data=df,
            palette="viridis",
            legend=False,
        )
        plt.title(
            f'GraphLIME Global Feature Importance - {task.replace("_", " ").capitalize()}'
        )
        plt.xlabel("Mean Absolute Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"graphlime_global_bar_plot_{task}.png"),
            dpi=300,
            facecolor="white",
        )
        plt.close()


class ExplainabilityBenchmark:
    """
    Benchmark metrics for evaluating and comparing GNN explainability methods.
    Mirrors transformer benchmark logic, adapted for graph inputs.
    """

    def __init__(self, model, device, task="activity", vocabs=None):
        self.model = model
        self.device = device
        self.task = task
        self.vocabs = vocabs or {}
        self.results = {}

    def _predict(self, graph):
        self.model.eval()
        graph = graph.to(self.device)
        with torch.no_grad():
            return self.model(graph)

    def _select_output(self, out):
        if self.task in ["activity", "next_activity"]:
            return out[0]
        if self.task == "event_time":
            return out[1]
        return out[2]

    def _prediction_value(self, out):
        pred = self._select_output(out)
        if self.task in ["activity", "next_activity"]:
            return pred.max()
        return pred.mean()

    def _feature_dims(self, graph):
        act_features = graph["activity"].x.shape[1] if "activity" in graph.x_dict else 0
        res_features = graph["resource"].x.shape[1] if "resource" in graph.x_dict else 0
        return act_features, res_features

    def _mask_graph(self, graph, act_mask=None, res_mask=None):
        masked = graph.clone()
        if act_mask is not None and "activity" in masked.x_dict:
            mask_tensor = torch.tensor(
                act_mask,
                device=masked["activity"].x.device,
                dtype=masked["activity"].x.dtype,
            )
            masked["activity"].x = masked["activity"].x * mask_tensor
        if res_mask is not None and "resource" in masked.x_dict:
            mask_tensor = torch.tensor(
                res_mask,
                device=masked["resource"].x.device,
                dtype=masked["resource"].x.dtype,
            )
            masked["resource"].x = masked["resource"].x * mask_tensor
        return masked

    def _masks_from_indices(self, indices, act_features, res_features, keep=False):
        if keep:
            act_mask = np.zeros(act_features, dtype=np.float32)
            res_mask = np.zeros(res_features, dtype=np.float32)
        else:
            act_mask = np.ones(act_features, dtype=np.float32)
            res_mask = np.ones(res_features, dtype=np.float32)

        for idx in indices:
            if idx < act_features:
                act_mask[idx] = 1.0 if keep else 0.0
            else:
                res_idx = idx - act_features
                if 0 <= res_idx < res_features:
                    res_mask[res_idx] = 1.0 if keep else 0.0
        return act_mask, res_mask

    def compute_gradient_attributions(self, graphs):
        self.model.eval()
        attributions = []
        node_contribs = []

        for graph in graphs:
            g = graph.to(self.device)
            for key in g.x_dict:
                g.x_dict[key].requires_grad = True

            out = self.model(g)
            if self.task in ["activity", "next_activity"]:
                logits = out[0]
                pred_idx = logits.argmax(dim=1)
                score = logits[0, pred_idx]
            elif self.task == "event_time":
                score = out[1].sum()
            else:
                score = out[2].sum()

            self.model.zero_grad()
            score.backward()

            act_features = g["activity"].x.shape[1] if "activity" in g.x_dict else 0
            res_features = g["resource"].x.shape[1] if "resource" in g.x_dict else 0
            combined = np.zeros(act_features + res_features, dtype=np.float32)

            if "activity" in g.x_dict and g.x_dict["activity"].grad is not None:
                grad = g.x_dict["activity"].grad
                inp = g.x_dict["activity"]
                act_attr = (grad * inp).abs().sum(dim=0).detach().cpu().numpy()
                combined[:act_features] = act_attr
                node_contrib = (grad * inp).abs().sum(dim=1).detach().cpu().numpy()
                node_contribs.append(node_contrib)
            else:
                node_contribs.append(np.array([]))

            if "resource" in g.x_dict and g.x_dict["resource"].grad is not None:
                grad = g.x_dict["resource"].grad
                inp = g.x_dict["resource"]
                res_attr = (grad * inp).abs().sum(dim=0).detach().cpu().numpy()
                combined[act_features:] = res_attr

            attributions.append(combined)

        if attributions:
            return np.stack(attributions, axis=0), node_contribs
        return np.array([]), []

    def _parse_feature_name(self, name, act_vocab, res_vocab):
        is_resource_hint = name.endswith(" [Resource]")
        clean = name.replace(" [Resource]", "")
        clean = re.sub(r"\s+\(x\d+\)$", "", clean).strip()

        if is_resource_hint:
            if clean in res_vocab:
                return "resource", res_vocab[clean]
            if clean.startswith("Resource_"):
                try:
                    return "resource", int(clean.split("_", 1)[1])
                except ValueError:
                    return None, None
            return "resource", None

        if clean in act_vocab:
            return "activity", act_vocab[clean]
        if clean in res_vocab:
            return "resource", res_vocab[clean]
        if clean.startswith("activity_") or clean.startswith("Act_"):
            try:
                # Handle both activity_idx and Act_idx
                parts = clean.split("_", 1)
                return "activity", int(parts[1])
            except (ValueError, IndexError):
                return None, None
        if clean.startswith("Resource_") or clean.startswith("Res_"):
            try:
                parts = clean.split("_", 1)
                return "resource", int(parts[1])
            except (ValueError, IndexError):
                return None, None
        return None, None

    def vectorize_graphlime(self, explanation, act_features, res_features):
        vec = np.zeros(act_features + res_features, dtype=np.float32)
        act_vocab = self.vocabs.get("Activity", {})
        res_vocab = self.vocabs.get("Resource", {})

        for item in explanation:
            name = item.get("Feature")
            if not name:
                continue
            node_type, idx = self._parse_feature_name(name, act_vocab, res_vocab)
            if node_type == "activity" and 0 <= idx < act_features:
                vec[idx] = float(item.get("Weight", 0.0))
            elif node_type == "resource" and 0 <= idx < res_features:
                vec[act_features + idx] = float(item.get("Weight", 0.0))
        return vec

    def faithfulness_correlation(self, graphs, attributions, k_values=[1, 3, 5, 10]):
        print("Computing Faithfulness Correlation...")
        n_samples = len(graphs)
        n_features = (
            attributions.shape[1] if attributions.ndim > 1 else len(attributions)
        )

        results = {}
        for k in k_values:
            if k > n_features:
                continue

            pred_changes = []
            importance_sums = []

            for i in range(n_samples):
                orig_out = self._predict(graphs[i])
                orig_pred = self._select_output(orig_out)

                sample_attr = (
                    np.abs(attributions[i])
                    if attributions.ndim > 1
                    else np.abs(attributions)
                )
                top_k_idx = np.argsort(sample_attr)[-k:]

                act_features, res_features = self._feature_dims(graphs[i])
                act_mask, res_mask = self._masks_from_indices(
                    top_k_idx, act_features, res_features, keep=False
                )
                masked_graph = self._mask_graph(graphs[i], act_mask, res_mask)
                masked_out = self._predict(masked_graph)
                masked_pred = self._select_output(masked_out)

                if self.task in ["activity", "next_activity"]:
                    pred_change = (orig_pred - masked_pred).abs().max().item()
                else:
                    pred_change = (orig_pred - masked_pred).abs().mean().item()

                pred_changes.append(pred_change)
                importance_sums.append(sample_attr[top_k_idx].sum())

            from scipy.stats import spearmanr, pearsonr

            if len(set(pred_changes)) > 1 and len(set(importance_sums)) > 1:
                spearman_corr, spearman_p = spearmanr(importance_sums, pred_changes)
                pearson_corr, pearson_p = pearsonr(importance_sums, pred_changes)
            else:
                spearman_corr, spearman_p = 0.0, 1.0
                pearson_corr, pearson_p = 0.0, 1.0

            results[f"faithfulness_k{k}"] = {
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(spearman_p),
                "pearson_correlation": float(pearson_corr),
                "pearson_p_value": float(pearson_p),
                "mean_pred_change": float(np.mean(pred_changes)),
                "std_pred_change": float(np.std(pred_changes)),
            }

        return results

    def comprehensiveness(self, graphs, attributions, k_values=[1, 3, 5, 10]):
        print("Computing Comprehensiveness...")
        n_samples = len(graphs)
        n_features = (
            attributions.shape[1] if attributions.ndim > 1 else len(attributions)
        )

        results = {}
        for k in k_values:
            if k > n_features:
                continue

            comp_scores = []

            for i in range(n_samples):
                orig_out = self._predict(graphs[i])
                orig_pred = self._select_output(orig_out)

                sample_attr = (
                    np.abs(attributions[i])
                    if attributions.ndim > 1
                    else np.abs(attributions)
                )
                top_k_idx = np.argsort(sample_attr)[-k:]

                act_features, res_features = self._feature_dims(graphs[i])
                act_mask, res_mask = self._masks_from_indices(
                    top_k_idx, act_features, res_features, keep=False
                )
                masked_graph = self._mask_graph(graphs[i], act_mask, res_mask)
                masked_out = self._predict(masked_graph)
                masked_pred = self._select_output(masked_out)

                if self.task in ["activity", "next_activity"]:
                    comp = (orig_pred.max() - masked_pred.max()).item()
                else:
                    comp = (orig_pred - masked_pred).abs().mean().item()
                comp_scores.append(comp)

            results[f"comprehensiveness_k{k}"] = {
                "mean": float(np.mean(comp_scores)),
                "std": float(np.std(comp_scores)),
                "median": float(np.median(comp_scores)),
            }

        return results

    def sufficiency(self, graphs, attributions, k_values=[1, 3, 5, 10]):
        print("Computing Sufficiency...")
        n_samples = len(graphs)
        n_features = (
            attributions.shape[1] if attributions.ndim > 1 else len(attributions)
        )

        results = {}
        for k in k_values:
            if k > n_features:
                continue

            suff_scores = []

            for i in range(n_samples):
                orig_out = self._predict(graphs[i])
                orig_pred = self._select_output(orig_out)

                sample_attr = (
                    np.abs(attributions[i])
                    if attributions.ndim > 1
                    else np.abs(attributions)
                )
                top_k_idx = np.argsort(sample_attr)[-k:]

                act_features, res_features = self._feature_dims(graphs[i])
                act_mask, res_mask = self._masks_from_indices(
                    top_k_idx, act_features, res_features, keep=True
                )
                masked_graph = self._mask_graph(graphs[i], act_mask, res_mask)
                masked_out = self._predict(masked_graph)
                masked_pred = self._select_output(masked_out)

                if self.task in ["activity", "next_activity"]:
                    suff = (orig_pred.max() - masked_pred.max()).item()
                else:
                    suff = (orig_pred - masked_pred).abs().mean().item()
                suff_scores.append(suff)

            results[f"sufficiency_k{k}"] = {
                "mean": float(np.mean(suff_scores)),
                "std": float(np.std(suff_scores)),
                "median": float(np.median(suff_scores)),
            }

        return results

    def stability(self, attributions, n_perturbations=10):
        print("Computing Stability...")
        n_samples = min(len(attributions), 20)
        stability_scores = []
        for i in range(n_samples):
            original_attr = attributions[i]
            perturbed_attrs = [original_attr for _ in range(n_perturbations)]
            attr_variance = np.var(perturbed_attrs, axis=0).mean()
            stability_scores.append(attr_variance)

        return {
            "stability": {
                "mean_variance": (
                    float(np.mean(stability_scores)) if stability_scores else 0.0
                ),
                "max_variance": (
                    float(np.max(stability_scores)) if stability_scores else 0.0
                ),
                "stability_score": (
                    float(1.0 / (1.0 + np.mean(stability_scores)))
                    if stability_scores
                    else 0.0
                ),
            }
        }

    def method_agreement(
        self, grad_attributions, lime_attributions, k_values=[3, 5, 10]
    ):
        print("Computing Method Agreement (Gradient vs GraphLIME)...")
        if grad_attributions is None or lime_attributions is None:
            return {"method_agreement": "N/A - Missing attributions"}

        n_samples = min(len(grad_attributions), len(lime_attributions))
        results = {}

        for k in k_values:
            jaccard_scores = []
            overlap_scores = []
            rank_correlations = []

            for i in range(n_samples):
                grad_attr = np.abs(grad_attributions[i])
                lime_attr = np.abs(lime_attributions[i])

                min_len = min(len(grad_attr), len(lime_attr))
                grad_attr = grad_attr[:min_len]
                lime_attr = lime_attr[:min_len]

                if k > min_len:
                    continue

                grad_top_k = set(np.argsort(grad_attr)[-k:])
                lime_top_k = set(np.argsort(lime_attr)[-k:])

                intersection = len(grad_top_k & lime_top_k)
                union = len(grad_top_k | lime_top_k)
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)

                overlap = intersection / k
                overlap_scores.append(overlap)

                from scipy.stats import spearmanr

                if len(grad_attr) > 1:
                    corr, _ = spearmanr(grad_attr, lime_attr)
                    if not np.isnan(corr):
                        rank_correlations.append(corr)

            results[f"agreement_k{k}"] = {
                "jaccard_similarity": (
                    float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
                ),
                "top_k_overlap": (
                    float(np.mean(overlap_scores)) if overlap_scores else 0.0
                ),
                "rank_correlation": (
                    float(np.mean(rank_correlations)) if rank_correlations else 0.0
                ),
            }

        return results

    def monotonicity(self, graphs, attributions):
        print("Computing Monotonicity...")
        n_samples = min(len(graphs), 20)
        monotonicity_scores = []

        for i in range(n_samples):
            orig_out = self._predict(graphs[i])
            predictions = [self._prediction_value(orig_out).item()]

            sample_attr = np.abs(attributions[i])
            sorted_indices = np.argsort(sample_attr)[::-1]

            act_features, res_features = self._feature_dims(graphs[i])
            removed = []
            for idx in sorted_indices[: min(10, len(sorted_indices))]:
                removed.append(int(idx))
                act_mask, res_mask = self._masks_from_indices(
                    removed, act_features, res_features, keep=False
                )
                masked_graph = self._mask_graph(graphs[i], act_mask, res_mask)
                pred = self._prediction_value(self._predict(masked_graph)).item()
                predictions.append(pred)

            n_monotonic = sum(
                1
                for j in range(1, len(predictions))
                if predictions[j] <= predictions[j - 1]
            )
            monotonicity = (
                n_monotonic / (len(predictions) - 1) if len(predictions) > 1 else 0
            )
            monotonicity_scores.append(monotonicity)

        return {
            "monotonicity": {
                "mean": (
                    float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0
                ),
                "std": (
                    float(np.std(monotonicity_scores)) if monotonicity_scores else 0.0
                ),
                "median": (
                    float(np.median(monotonicity_scores))
                    if monotonicity_scores
                    else 0.0
                ),
            }
        }

    def temporal_consistency(self, node_contribs):
        print("Computing Temporal Consistency...")
        if not node_contribs:
            return {"temporal_consistency": "N/A - Missing node attributions"}

        max_len = max(len(x) for x in node_contribs if len(x) > 0)
        position_importance = np.zeros(max_len)
        position_counts = np.zeros(max_len)

        for contrib in node_contribs:
            if len(contrib) == 0:
                continue
            vals = np.abs(contrib)
            position_importance[: len(vals)] += vals
            position_counts[: len(vals)] += 1

        avg_importance = np.divide(
            position_importance,
            position_counts,
            where=position_counts > 0,
            out=np.zeros_like(position_importance),
        )

        positions = np.arange(max_len)
        valid_mask = position_counts > 0
        from scipy.stats import spearmanr

        if valid_mask.sum() > 2:
            recency_corr, recency_p = spearmanr(
                positions[valid_mask], avg_importance[valid_mask]
            )
        else:
            recency_corr, recency_p = 0.0, 1.0

        return {
            "temporal_consistency": {
                "recency_correlation": (
                    float(recency_corr) if not np.isnan(recency_corr) else 0.0
                ),
                "recency_p_value": float(recency_p) if not np.isnan(recency_p) else 1.0,
                "position_importance": avg_importance.tolist(),
                "most_important_position": (
                    int(np.argmax(avg_importance)) if valid_mask.any() else 0
                ),
                "least_important_position": (
                    int(np.argmin(avg_importance[valid_mask]))
                    if valid_mask.any()
                    else 0
                ),
            }
        }

    def run_full_benchmark(
        self,
        graphs,
        grad_values,
        lime_values=None,
        node_contribs=None,
        k_values=[1, 3, 5, 10],
    ):
        print("\n" + "=" * 60)
        print("EXPLAINABILITY BENCHMARK EVALUATION")
        print("=" * 60)

        act_features, res_features = self._feature_dims(graphs[0]) if graphs else (0, 0)
        results = {
            "metadata": {
                "task": self.task,
                "n_samples": len(graphs),
                "n_features": (
                    int(grad_values.shape[1])
                    if grad_values is not None and grad_values.size > 0
                    else 0
                ),
                "activity_features": int(act_features),
                "resource_features": int(res_features),
                "k_values": k_values,
            }
        }

        try:
            results["faithfulness"] = self.faithfulness_correlation(
                graphs, grad_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Faithfulness computation failed: {e}")
            results["faithfulness"] = {"error": str(e)}

        try:
            results["comprehensiveness"] = self.comprehensiveness(
                graphs, grad_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Comprehensiveness computation failed: {e}")
            results["comprehensiveness"] = {"error": str(e)}

        try:
            results["sufficiency"] = self.sufficiency(graphs, grad_values, k_values)
        except Exception as e:
            print(f"[WARNING] Sufficiency computation failed: {e}")
            results["sufficiency"] = {"error": str(e)}

        try:
            results["monotonicity"] = self.monotonicity(graphs, grad_values)
        except Exception as e:
            print(f"[WARNING] Monotonicity computation failed: {e}")
            results["monotonicity"] = {"error": str(e)}

        try:
            results["stability"] = self.stability(grad_values)
        except Exception as e:
            print(f"[WARNING] Stability computation failed: {e}")
            results["stability"] = {"error": str(e)}

        if lime_values is not None:
            try:
                results["method_agreement"] = self.method_agreement(
                    grad_values, lime_values, k_values
                )
            except Exception as e:
                print(f"[WARNING] Method agreement computation failed: {e}")
                results["method_agreement"] = {"error": str(e)}

        try:
            results["temporal_consistency"] = self.temporal_consistency(
                node_contribs or []
            )
        except Exception as e:
            print(f"[WARNING] Temporal consistency computation failed: {e}")
            results["temporal_consistency"] = {"error": str(e)}

        self.results = results
        return results

    def save_results(self, output_dir, filename="benchmark_results.json"):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[OK] Benchmark results saved to: {filepath}")

        summary_rows = self.summary_rows()

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(
                output_dir, filename.replace(".json", "_summary.csv")
            )
            summary_df.to_csv(summary_path, index=False)
            print(f"[OK] Benchmark summary saved to: {summary_path}")

        return filepath

    def summary_rows(self, task_prefix=None):
        summary_rows = []
        for metric_name, metric_data in self.results.items():
            if metric_name == "metadata":
                continue
            if isinstance(metric_data, dict):
                for sub_key, sub_val in metric_data.items():
                    if isinstance(sub_val, dict):
                        for k, v in sub_val.items():
                            if isinstance(v, (int, float)):
                                metric_label = f"{sub_key}_{k}"
                                if task_prefix:
                                    metric_label = f"{task_prefix}_{metric_label}"
                                summary_rows.append(
                                    {
                                        "category": metric_name,
                                        "metric": metric_label,
                                        "value": v,
                                    }
                                )
                    elif isinstance(sub_val, (int, float)):
                        metric_label = sub_key
                        if task_prefix:
                            metric_label = f"{task_prefix}_{metric_label}"
                        summary_rows.append(
                            {
                                "category": metric_name,
                                "metric": metric_label,
                                "value": sub_val,
                            }
                        )
        return summary_rows

    def print_summary(self):
        if not self.results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if (
            "faithfulness" in self.results
            and "error" not in self.results["faithfulness"]
        ):
            print("\nFAITHFULNESS (Higher = Better)")
            for k, v in self.results["faithfulness"].items():
                if isinstance(v, dict):
                    corr = v.get("spearman_correlation", "N/A")
                    print(
                        f"   {k}: Spearman={corr:.4f}"
                        if isinstance(corr, float)
                        else f"   {k}: {corr}"
                    )

        if (
            "comprehensiveness" in self.results
            and "error" not in self.results["comprehensiveness"]
        ):
            print("\nCOMPREHENSIVENESS (Higher = Better)")
            for k, v in self.results["comprehensiveness"].items():
                if isinstance(v, dict):
                    mean = v.get("mean", "N/A")
                    print(
                        f"   {k}: Mean={mean:.4f}"
                        if isinstance(mean, float)
                        else f"   {k}: {mean}"
                    )

        if "sufficiency" in self.results and "error" not in self.results["sufficiency"]:
            print("\nSUFFICIENCY (Lower = Better)")
            for k, v in self.results["sufficiency"].items():
                if isinstance(v, dict):
                    mean = v.get("mean", "N/A")
                    print(
                        f"   {k}: Mean={mean:.4f}"
                        if isinstance(mean, float)
                        else f"   {k}: {mean}"
                    )

        if (
            "monotonicity" in self.results
            and "error" not in self.results["monotonicity"]
        ):
            mono = self.results["monotonicity"].get("monotonicity", {})
            mean = mono.get("mean", "N/A")
            print(
                f"\nMONOTONICITY (Higher = Better): {mean:.4f}"
                if isinstance(mean, float)
                else f"\nMONOTONICITY: {mean}"
            )

        if (
            "method_agreement" in self.results
            and "error" not in self.results["method_agreement"]
        ):
            print("\nMETHOD AGREEMENT (Gradient vs GraphLIME)")
            for k, v in self.results["method_agreement"].items():
                if isinstance(v, dict):
                    jaccard = v.get("jaccard_similarity", "N/A")
                    overlap = v.get("top_k_overlap", "N/A")
                    print(
                        f"   {k}: Jaccard={jaccard:.4f}, Overlap={overlap:.2%}"
                        if isinstance(jaccard, float)
                        else f"   {k}: {jaccard}"
                    )

        if (
            "temporal_consistency" in self.results
            and "error" not in self.results["temporal_consistency"]
        ):
            tc = self.results["temporal_consistency"].get("temporal_consistency", {})
            recency = tc.get("recency_correlation", "N/A")
            print(
                f"\nTEMPORAL CONSISTENCY (Recency Correlation): {recency:.4f}"
                if isinstance(recency, float)
                else f"\nTEMPORAL CONSISTENCY: {recency}"
            )

        print("\n" + "=" * 60)


def generate_feature_importance_summary(
    grad_dir, lime_dir, output_dir, task="activity"
):
    grad_importance = {}
    feature_types = {}

    grad_file = os.path.join(grad_dir, f"gradient_global_{task}.csv")
    if os.path.exists(grad_file):
        grad_df = pd.read_csv(grad_file)
        for _, row in grad_df.iterrows():
            name = row["activity"]
            grad_importance[name] = row["importance"]
            feature_types[name] = "Resource" if " [Resource]" in name else "Activity"

    lime_importance = {}
    if os.path.exists(lime_dir):
        # Try global file first
        global_lime_file = os.path.join(lime_dir, f"graphlime_global_{task}.csv")
        if os.path.exists(global_lime_file):
            lime_df = pd.read_csv(global_lime_file)
            for _, row in lime_df.iterrows():
                name = row["activity"]
                lime_importance[name] = row["importance"]
                if name not in feature_types:
                    feature_types[name] = (
                        "Resource" if " [Resource]" in name else "Activity"
                    )

        # Fallback to aggregation
        if not lime_importance:
            lime_files = [
                f
                for f in os.listdir(lime_dir)
                if f.startswith("graphlime_")
                and f.endswith(".csv")
                and "global" not in f
            ]
            if lime_files:
                lime_aggregated = {}
                for lime_file in lime_files:
                    lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                    for _, row in lime_df.iterrows():
                        feature = row["activity"]
                        weight = abs(row["importance"])
                        if feature not in lime_aggregated:
                            lime_aggregated[feature] = []
                        lime_aggregated[feature].append(weight)

                for feature, weights in lime_aggregated.items():
                    lime_importance[feature] = np.mean(weights)
                    if feature not in feature_types:
                        feature_types[feature] = (
                            "Resource" if " [Resource]" in feature else "Activity"
                        )

    summary_data = []
    all_features = set(grad_importance.keys()) | set(lime_importance.keys())
    for feature in all_features:
        grad_score = grad_importance.get(feature, 0.0)
        lime_score = lime_importance.get(feature, 0.0)

        if grad_score == 0.0 and lime_score == 0.0:
            continue

        avg_score = (
            (grad_score + lime_score) / 2.0
            if (grad_score > 0 and lime_score > 0)
            else max(grad_score, lime_score)
        )

        summary_data.append(
            {
                "activity": feature,
                "gradient_importance": grad_score,
                "graphlime_importance": lime_score,
                "average_importance": avg_score,
            }
        )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("average_importance", ascending=False)
        summary_df.to_csv(
            os.path.join(output_dir, "feature_importance_summary.csv"), index=False
        )
        print(f"[OK] Feature importance summary saved for {task}")
    else:
        print(
            f"[DEBUG] summary_data is empty for task {task}. No feature_importance_summary.csv generated."
        )

    return pd.DataFrame(summary_data) if summary_data else None


def generate_comparison_report(
    grad_dir, lime_dir, output_dir, task="activity", summary_df=None
):
    report_path = os.path.join(output_dir, f"comparison_report_{task}.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"GNN EXPLAINABILITY COMPARISON REPORT - {task.upper()}\n")
        f.write("=" * 70 + "\n\n")

        grad_file = os.path.join(grad_dir, f"gradient_global_{task}.csv")
        if os.path.exists(grad_file):
            grad_df = pd.read_csv(grad_file)
            f.write("GRADIENT-BASED SALIENCY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total features analyzed: {len(grad_df)}\n")
            f.write(
                f"Top feature: {grad_df.iloc[-1]['activity']} ({grad_df.iloc[-1]['importance']:.4f})\n"
            )
            f.write("\n")

        global_lime_file = os.path.join(lime_dir, f"graphlime_global_{task}.csv")
        if os.path.exists(global_lime_file):
            lime_df = pd.read_csv(global_lime_file)
            f.write("GRAPHLIME GLOBAL ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"Top feature: {lime_df.iloc[0]['activity']} ({lime_df.iloc[0]['importance']:.4f})\n"
            )
            f.write("\n")
        else:
            lime_files = [
                f
                for f in os.listdir(lime_dir)
                if f.startswith("graphlime_") and f.endswith(".csv")
            ]
            if lime_files:
                f.write("GRAPHLIME LOCAL ANALYSIS (Aggregated):\n")
                f.write("-" * 70 + "\n")
                f.write(f"Samples analyzed: {len(lime_files)}\n")
                f.write("\n")
            all_features = []
            for lime_file in lime_files:
                lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                all_features.extend(lime_df["activity"].tolist())

            unique_features = len(set(all_features))
            f.write(f"Unique features identified: {unique_features}\n")
            f.write(
                f"Most common feature: {max(set(all_features), key=all_features.count)}\n\n"
            )

        if summary_df is not None and not summary_df.empty:
            f.write("TOP 10 MOST IMPORTANT FEATURES (AVERAGE):\n")
            f.write("-" * 70 + "\n")
            for i, row in enumerate(summary_df.head(10).to_dict("records"), 1):
                f.write(
                    f"{i:2d}. {row['activity']:<30} | Avg: {row['average_importance']:.4f}\n"
                )
                f.write(
                    f"    Grad: {row['gradient_importance']:.4f} | LIME: {row['graphlime_importance']:.4f}\n\n"
                )

        f.write("METHOD COMPARISON:\n")
        f.write("-" * 70 + "\n")
        f.write("Gradient Method:\n")
        f.write("  + Global feature importance\n")
        f.write("  + Considers all node types (activity + resource)\n")
        f.write("  + Fast computation\n")
        f.write("  - Less interpretable for individual predictions\n\n")

        f.write("GraphLIME Method:\n")
        f.write("  + Local explanations for individual samples\n")
        f.write("  + Highly interpretable\n")
        f.write("  + Now includes resource features\n")
        f.write("  - Slower computation\n")
        f.write("  - May vary between samples\n\n")

        f.write("Temporal Gradient Method:\n")
        f.write("  + Shows contribution over time steps\n")
        f.write("  + Visualizes observed data alongside gradients\n")
        f.write("  + Good for understanding sequential patterns\n")
        f.write("  - Requires sequential/temporal data structure\n\n")

        f.write("=" * 70 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("-" * 70 + "\n")
        f.write("Use Gradient for: Overall model behavior understanding\n")
        f.write("Use GraphLIME for: Understanding specific predictions\n")
        f.write("Use Temporal Gradient for: Time-series attribution analysis\n")
        f.write("=" * 70 + "\n")

    print(f"[OK] Comparison report saved for {task}")


def run_gnn_explainability(
    model,
    data,
    output_dir,
    device,
    vocabularies=None,
    num_samples=50,
    methods="all",
    tasks=None,
    scaler=None,
    y_true=None,
    run_benchmark=True,
):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GNN EXPLAINABILITY - SHAP-STYLE VISUALIZATION")
    print("=" * 70)

    if "test_graphs" in data:
        graphs = data["test_graphs"]
    elif "test" in data:
        graphs = data["test"]
    else:
        print("Error: Could not find test graphs in data object.")
        return

    if tasks is None:
        print("[!] WARNING: No tasks specified")
        return {}

    if isinstance(tasks, str):
        tasks = [tasks]

    print(f"\n[->] Tasks: {tasks}")
    print(f"[->] Samples: {num_samples}")

    if methods in ["gradient", "all"]:
        print("\n" + "─" * 70)
        print("GRADIENT ANALYSIS (SHAP-STYLE)")
        print("─" * 70)

        explainer = GradientExplainer(model, device, vocabularies)
        grad_dir = os.path.join(output_dir, "gradient")

        for task in tasks:
            print(f"\n[{task.upper()}]")

            try:
                print(f"  Creating global summary for {task}...")
                imp = explainer.explain_global_importance(
                    graphs, task=task, num_samples=num_samples
                )
                explainer.plot_global_importance(imp, grad_dir, task=task)

            except Exception as e:
                print(f"[ERROR] Failed {task}: {e}")
        if not _dir_has_png(grad_dir):
            print("[WARNING] No gradient plots generated.")

    if methods in ["temporal", "all"]:
        print("\n[Temporal Gradient Attribution - Global Only]")
        # Individual temporal plots are skipped in global mode
        pass

    if methods in ["lime", "all"]:
        print("\n" + "─" * 70)
        print("GRAPHLIME LOCAL ANALYSIS")
        print("─" * 70)

        lime_explainer = GraphLIMEExplainer(model, device, vocabularies)
        lime_dir = os.path.join(output_dir, "graphlime")

        for task in tasks:
            print(f"\n[{task.upper()}] Global GraphLIME")
            try:
                print(f"  Creating global summary for {task}...")
                imp = lime_explainer.explain_global_importance(
                    graphs, task=task, num_samples=num_samples
                )
                lime_explainer.plot_global_importance(imp, lime_dir, task=task)
            except Exception as e:
                print(f"[ERROR] Failed global GraphLIME for {task}: {e}")

        if not _dir_has_png(lime_dir):
            print("[WARNING] No GraphLIME plots generated.")

    # Always generate comprehensive analysis if any method was run
    if methods in ["gradient", "lime", "all"]:
        print("\n[Generating Comprehensive Analysis]")
        grad_dir = os.path.join(output_dir, "gradient")
        lime_dir = os.path.join(output_dir, "graphlime")

        for task in tasks:
            try:
                summary_df = generate_feature_importance_summary(
                    grad_dir, lime_dir, output_dir, task
                )
                generate_comparison_report(
                    grad_dir, lime_dir, output_dir, task, summary_df
                )
            except Exception as e:
                print(f"[ERROR] Failed to generate summary for {task}: {e}")

    benchmark_results = None
    combined_summary_rows = []
    if run_benchmark and methods in ["gradient", "all"]:
        print("\n[Benchmark Evaluation]")
        bench_dir = os.path.join(output_dir, "benchmark")
        os.makedirs(bench_dir, exist_ok=True)

        if graphs:
            sample_count = min(num_samples, len(graphs))
            sample_indices = np.random.choice(len(graphs), sample_count, replace=False)
            bench_graphs = [graphs[i] for i in sample_indices]
        else:
            bench_graphs = []

        for task in tasks:
            try:
                benchmark = ExplainabilityBenchmark(
                    model, device, task=task, vocabs=vocabularies
                )
                grad_attr, node_contribs = benchmark.compute_gradient_attributions(
                    bench_graphs
                )

                lime_attr = None
                if methods in ["lime", "all"]:
                    lime_explainer = GraphLIMEExplainer(model, device, vocabularies)
                    if bench_graphs:
                        act_features, res_features = benchmark._feature_dims(
                            bench_graphs[0]
                        )
                        lime_vectors = []
                        for g in bench_graphs:
                            exp_list, _ = lime_explainer.explain_local(g, task)
                            lime_vectors.append(
                                benchmark.vectorize_graphlime(
                                    exp_list, act_features, res_features
                                )
                            )
                        if lime_vectors:
                            lime_attr = np.stack(lime_vectors, axis=0)

                if grad_attr.size > 0:
                    benchmark_results = benchmark.run_full_benchmark(
                        bench_graphs,
                        grad_attr,
                        lime_values=lime_attr,
                        node_contribs=node_contribs,
                        k_values=[1, 3, 5, 10],
                    )
                    benchmark.save_results(
                        bench_dir, filename=f"benchmark_results_{task}.json"
                    )
                    combined_summary_rows.extend(
                        benchmark.summary_rows(task_prefix=task)
                    )
                    benchmark.print_summary()
                else:
                    print(
                        "[WARNING] Benchmark skipped: no gradient attributions available."
                    )
            except Exception as e:
                print(f"[ERROR] Benchmark evaluation failed for {task}: {e}")

        if combined_summary_rows:
            combined_df = pd.DataFrame(combined_summary_rows)
            combined_path = os.path.join(bench_dir, "benchmark_summary.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"[OK] Combined benchmark summary saved to: {combined_path}")

    if methods in ["gradient", "all"] and not _dir_has_png(
        os.path.join(output_dir, "gradient")
    ):
        print("[WARNING] Gradient plots not found, but continuing...")

    if methods == "temporal" and not _dir_has_png(os.path.join(output_dir, "temporal")):
        # In global mode, we skip individual temporal plots, so this check should be flexible
        print("[INFO] Temporal plots skipped or not generated in global mode.")

    if methods in ["lime", "all"] and not _dir_has_png(
        os.path.join(output_dir, "graphlime")
    ):
        raise RuntimeError(
            "GNN explainability failed: GraphLIME plots were not generated."
        )

    print("\n" + "=" * 70)
    print("GNN EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {}


class GNNExplainerWrapper:
    def __init__(self, model, device, vocabularies=None, scaler=None):
        self.model = model
        self.device = device
        self.vocabularies = vocabularies
        self.scaler = scaler

    def run(
        self,
        data,
        output_dir,
        num_samples=50,
        methods="all",
        tasks=None,
        y_true=None,
        run_benchmark=True,
    ):
        return run_gnn_explainability(
            model=self.model,
            data=data,
            output_dir=output_dir,
            device=self.device,
            vocabularies=self.vocabularies,
            num_samples=num_samples,
            methods=methods,
            tasks=tasks,
            scaler=self.scaler,
            y_true=y_true,
            run_benchmark=run_benchmark,
        )
