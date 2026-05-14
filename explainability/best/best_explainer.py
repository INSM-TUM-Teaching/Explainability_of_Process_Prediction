import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})


class BESTExplainer:
    """Native pattern-based explainability for a fitted BESTPredictor.

    After BESTPredictor.predict() has been called the model exposes:
        model.choice_tracker_nap  – {'prob': [], 'len': [], 'dist': []}
        model.choice_tracker_rtp  – {'prob': [], 'len': [], 'dist': []}

    This class reads those trackers and writes visual + tabular outputs to
    output_dir/explainability/.
    """

    def __init__(self, model, output_dir: str, task: str):
        """
        Args:
            model:      A fitted BESTPredictor instance (after predict() was called).
            output_dir: Base artifacts directory; outputs go to output_dir/explainability/.
            task:       'nap' or 'rtp'.
        """
        self.model = model
        self.output_dir = os.path.join(output_dir, "explainability")
        self.task = task

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
        lengths = self._clean(tracker.get("len", []))
        distances = self._clean(tracker.get("dist", []))

        if len(probs) == 0:
            self._write_placeholder("Tracker contains no data points.")
            return

        self._plot_probabilities(probs)
        self._plot_lengths(lengths)
        self._plot_distances(distances)
        self._save_top_patterns(probs, lengths, distances)

        print(f"[BEST Explainer] Artefacts written to {self.output_dir}")

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
    # Plots
    # ------------------------------------------------------------------

    def _plot_probabilities(self, probs: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(probs, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Pattern conditional probability")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of chosen-pattern probabilities")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "pattern_probabilities.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _plot_lengths(self, lengths: np.ndarray) -> None:
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

    def _plot_distances(self, distances: np.ndarray) -> None:
        if len(distances) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=30, color="#C44E52", edgecolor="white", alpha=0.85)
        ax.set_xlabel("RPIF distance")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of RPIF distance scores")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "rpif_distances.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Top-patterns CSV
    # ------------------------------------------------------------------

    def _save_top_patterns(
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

        # Round into probability buckets and aggregate
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
