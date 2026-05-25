import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_research_grade_local(df, current_seq_names, output_path, title="Trace History"):
    # --- Bar Plot Only ---
    fig, ax_bar = plt.subplots(figsize=(11, 5))
    
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df['importance']]
    bars = ax_bar.barh(df['activity'], df['importance'], color=colors, height=0.6)
    
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.6)
    ax_bar.margins(x=0.15)
    
    ax_bar.set_xlabel("Contribution to Prediction", fontsize=11)
    
    for rect in bars:
        w = rect.get_width()
        y = rect.get_y() + rect.get_height()/2
        padding = 0.0005 if w > 0 else -0.0005
        ha = 'left' if w > 0 else 'right'
        ax_bar.text(w + padding, y, f'{w:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')

    ax_bar.legend(handles=[
        Patch(facecolor='#2ca02c', label='Supports'),
        Patch(facecolor='#d62728', label='Contradicts')
    ], loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close()
