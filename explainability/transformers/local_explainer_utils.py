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

def plot_waterfall_local(df, current_seq_names, output_path, title="Trace History (Waterfall)", base_value=0.0):
    fig, ax = plt.subplots(figsize=(11, 6))
    
    activities = df['activity'].tolist()
    importances = df['importance'].tolist()
    
    final_value = base_value + sum(importances)
    
    # Positive Steps (Green/Pushing time up), Negative Steps (Red/Pulling time down)
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in importances]
    
    # Diverging horizontal bar chart centered at 0
    bars = ax.barh(activities, importances, color=colors, height=0.6)
    
    # X-axis centered at 0
    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    ax.set_xlabel("Impact on Remaining Time (Days)", fontsize=11, fontweight='bold')
    
    # Place Base Value directly above the zero-line
    ax.text(0, 1.02, f"Base Value: {base_value:.2f} d", transform=ax.get_xaxis_transform(),
            va='bottom', ha='center', fontsize=11, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
    
    # Add data labels
    for rect in bars:
        w = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        padding = 0.02 * (np.max(np.abs(importances)) if len(importances) > 0 else 1)
        if w < 0:
            padding = -padding
        ha = 'left' if w > 0 else 'right'
        ax.text(w + padding, y, f'{w:+.2f} d', va='center', ha=ha, fontsize=10, fontweight='bold')
        
    ax.legend(handles=[
        Patch(facecolor='#2ca02c', label='Increases Duration'),
        Patch(facecolor='#d62728', label='Decreases Duration')
    ], loc='lower right', frameon=True)

    # Adjust margins to fit labels comfortably
    ax.margins(x=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="white", bbox_inches='tight')
    plt.close()

def _ensure_stub_csv(path, columns):
    import pandas as pd
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)
