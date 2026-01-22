import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class ReadableTableExplainer:
    """
    Feature-level explainer with READABLE table.
    Shows activities and resources at each step in a clean, understandable format.
    """
    
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
        """Get activity name, truncated if too long"""
        if 'Activity' in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs['Activity'].items()}
            name = inv_vocab.get(int(idx), f"Act_{idx}")
            # Truncate long names
            return name[:18] + ".." if len(name) > 18 else name
        return f"Activity_{idx}"

    def _get_resource_name(self, idx):
        """Get resource name, truncated if too long"""
        if 'Resource' in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs['Resource'].items()}
            name = inv_vocab.get(int(idx), f"Res_{idx}")
            return name[:12] + ".." if len(name) > 12 else name
        return f"Resource_{idx}"

    def explain_time_series_with_features(self, graphs, task='event_time', num_samples=50):
        """Compute contributions with feature information"""
        self.model.eval()
        
        time_step_data = []
        predictions = []
        true_values = []
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        selected_graphs = [graphs[i] for i in sample_indices]
        
        print(f"Computing feature-level contributions for {len(selected_graphs)} graphs ({task})...")
        
        for graph in tqdm(selected_graphs, desc=f"Analyzing ({task})"):
            graph = graph.to(self.device)
            seq_len = graph['activity'].x.shape[0]
            
            # Enable gradients
            for key in graph.x_dict:
                if key in ['activity', 'resource', 'time']:
                    graph.x_dict[key] = graph.x_dict[key].detach().clone()
                    graph.x_dict[key].requires_grad = True
            
            out = self.model(graph)
            
            if task == 'event_time':
                score = out[1]
                true_val = graph.y_timestamp.item()
            elif task == 'remaining_time':
                score = out[2]
                true_val = graph.y_remaining_time.item()
            else:
                continue
            
            self.model.zero_grad()
            score.backward()
            
            with torch.no_grad():
                for step in range(seq_len):
                    step_info = {'step': step}
                    contrib_sum = 0.0
                    
                    # FOR TIME TASKS: Only use timestamp gradients
                    if task in ['event_time', 'remaining_time']:
                        # ONLY timestamp contribution
                        if 'time' in graph.x_dict and graph.x_dict['time'].grad is not None:
                            time_grad = graph.x_dict['time'].grad[step]
                            time_inp = graph.x_dict['time'][step]
                            contrib_sum += (time_grad * time_inp).abs().sum().item()
                        
                        # Still extract activity/resource for CONTEXT (but don't add to contribution)
                        if 'activity' in graph.x_dict:
                            act_inp = graph.x_dict['activity'][step]
                            act_idx = act_inp.argmax().item()
                            step_info['activity'] = self._get_activity_name(act_idx)
                        
                        if 'resource' in graph.x_dict:
                            res_inp = graph.x_dict['resource'][step]
                            res_idx = res_inp.argmax().item()
                            step_info['resource'] = self._get_resource_name(res_idx)
                    
                    else:
                        # FOR ACTIVITY TASK: Use all features
                        if 'activity' in graph.x_dict and graph.x_dict['activity'].grad is not None:
                            act_grad = graph.x_dict['activity'].grad[step]
                            act_inp = graph.x_dict['activity'][step]
                            contrib_sum += (act_grad * act_inp).abs().sum().item()
                            act_idx = act_inp.argmax().item()
                            step_info['activity'] = self._get_activity_name(act_idx)
                        
                        if 'resource' in graph.x_dict and graph.x_dict['resource'].grad is not None:
                            res_grad = graph.x_dict['resource'].grad[step]
                            res_inp = graph.x_dict['resource'][step]
                            contrib_sum += (res_grad * res_inp).abs().sum().item()
                            res_idx = res_inp.argmax().item()
                            step_info['resource'] = self._get_resource_name(res_idx)
                        
                        if 'time' in graph.x_dict and graph.x_dict['time'].grad is not None:
                            time_grad = graph.x_dict['time'].grad[step]
                            time_inp = graph.x_dict['time'][step]
                            contrib_sum += (time_grad * time_inp).abs().sum().item()
                    
                    step_info['contribution'] = contrib_sum
                    time_step_data.append(step_info)
                
                predictions.append(score.item())
                true_values.append(true_val)
        
        df = pd.DataFrame(time_step_data)
        
        # Aggregate by step
        step_summary = []
        for step in range(df['step'].max() + 1):
            step_data = df[df['step'] == step]
            
            summary = {
                'step': step,
                'avg_contribution': step_data['contribution'].mean(),
                'std_contribution': step_data['contribution'].std(),
            }
            
            if 'activity' in step_data.columns:
                top_activity = step_data['activity'].mode()
                if len(top_activity) > 0:
                    summary['top_activity'] = top_activity.iloc[0]
                    summary['activity_freq'] = (step_data['activity'] == top_activity.iloc[0]).sum() / len(step_data)
                else:
                    summary['top_activity'] = "N/A"
                    summary['activity_freq'] = 0
            
            if 'resource' in step_data.columns:
                top_resource = step_data['resource'].mode()
                if len(top_resource) > 0:
                    summary['top_resource'] = top_resource.iloc[0]
                    summary['resource_freq'] = (step_data['resource'] == top_resource.iloc[0]).sum() / len(step_data)
                else:
                    summary['top_resource'] = "N/A"
                    summary['resource_freq'] = 0
            
            step_summary.append(summary)
        
        summary_df = pd.DataFrame(step_summary)
        
        return {
            'summary': summary_df,
            'detailed_data': df,
            'predictions': predictions,
            'true_values': true_values
        }
    
    def explain_individual_sample(self, graph, task='event_time'):
        """
        Explain a single sample using gradients.
        Returns per-step contributions and feature info.
        """
        self.model.eval()
        graph = graph.to(self.device)
        
        seq_len = graph['activity'].x.shape[0]
        
        # Enable gradients
        for key in graph.x_dict:
            if key in ['activity', 'resource', 'time']:
                graph.x_dict[key] = graph.x_dict[key].detach().clone()
                graph.x_dict[key].requires_grad = True
        
        out = self.model(graph)
        
        if task == 'event_time':
            score = out[1]
            true_val = graph.y_timestamp.item()
        elif task == 'remaining_time':
            score = out[2]
            true_val = graph.y_remaining_time.item()
        elif task == 'activity':
            # For activity, use the logits for the predicted class
            predicted_class = out[0].argmax()
            score = out[0][predicted_class]  # Score for the predicted class
            true_val = graph.y_activity.item()
        else:
            return None, None, None, None
        
        self.model.zero_grad()
        score.backward()
        
        # Extract per-step contributions and features
        step_contributions = []
        step_info = []
        
        with torch.no_grad():
            for step in range(seq_len):
                contrib_sum = 0.0
                info = {'step': step}
                
                # FOR TIME TASKS: Only use timestamp gradients
                if task in ['event_time', 'remaining_time']:
                    # ONLY timestamp contribution
                    if 'time' in graph.x_dict and graph.x_dict['time'].grad is not None:
                        time_grad = graph.x_dict['time'].grad[step]
                        time_inp = graph.x_dict['time'][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                    
                    # Still extract activity/resource for CONTEXT (but don't add to contribution)
                    if 'activity' in graph.x_dict:
                        act_inp = graph.x_dict['activity'][step]
                        act_idx = act_inp.argmax().item()
                        info['activity'] = self._get_activity_name(act_idx)
                    
                    if 'resource' in graph.x_dict:
                        res_inp = graph.x_dict['resource'][step]
                        res_idx = res_inp.argmax().item()
                        info['resource'] = self._get_resource_name(res_idx)
                
                elif task == 'activity':
                    # FOR ACTIVITY TASK: Only use activity gradients (not time/resource)
                    if 'activity' in graph.x_dict and graph.x_dict['activity'].grad is not None:
                        act_grad = graph.x_dict['activity'].grad[step]
                        act_inp = graph.x_dict['activity'][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info['activity'] = self._get_activity_name(act_idx)
                    
                    # Still extract resource/time for CONTEXT (but don't add to contribution)
                    if 'resource' in graph.x_dict:
                        res_inp = graph.x_dict['resource'][step]
                        res_idx = res_inp.argmax().item()
                        info['resource'] = self._get_resource_name(res_idx)
                    
                    if 'time' in graph.x_dict:
                        time_inp = graph.x_dict['time'][step]
                        # Just for display, not contribution
                
                else:
                    # FOR OTHER TASKS: Use all features
                    if 'activity' in graph.x_dict and graph.x_dict['activity'].grad is not None:
                        act_grad = graph.x_dict['activity'].grad[step]
                        act_inp = graph.x_dict['activity'][step]
                        contrib_sum += (act_grad * act_inp).abs().sum().item()
                        act_idx = act_inp.argmax().item()
                        info['activity'] = self._get_activity_name(act_idx)
                    
                    if 'resource' in graph.x_dict and graph.x_dict['resource'].grad is not None:
                        res_grad = graph.x_dict['resource'].grad[step]
                        res_inp = graph.x_dict['resource'][step]
                        contrib_sum += (res_grad * res_inp).abs().sum().item()
                        res_idx = res_inp.argmax().item()
                        info['resource'] = self._get_resource_name(res_idx)
                    
                    if 'time' in graph.x_dict and graph.x_dict['time'].grad is not None:
                        time_grad = graph.x_dict['time'].grad[step]
                        time_inp = graph.x_dict['time'][step]
                        contrib_sum += (time_grad * time_inp).abs().sum().item()
                
                step_contributions.append(contrib_sum)
                step_info.append(info)
        
        return np.array(step_contributions), score.item(), true_val, step_info
    
    def plot_individual_gradient_explanation(self, contributions, pred, true_val, step_info, output_dir, task, sample_id):
        """Plot SHAP-style explanation with activity labels on ALL bars"""
        os.makedirs(output_dir, exist_ok=True)
        
        if contributions is None:
            return
        
        num_steps = len(contributions)
        
        # Create figure with single plot
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Center contributions around mean
        centered_contrib = contributions - np.mean(contributions)
        
        time_steps = np.arange(num_steps)
        positive = np.maximum(centered_contrib, 0)
        negative = np.minimum(centered_contrib, 0)
        
        # Plot bars
        ax.bar(time_steps, positive, color='#d62728', alpha=0.85, 
               label='Positive', width=0.75, edgecolor='darkred', linewidth=0.8)
        ax.bar(time_steps, negative, color='#1f77b4', alpha=0.85, 
               label='Negative', width=0.75, edgecolor='darkblue', linewidth=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        
        # Add activity labels ON ALL BARS (no threshold filtering)
        max_contrib = np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
        label_offset = max_contrib * 0.10  # Increased from 0.08
        
        # LABEL ALL STEPS (removed threshold filter)
        for i in range(num_steps):
            info = step_info[i]
            activity = info.get('activity', 'N/A')
            
            # Truncate long activity names
            if len(activity) > 12:
                activity = activity[:10] + '..'
            
            contrib_val = centered_contrib[i]
            
            # Position label above positive bars, below negative bars
            if contrib_val > 0:
                y_pos = contrib_val + label_offset
                va = 'bottom'
                rotation = 45
            else:
                y_pos = contrib_val - label_offset
                va = 'top'
                rotation = 45
            
            # Add yellow box background for labels (like SHAP)
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', 
                             edgecolor='black', linewidth=0.8, alpha=0.9)
            
            ax.text(i, y_pos, activity, 
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   rotation=rotation, bbox=bbox_props)
        
        # NO second y-axis or observed data line
        
        # Formatting
        ax.set_xlabel(f'Time step (Sample {sample_id})', 
                     fontweight='bold', fontsize=14)
        ax.set_ylabel('Feature Contribution', 
                     fontweight='bold', fontsize=14, color='black')
        
        # Title
        task_name = task.replace('_', ' ').title()
        ax.set_title(f'Graph Neural Network (GNN) - Gradient Descent\n{task_name}',
                    fontweight='bold', fontsize=16, pad=20)
        
        # Legend
        ax.legend(loc='upper left', framealpha=0.95, fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-axis ticks
        if num_steps <= 25:
            ax.set_xticks(time_steps)
            ax.set_xticklabels([f'{i}' for i in time_steps], fontsize=10)
        else:
            skip = max(1, num_steps // 25)
            ax.set_xticks(time_steps[::skip])
            ax.set_xticklabels([f'{i}' for i in time_steps[::skip]], fontsize=10)
        
        # Set y-limits for better visibility (increased margin)
        y_margin = max_contrib * 0.40  # Increased from 0.35
        ax.set_ylim(negative.min() - y_margin if len(negative) > 0 else -1, 
                   positive.max() + y_margin if len(positive) > 0 else 1)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'gradient_sample_{sample_id}_{task}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  [✓] SHAP-style plot saved: gradient_sample_{sample_id}_{task}.png")
        
        # Save CSV with appropriate columns based on task
        if step_info:
            df_info = pd.DataFrame(step_info)
            df_info['shap_contribution'] = centered_contrib
            
            # Add note about what contributes to the values
            if task in ['event_time', 'remaining_time']:
                # For time tasks, make it clear contributions are from timestamps only
                csv_path = os.path.join(output_dir, f'gradient_sample_{sample_id}_{task}_details.csv')
                
                # Reorder columns to emphasize timestamp contribution
                cols = ['step']
                if 'activity' in df_info.columns:
                    cols.append('activity')
                if 'resource' in df_info.columns:
                    cols.append('resource')
                cols.append('shap_contribution')
                
                df_info = df_info[cols]
                
                # Add header comment to CSV
                with open(csv_path, 'w') as f:
                    f.write(f"# Task: {task}\n")
                    f.write(f"# SHAP contributions calculated from: TIMESTAMP GRADIENTS ONLY\n")
                    f.write(f"# Activity and Resource columns shown for CONTEXT only\n")
                    f.write(f"# (they do NOT contribute to the shap_contribution values)\n")
                    df_info.to_csv(f, index=False)
                
                print(f"  [✓] Details CSV saved (timestamp-only contributions)")
            else:
                # For activity task, all features contribute
                df_info.to_csv(os.path.join(output_dir, f'gradient_sample_{sample_id}_{task}_details.csv'), index=False)
                print(f"  [✓] Details CSV saved (all features)")

    def plot_with_readable_table(self, results, output_dir, task):
        """Create SHAP-style visualization (no table, labels on bars)"""
        os.makedirs(output_dir, exist_ok=True)
        
        summary_df = results['summary']
        predictions = results['predictions']
        true_values = results['true_values']
        
        max_len = len(summary_df)
        
        # Aggregate for long sequences
        if max_len > 40:
            bin_size = max(2, max_len // 35)
            print(f"[INFO] Long sequence ({max_len} steps) - binning into ~{max_len//bin_size} bins")
            
            # Bin the data
            binned_data = []
            for bin_idx in range((max_len + bin_size - 1) // bin_size):
                start_step = bin_idx * bin_size
                end_step = min((bin_idx + 1) * bin_size, max_len)
                bin_df = summary_df[(summary_df['step'] >= start_step) & (summary_df['step'] < end_step)]
                
                binned_data.append({
                    'step': bin_idx,
                    'step_range': f"{start_step}-{end_step-1}",
                    'avg_contribution': bin_df['avg_contribution'].mean(),
                    'top_activity': bin_df['top_activity'].mode().iloc[0] if len(bin_df['top_activity'].mode()) > 0 else "N/A",
                })
            
            plot_df = pd.DataFrame(binned_data)
            x_values = plot_df['step'].values
            contributions = plot_df['avg_contribution'].values
            activities = plot_df['top_activity'].values
            xlabel_text = 'Event sequence position (binned)'
        else:
            # Use raw data
            x_values = summary_df['step'].values
            contributions = summary_df['avg_contribution'].values
            activities = summary_df['top_activity'].values
            xlabel_text = 'Event sequence position (BPI Dataset)'
        
        # Center contributions
        mean_val = np.mean(contributions[contributions > 0]) if np.any(contributions > 0) else 0
        centered_contrib = contributions - mean_val
        
        # Create single plot
        fig, ax = plt.subplots(figsize=(20, 8))
        
        positive = np.maximum(centered_contrib, 0)
        negative = np.minimum(centered_contrib, 0)
        
        # Plot bars
        ax.bar(x_values, positive, color='#d62728', alpha=0.85, 
               label='Positive', width=0.75, edgecolor='darkred', linewidth=0.8)
        ax.bar(x_values, negative, color='#1f77b4', alpha=0.85, 
               label='Negative', width=0.75, edgecolor='darkblue', linewidth=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        
        # Add activity labels on bars
        max_contrib = np.max(np.abs(centered_contrib)) if len(centered_contrib) > 0 else 1
        label_offset = max_contrib * 0.12
        
        # Only label significant bars (top contributors)
        threshold = np.percentile(np.abs(centered_contrib), 70) if len(centered_contrib) > 0 else 0  # Top 30%
        
        for i, (x, contrib, activity) in enumerate(zip(x_values, centered_contrib, activities)):
            if abs(contrib) >= threshold:  # Only label important steps
                # Truncate activity name
                activity_str = str(activity)
                if len(activity_str) > 12:
                    activity_str = activity_str[:10] + '..'
                
                # Position
                if contrib > 0:
                    y_pos = contrib + label_offset
                    va = 'bottom'
                else:
                    y_pos = contrib - label_offset
                    va = 'top'
                
                # Yellow box label
                bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', 
                                 edgecolor='black', linewidth=0.8, alpha=0.9)
                
                ax.text(x, y_pos, activity_str, 
                       ha='center', va=va, fontsize=10, fontweight='bold',
                       rotation=45, bbox=bbox_props)
        
        # NO second y-axis for aggregated view (flat line doesn't add value)
        
        # Formatting
        num_samples = len(predictions) if len(predictions) > 0 else 0
        ax.set_xlabel(xlabel_text, fontweight='bold', fontsize=14)
        ax.set_ylabel('Feature Contribution', 
                     fontweight='bold', fontsize=14)
        
        task_name = task.replace('_', ' ').title()
        # Updated title to show aggregation
        ax.set_title(f'Graph Neural Network (GNN) Model - SHAP Explainability (Averaged Over {num_samples} Samples)\n{task_name}',
                    fontweight='bold', fontsize=17, pad=20)
        
        # Legend
        ax.legend(loc='upper left', framealpha=0.95, fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-ticks
        if len(x_values) <= 30:
            ax.set_xticks(x_values)
            if max_len > 40:
                labels = [plot_df.iloc[i]['step_range'] for i in range(len(x_values))]
                ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
            else:
                ax.set_xticklabels([f'{int(x)}' for x in x_values], fontsize=10)
        else:
            skip = max(1, len(x_values) // 30)
            ax.set_xticks(x_values[::skip])
            ax.set_xticklabels([f'{int(x)}' for x in x_values[::skip]], fontsize=10)
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'feature_level_{task}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[✓] SHAP-style plot saved")
        
        # Still save CSV
        summary_df.to_csv(os.path.join(output_dir, f'feature_summary_{task}.csv'), index=False)
        print(f"[✓] CSV saved")

    def explain_global_activity(self, graphs, num_samples=50):
        """Activity classification bar chart"""
        self.model.eval()
        importances = {'activity': [], 'resource': []}
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        selected_graphs = [graphs[i] for i in sample_indices]
        
        print(f"Computing gradients for {len(selected_graphs)} graphs (activity)...")
        
        for graph in tqdm(selected_graphs, desc="Activity Analysis"):
            graph = graph.to(self.device)
            for key in graph.x_dict:
                if key in ['activity', 'resource']:
                    graph.x_dict[key] = graph.x_dict[key].detach().clone()
                    graph.x_dict[key].requires_grad = True
            
            out = self.model(graph)
            logits = out[0]
            pred_idx = logits.argmax(dim=1)
            score = logits[0, pred_idx]
            
            self.model.zero_grad()
            score.backward()
            
            with torch.no_grad():
                for node_type in ['activity', 'resource']:
                    if node_type in graph.x_dict and graph.x_dict[node_type].grad is not None:
                        grad = graph.x_dict[node_type].grad
                        inp = graph.x_dict[node_type]
                        imp = (grad * inp).abs().sum(dim=0).cpu().numpy()
                        importances[node_type].append(imp)
        
        global_imp = {}
        for n_type in importances:
            if len(importances[n_type]) > 0:
                global_imp[n_type] = np.mean(np.stack(importances[n_type]), axis=0)
        
        return global_imp

    def plot_global_importance_activity(self, importances, output_dir):
        """Bar chart for activity prediction"""
        os.makedirs(output_dir, exist_ok=True)
        
        data = []
        for n_type, scores in importances.items():
            top_k_idx = np.argsort(scores)[-10:]
            for idx in top_k_idx:
                if scores[idx] > 0:
                    if n_type == 'activity':
                        name = self._get_activity_name(idx)
                    elif n_type == 'resource':
                        name = self._get_resource_name(idx)
                    else:
                        name = f"{n_type}_{idx}"
                    
                    data.append({
                        'Feature': name,
                        'Importance': scores[idx],
                        'Type': n_type.capitalize()
                    })
        
        df = pd.DataFrame(data).sort_values('Importance', ascending=True)
        df.to_csv(os.path.join(output_dir, 'gradient_global_activity.csv'), index=False)
        
        if df.empty:
            return
        
        plt.figure(figsize=(12, 7))
        colors = {'Activity': '#1f77b4', 'Resource': '#ff7f0e'}
        
        plt.barh(df['Feature'], df['Importance'],
                 color=[colors.get(t, 'grey') for t in df['Type']])
        
        plt.title("Global Feature Importance (Next Activity)", fontweight='bold', fontsize=14)
        plt.xlabel("Mean Gradient Magnitude (Impact)", fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_global_activity.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[✓] Activity bar chart saved")


class GraphLIMEExplainer:
    """
    GraphLIME local explainer for individual predictions.
    """
    
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
        if 'Activity' in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs['Activity'].items()}
            return inv_vocab.get(int(idx), f"Act_{idx}")[:18]
        return f"Activity_{idx}"
    
    def _get_resource_name(self, idx):
        if 'Resource' in self.vocabs:
            inv_vocab = {v: k for k, v in self.vocabs['Resource'].items()}
            return inv_vocab.get(int(idx), f"Res_{idx}")[:12]
        return f"Resource_{idx}"
    
    def explain_local(self, graph, task='event_time', num_perturbations=200):
        """
        Local explanation for a single graph showing which steps matter.
        """
        self.model.eval()
        graph = graph.to(self.device)
        
        # Get base prediction
        predicted_class = None  # For activity task
        with torch.no_grad():
            out = self.model(graph)
            if task == 'event_time':
                base_score = out[1].item()
                true_val = graph.y_timestamp.item()
            elif task == 'remaining_time':
                base_score = out[2].item()
                true_val = graph.y_remaining_time.item()
            elif task == 'activity':
                # For activity, use the probability of the predicted class (not just index)
                probs = torch.softmax(out[0], dim=-1)  # Softmax over last dimension
                predicted_class = probs.argmax().item()  # Get predicted class from probs
                base_score = probs[predicted_class].item()  # Confidence score
                true_val = graph.y_activity.item()
            else:
                return None, None, None, None
        
        seq_len = graph['activity'].x.shape[0]
        
        # Extract activity/resource names for display
        step_info = []
        with torch.no_grad():
            for step in range(seq_len):
                info = {'step': step}
                if 'activity' in graph.x_dict:
                    act_idx = graph['activity'].x[step].argmax().item()
                    info['activity'] = self._get_activity_name(act_idx)
                if 'resource' in graph.x_dict:
                    res_idx = graph['resource'].x[step].argmax().item()
                    info['resource'] = self._get_resource_name(res_idx)
                step_info.append(info)
        
        # Perturb by masking time steps
        X_perturb = []
        y_perturb = []
        
        for _ in range(num_perturbations):
            # Create mask
            mask = np.ones(seq_len)
            num_to_mask = max(1, np.random.randint(1, max(2, int(seq_len * 0.4))))
            mask_indices = np.random.choice(seq_len, size=num_to_mask, replace=False)
            mask[mask_indices] = 0
            
            X_perturb.append(mask)
            
            # Apply mask
            masked_graph = graph.clone()
            mask_tensor = torch.tensor(mask, device=self.device, dtype=torch.float32).view(-1, 1)
            
            # FOR TIME TASKS: Only mask timestamps (not activity/resource)
            if task in ['event_time', 'remaining_time']:
                # Only perturb time features
                masked_graph['time'].x = masked_graph['time'].x * mask_tensor
                # Keep activity and resource unchanged
            elif task == 'activity':
                # FOR ACTIVITY TASK: Only mask activity features (not time/resource)
                masked_graph['activity'].x = masked_graph['activity'].x * mask_tensor
                # Keep resource and time unchanged
            else:
                # FOR OTHER TASKS: Mask all features
                masked_graph['activity'].x = masked_graph['activity'].x * mask_tensor
                masked_graph['resource'].x = masked_graph['resource'].x * mask_tensor
                masked_graph['time'].x = masked_graph['time'].x * mask_tensor
            
            # Get prediction
            with torch.no_grad():
                out_p = self.model(masked_graph)
                if task == 'event_time':
                    score = out_p[1].item()
                elif task == 'remaining_time':
                    score = out_p[2].item()
                elif task == 'activity':
                    # Use probability of the predicted class (same as base prediction)
                    probs = torch.softmax(out_p[0], dim=-1)  # Softmax over last dimension
                    score = probs[predicted_class].item()  # How confident is it now?
                y_perturb.append(score)
        
        X_perturb = np.array(X_perturb)
        y_perturb = np.array(y_perturb)
        
        # Fit Ridge regression
        distances = pairwise_distances(X_perturb, X_perturb[0].reshape(1, -1), metric='cosine').ravel()
        weights = np.sqrt(np.exp(-(distances**2) / 0.25))
        
        simpler_model = Ridge(alpha=1.0)
        simpler_model.fit(X_perturb, y_perturb, sample_weight=weights)
        
        # Coefficients = importance of each step
        step_importance = simpler_model.coef_
        
        return step_importance, base_score, true_val, step_info, predicted_class
    
    def plot_local_explanation(self, importance, base_score, true_val, step_info, output_dir, task, sample_id, predicted_class=None):
        """Plot local explanation with horizontal bars (original style with updated labels)"""
        os.makedirs(output_dir, exist_ok=True)
        
        if importance is None:
            return
        
        num_steps = len(importance)
        
        # Get feature names (activities) with O_ prefix
        feature_names = []
        for info in step_info:
            activity = info.get('activity', f'Step_{info["step"]}')
            feature_names.append(f'O_{activity}')
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, num_steps * 0.5)))
        
        # Sort by importance (absolute value)
        sorted_indices = np.argsort(np.abs(importance))
        sorted_importance = importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Color bars: green for positive, red for negative
        colors = ['green' if x > 0 else 'red' for x in sorted_importance]
        
        # Plot horizontal bars
        y_pos = np.arange(len(sorted_names))
        
        # Track which legend entries we've added
        has_positive = False
        has_negative = False
        
        # Create bars with labels for legend
        for i, (y, importance_val, color) in enumerate(zip(y_pos, sorted_importance, colors)):
            if importance_val > 0 and not has_positive:
                # First positive bar - add to legend
                ax.barh(y, importance_val, color=color, alpha=0.7, edgecolor='black', 
                       linewidth=1, label='Support')
                has_positive = True
            elif importance_val < 0 and not has_negative:
                # First negative bar - add to legend
                ax.barh(y, importance_val, color=color, alpha=0.7, edgecolor='black', 
                       linewidth=1, label='Contradict')
                has_negative = True
            else:
                # Regular bar without label
                ax.barh(y, importance_val, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Labels - Updated as requested
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel('Feature Contribution (LIME)', fontweight='bold', fontsize=12)
        # No Y-axis label
        
        # Title - Updated as requested
        task_name = task.replace('_', ' ').title()
        if task == 'activity':
            # For activity, show the predicted activity name and confidence
            pred_activity = self._get_activity_name(int(predicted_class)) if predicted_class is not None else "Unknown"
            confidence_pct = base_score * 100  # Convert probability to percentage
            ax.set_title(f'Graph Neural Network (GNN) - GraphLIME\n{task_name} (Sample {sample_id})\nPrediction: {pred_activity} ({confidence_pct:.1f}%)',
                        fontweight='bold', fontsize=13, pad=15)
        else:
            # For time tasks, show numeric prediction
            ax.set_title(f'Graph Neural Network (GNN) - GraphLIME\n{task_name} (Sample {sample_id})\nPrediction: {base_score:.2f}',
                        fontweight='bold', fontsize=13, pad=15)
        
        # Legend
        ax.legend(loc='best', framealpha=0.95, fontsize=11)
        
        # Vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Grid
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'graphlime_sample_{sample_id}_{task}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  [✓] LIME plot saved: graphlime_sample_{sample_id}_{task}.png")
        
        # Save step details to CSV
        if step_info:
            df_info = pd.DataFrame(step_info)
            df_info['lime_contribution'] = importance
            df_info.to_csv(os.path.join(output_dir, f'graphlime_sample_{sample_id}_{task}_details.csv'), index=False)
            print(f"  [✓] Step details CSV saved")


def run_gnn_explainability(model, data, output_dir, device, vocabularies=None, 
                          num_samples=50, methods='all', tasks=None):
    """Run explainability with readable tables and LIME support"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY - READABLE TABLE VERSION")
    print("="*70)
    
    if 'test_graphs' in data:
        graphs = data['test_graphs']
    elif 'test' in data:
        graphs = data['test']
    else:
        print("[✗] Error: Could not find test graphs")
        return {}
    
    if tasks is None:
        print("[!] WARNING: No tasks specified")
        return {}
    
    if isinstance(tasks, str):
        tasks = [tasks]
    
    print(f"\n[→] Tasks: {tasks}")
    print(f"[→] Samples: {num_samples}")
    
    # ========== GRADIENT-BASED EXPLANATIONS ==========
    if methods in ['gradient', 'all']:
        print("\n" + "─"*70)
        print("GRADIENT ANALYSIS")
        print("─"*70)
        
        explainer = ReadableTableExplainer(model, device, vocabularies)
        grad_dir = os.path.join(output_dir, 'gradient')
        
        for task in tasks:
            print(f"\n[{task.upper()}]")
            
            try:
                if task == 'activity':
                    imp = explainer.explain_global_activity(graphs, num_samples)
                    explainer.plot_global_importance_activity(imp, grad_dir)
                    
                elif task in ['event_time', 'remaining_time']:
                    # Global aggregated explanation (DISABLED - only showing individual samples)
                    # print("  Creating global aggregated explanation...")
                    # results = explainer.explain_time_series_with_features(graphs, task, num_samples)
                    # explainer.plot_with_readable_table(results, grad_dir, task)
                    
                    # Individual sample explanations only
                    print(f"\n  Creating individual sample explanations...")
                    num_individual = min(5, len(graphs))
                    
                    # Pick samples from different parts of dataset (not just first 5!)
                    sample_indices = []
                    if len(graphs) > 100:
                        # Pick from different quartiles for variety
                        sample_indices = [
                            len(graphs) // 4,      # 25% through dataset
                            len(graphs) // 2,      # 50% through dataset  
                            3 * len(graphs) // 4,  # 75% through dataset
                            len(graphs) - 100,     # Near end
                            len(graphs) - 50,      # Very near end
                        ]
                    else:
                        # Small dataset - skip first few short prefixes
                        start = min(10, len(graphs) // 3)
                        sample_indices = list(range(start, min(start + num_individual, len(graphs))))
                    
                    for i, idx in enumerate(sample_indices):
                        print(f"  Sample {i} (graph index {idx}):")
                        graph = graphs[idx]
                        contrib, pred, true_val, step_info = explainer.explain_individual_sample(graph, task)
                        explainer.plot_individual_gradient_explanation(contrib, pred, true_val, step_info, grad_dir, task, i)
                    
            except Exception as e:
                print(f"  [✗] Error: {e}")
                import traceback
                traceback.print_exc()
    
    # ========== GRAPHLIME LOCAL ANALYSIS ==========
    if methods in ['lime', 'all']:
        print("\n" + "─"*70)
        print("GRAPHLIME LOCAL ANALYSIS")
        print("─"*70)
        
        lime_explainer = GraphLIMEExplainer(model, device, vocabularies)
        lime_dir = os.path.join(output_dir, 'graphlime')
        
        # Analyze first 5 samples
        num_local_samples = min(5, len(graphs))
        print(f"\n[→] Analyzing {num_local_samples} individual samples...")
        
        for idx in range(num_local_samples):
            graph = graphs[idx]
            print(f"\nSample {idx}:")
            
            for task in tasks:
                try:
                    print(f"  Processing {task}...")
                    imp, score, true_val, step_info, pred_class = lime_explainer.explain_local(graph, task)
                    lime_explainer.plot_local_explanation(imp, score, true_val, step_info, lime_dir, task, idx, pred_class)
                    
                except Exception as e:
                    print(f"  [✗] Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"✓ COMPLETE - Results in: {output_dir}")
    print("="*70)
    
    return {}


class GNNExplainerWrapper:
    def __init__(self, model, device, vocabularies=None):
        self.model = model
        self.device = device
        self.vocabularies = vocabularies

    def run(self, data, output_dir, num_samples=50, methods='all', tasks=None):
        return run_gnn_explainability(
            model=self.model,
            data=data,
            output_dir=output_dir,
            device=self.device,
            vocabularies=self.vocabularies,
            num_samples=num_samples,
            methods=methods,
            tasks=tasks
        )