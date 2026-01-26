import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import Lasso

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

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

    def _get_feature_name(self, node_type, feature_idx):
        if node_type == 'activity' and 'Activity' in self.vocabs:
            vocab = self.vocabs['Activity']
            inv_vocab = {v: k for k, v in vocab.items()}
            return inv_vocab.get(int(feature_idx), f"Activity_{feature_idx}")
        elif node_type == 'resource' and 'Resource' in self.vocabs:
            vocab = self.vocabs['Resource']
            inv_vocab = {v: k for k, v in vocab.items()}
            return inv_vocab.get(int(feature_idx), f"Resource_{feature_idx}")
        return f"{node_type}_{feature_idx}"

    def explain_global(self, graphs, task='activity', num_samples=50):
        self.model.eval()
        importances = {'activity': [], 'resource': []}
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        selected_graphs = [graphs[i] for i in sample_indices]
        
        print(f"Computing gradients for {len(selected_graphs)} graphs ({task})...")
        
        for graph in tqdm(selected_graphs):
            graph = graph.to(self.device)
            for key in graph.x_dict:
                graph.x_dict[key].requires_grad = True
                
            out = self.model(graph)
            
            if task == 'activity':
                logits = out[0]
                pred_idx = logits.argmax(dim=1)
                score = logits[0, pred_idx]
            elif task == 'event_time':
                score = out[1]
            elif task == 'remaining_time':
                score = out[2]
                
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

    def plot_global_importance(self, importances, output_dir, task):
        os.makedirs(output_dir, exist_ok=True)
        
        data = []
        for n_type, scores in importances.items():
            top_k_idx = np.argsort(scores)[-10:] 
            for idx in top_k_idx:
                if scores[idx] > 0:
                    name = self._get_feature_name(n_type, idx)
                    data.append({
                        'Feature': name, 
                        'Importance': scores[idx], 
                        'Type': n_type.capitalize()
                    })
                    
        df = pd.DataFrame(data).sort_values('Importance', ascending=True)
        df.to_csv(os.path.join(output_dir, f'gradient_global_{task}.csv'), index=False)
        
        if df.empty: return

        plt.figure(figsize=(10, 6))
        colors = {'Activity': '#1f77b4', 'Resource': '#ff7f0e'}
        
        plt.barh(df['Feature'], df['Importance'], 
                 color=[colors.get(t, 'grey') for t in df['Type']])
        
        plt.title(f"Global Feature Importance ({task.replace('_', ' ').title()})", fontweight='bold')
        plt.xlabel("Mean Gradient Magnitude (Impact)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gradient_global_{task}.png'), dpi=300)
        plt.close()


class TemporalGradientExplainer:
    """
    Generates temporal attribution plots showing gradient contributions over time steps
    along with observed data values - similar to SHAP temporal plots.
    """
    def __init__(self, model, device, vocabularies=None, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler
        
        if callable(vocabularies):
            try:
                self.vocabs = vocabularies()
            except:
                self.vocabs = {}
        else:
            self.vocabs = vocabularies or {}
        if not isinstance(self.vocabs, dict):
            self.vocabs = {}

    def _maybe_unscale(self, arr):
        if self.scaler is None or arr is None:
            return arr
        
        try:
            arr_reshaped = arr.reshape(-1, 1)
            if hasattr(self.scaler, 'n_features_in_'):
                if self.scaler.n_features_in_ == 1:
                    return self.scaler.inverse_transform(arr_reshaped).flatten()
                else:
                    return arr
            else:
                return self.scaler.inverse_transform(arr_reshaped).flatten()
        except (ValueError, AttributeError) as e:
            return arr

    def compute_temporal_gradients(self, graphs, task='activity', num_samples=10):
        """
        Compute gradient-based attributions for each node/time step in the sequence.
        Returns per-sample temporal gradient contributions.
        """
        self.model.eval()
        temporal_contributions = []
        observed_sequences = []
        predictions = []
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        selected_graphs = [graphs[i] for i in sample_indices]
        
        print(f"Computing temporal gradients for {len(selected_graphs)} graphs ({task})...")
        
        for graph in tqdm(selected_graphs):
            graph = graph.to(self.device)
            
            for key in graph.x_dict:
                graph.x_dict[key].requires_grad = True
            
            out = self.model(graph)
            
            if task == 'activity':
                logits = out[0]
                pred_idx = logits.argmax(dim=1)
                score = logits[0, pred_idx]
                pred_value = pred_idx.item()
            elif task == 'event_time':
                score = out[1]
                pred_value = score.item()
            elif task == 'remaining_time':
                score = out[2]
                pred_value = score.item()
            
            self.model.zero_grad()
            score.backward()
            
            with torch.no_grad():
                if 'activity' in graph.x_dict and graph.x_dict['activity'].grad is not None:
                    grad = graph.x_dict['activity'].grad
                    inp = graph.x_dict['activity']
                    
                    # Compute contribution per node (row) by summing across features
                    # grad shape: (num_nodes, num_features)
                    # We want contribution per node/time step
                    contribution = (grad * inp).sum(dim=1).cpu().numpy()
                    
                    # Get observed values (sum of features per node)
                    observed = inp.sum(dim=1).cpu().numpy()
                    
                    # Only keep if we have more than 1 time step
                    if len(contribution) > 1:
                        temporal_contributions.append(contribution)
                        observed_sequences.append(observed)
                        predictions.append(pred_value)
                    else:
                        print(f"[WARNING] Graph has only {len(contribution)} node(s), skipping.")
        
        return temporal_contributions, observed_sequences, predictions

    def generate_temporal_plots(self, graphs, output_dir, task='activity', num_samples=10, y_true=None):
        """
        Generate temporal attribution plots showing gradient contributions and observed values.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        contributions, observed_seqs, preds = self.compute_temporal_gradients(graphs, task, num_samples)
        
        if not contributions:
            print("[WARNING] Temporal gradient plot skipped: No valid gradients computed.")
            return
        
        preds = self._maybe_unscale(np.array(preds))
        if y_true is not None:
            y_true = self._maybe_unscale(y_true[:len(contributions)])
        
        print(f"Generating temporal gradient plots for {len(contributions)} samples...")
        
        for i in range(len(contributions)):
            signal = contributions[i]
            observed = observed_seqs[i] if i < len(observed_seqs) else None
            
            # Get sequence length
            seq_len = len(signal)
            
            # Skip if only 1 or 0 time steps
            if seq_len <= 1:
                print(f"[WARNING] Sample {i} has only {seq_len} time step(s), skipping plot.")
                continue
            
            # Find valid (non-zero) length
            non_zero_indices = np.nonzero(signal)[0]
            if len(non_zero_indices) > 1:
                valid_len = non_zero_indices[-1] + 1
            else:
                valid_len = seq_len
            
            # Trim to valid length (from beginning)
            if valid_len > 1:
                signal = signal[:valid_len]
                if observed is not None and len(observed) >= valid_len:
                    observed = observed[:valid_len]
                elif observed is not None:
                    # Align lengths
                    min_len = min(len(signal), len(observed))
                    signal = signal[:min_len]
                    observed = observed[:min_len]
            
            seq_len = len(signal)
            
            # Final check
            if seq_len <= 1:
                print(f"[WARNING] Sample {i} has only {seq_len} valid time step(s), skipping plot.")
                continue
                
            x = np.arange(1, seq_len + 1)
            x_label = "Time steps"
            
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            pos_color = '#d62728'
            neg_color = '#1f77b4'
            colors = np.where(signal >= 0, pos_color, neg_color)
            ax.bar(x, signal, color=colors, width=0.8, alpha=0.9)
            ax.axhline(0, color='#222222', linewidth=1.0, alpha=0.6)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Gradient values")
            
            plt.title("Observed values and contribution scores", fontsize=12, fontweight='bold')
            
            legend_handles = []
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_handles.append(Patch(facecolor=pos_color, label="Positive Gradient values"))
            legend_handles.append(Patch(facecolor=neg_color, label="Negative Gradient values"))
            
            # Check if y_true has sequence data for this sample
            if y_true is not None and i < len(y_true):
                if hasattr(y_true[i], '__len__') and len(y_true[i]) > 1:
                    observed = np.array(y_true[i])[:seq_len]
            
            if observed is not None and len(observed) == len(signal):
                ax2 = ax.twinx()
                ax2.plot(x, observed, color='#555555', linewidth=1.5)
                ax2.set_ylabel("Observed data values")
                legend_handles.append(Line2D([0], [0], color='#555555', linewidth=1.5, label="Observed data"))
            
            ax.legend(handles=legend_handles, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'temporal_gradient_sample_{i}.png'), dpi=300)
            plt.close()
            
            df = pd.DataFrame({
                'TimeStep': x,
                'Contribution': signal
            })
            if observed is not None and len(observed) == len(signal):
                df['Observed'] = observed
            df.to_csv(os.path.join(output_dir, f'temporal_gradient_sample_{i}.csv'), index=False)
        
        print(f"[OK] Temporal gradient plots saved: {len(contributions)} samples")


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
        
    def _get_feature_name(self, node_type, feature_idx):
        if node_type == 'activity' and 'Activity' in self.vocabs:
            vocab = self.vocabs['Activity']
            inv_vocab = {v: k for k, v in vocab.items()}
            return inv_vocab.get(int(feature_idx), f"Activity_{feature_idx}")
        elif node_type == 'resource' and 'Resource' in self.vocabs:
            vocab = self.vocabs['Resource']
            inv_vocab = {v: k for k, v in vocab.items()}
            return inv_vocab.get(int(feature_idx), f"Resource_{feature_idx}")
        return f"{node_type}_{feature_idx}"

    def _aggregate_features(self, explanation):
        activity_stats = {}
        for item in explanation:
            name = item['Feature']
            weight = item['Weight']
            
            if name not in activity_stats:
                activity_stats[name] = {'weight': 0.0, 'count': 0}
            activity_stats[name]['weight'] += weight
            activity_stats[name]['count'] += 1
        
        aggregated = []
        for name, stats in activity_stats.items():
            label = f"{name} (x{stats['count']})" if stats['count'] > 1 else name
            aggregated.append({
                'Feature': label,
                'Weight': stats['weight'],
                'AbsWeight': abs(stats['weight'])
            })
        
        return sorted(aggregated, key=lambda x: x['AbsWeight'], reverse=False)

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
        l_mat = np.exp(-(y_dist ** 2) / (2 * sigma_y ** 2))
        l_bar = self._center_and_normalize(l_mat)

        features = []
        for k in range(d):
            xk = x_mat[:, k].reshape(-1, 1)
            x_dist = xk - xk.T
            sigma_x = self._median_sigma(np.abs(x_dist))
            k_mat = np.exp(-(x_dist ** 2) / (2 * sigma_x ** 2))
            k_bar = self._center_and_normalize(k_mat)
            features.append(k_bar.reshape(-1))

        x_feat = np.stack(features, axis=1)
        y_target = l_bar.reshape(-1)

        model = Lasso(alpha=self.hsic_lambda, fit_intercept=False, positive=True, max_iter=5000)
        model.fit(x_feat, y_target)
        return model.coef_

    def explain_local(self, graph, task='activity', num_perturbations=200):
        self.model.eval()
        graph = graph.to(self.device)
        
        with torch.no_grad():
            out = self.model(graph)
            if task == 'activity':
                base_pred = out[0].softmax(dim=1).cpu().numpy().flatten()
                target_class = base_pred.argmax()
                base_score = base_pred[target_class]
            elif task == 'event_time':
                base_score = out[1].item()
            elif task == 'remaining_time':
                base_score = out[2].item()
        
        activity_features = graph['activity'].x.shape[1] if 'activity' in graph.x_dict else 0
        resource_features = graph['resource'].x.shape[1] if 'resource' in graph.x_dict else 0
        
        active_activity = torch.where(graph['activity'].x.sum(dim=0) > 0)[0].cpu().numpy() if activity_features > 0 else np.array([])
        active_resource = torch.where(graph['resource'].x.sum(dim=0) > 0)[0].cpu().numpy() if resource_features > 0 else np.array([])
        
        active_features = [("activity", idx) for idx in active_activity] + [
            ("resource", idx) for idx in active_resource
        ]
        if not active_features:
            return [], base_score

        X_perturb = []
        y_perturb = []
        
        print(f"[GraphLIME] Activity features: {activity_features}, Resource features: {resource_features}")
        
        for _ in range(num_perturbations):
            mask_activity = np.ones(activity_features)
            mask_resource = np.ones(resource_features)
            
            if len(active_activity) > 0:
                subset = np.random.choice(active_activity, size=int(max(1, len(active_activity)*0.3)), replace=False)
                mask_activity[subset] = 0
            
            if len(active_resource) > 0:
                subset = np.random.choice(active_resource, size=int(max(1, len(active_resource)*0.3)), replace=False)
                mask_resource[subset] = 0
            
            mask_activity_active = mask_activity[active_activity] if len(active_activity) > 0 else np.array([])
            mask_resource_active = mask_resource[active_resource] if len(active_resource) > 0 else np.array([])
            combined_mask = np.concatenate([mask_activity_active, mask_resource_active])
            X_perturb.append(combined_mask)
            
            masked_graph = graph.clone()
            if activity_features > 0:
                mask_tensor_activity = torch.tensor(mask_activity, device=self.device, dtype=torch.float32)
                masked_graph['activity'].x = masked_graph['activity'].x * mask_tensor_activity
            if resource_features > 0:
                mask_tensor_resource = torch.tensor(mask_resource, device=self.device, dtype=torch.float32)
                masked_graph['resource'].x = masked_graph['resource'].x * mask_tensor_resource
            
            with torch.no_grad():
                out_p = self.model(masked_graph)
                if task == 'activity':
                    score = out_p[0].softmax(dim=1)[0, target_class].item()
                elif task == 'event_time':
                    score = out_p[1].item()
                elif task == 'remaining_time':
                    score = out_p[2].item()
                y_perturb.append(score)
                
        X_perturb = np.array(X_perturb)
        y_perturb = np.array(y_perturb)
        
        coefs = self._hsic_lasso(X_perturb, y_perturb)
        if coefs is None or np.allclose(coefs, 0):
            weights = []
            y_std = np.std(y_perturb)
            for k in range(X_perturb.shape[1]):
                x_std = np.std(X_perturb[:, k])
                if x_std == 0 or y_std == 0:
                    weights.append(0.0)
                else:
                    weights.append(float(np.corrcoef(X_perturb[:, k], y_perturb)[0, 1]))
            coefs = np.array(weights)

        explanation = []
        for coef, (node_type, feat_idx) in zip(coefs, active_features):
            explanation.append({
                'Feature': self._get_feature_name(node_type, feat_idx),
                'Weight': float(coef),
                'AbsWeight': float(abs(coef))
            })

        aggregated_explanation = self._aggregate_features(explanation)
        if self.top_k and len(aggregated_explanation) > self.top_k:
            aggregated_explanation = sorted(
                aggregated_explanation, key=lambda x: x['AbsWeight'], reverse=True
            )[: self.top_k]
            aggregated_explanation = sorted(
                aggregated_explanation, key=lambda x: x['AbsWeight'], reverse=False
            )
                
        return aggregated_explanation, base_score

    def plot_local(self, explanation, base_score, output_dir, task, sample_id):
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(explanation)
        if df.empty: return
        
        df.to_csv(os.path.join(output_dir, f'graphlime_sample_{sample_id}_{task}.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        colors = ['#2ca02c' if w > 0 else '#d62728' for w in df['Weight']]
        
        bars = plt.barh(df['Feature'], df['Weight'], color=colors)
        
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        title = f"Local Explanation (Sample {sample_id})\nPrediction: {base_score:.2f}"
        plt.title(title, fontweight='bold')
        plt.xlabel("Feature Contribution (LIME)")
        
        for rect in bars:
            w = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            padding = 0.001 if w > 0 else -0.001
            ha = 'left' if w > 0 else 'right'
            plt.text(w + padding, y, f'{w:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'graphlime_sample_{sample_id}_{task}.png'), dpi=300)
        plt.close()


def generate_feature_importance_summary(grad_dir, lime_dir, output_dir, task='activity'):
    summary_data = []
    
    grad_file = os.path.join(grad_dir, f'gradient_global_{task}.csv')
    if os.path.exists(grad_file):
        grad_df = pd.read_csv(grad_file)
        for _, row in grad_df.iterrows():
            summary_data.append({
                'Feature': row['Feature'],
                'Method': 'Gradient',
                'Importance': row['Importance'],
                'Type': row.get('Type', 'Unknown')
            })
    
    lime_files = [f for f in os.listdir(lime_dir) if f.startswith('graphlime_sample_') and f.endswith(f'_{task}.csv')]
    if lime_files:
        lime_aggregated = {}
        for lime_file in lime_files:
            lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
            for _, row in lime_df.iterrows():
                feature = row['Feature']
                weight = abs(row['Weight'])
                if feature not in lime_aggregated:
                    lime_aggregated[feature] = []
                lime_aggregated[feature].append(weight)
        
        for feature, weights in lime_aggregated.items():
            summary_data.append({
                'Feature': feature,
                'Method': 'GraphLIME',
                'Importance': np.mean(weights),
                'Type': 'Activity' if 'Activity' in feature else 'Resource'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Importance', ascending=False)
        summary_df.to_csv(os.path.join(output_dir, f'feature_importance_summary_{task}.csv'), index=False)
        print(f"[OK] Feature importance summary saved for {task}")
    
    return summary_df if summary_data else None


def generate_comparison_report(grad_dir, lime_dir, output_dir, task='activity'):
    report_path = os.path.join(output_dir, f'comparison_report_{task}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"GNN EXPLAINABILITY COMPARISON REPORT - {task.upper()}\n")
        f.write("="*70 + "\n\n")
        
        grad_file = os.path.join(grad_dir, f'gradient_global_{task}.csv')
        if os.path.exists(grad_file):
            grad_df = pd.read_csv(grad_file)
            f.write("GRADIENT-BASED SALIENCY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total features analyzed: {len(grad_df)}\n")
            f.write(f"Top feature: {grad_df.iloc[-1]['Feature']} ({grad_df.iloc[-1]['Importance']:.4f})\n")
            
            activity_count = len(grad_df[grad_df['Type'] == 'Activity'])
            resource_count = len(grad_df[grad_df['Type'] == 'Resource'])
            f.write(f"Activity features: {activity_count}\n")
            f.write(f"Resource features: {resource_count}\n\n")
        
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('graphlime_sample_') and f.endswith(f'_{task}.csv')]
        if lime_files:
            f.write("GRAPHLIME LOCAL ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Samples analyzed: {len(lime_files)}\n")
            
            all_features = []
            for lime_file in lime_files:
                lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                all_features.extend(lime_df['Feature'].tolist())
            
            unique_features = len(set(all_features))
            f.write(f"Unique features identified: {unique_features}\n")
            f.write(f"Most common feature: {max(set(all_features), key=all_features.count)}\n\n")
        
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
        
        f.write("="*70 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("-" * 70 + "\n")
        f.write("Use Gradient for: Overall model behavior understanding\n")
        f.write("Use GraphLIME for: Understanding specific predictions\n")
        f.write("Use Temporal Gradient for: Time-series attribution analysis\n")
        f.write("="*70 + "\n")
    
    print(f"[OK] Comparison report saved for {task}")


def run_gnn_explainability(model, data, output_dir, device, vocabularies=None, num_samples=50, methods='all', tasks=None, scaler=None, y_true=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY ANALYSIS (Focused: Next Activity Only)")
    print("="*70)
    
    if 'test_graphs' in data:
        graphs = data['test_graphs']
    elif 'test' in data:
        graphs = data['test']
    else:
        print("Error: Could not find test graphs in data object.")
        return

    if tasks is None:
        tasks = ['activity']
    elif isinstance(tasks, str):
        tasks = [tasks]
    
    print(f"Tasks to explain: {', '.join(tasks)}")
    print(f"Number of samples: {num_samples}")
    
    if methods in ['gradient', 'all']:
        print("\n[Gradient-Based Saliency]")
        g_explainer = GradientExplainer(model, device, vocabularies)
        grad_dir = os.path.join(output_dir, 'gradient')
        
        for task in tasks:
            try:
                global_imp = g_explainer.explain_global(graphs, task, num_samples)
                g_explainer.plot_global_importance(global_imp, grad_dir, task)
                print(f"[OK] {task.capitalize()} Global Importance saved.")
            except Exception as e:
                print(f"[ERROR] Failed {task}: {e}")

    if methods in ['temporal', 'all']:
        print("\n[Temporal Gradient Attribution]")
        temporal_explainer = TemporalGradientExplainer(model, device, vocabularies, scaler)
        temporal_dir = os.path.join(output_dir, 'temporal')
        
        for task in tasks:
            try:
                temporal_explainer.generate_temporal_plots(
                    graphs, temporal_dir, task, 
                    num_samples=min(10, num_samples),
                    y_true=y_true
                )
                print(f"[OK] {task.capitalize()} Temporal Gradient plots saved.")
            except Exception as e:
                print(f"[ERROR] Failed temporal {task}: {e}")
                import traceback
                traceback.print_exc()

    if methods in ['lime', 'all']:
        print("\n[GraphLIME Local Analysis]")
        lime_explainer = GraphLIMEExplainer(model, device, vocabularies)
        lime_dir = os.path.join(output_dir, 'graphlime')
        
        max_lime_samples = min(10, len(graphs))
        step = max(1, len(graphs) // max_lime_samples)
        sample_ids = list(range(0, len(graphs), step))[:max_lime_samples]
        
        print(f"Analyzing {len(sample_ids)} diverse samples: {sample_ids}")
        
        for idx in sample_ids:
            if idx >= len(graphs): break
            graph = graphs[idx]
            print(f"Analyzing Sample {idx}...")
            
            for task in tasks:
                try:
                    exp_list, score = lime_explainer.explain_local(graph, task)
                    lime_explainer.plot_local(exp_list, score, lime_dir, task, idx)
                except Exception as e:
                    print(f"[ERROR] Failed Sample {idx} {task}: {e}")

    if methods == 'all':
        print("\n[Generating Comprehensive Analysis]")
        grad_dir = os.path.join(output_dir, 'gradient')
        lime_dir = os.path.join(output_dir, 'graphlime')
        
        for task in tasks:
            try:
                generate_feature_importance_summary(grad_dir, lime_dir, output_dir, task)
                generate_comparison_report(grad_dir, lime_dir, output_dir, task)
            except Exception as e:
                print(f"[ERROR] Failed to generate summary for {task}: {e}")

    print("\n" + "="*70)
    print(f"GNN EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    print("\nGenerated outputs:")
    print("  [OK] Gradient global importance plots")
    if methods in ['temporal', 'all']:
        print("  [OK] Temporal gradient attribution plots")
    print("  [OK] GraphLIME local explanations (10 diverse samples)")
    print("  [OK] Feature importance summary CSV")
    print("  [OK] Method comparison report")
    print("="*70)
    
    return {}


class GNNExplainerWrapper:
    def __init__(self, model, device, vocabularies=None, scaler=None):
        self.model = model
        self.device = device
        self.vocabularies = vocabularies
        self.scaler = scaler

    def run(self, data, output_dir, num_samples=50, methods='all', tasks=None, y_true=None):
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
            y_true=y_true
        )