import os
import json
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
        elif node_type == 'time':
            return "time"
        elif node_type == 'trace':
            return f"trace_feat_{feature_idx}"
        return f"{node_type}_{feature_idx}"

    def explain_global(self, graphs, task='activity', num_samples=50):
        self.model.eval()
        importances: dict[str, list[np.ndarray]] = {}
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        selected_graphs = [graphs[i] for i in sample_indices]
        
        print(f"Computing gradients for {len(selected_graphs)} graphs ({task})...")
        
        for graph in tqdm(selected_graphs):
            graph = graph.to(self.device)
            for key, x in graph.x_dict.items():
                if torch.is_tensor(x) and x.dtype.is_floating_point:
                    graph.x_dict[key].requires_grad = True
                
            out = self.model(graph)
            
            if task == 'activity':
                logits = out[0]
                pred_idx = logits[0].argmax()
                score = logits[0, pred_idx]
            elif task == 'event_time':
                score = out[1].view(-1)[0]
            elif task == 'remaining_time':
                score = out[2].view(-1)[0]
                
            self.model.zero_grad()
            score.backward()
            
            with torch.no_grad():
                for node_type, x in graph.x_dict.items():
                    grad = getattr(x, "grad", None)
                    if grad is None:
                        continue

                    imp = (grad * x).abs().sum(dim=0).cpu().numpy()
                    if node_type not in importances:
                        importances[node_type] = []
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
            top_k = 10 if scores.shape[0] >= 10 else scores.shape[0]
            top_k_idx = np.argsort(scores)[-top_k:]
            for idx in top_k_idx:
                if scores[idx] > 0:
                    name = self._get_feature_name(n_type, idx)
                    data.append({
                        'Feature': name, 
                        'Importance': scores[idx], 
                        'Type': n_type.capitalize()
                    })
                    
        df = pd.DataFrame(data).sort_values('Importance', ascending=True)
        df.to_csv(os.path.join(output_dir, 'gradient_global.csv'), index=False)
        
        if df.empty: return

        plt.figure(figsize=(10, 6))
        colors = {
            'Activity': '#1f77b4',
            'Resource': '#ff7f0e',
            'Time': '#2ca02c',
            'Trace': '#9467bd',
        }
        
        plt.barh(df['Feature'], df['Importance'], 
                 color=[colors.get(t, 'grey') for t in df['Type']])
        
        plt.title(f"Global Feature Importance ({task.replace('_', ' ').title()})", fontweight='bold')
        plt.xlabel("Mean Gradient Magnitude (Impact)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_global.png'), dpi=300)
        plt.close()


class GraphLIMEExplainer:
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
        
        num_features = graph['activity'].x.shape[1]
        active_features = torch.where(graph['activity'].x.sum(dim=0) > 0)[0].cpu().numpy()
        
        X_perturb = []
        y_perturb = []
        
        for _ in range(num_perturbations):
            mask = np.ones(num_features)
            if len(active_features) > 0:
                subset = np.random.choice(active_features, size=int(max(1, len(active_features)*0.3)), replace=False)
                mask[subset] = 0
            X_perturb.append(mask)
            
            masked_graph = graph.clone()
            mask_tensor = torch.tensor(mask, device=self.device, dtype=torch.float32)
            masked_graph['activity'].x = masked_graph['activity'].x * mask_tensor
            
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
        
        if len(X_perturb) > 1:
            distances = pairwise_distances(X_perturb, X_perturb[0].reshape(1, -1), metric='cosine').ravel()
            weights = np.sqrt(np.exp(-(distances**2) / 0.25))
        else:
            weights = np.ones(len(X_perturb))
        
        simpler_model = Ridge(alpha=1.0)
        simpler_model.fit(X_perturb, y_perturb, sample_weight=weights)
        
        coefs = simpler_model.coef_
        explanation = []
        for idx in active_features:
            if abs(coefs[idx]) > 0.0001:
                explanation.append({
                    'Feature': self._get_feature_name('activity', idx),
                    'Weight': coefs[idx],
                    'AbsWeight': abs(coefs[idx])
                })
                
        return sorted(explanation, key=lambda x: x['AbsWeight'], reverse=False), base_score

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


def run_gnn_explainability(model, data, output_dir, device, vocabularies=None, num_samples=10, methods='all', tasks=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY ANALYSIS")
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
    else:
        tasks = list(tasks)
    
    # Normalize common aliases from UI/other callers
    if isinstance(methods, str):
        m = methods.strip().lower()
        if m == "gradient_based":
            methods = "gradient"
        elif m == "graphlime":
            methods = "lime"

    if methods in ['gradient', 'all']:
        print("\n[Gradient-Based Saliency]")
        g_explainer = GradientExplainer(model, device, vocabularies)
        grad_root = os.path.join(output_dir, 'gradient')
        
        for task_name in tasks:
            try:
                task_dir = os.path.join(grad_root, task_name)
                global_imp = g_explainer.explain_global(graphs, task_name, num_samples)
                g_explainer.plot_global_importance(global_imp, task_dir, task_name)
                print(f"✓ {task_name.capitalize()} Global Importance saved.")
            except Exception as e:
                print(f"✗ Failed {task_name}: {e}")

    if methods in ['lime', 'all']:
        print("\n[GraphLIME Local Analysis]")
        lime_explainer = GraphLIMEExplainer(model, device, vocabularies)
        lime_root = os.path.join(output_dir, 'graphlime')
        
        sample_ids = [0]
        for idx in sample_ids:
            if idx >= len(graphs): break
            graph = graphs[idx]
            print(f"Analyzing Sample {idx}...")
            
            for task_name in tasks:
                try:
                    task_dir = os.path.join(lime_root, task_name)
                    exp_list, score = lime_explainer.explain_local(graph, task_name)
                    lime_explainer.plot_local(exp_list, score, task_dir, task_name, idx)
                except Exception as e:
                    print(f"✗ Failed Sample {idx} {task_name}: {e}")

    print("\n" + "="*70)
    print(f"DONE. Results saved to: {output_dir}")
    print("="*70)
    
    return {}

class GNNExplainerWrapper:
    def __init__(self, model, device, vocabularies=None):
        self.model = model
        self.device = device
        self.vocabularies = vocabularies

    def run(self, data, output_dir, num_samples=10, methods='all'):
        return run_gnn_explainability(
            model=self.model,
            data=data,
            output_dir=output_dir,
            device=self.device,
            vocabularies=self.vocabularies,
            num_samples=num_samples,
            methods=methods
        )
