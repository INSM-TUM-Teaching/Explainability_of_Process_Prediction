import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm
import pickle
import json
import sys

# For gradient explainer
try:
    from gnns import dataset_builder
    from gnns.model import HeteroGNN
except:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    try:
        from gnns import dataset_builder
        from gnns.model import HeteroGNN
    except:
        pass  # Will be loaded when needed


class GNNExplainerWrapper:
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.explainer = None
        self.explanations = []
        
    def initialize_explainer(self, sample_graph, task='activity'):
        self.task = task
        
        self.explainer = Explainer(
            model=self.model,
            algorithm=PyGGNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification' if task == 'activity' else 'regression',
                task_level='graph',
                return_type='log_prob' if task == 'activity' else 'raw'
            )
        )
        
    def explain_graph(self, graph, target_class=None):
        graph = graph.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            if target_class is None:
                act_logits, _, _ = self.model(graph)
                target_class = act_logits.argmax(dim=1).item()
        
        explanation = self.explainer(
            x=graph.x_dict['activity'],
            edge_index=graph.edge_index_dict,
            target=torch.tensor([target_class], device=self.device)
        )
        
        node_importance = explanation.node_mask.detach().cpu().numpy() if hasattr(explanation, 'node_mask') else None
        edge_importance = explanation.edge_mask.detach().cpu().numpy() if hasattr(explanation, 'edge_mask') else None
        
        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'target_class': target_class,
            'explanation': explanation
        }
    
    def explain_batch(self, graphs, num_samples=10, task='activity'):
        explanations = []
        
        sample_indices = np.random.choice(len(graphs), min(num_samples, len(graphs)), replace=False)
        
        for idx in tqdm(sample_indices, desc="Generating explanations"):
            graph = graphs[idx]
            exp = self.explain_graph(graph)
            explanations.append(exp)
        
        self.explanations = explanations
        return explanations
    
    def aggregate_explanations(self, explanations):
        node_importances = []
        edge_importances = []
        
        for exp in explanations:
            if exp['node_importance'] is not None:
                node_importances.append(exp['node_importance'])
            if exp['edge_importance'] is not None:
                edge_importances.append(exp['edge_importance'])
        
        aggregated = {
            'avg_node_importance': np.mean(node_importances, axis=0) if node_importances else None,
            'std_node_importance': np.std(node_importances, axis=0) if node_importances else None,
            'avg_edge_importance': np.mean(edge_importances, axis=0) if edge_importances else None,
            'std_edge_importance': np.std(edge_importances, axis=0) if edge_importances else None
        }
        
        return aggregated
    
    def save_explanations(self, explanations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        exp_data = []
        for i, exp in enumerate(explanations):
            exp_dict = {
                'sample_id': i,
                'target_class': exp['target_class'],
                'node_importance': exp['node_importance'].tolist() if exp['node_importance'] is not None else None,
                'edge_importance': exp['edge_importance'].tolist() if exp['edge_importance'] is not None else None
            }
            exp_data.append(exp_dict)
        
        with open(os.path.join(output_dir, 'gnnexplainer_explanations.json'), 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        df = pd.DataFrame([{
            'sample_id': e['sample_id'],
            'target_class': e['target_class'],
            'avg_node_importance': np.mean(e['node_importance']) if e['node_importance'] else 0,
            'avg_edge_importance': np.mean(e['edge_importance']) if e['edge_importance'] else 0
        } for e in exp_data])
        
        df.to_csv(os.path.join(output_dir, 'gnnexplainer_summary.csv'), index=False)
    
    def plot_feature_importance(self, aggregated, output_dir, top_k=15):
        os.makedirs(output_dir, exist_ok=True)
        
        if aggregated['avg_node_importance'] is not None:
            plt.figure(figsize=(12, 6))
            
            importances = aggregated['avg_node_importance']
            top_indices = np.argsort(importances)[-top_k:][::-1]
            
            plt.barh(range(len(top_indices)), importances[top_indices], color='steelblue')
            plt.yticks(range(len(top_indices)), [f'Node {i}' for i in top_indices])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {top_k} Node Feature Importance (GNNExplainer)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gnnexplainer_node_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_edge_importance(self, aggregated, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        if aggregated['avg_edge_importance'] is not None:
            plt.figure(figsize=(10, 6))
            
            importances = aggregated['avg_edge_importance']
            plt.hist(importances, bins=30, color='coral', edgecolor='black', alpha=0.7)
            plt.xlabel('Edge Importance Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Edge Importance Distribution (GNNExplainer)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gnnexplainer_edge_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_explanation(self, explanation, output_dir, sample_id=0):
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        if explanation['node_importance'] is not None:
            node_imp = explanation['node_importance']
            axes[0].bar(range(len(node_imp)), node_imp, color='steelblue', alpha=0.7)
            axes[0].set_xlabel('Node Index', fontsize=11)
            axes[0].set_ylabel('Importance Score', fontsize=11)
            axes[0].set_title(f'Node Importance (Sample {sample_id})', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        if explanation['edge_importance'] is not None:
            edge_imp = explanation['edge_importance']
            axes[1].bar(range(len(edge_imp)), edge_imp, color='coral', alpha=0.7)
            axes[1].set_xlabel('Edge Index', fontsize=11)
            axes[1].set_ylabel('Importance Score', fontsize=11)
            axes[1].set_title(f'Edge Importance (Sample {sample_id})', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gnnexplainer_sample_{sample_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()


class GraphLIMEExplainer:
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.explanations = []
        
    def explain_graph(self, graph, num_samples=100):
        graph = graph.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            act_logits, _, _ = self.model(graph)
            original_pred = act_logits.argmax(dim=1).item()
            original_probs = F.softmax(act_logits, dim=1).cpu().numpy()[0]
        
        num_nodes = graph['activity'].x.size(0)
        node_importance = np.zeros(num_nodes)
        
        for node_idx in range(num_nodes):
            perturbed_preds = []
            
            for _ in range(num_samples):
                perturbed_graph = graph.clone()
                
                mask = torch.rand(num_nodes) > 0.5
                mask[node_idx] = False
                
                for node_type in perturbed_graph.x_dict.keys():
                    if node_type == 'activity':
                        perturbed_graph[node_type].x = perturbed_graph[node_type].x.clone()
                        perturbed_graph[node_type].x[mask] = 0
                
                with torch.no_grad():
                    act_logits_pert, _, _ = self.model(perturbed_graph)
                    pred_pert = act_logits_pert.argmax(dim=1).item()
                    perturbed_preds.append(pred_pert)
            
            importance = np.mean([1 if p == original_pred else 0 for p in perturbed_preds])
            node_importance[node_idx] = importance
        
        return {
            'node_importance': node_importance,
            'original_prediction': original_pred,
            'original_probabilities': original_probs
        }
    
    def explain_batch(self, graphs, num_samples_per_graph=100, num_graphs=10):
        explanations = []
        
        sample_indices = np.random.choice(len(graphs), min(num_graphs, len(graphs)), replace=False)
        
        for idx in tqdm(sample_indices, desc="GraphLIME explanations"):
            graph = graphs[idx]
            exp = self.explain_graph(graph, num_samples=num_samples_per_graph)
            exp['graph_id'] = int(idx)
            explanations.append(exp)
        
        self.explanations = explanations
        return explanations
    
    def aggregate_explanations(self, explanations):
        node_importances = [exp['node_importance'] for exp in explanations]
        
        max_len = max(len(imp) for imp in node_importances)
        padded = np.zeros((len(node_importances), max_len))
        
        for i, imp in enumerate(node_importances):
            padded[i, :len(imp)] = imp
        
        return {
            'avg_node_importance': np.mean(padded, axis=0),
            'std_node_importance': np.std(padded, axis=0)
        }
    
    def save_explanations(self, explanations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        exp_data = []
        for exp in explanations:
            exp_dict = {
                'graph_id': exp['graph_id'],
                'original_prediction': int(exp['original_prediction']),
                'node_importance': exp['node_importance'].tolist(),
                'original_probabilities': exp['original_probabilities'].tolist()
            }
            exp_data.append(exp_dict)
        
        with open(os.path.join(output_dir, 'graphlime_explanations.json'), 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        df = pd.DataFrame([{
            'graph_id': e['graph_id'],
            'original_prediction': e['original_prediction'],
            'avg_node_importance': np.mean(e['node_importance']),
            'max_node_importance': np.max(e['node_importance']),
            'num_nodes': len(e['node_importance'])
        } for e in exp_data])
        
        df.to_csv(os.path.join(output_dir, 'graphlime_summary.csv'), index=False)
    
    def plot_importance(self, aggregated, output_dir, top_k=15):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        importances = aggregated['avg_node_importance']
        non_zero = importances[importances > 0]
        
        if len(non_zero) > 0:
            top_indices = np.argsort(importances)[-min(top_k, len(non_zero)):][::-1]
            
            plt.barh(range(len(top_indices)), importances[top_indices], color='forestgreen')
            plt.yticks(range(len(top_indices)), [f'Node {i}' for i in top_indices])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {len(top_indices)} Node Importance (GraphLIME)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'graphlime_node_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_explanation(self, explanation, output_dir, sample_id=0):
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        node_imp = explanation['node_importance']
        axes[0].bar(range(len(node_imp)), node_imp, color='forestgreen', alpha=0.7)
        axes[0].set_xlabel('Node Index', fontsize=11)
        axes[0].set_ylabel('Importance Score', fontsize=11)
        axes[0].set_title(f'Node Importance (GraphLIME, Sample {sample_id})', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        probs = explanation['original_probabilities']
        top_5 = np.argsort(probs)[-5:][::-1]
        axes[1].barh(range(len(top_5)), probs[top_5], color='coral', alpha=0.7)
        axes[1].set_yticks(range(len(top_5)))
        axes[1].set_yticklabels([f'Class {i}' for i in top_5])
        axes[1].set_xlabel('Probability', fontsize=11)
        axes[1].set_title(f'Top 5 Class Probabilities', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'graphlime_sample_{sample_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()


class GradientExplainer:
    """
    Gradient-based feature attribution for heterogeneous GNN models.
    Analyzes importance across all node types and all prediction tasks.
    """
    
    def __init__(self, model, device, model_dir, dataset_path):
        """
        Args:
            model: Trained HeteroGNN model
            device: torch device (cuda/cpu)
            model_dir: Directory containing model files
            dataset_path: Path to dataset CSV
        """
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.vocabs = None
        self.trace_attributes = None
        self.sample_graphs = []
        
        print(f"\n[GradientExplainer Initialized]")
        print(f"Device: {self.device}")
        print(f"Model dir: {self.model_dir}")
        print(f"Dataset: {os.path.basename(self.dataset_path)}")
    
    def prepare_graphs(self, num_graphs=50):
        """Prepare prefix graphs from dataset."""
        print(f"\nPreparing {num_graphs} graphs for explanation...")
        
        df = pd.read_csv(self.dataset_path)
        
        # Column mapping
        mapping = {
            'time:timestamp': 'Timestamp',
            'case:id': 'CaseID',
            'case:concept:name': 'CaseID',
            'concept:name': 'Activity',
            'org:resource': 'Resource'
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)
        
        # Detect trace attributes
        trace_cols = []
        grouped = df.groupby("CaseID")
        for col in df.columns:
            if col in ["CaseID", "Activity", "Resource", "Timestamp"]:
                continue
            if grouped[col].nunique().max() == 1:
                trace_cols.append(col)
        
        self.trace_attributes = trace_cols
        
        # Generate prefixes
        rows = []
        for case_id, group in df.groupby("CaseID"):
            group = group.sort_values("Timestamp").reset_index(drop=True)
            trace_len = len(group)
            trace_attrs = {c: group[c].iloc[0] for c in trace_cols}
            
            for k in range(1, trace_len):
                label_next = group.iloc[k]["Activity"]
                prefix = group.iloc[:k]
                
                for pos, (_, ev) in enumerate(prefix.iterrows(), start=1):
                    row = {
                        "CaseID": case_id,
                        "prefix_id": k,
                        "prefix_pos": pos,
                        "prefix_length": k,
                        "Activity": ev["Activity"],
                        "Resource": ev["Resource"],
                        "Timestamp": ev["Timestamp"],
                        "next_activity": label_next,
                    }
                    row.update(trace_attrs)
                    rows.append(row)
        
        prefix_df = pd.DataFrame(rows)
        prefix_df["__ts_log"] = np.log1p(
            prefix_df["Timestamp"].astype("int64") // 1_000_000_000
        ).astype("float32")
        
        # Build vocabularies
        self.vocabs = {}
        all_acts = set(prefix_df["Activity"].unique()) | set(prefix_df["next_activity"].unique())
        self.vocabs["Activity"] = {v: i for i, v in enumerate(sorted(all_acts))}
        self.vocabs["Resource"] = {v: i for i, v in enumerate(sorted(prefix_df["Resource"].unique()))}
        
        IGNORE = {"CaseID", "prefix_id", "prefix_pos", "prefix_length", 
                  "Activity", "Resource", "Timestamp", "next_activity", "__ts_log"}
        
        for col in self.trace_attributes:
            if not pd.api.types.is_numeric_dtype(prefix_df[col]):
                vals = sorted(prefix_df[col].fillna("NaN").unique())
                self.vocabs[col] = {v: i for i, v in enumerate(vals)}
        
        # Build graphs
        groups = list(prefix_df.groupby(["CaseID", "prefix_id"]))
        sample_indices = np.random.choice(len(groups), min(num_graphs, len(groups)), replace=False)
        
        print(f"Building {len(sample_indices)} graphs...")
        for idx in tqdm(sample_indices, desc="Building graphs"):
            (_, _), p = groups[idx]
            p = p.sort_values("prefix_pos")
            graph = self._build_graph(p)
            if graph is not None:
                self.sample_graphs.append(graph)
        
        print(f"✓ Prepared {len(self.sample_graphs)} graphs")
    
    def _build_graph(self, prefix):
        """Build a single heterogeneous graph."""
        data = HeteroData()
        k = len(prefix)
        
        act_map = self.vocabs["Activity"]
        res_map = self.vocabs["Resource"]
        
        # Node features
        act_ids = np.array([act_map[a] for a in prefix["Activity"]])
        data["activity"].x = F.one_hot(
            torch.tensor(act_ids, dtype=torch.long), 
            num_classes=len(act_map)
        ).float()
        
        res_ids = np.array([res_map[r] for r in prefix["Resource"]])
        data["resource"].x = F.one_hot(
            torch.tensor(res_ids, dtype=torch.long),
            num_classes=len(res_map)
        ).float()
        
        data["time"].x = torch.tensor(
            prefix["__ts_log"].to_numpy(), 
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Trace features
        trace_features = []
        first = prefix.iloc[0]
        for col in self.trace_attributes:
            if col not in prefix.columns:
                continue
            val = first[col]
            if col in self.vocabs:
                idx = self.vocabs[col].get(val, 0)
                trace_features.append(
                    F.one_hot(torch.tensor(idx), num_classes=len(self.vocabs[col])).float()
                )
            else:
                try:
                    trace_features.append(torch.tensor([np.log1p(float(val))], dtype=torch.float32))
                except:
                    trace_features.append(torch.zeros(1))
        
        if not trace_features:
            trace_features = [torch.zeros(1)]
        data["trace"].x = torch.cat(trace_features).unsqueeze(0)
        
        # Edges
        idx = torch.arange(k)
        if k > 1:
            dfr = torch.stack([idx[:-1], idx[1:]])
        else:
            dfr = torch.empty((2, 0), dtype=torch.long)
        
        data["activity", "next", "activity"].edge_index = dfr
        data["resource", "next", "resource"].edge_index = dfr.clone()
        data["time", "next", "time"].edge_index = dfr.clone()
        
        same_ev = torch.stack([idx, idx])
        data["activity", "same_event", "resource"].edge_index = same_ev
        data["resource", "same_event", "activity"].edge_index = same_ev.clone()
        data["activity", "same_time", "time"].edge_index = same_ev.clone()
        data["time", "same_time", "activity"].edge_index = same_ev.clone()
        
        trace_src = torch.zeros(k, dtype=torch.long)
        data["activity", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
        data["resource", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
        data["time", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
        
        # Labels
        next_act = act_map[first["next_activity"]]
        data.y_activity = torch.tensor([next_act], dtype=torch.long)
        
        if k > 1:
            t_next = prefix.iloc[1]["Timestamp"].timestamp()
        else:
            t_next = first["Timestamp"].timestamp()
        data.y_timestamp = torch.tensor([np.log1p(t_next)], dtype=torch.float32)
        
        t_end = prefix.iloc[-1]["Timestamp"].timestamp()
        t_now = first["Timestamp"].timestamp()
        remaining = max(0, t_end - t_now)
        data.y_remaining_time = torch.tensor([np.log1p(remaining)], dtype=torch.float32)
        
        return data
    
    def compute_gradient_importance(self, graph, task='activity'):
        """Compute gradient-based feature importance for a single graph."""
        graph = graph.to(self.device)
        self.model.eval()
        
        # Enable gradients for input features
        for node_type in graph.x_dict.keys():
            graph[node_type].x.requires_grad = True
        
        # Forward pass
        act_logits, time_pred, rem_pred = self.model(graph)
        
        # Compute loss based on task
        if task == 'activity':
            loss = F.cross_entropy(act_logits, graph.y_activity.view(-1))
        elif task == 'event_time':
            loss = F.l1_loss(time_pred, graph.y_timestamp.view(-1))
        elif task == 'remaining_time':
            loss = F.l1_loss(rem_pred, graph.y_remaining_time.view(-1))
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        importance_dict = {}
        for node_type in graph.x_dict.keys():
            if graph[node_type].x.grad is not None:
                # Gradient * Input (integrated gradients approximation)
                grad = graph[node_type].x.grad.detach().cpu().numpy()
                features = graph[node_type].x.detach().cpu().numpy()
                importance = np.abs(grad * features)
                
                # Average across nodes
                importance_dict[node_type] = np.mean(importance, axis=0)
        
        return importance_dict
    
    def explain_batch(self, num_samples=50):
        """Generate explanations for multiple graphs across all tasks."""
        if not self.sample_graphs:
            self.prepare_graphs(num_samples)
        
        graphs_to_explain = self.sample_graphs[:num_samples]
        
        print(f"\n[Computing Gradient Importance]")
        print(f"Analyzing {len(graphs_to_explain)} graphs across 3 tasks...")
        
        tasks = ['activity', 'event_time', 'remaining_time']
        all_importances = {task: [] for task in tasks}
        
        for graph in tqdm(graphs_to_explain, desc="Processing graphs"):
            for task in tasks:
                importance = self.compute_gradient_importance(graph, task=task)
                all_importances[task].append(importance)
        
        # Aggregate across samples
        aggregated = {}
        for task in tasks:
            task_agg = {}
            # Get all node types
            node_types = set()
            for imp_dict in all_importances[task]:
                node_types.update(imp_dict.keys())
            
            for node_type in node_types:
                importances = [imp[node_type] for imp in all_importances[task] if node_type in imp]
                if importances:
                    task_agg[node_type] = np.mean(importances, axis=0)
            
            aggregated[task] = task_agg
        
        print(f"✓ Gradient importance computed for all tasks")
        
        # Save and visualize
        output_dir = os.path.join(self.model_dir, 'explanations')
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_explanations(aggregated, output_dir, num_samples)
        self.plot_importance(aggregated, output_dir)
        self.create_report(aggregated, output_dir, num_samples)
        
        return aggregated
    
    def save_explanations(self, aggregated, output_dir, num_samples):
        """Save explanations to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for task, importance_dict in aggregated.items():
            export_data[task] = {}
            for node_type, importance in importance_dict.items():
                export_data[task][node_type] = importance.tolist()
        
        export_data['metadata'] = {
            'num_samples': num_samples,
            'tasks': list(aggregated.keys()),
            'node_types': list(aggregated['activity'].keys()) if 'activity' in aggregated else []
        }
        
        output_path = os.path.join(output_dir, 'gradient_explanations.json')
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Saved explanations: gradient_explanations.json")
    
    def plot_importance(self, aggregated, output_dir):
        """Generate visualization plots for each task."""
        print("\nGenerating visualizations...")
        
        for task, importance_dict in aggregated.items():
            if not importance_dict:
                continue
            
            # Filter out trace node type
            filtered_dict = {k: v for k, v in importance_dict.items() if k != 'trace'}
            
            if not filtered_dict:
                continue
            
            # Separate time from others
            time_importance = filtered_dict.pop('time', None)
            
            n_plots = len(filtered_dict) + (1 if time_importance is not None else 0)
            if n_plots == 0:
                continue
            
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # Plot activity and resource
            for node_type, imp in filtered_dict.items():
                ax = axes[plot_idx]
                
                if len(imp.shape) > 1:
                    imp = imp.flatten()
                
                top_k = min(15, len(imp))
                top_idx = np.argsort(imp)[-top_k:]
                top_val = imp[top_idx]
                
                feature_names = self._get_feature_names(node_type, top_idx)
                
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_k))
                ax.barh(range(top_k), top_val, color=colors)
                ax.set_yticks(range(top_k))
                ax.set_yticklabels(feature_names, fontsize=9)
                ax.set_xlabel('Average Importance', fontsize=10)
                ax.set_title(f'{node_type.capitalize()}\n({task.replace("_", " ").title()})', 
                           fontsize=11, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plot_idx += 1
            
            # Plot time
            if time_importance is not None:
                ax = axes[plot_idx]
                
                if len(time_importance.shape) > 1:
                    time_importance = time_importance.flatten()
                
                time_value = time_importance[0] if len(time_importance) > 0 else 0
                
                ax.barh([0], [time_value], color='#FF9500', height=0.5)
                ax.set_yticks([0])
                ax.set_yticklabels(['Timestamp'], fontsize=10)
                ax.set_xlabel('Average Importance', fontsize=10)
                ax.set_title(f'Time\n({task.replace("_", " ").title()})', 
                           fontsize=11, fontweight='bold')
                ax.set_ylim(-0.5, 0.5)
                ax.grid(axis='x', alpha=0.3)
                ax.text(time_value, 0, f' {time_value:.4f}', 
                       va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            filename = f'global_importance_{task}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {filename}")
    
    def _get_feature_names(self, node_type, indices):
        """Get human-readable feature names."""
        if node_type == 'activity':
            vocab = self.vocabs['Activity']
            inv_vocab = {v: k for k, v in vocab.items()}
            return [inv_vocab.get(i, f'Feature_{i}') for i in indices]
        elif node_type == 'resource':
            vocab = self.vocabs['Resource']
            inv_vocab = {v: k for k, v in vocab.items()}
            return [inv_vocab.get(i, f'Feature_{i}') for i in indices]
        else:
            return [f'Feature_{i}' for i in indices]
    
    def create_report(self, aggregated, output_dir, num_samples):
        """Create text report of explanations."""
        report_path = os.path.join(output_dir, 'gradient_explanation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GRADIENT-BASED FEATURE ATTRIBUTION REPORT\n")
            f.write(f"Analyzed {num_samples} samples\n")
            f.write("="*70 + "\n\n")
            
            for task, importance_dict in aggregated.items():
                if not importance_dict:
                    continue
                
                f.write("="*70 + "\n")
                f.write(f"{task.replace('_', ' ').upper()}\n")
                f.write("="*70 + "\n\n")
                
                for node_type, imp in importance_dict.items():
                    if node_type == 'trace':
                        continue
                    
                    if len(imp.shape) > 1:
                        imp = imp.flatten()
                    
                    f.write(f"{node_type.capitalize()} Features:\n")
                    f.write("-"*70 + "\n")
                    
                    if node_type == 'time':
                        f.write(f"  Timestamp importance: {imp[0]:.6f}\n\n")
                    else:
                        top_k = min(10, len(imp))
                        top_idx = np.argsort(imp)[-top_k:][::-1]
                        feature_names = self._get_feature_names(node_type, top_idx)
                        
                        for rank, (idx, name) in enumerate(zip(top_idx, feature_names), 1):
                            f.write(f"  {rank}. {name}: {imp[idx]:.6f}\n")
                        f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*70 + "\n\n")
            f.write("Higher importance values indicate features with stronger influence\n")
            f.write("on model predictions. These are computed using gradient-based\n")
            f.write("attribution (gradient × input approximation).\n")
        
        print(f"✓ Saved: gradient_explanation_report.txt")


def run_gnn_explainability(model, data, output_dir, device, model_dir, dataset_path, num_samples=10, methods='all'):
    os.makedirs(output_dir, exist_ok=True)
    
    test_graphs = data['test']
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    run_gradient = methods in ['gradient', 'all']
    run_graphlime = methods in ['lime', 'all']
    
    results = {}
    
    if run_gradient:
        print("\n[Gradient-Based Attribution]")
        print("-"*70)
        try:
            gradient_explainer = GradientExplainer(
                model=model,
                device=device,
                model_dir=model_dir,
                dataset_path=dataset_path
            )
            
            # Run batch explanation
            gradient_results = gradient_explainer.explain_batch(num_samples=num_samples)
            
            print(f"✓ Gradient-based results saved to: {model_dir}/explanations/")
            results['gradient'] = gradient_results
            
        except Exception as e:
            print(f"✗ Gradient-based explainability failed: {e}")
            import traceback
            traceback.print_exc()
            results['gradient'] = None
    
    if run_graphlime:
        print("\n[GraphLIME Method]")
        print("-"*70)
        graphlime = GraphLIMEExplainer(model, device)
        
        lime_explanations = graphlime.explain_batch(test_graphs, num_samples_per_graph=50, num_graphs=num_samples)
        lime_aggregated = graphlime.aggregate_explanations(lime_explanations)
        
        graphlime_dir = os.path.join(output_dir, 'graphlime')
        graphlime.save_explanations(lime_explanations, graphlime_dir)
        graphlime.plot_importance(lime_aggregated, graphlime_dir)
        
        for i in range(min(3, len(lime_explanations))):
            graphlime.visualize_explanation(lime_explanations[i], graphlime_dir, sample_id=i)
        
        print(f"✓ GraphLIME results saved to: {graphlime_dir}")
        results['graphlime'] = lime_explanations
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY COMPLETE")
    print("="*70)
    
    return results