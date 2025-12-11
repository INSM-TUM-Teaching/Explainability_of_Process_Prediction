"""
GNN Model Explainability using Gradient-based Feature Attribution

This script explains GNN predictions for:
- Next Activity Prediction
- Event Time Prediction  
- Remaining Time Prediction

Save this file as: gnns/gnn_explainability.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings('ignore')

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import local modules
import dataset_builder
from model import HeteroGNN


class GraphExplainer:
    """
    Gradient-based explainer for heterogeneous GNN predictions.
    """
    
    def __init__(self, model_dir, dataset_path):
        """
        Args:
            model_dir: Directory containing gnn_model.pt
            dataset_path: Path to dataset CSV
        """
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print("GNN MODEL EXPLAINABILITY")
        print(f"{'='*70}\n")
        print(f"Model dir: {model_dir}")
        print(f"Dataset: {dataset_path}")
        print(f"Device: {self.device}\n")
        
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """Load and prepare dataset."""
        print("Loading dataset...")
        
        df = pd.read_csv(self.dataset_path)
        
        # Column mapping
        mapping = {
            'time:timestamp': 'Timestamp',
            'case:id': 'CaseID',
            'concept:name': 'Activity',
            'org:resource': 'Resource'
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)
        
        self.df = df
        print(f"✓ Loaded {len(df):,} events\n")
        
        # Prepare graphs
        self.prepare_graphs()
        
    def prepare_graphs(self):
        """Prepare prefix graphs from dataset."""
        print("Preparing graphs...")
        
        # Detect trace attributes
        trace_cols = []
        grouped = self.df.groupby("CaseID")
        for col in self.df.columns:
            if col in ["CaseID", "Activity", "Resource", "Timestamp"]:
                continue
            if grouped[col].nunique().max() == 1:
                trace_cols.append(col)
        
        print(f"Trace attributes: {trace_cols}")
        
        # Generate prefixes
        rows = []
        for case_id, group in self.df.groupby("CaseID"):
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
        trace_attributes = [c for c in prefix_df.columns if c not in IGNORE]
        
        for col in trace_attributes:
            if not pd.api.types.is_numeric_dtype(prefix_df[col]):
                vals = sorted(prefix_df[col].fillna("NaN").unique())
                self.vocabs[col] = {v: i for i, v in enumerate(vals)}
        
        self.trace_attributes = trace_attributes
        
        # Build graphs (use test set only for speed)
        groups = list(prefix_df.groupby(["CaseID", "prefix_id"]))
        n_total = len(groups)
        n_test = max(10, int(n_total * 0.1))  # Use 10% or minimum 10
        
        print(f"Building {n_test} test graphs...")
        self.sample_graphs = []
        
        for (_, _), p in groups[-n_test:]:
            p = p.sort_values("prefix_pos")
            graph = self._build_graph(p)
            if graph is not None:
                self.sample_graphs.append(graph)
        
        print(f"✓ Built {len(self.sample_graphs)} graphs\n")
        
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
        dfr = torch.stack([idx[:-1], idx[1:]]) if k > 1 else torch.empty((2, 0), dtype=torch.long)
        
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
        next_act_name = first["next_activity"]
        if next_act_name not in act_map:
            return None
        
        data.y_activity = torch.tensor([act_map[next_act_name]], dtype=torch.long)
        
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
        
    def load_model(self):
        """Load trained model."""
        print("Loading model...")
        
        model_path = os.path.join(self.model_dir, 'gnn_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        sample = self.sample_graphs[0]
        metadata = sample.metadata()
        proj_dims = {k: v.size(-1) for k, v in sample.x_dict.items()}
        num_classes = len(self.vocabs['Activity'])
        
        self.model = HeteroGNN(
            metadata=metadata,
            hidden_channels=64,
            proj_dims=proj_dims,
            num_activity_classes=num_classes,
            dropout=0.1,
            loss_weights=(1.0, 0.1, 0.1),
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ Model loaded ({num_classes} activities)\n")
        
    def explain_graph(self, graph_idx, task='activity'):
        """
        Explain prediction for a specific graph and task.
        
        Args:
            graph_idx: Index of graph to explain
            task: 'activity', 'time', or 'remaining'
        """
        print(f"{'='*70}")
        print(f"EXPLAINING: {task.upper()} - Graph {graph_idx}")
        print(f"{'='*70}\n")
        
        graph = self.sample_graphs[graph_idx].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            act_logits, time_pred, rem_pred = self.model(graph)
        
        # Display prediction
        if task == 'activity':
            pred_idx = torch.argmax(act_logits, dim=1).item()
            true_idx = graph.y_activity.item()
            conf = F.softmax(act_logits, dim=1)[0, pred_idx].item()
            
            pred_name = self._get_activity_name(pred_idx)
            true_name = self._get_activity_name(true_idx)
            
            print(f"Predicted: {pred_name} (confidence: {conf*100:.2f}%)")
            print(f"True: {true_name}")
            print(f"Correct: {'✓' if pred_idx == true_idx else '✗'}\n")
            
        elif task == 'time':
            pred_val = np.expm1(time_pred.item())
            true_val = np.expm1(graph.y_timestamp.item())
            error = abs(pred_val - true_val)
            
            print(f"Predicted: {pred_val:.2f} seconds")
            print(f"True: {true_val:.2f} seconds")
            print(f"Error: {error:.2f} seconds\n")
            
        else:  # remaining
            pred_val = np.expm1(rem_pred.item())
            true_val = np.expm1(graph.y_remaining_time.item())
            error = abs(pred_val - true_val)
            
            print(f"Predicted: {pred_val:.2f} seconds")
            print(f"True: {true_val:.2f} seconds")
            print(f"Error: {error:.2f} seconds\n")
        
        # Compute feature importance
        print("Computing feature importance...")
        importance = self._compute_importance(graph, task)
        
        # Visualize
        self._visualize_importance(importance, task, graph_idx)
        
        return importance
        
    def _compute_importance(self, graph, task):
        """Compute feature importance using gradients."""
        self.model.train()
        
        # Enable gradients
        x_dict = {k: v.clone().requires_grad_(True) for k, v in graph.x_dict.items()}
        
        temp = HeteroData()
        temp.x_dict = x_dict
        temp.edge_index_dict = graph.edge_index_dict
        
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'batch'):
                temp[node_type].batch = graph[node_type].batch
        
        # Forward
        act_logits, time_pred, rem_pred = self.model(temp)
        
        # Select output
        if task == 'activity':
            output = act_logits[0, torch.argmax(act_logits, dim=1)]
        elif task == 'time':
            output = time_pred[0]
        else:
            output = rem_pred[0]
        
        # Backward
        output.backward()
        
        # Extract importance
        importance = {}
        for node_type, x in x_dict.items():
            if x.grad is not None:
                grad = torch.abs(x.grad).cpu().numpy()
                if len(grad.shape) == 2:
                    grad = grad.mean(axis=0)
                importance[node_type] = grad
        
        self.model.eval()
        return importance
        
    def _visualize_importance(self, importance, task, graph_idx):
        """Create visualization of feature importance."""
        output_dir = os.path.join(self.model_dir, 'explanations')
        os.makedirs(output_dir, exist_ok=True)
        
        n = len(importance)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1:
            axes = [axes]
        
        for idx, (node_type, imp) in enumerate(importance.items()):
            ax = axes[idx]
            
            if len(imp.shape) > 1:
                imp = imp.flatten()
            
            top_k = min(15, len(imp))
            top_idx = np.argsort(imp)[-top_k:]
            top_val = imp[top_idx]
            
            # Get feature names instead of dimensions
            feature_names = self._get_feature_names(node_type, top_idx)
            
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_k))
            ax.barh(range(top_k), top_val, color=colors)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(feature_names, fontsize=8)
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(f'{node_type.capitalize()}\n({task.capitalize()})', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = f'importance_{task}_graph{graph_idx}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}\n")
        
    def _get_activity_name(self, idx):
        """Get activity name from index."""
        inv = {v: k for k, v in self.vocabs['Activity'].items()}
        return inv.get(idx, f"Activity_{idx}")
    
    def _get_feature_names(self, node_type, indices):
        """
        Get human-readable feature names for given indices.
        
        Args:
            node_type: Type of node (activity, resource, time, trace)
            indices: Array of feature indices
        
        Returns:
            List of feature names
        """
        names = []
        
        if node_type == 'activity':
            # Map to actual activity names
            inv_vocab = {v: k for k, v in self.vocabs['Activity'].items()}
            for idx in indices:
                activity_name = inv_vocab.get(idx, f"Activity_{idx}")
                # Shorten long names
                if len(activity_name) > 25:
                    activity_name = activity_name[:22] + "..."
                names.append(activity_name)
                
        elif node_type == 'resource':
            # Map to actual resource names
            inv_vocab = {v: k for k, v in self.vocabs['Resource'].items()}
            for idx in indices:
                resource_name = inv_vocab.get(idx, f"Resource_{idx}")
                # Shorten long names
                if len(resource_name) > 25:
                    resource_name = resource_name[:22] + "..."
                names.append(resource_name)
                
        elif node_type == 'time':
            # Time has only one feature
            for idx in indices:
                names.append("Timestamp")
                
        elif node_type == 'trace':
            # Map to trace attribute names
            for idx in indices:
                if idx < len(self.trace_attributes):
                    attr_name = self.trace_attributes[idx]
                    # Shorten long names
                    if len(attr_name) > 25:
                        attr_name = attr_name[:22] + "..."
                    names.append(attr_name)
                else:
                    # This is a one-hot encoded dimension
                    # Try to find which trace attribute it belongs to
                    current_idx = 0
                    for attr in self.trace_attributes:
                        if attr in self.vocabs:
                            vocab_size = len(self.vocabs[attr])
                            if current_idx <= idx < current_idx + vocab_size:
                                value_idx = idx - current_idx
                                inv_vocab = {v: k for k, v in self.vocabs[attr].items()}
                                value = inv_vocab.get(value_idx, f"Val_{value_idx}")
                                name = f"{attr}={value}"
                                if len(name) > 25:
                                    name = name[:22] + "..."
                                names.append(name)
                                break
                            current_idx += vocab_size
                        else:
                            if current_idx == idx:
                                names.append(attr)
                                break
                            current_idx += 1
                    else:
                        names.append(f"Trace_Dim_{idx}")
        else:
            # Fallback for unknown node types
            names = [f"{node_type}_Dim_{i}" for i in indices]
        
        return names
        
    def explain_all_tasks(self, graph_idx):
        """Explain all three tasks for one graph."""
        results = {}
        for task in ['activity', 'time', 'remaining']:
            try:
                imp = self.explain_graph(graph_idx, task)
                results[task] = imp
            except Exception as e:
                print(f"✗ Failed {task}: {e}\n")
                results[task] = None
        return results
        
    def batch_explain(self, num_samples):
        """
        Explain the model by aggregating feature importance across multiple samples.
        
        Args:
            num_samples: Number of samples to aggregate for model explanation
        """
        print(f"\n{'='*70}")
        print(f"GLOBAL MODEL EXPLANATION")
        print(f"Analyzing {num_samples} samples to understand model behavior")
        print(f"{'='*70}\n")
        
        all_results = {'activity': {}, 'time': {}, 'remaining': {}}
        
        # Collect importance scores across samples
        for i in range(min(num_samples, len(self.sample_graphs))):
            print(f"Processing sample {i+1}/{num_samples}...", end='\r')
            
            graph = self.sample_graphs[i].to(self.device)
            
            # Compute importance for each task
            for task in ['activity', 'time', 'remaining']:
                importance = self._compute_importance(graph, task)
                
                # Aggregate by node type
                for node_type, imp in importance.items():
                    if node_type not in all_results[task]:
                        all_results[task][node_type] = []
                    all_results[task][node_type].append(imp)
        
        print(f"\n✓ Processed {num_samples} samples\n")
        
        # Average importance scores
        aggregated = {}
        for task in ['activity', 'time', 'remaining']:
            aggregated[task] = {}
            for node_type, imp_list in all_results[task].items():
                if imp_list:
                    stacked = np.stack(imp_list)
                    aggregated[task][node_type] = stacked.mean(axis=0)
        
        # Create global visualizations and report
        self._visualize_global_importance(aggregated, num_samples)
        self._create_global_report(aggregated, num_samples)
        
        output_dir = os.path.join(self.model_dir, 'explanations')
        print(f"\n{'='*70}")
        print(f"MODEL EXPLANATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nResults saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - global_importance_activity.png")
        print("  - global_importance_time.png")
        print("  - global_importance_remaining.png")
        print("  - global_model_explanation.txt")
        print(f"{'='*70}\n")
        
        return aggregated
    
    def _visualize_global_importance(self, aggregated, num_samples):
        """Visualize global model-level feature importance."""
        output_dir = os.path.join(self.model_dir, 'explanations')
        os.makedirs(output_dir, exist_ok=True)
        
        for task, importance_dict in aggregated.items():
            if not importance_dict:
                continue
            
            # Filter out trace and handle time differently
            filtered_dict = {}
            time_importance = None
            
            for node_type, imp in importance_dict.items():
                if node_type == 'trace':
                    continue  # Skip trace
                elif node_type == 'time':
                    time_importance = imp  # Handle separately
                else:
                    filtered_dict[node_type] = imp
            
            # Determine number of subplots (activity + resource + time if exists)
            n_plots = len(filtered_dict)
            if time_importance is not None:
                n_plots += 1
            
            if n_plots == 0:
                continue
            
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # Plot activity and resource with bars
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
                ax.set_yticklabels(feature_names, fontsize=8)
                ax.set_xlabel('Average Importance', fontsize=10)
                ax.set_title(f'{node_type.capitalize()}\n({task.capitalize()})', 
                           fontsize=11, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plot_idx += 1
            
            # Plot time as a line/point
            if time_importance is not None:
                ax = axes[plot_idx]
                
                if len(time_importance.shape) > 1:
                    time_importance = time_importance.flatten()
                
                # Since time has only 1 feature, show it as a single point with context
                time_value = time_importance[0] if len(time_importance) > 0 else 0
                
                # Create a simple visualization
                ax.barh([0], [time_value], color='#FF9500', height=0.5)
                ax.set_yticks([0])
                ax.set_yticklabels(['Timestamp'], fontsize=10)
                ax.set_xlabel('Average Importance', fontsize=10)
                ax.set_title(f'Time\n({task.capitalize()})', 
                           fontsize=11, fontweight='bold')
                ax.set_ylim(-0.5, 0.5)
                ax.grid(axis='x', alpha=0.3)
                
                # Add annotation
                ax.text(time_value, 0, f' {time_value:.4f}', 
                       va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            filename = f'global_importance_{task}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved global explanation: {filename}")
    
    def _create_global_report(self, aggregated, num_samples):
        """Create a global model explanation report."""
        output_dir = os.path.join(self.model_dir, 'explanations')
        report_path = os.path.join(output_dir, 'global_model_explanation.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GLOBAL MODEL EXPLANATION\n")
            f.write(f"Aggregated across {num_samples} test samples\n")
            f.write("="*70 + "\n\n")
            
            f.write("This report shows which features are MOST IMPORTANT for the model\n")
            f.write("across multiple predictions, providing insight into the model's\n")
            f.write("overall decision-making process.\n\n")
            
            for task, importance_dict in aggregated.items():
                if not importance_dict:
                    continue
                
                f.write("="*70 + "\n")
                f.write(f"{task.upper()} PREDICTION\n")
                f.write("="*70 + "\n\n")
                
                for node_type, imp in importance_dict.items():
                    if node_type == 'trace':
                        continue  # Skip trace
                    
                    if len(imp.shape) > 1:
                        imp = imp.flatten()
                    
                    f.write(f"{node_type.capitalize()} Features:\n")
                    f.write("-"*70 + "\n")
                    
                    if node_type == 'time':
                        f.write(f"  Timestamp importance: {imp[0]:.6f}\n")
                        f.write("  Interpretation: Shows how much the model relies on timing\n")
                        f.write("  information for this prediction task.\n\n")
                    else:
                        # Show top 10 features
                        top_k = min(10, len(imp))
                        top_idx = np.argsort(imp)[-top_k:][::-1]
                        
                        feature_names = self._get_feature_names(node_type, top_idx)
                        
                        for rank, (idx, name) in enumerate(zip(top_idx, feature_names), 1):
                            f.write(f"  {rank}. {name}: {imp[idx]:.6f}\n")
                        f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*70 + "\n\n")
            f.write("Higher importance values indicate features that have stronger\n")
            f.write("influence on the model's predictions. These are the features\n")
            f.write("the model 'pays attention to' when making decisions.\n\n")
            f.write("Activity features: Which previous activities are most predictive\n")
            f.write("Resource features: Which resources/actors are most influential\n")
            f.write("Time features: How much the model relies on temporal patterns\n")
        
        print(f"✓ Saved global report: global_model_explanation.txt\n")


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='GNN Model Explainability')
    parser.add_argument('--model_dir', required=False, help='Model directory')
    parser.add_argument('--dataset', required=False, help='Dataset CSV path')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of samples to aggregate for model explanation (default: 50)')
    parser.add_argument('--mode', default='model', 
                       choices=['model', 'single'],
                       help='Explanation mode: model (global) or single (one graph)')
    parser.add_argument('--graph_idx', type=int, default=0, 
                       help='Graph index (only for single mode)')
    
    args = parser.parse_args()
    
    # Auto-detect model and dataset if not provided
    if not args.model_dir or not args.dataset:
        print("\n" + "="*70)
        print("AUTO-DETECTING MODEL AND DATASET")
        print("="*70 + "\n")
        
        # Find most recent model directory
        results_dirs = glob.glob("results/B2L_gnn_unified_prediction_*")
        if not results_dirs:
            print("✗ No model directories found in results/")
            print("Please train a model first by running main.py")
            return
        
        # Sort by timestamp (most recent first)
        results_dirs.sort(reverse=True)
        args.model_dir = results_dirs[0]
        
        print(f"✓ Found model: {args.model_dir}")
        
        # Read dataset from dataset_info.txt
        info_file = os.path.join(args.model_dir, 'dataset_info.txt')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    if line.startswith('Full Path:'):
                        args.dataset = line.split('Full Path:')[1].strip()
                        break
        
        if not args.dataset:
            print("✗ Could not detect dataset from model info")
            print("Please specify --dataset manually")
            return
        
        print(f"✓ Found dataset: {os.path.basename(args.dataset)}\n")
    
    explainer = GraphExplainer(args.model_dir, args.dataset)
    
    if args.mode == 'single':
        print("\n⚠️  Running in SINGLE mode - explaining one specific graph")
        explainer.explain_all_tasks(args.graph_idx)
    else:
        print(f"\n✓ Running in MODEL mode - explaining model behavior")
        explainer.batch_explain(args.num_samples)


if __name__ == "__main__":
    main()