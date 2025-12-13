import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
from torch_geometric.data import Batch
from tqdm import tqdm
import pickle
import json


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


def run_gnn_explainability(model, data, output_dir, device, num_samples=10, methods='all'):
    os.makedirs(output_dir, exist_ok=True)
    
    test_graphs = data['test']
    
    print("\n" + "="*70)
    print("GNN EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    run_gnnexplainer = methods in ['shap', 'all']
    run_graphlime = methods in ['lime', 'all']
    
    results = {}
    
    if run_gnnexplainer:
        print("\n[GNNExplainer Method]")
        print("-"*70)
        gnn_explainer = GNNExplainerWrapper(model, device)
        gnn_explainer.initialize_explainer(test_graphs[0], task='activity')
        
        gnn_explanations = gnn_explainer.explain_batch(test_graphs, num_samples=num_samples)
        gnn_aggregated = gnn_explainer.aggregate_explanations(gnn_explanations)
        
        gnnexp_dir = os.path.join(output_dir, 'gnnexplainer')
        gnn_explainer.save_explanations(gnn_explanations, gnnexp_dir)
        gnn_explainer.plot_feature_importance(gnn_aggregated, gnnexp_dir)
        gnn_explainer.plot_edge_importance(gnn_aggregated, gnnexp_dir)
        
        for i in range(min(3, len(gnn_explanations))):
            gnn_explainer.visualize_explanation(gnn_explanations[i], gnnexp_dir, sample_id=i)
        
        print(f"✓ GNNExplainer results saved to: {gnnexp_dir}")
        results['gnnexplainer'] = gnn_explanations
    
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