import sys
import os
import json
import traceback
import argparse
import pickle
import pandas as pd
import numpy as np

# ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ppm_pipeline import detect_and_standardize_columns, _safe_rename_columns
from transformers.prediction.next_activity import NextActivityPredictor
from explainability.transformers.transformer_explainer import SHAPExplainer, LIMEExplainer

def run_transformer_explainability_on_demand(run_dir, dataset_path, case_id, case_index, method, task="activity"):
    artifacts_dir = os.path.join(run_dir, "artifacts")
    pkl_path = os.path.join(artifacts_dir, "transformer_artifacts.pkl")
    model_path = os.path.join(artifacts_dir, "next_activity_transformer.keras")

    if not os.path.exists(pkl_path) or not os.path.exists(model_path):
        raise RuntimeError("Missing transformer artifacts for on-demand explainability.")

    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
    
    label_encoder = artifacts["label_encoder"]
    vocab_size = artifacts["vocab_size"]
    max_len = artifacts["max_len"]

    # load data
    mapped_path = os.path.join(run_dir, "input", "dataset_mapped.csv")
    if os.path.exists(mapped_path):
        df = pd.read_csv(mapped_path)
    else:
        df = pd.read_csv(dataset_path)
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    # Prepare data to get sequence
    # case_id from the frontend is stripped of "Case ". We will do the same to dataset case variables for matching.
    case_col_str = df['case:id'].astype(str).str.replace(r'^[Cc]ase\s+', '', regex=True).str.strip()
    
    case_df = df[case_col_str == str(case_id)].copy()
    if case_df.empty:
        raise ValueError(f"Case {case_id} not found in dataset.")
    
    # Sort by time
    case_df = case_df.sort_values(by='time:timestamp')
    activities = case_df['concept:name'].values
    
    encoded_activities = label_encoder.transform(activities) + 1  # 0 is padding

    target_idx = case_index # case_index comes from save_results, it's 1-indexed, meaning 1st prefix (len 1, to predict 2nd)
    if target_idx is None:
        target_idx = len(encoded_activities) - 1
    
    if target_idx < 1 or target_idx >= len(encoded_activities):
        raise ValueError(f"Invalid case index {case_index} for case length {len(encoded_activities)}")

    prefix = encoded_activities[:target_idx]
    
    # pad sequence
    padded_prefix = np.zeros(max_len, dtype=int)
    if len(prefix) > max_len:
        padded_prefix[:] = prefix[-max_len:]
    else:
        padded_prefix[-len(prefix):] = prefix
    
    X_sample = np.array([padded_prefix])
    
    # Load model
    from tensorflow import keras
    from transformers.model import PositionalEncoding, TransformerBlock
    model = keras.models.load_model(model_path, custom_objects={
        'PositionalEncoding': PositionalEncoding,
        'TransformerBlock': TransformerBlock
    })
    
    # We need background data. Let's just use a sample of all data
    # Wait, fetching all data through prepare_data is expensive. Let's sample directly:
    grouped = df.groupby('case:id')
    bg_seqs = []
    for cid, group in grouped:
        act = label_encoder.transform(group['concept:name'].values) + 1
        for i in range(1, len(act)):
            p = act[:i]
            ps = np.zeros(max_len, dtype=int)
            if len(p) > max_len:
                ps[:] = p[-max_len:]
            else:
                ps[-len(p):] = p
            bg_seqs.append(ps)
        if len(bg_seqs) > 100:
            break
            
    bg_data = np.array(bg_seqs[:100])
    
    output_dir = os.path.join(artifacts_dir, "explainability", f"{case_id}_{case_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    result = {}

    if method.lower() == "shap":
        explainer = SHAPExplainer(model, task=task, label_encoder=label_encoder)
        explainer.initialize_explainer(bg_data)
        explainer.explain_samples(X_sample, num_samples=1, sample_ids=[case_id], sample_indexes=[case_index])
        
        # Plot local SHAP logic just like LIME
        explainer.plot_explanation(output_dir, sample_idx=0, case_id=case_id, case_index=case_index)
        
        import shutil, glob
        shap_files = glob.glob(os.path.join(output_dir, "shap_explanation_*.png"))
        if shap_files:
            shutil.copy(shap_files[0], os.path.join(output_dir, "shap_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"SHAP Explanation Failed\nGraph could not be generated", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "shap_summary.png"))
            plt.close()
            
        result["method"] = "shap"
        result["files"] = ["shap_summary.png"]
    elif method.lower() == "lime":
        explainer = LIMEExplainer(model, task=task, label_encoder=label_encoder)
        explainer.initialize_explainer(bg_data)
        explainer.explain_samples(X_sample, num_samples=1, sample_case_ids=[case_id], sample_indexes=[case_index])
        
        # LIME explainer uses plot_explanation for individual samples (which acts as a summary plot for that case)
        explainer.plot_explanation(output_dir, sample_idx=0, case_id=case_id, case_index=case_index)
        
        # To match the expected names from the UI: 
        # The LIME method generates a file like: lime_explanation_case_{case_id}_idx_{case_index}.png
        import shutil, glob
        lime_files = glob.glob(os.path.join(output_dir, "lime_explanation_*.png"))
        if lime_files:
            shutil.copy(lime_files[0], os.path.join(output_dir, "lime_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"LIME Explanation Failed\nGraph could not be generated", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "lime_summary.png"))
            plt.close()
        
        result["method"] = "lime"
        result["files"] = ["lime_summary.png"]
        
    # Write result.json
    res_path = os.path.join(output_dir, f"{method}_result.json")
    with open(res_path, "w") as f:
        json.dump(result, f)
        
    return res_path

from explainability.gnns.gnn_explainer import GradientExplainer, GraphLIMEExplainer

def run_gnn_explainability_on_demand(run_dir, dataset_path, case_id, case_index, method, task="activity"):
    artifacts_dir = os.path.join(run_dir, "artifacts")
    output_dir = os.path.join(artifacts_dir, "explainability", f"{case_id}_{case_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if a model file exists
    model_path = os.path.join(artifacts_dir, "gnn_model.pt")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing GNN model at {model_path}.")
        
    mapped_path = os.path.join(run_dir, "input", "dataset_mapped.csv")
    if os.path.exists(mapped_path):
        df = pd.read_csv(mapped_path)
    else:
        df = pd.read_csv(dataset_path)
        from utils.column_detector import detect_and_standardize_columns
        df, _, _ = detect_and_standardize_columns(df, verbose=False)
        
    from ppm_pipeline import _safe_rename_columns
    df = _safe_rename_columns(df, {
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp',
        'Resource': 'Resource'
    })
    
    case_col = 'case:id'
    act_col = 'concept:name'
    time_col = 'time:timestamp'
    res_col = 'Resource'
    
    if res_col not in df.columns:
        df[res_col] = 'Unknown'
        
    # Build vocabs identically to GNN predictor
    all_activities = set(df[act_col].unique().tolist())
    vocabs = {}
    vocabs["Activity"] = {v: i for i, v in enumerate(sorted(all_activities))}
    vocabs["Resource"] = {v: i for i, v in enumerate(sorted(df[res_col].unique().tolist()))}
    
    # We only reconstruct ignoring trace attributes for on-demand since we don't have them easily
    # It might result in trace nodes having zeros, which is acceptable for activity/resource explanations.
    
    # Get case
    case_col_str = df[case_col].astype(str).str.replace(r'^[Cc]ase\s+', '', regex=True).str.strip()
    case_df = df[case_col_str == str(case_id)].copy()
    if case_df.empty:
        raise ValueError(f"Case {case_id} not found in dataset.")
    
    case_df[time_col] = pd.to_datetime(case_df[time_col])
    case_df = case_df.sort_values(by=time_col)
    
    # Target idx comes from case_index (1-indexed for prefix length)
    target_idx = case_index
    if target_idx < 1 or target_idx >= len(case_df):
        raise ValueError(f"Invalid case index {case_index} for case length {len(case_df)}")
        
    prefix_df = case_df.iloc[:target_idx].copy()
    prefix_df["__ts_log"] = np.log1p(prefix_df[time_col].astype("int64") // 1_000_000_000).astype("float32")
    
    from torch_geometric.data import HeteroData
    import torch
    import torch.nn.functional as F
    
    graph = HeteroData()
    act_map = vocabs["Activity"]
    res_map = vocabs["Resource"]
    k = len(prefix_df)
    
    act_ids = np.array([act_map.get(a, 0) for a in prefix_df[act_col]])
    graph["activity"].x = F.one_hot(torch.tensor(act_ids, dtype=torch.long), num_classes=len(act_map)).float()
    graph["activity"].num_nodes = k
    
    res_ids = np.array([res_map.get(r, 0) for r in prefix_df[res_col]])
    graph["resource"].x = F.one_hot(torch.tensor(res_ids, dtype=torch.long), num_classes=len(res_map)).float()
    graph["resource"].num_nodes = k
    
    graph["time"].x = torch.tensor(prefix_df["__ts_log"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    graph["time"].num_nodes = k
    
    graph["trace"].x = torch.zeros((1, 1), dtype=torch.float32)
    graph["trace"].num_nodes = 1
    
    idx = torch.arange(k)
    dfr = torch.stack([idx[:-1], idx[1:]]) if k > 1 else torch.empty((2, 0), dtype=torch.long)
    graph["activity", "next", "activity"].edge_index = dfr
    graph["resource", "next", "resource"].edge_index = dfr.clone()
    graph["time", "next", "time"].edge_index = dfr.clone()
    
    same_ev = torch.stack([idx, idx])
    graph["activity", "same_event", "resource"].edge_index = same_ev
    graph["resource", "same_event", "activity"].edge_index = same_ev.clone()
    graph["activity", "same_time", "time"].edge_index = same_ev.clone()
    graph["time", "same_time", "activity"].edge_index = same_ev.clone()

    trace_src = torch.zeros(k, dtype=torch.long)
    graph["activity", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
    graph["resource", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
    graph["time", "to_trace", "trace"].edge_index = torch.stack([idx, trace_src])
    
    # Ground truth labels
    next_act_lbl = case_df.iloc[target_idx][act_col]
    graph.y_activity = torch.tensor([act_map.get(next_act_lbl, 0)], dtype=torch.long)
    graph.y_timestamp = torch.tensor([0.0]) # Add real calc if needed later
    graph.y_remaining_time = torch.tensor([0.0]) # Add real calc if needed later
    graph.case_id = case_id
    graph.case_index = case_index

    # Build model (config would ideally be loaded, but assuming defaults for test)
    from gnns.model import HeteroGNN
    metadata = graph.metadata()
    proj_dims = {key: v.size(-1) for key, v in graph.x_dict.items()}
    num_classes = len(act_map)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assume default config or try to load, but default is 64 hidden
    model = HeteroGNN(
        metadata=metadata,
        hidden_channels=64, # Default from unified prediction
        proj_dims=proj_dims,
        num_activity_classes=num_classes,
        dropout=0.1
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"Warning: partial load or error: {e}")
        
    result = {"method": method, "files": [f"{method}_summary.png"]}
    
    # Run the explainer
    if method.lower() == "gradient":
        explainer = GradientExplainer(model, device, vocabs)
        contrib, pred, true_val, step_info = explainer.explain_individual_sample(graph, task)
        
        if contrib is not None and not np.allclose(contrib, 0):
            explainer.plot_individual_gradient_explanation(contrib, pred, true_val, step_info, output_dir, task, f"case_{case_id}_idx_{case_index}")
            import shutil, glob
            plots = glob.glob(os.path.join(output_dir, "gradient_timestep_heatmap*.png"))
            if plots:
                shutil.copy(plots[0], os.path.join(output_dir, "gradient_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"Gradient Explanation Failed\nAll contributions were zero", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "gradient_summary.png"))
            plt.close()
            
    elif method.lower() == "graphlime":
        explainer = GraphLIMEExplainer(model, device, vocabs)
        imp, score, true_val, step_info, pred_class = explainer.explain_local(graph, task)
        
        explainer.plot_local_explanation(imp, score, true_val, step_info, output_dir, task, f"case_{case_id}_idx_{case_index}", pred_class)
        import shutil, glob
        plots = glob.glob(os.path.join(output_dir, "graphlime_sample*.png"))
        if plots:
            shutil.copy(plots[0], os.path.join(output_dir, "graphlime_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"LIME Explanation Failed\nValid features could not be found", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "graphlime_summary.png"))
            plt.close()

    res_path = os.path.join(output_dir, f"{method}_result.json")
    with open(res_path, "w") as f:
        json.dump(result, f)
        
    return res_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-type", required=True)
    
    args = parser.parse_args()
    
    if args.model_type == "transformer":
        res_path = run_transformer_explainability_on_demand(
            args.run_dir, args.dataset, args.case_id, args.case_index, args.method
        )
        print(json.dumps({"success": True, "result_file": res_path}))
    elif args.model_type == "gnn":
        res_path = run_gnn_explainability_on_demand(
            args.run_dir, args.dataset, args.case_id, args.case_index, args.method
        )
        print(json.dumps({"success": True, "result_file": res_path}))
    else:
        print(json.dumps({"success": False, "error": f"Unknown model type {args.model_type}"}))

if __name__ == "__main__":
    main()
