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
from explainability.gnns.gnn_explainer import GradientExplainer, GraphLIMEExplainer

def run_transformer_explainability_on_demand(run_dir, dataset_path, case_id, case_index, method, task="activity"):
    artifacts_dir = os.path.join(run_dir, "artifacts")
    storage_dir = os.path.dirname(os.path.dirname(run_dir))
    dataset_artifacts_path = os.path.join(storage_dir, "transformer_artifacts.pkl")
    
    if not os.path.exists(dataset_artifacts_path):
        if artifacts_dir:
            dataset_artifacts_path = os.path.join(artifacts_dir, "transformer_artifacts.pkl")

    if os.path.exists(dataset_artifacts_path):
        with open(dataset_artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
    else:
        if artifacts_dir:
            print(f"[WARNING] Missing transformer_artifacts.pkl. Reconstructing from {dataset_path}...")
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            label_encoder = LabelEncoder()
            df_temp, _, _ = detect_and_standardize_columns(pd.read_csv(dataset_path), verbose=False)
            df_temp = _safe_rename_columns(df_temp, {
                'CaseID': 'case:id',
                'Activity': 'concept:name',
                'Timestamp': 'time:timestamp'
            })
            case_col = next((c for c in ['CaseID', 'case:id', 'case:concept:name', 'case_id', 'caseid', 'Case ID', 'Case_ID'] if c in df_temp.columns), df_temp.columns[0])
            act_col = next((c for c in ['Activity', 'concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task'] if c in df_temp.columns), df_temp.columns[1])
            time_col = next((c for c in ['Timestamp', 'time:timestamp', 'timestamp', 'Complete Timestamp', 'time'] if c in df_temp.columns), df_temp.columns[2] if len(df_temp.columns) > 2 else df_temp.columns[0])
            label_encoder.fit(df_temp[act_col])
            
            vocab_size = len(label_encoder.classes_) + 2
            df_temp = df_temp.sort_values(by=[case_col, time_col])
            max_seq_len = df_temp.groupby(case_col).size().max()
            
            scaler = None
            if task in ["remaining_time", "time", "event_time"]:
                print("[INFO] Reconstructing StandardScaler for temporal features exactly as during training...")
                df_temp[time_col] = pd.to_datetime(df_temp[time_col])
                df_temp['fvt1'] = df_temp.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)
                case_starts = df_temp.groupby(case_col)[time_col].transform('min')
                df_temp['fvt2'] = (df_temp[time_col] - case_starts).dt.total_seconds()
                df_temp['fvt3'] = df_temp[time_col].dt.hour
                
                # Check for train split
                if 'split' in df_temp.columns:
                    train_df = df_temp[df_temp['split'] == 'train']
                else:
                    train_df = df_temp
                
                # Extract temporal features EXACTLY as in build_samples (skip i=0)
                train_feats = []
                for cid, group in train_df.groupby(case_col):
                    f1 = group['fvt1'].values
                    f2 = group['fvt2'].values
                    f3 = group['fvt3'].values
                    # build_samples only includes range(1, len)
                    for i in range(1, len(group)):
                        train_feats.append([f1[i], f2[i], f3[i]])
                        
                scaler = StandardScaler()
                if train_feats:
                    scaler.fit(train_feats)
                else:
                    scaler.fit(df_temp[['fvt1', 'fvt2', 'fvt3']].values)

            artifacts = {"label_encoder": label_encoder, "scaler": scaler, "vocab_size": vocab_size, "max_seq_len": max_seq_len}
        else:
            raise RuntimeError(f"Missing transformer_artifacts.pkl for on-demand explainability.")

    label_encoder = artifacts["label_encoder"]
    vocab_size = artifacts["vocab_size"]
    max_len = int(artifacts.get("max_len", artifacts.get("max_seq_len", 16)))

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

    case_col_str = df['case:id'].astype(str).str.replace(r'^[Cc]ase\s+', '', regex=True).str.strip()
    clean_case_id = str(case_id).replace("Case ", "").replace("case ", "").strip()
    case_df = df[case_col_str == clean_case_id].copy()
    
    if case_df.empty:
        raise ValueError(f"Case {case_id} not found in dataset.")
    
    case_df = case_df.sort_values(by='time:timestamp')
    activities = case_df['concept:name'].values
    encoded_activities = label_encoder.transform(activities) + 1
    
    target_idx = case_index
    if target_idx is None:
        target_idx = len(encoded_activities) - 1
    
    if target_idx < 1 or target_idx >= len(encoded_activities):
        raise ValueError(f"Invalid case index {case_index} for case length {len(encoded_activities)}")

    prefix = encoded_activities[:target_idx]
    padded_prefix = np.zeros(max_len, dtype=int)
    if len(prefix) > max_len:
        padded_prefix[:] = prefix[-max_len:]
    else:
        padded_prefix[-len(prefix):] = prefix
    
    X_sample_seq = np.array([padded_prefix])

    if task in ["remaining_time", "time", "event_time"]:
        case_start = pd.to_datetime(case_df['time:timestamp'].iloc[0])
        curr_time = pd.to_datetime(case_df['time:timestamp'].iloc[target_idx])
        if target_idx > 0:
            prev_time = pd.to_datetime(case_df['time:timestamp'].iloc[target_idx - 1])
        else:
            prev_time = curr_time
            
        fvt1 = (curr_time - prev_time).total_seconds()
        fvt2 = (curr_time - case_start).total_seconds()
        fvt3 = curr_time.hour
        
        X_sample_temp = np.array([[fvt1, fvt2, fvt3]])
        
        if artifacts.get("scaler"):
            X_sample_temp = artifacts["scaler"].transform(X_sample_temp)
            
        X_sample = [X_sample_seq, X_sample_temp]
        
        bg_seqs = []
        bg_temps = []
        grouped = df.groupby('case:id')
        for cid, group in grouped:
            group = group.sort_values(by='time:timestamp')
            act = label_encoder.transform(group['concept:name'].values) + 1
            timestamps = pd.to_datetime(group['time:timestamp'].values)
            cstart = timestamps[0]
            for i in range(1, len(act)):
                p = act[:i]
                ps = np.zeros(max_len, dtype=int)
                if len(p) > max_len:
                    ps[:] = p[-max_len:]
                else:
                    ps[-len(p):] = p
                bg_seqs.append(ps)
                
                curr = timestamps[i]
                prev = timestamps[i-1]
                t1 = (curr - prev).total_seconds()
                t2 = (curr - cstart).total_seconds()
                t3 = curr.hour
                bg_temps.append([t1, t2, t3])
            if len(bg_seqs) > 100:
                break
                
        bg_temps_arr = np.array(bg_temps[:100])
        if artifacts.get("scaler"):
            bg_temps_arr = artifacts["scaler"].transform(bg_temps_arr)
            
        bg_data = [np.array(bg_seqs[:100]), bg_temps_arr]
        
        from transformers.model import build_time_prediction_model
        model = build_time_prediction_model(vocab_size=vocab_size, max_len=max_len, num_heads=4, d_model=64, num_blocks=2, use_timestep_explainability=True)
        model_path = os.path.join(artifacts_dir, "remaining_time_transformer.keras")
    else:
        X_sample = X_sample_seq
        bg_data = []
        grouped = df.groupby('case:id')
        for cid, group in grouped:
            group = group.sort_values(by='time:timestamp')
            act = label_encoder.transform(group['concept:name'].values) + 1
            for i in range(1, len(act)):
                p = act[:i]
                ps = np.zeros(max_len, dtype=int)
                if len(p) > max_len:
                    ps[:] = p[-max_len:]
                else:
                    ps[-len(p):] = p
                bg_data.append(ps)
            if len(bg_data) > 100:
                break
        bg_data = np.array(bg_data[:100])
        
        from transformers.model import build_next_activity_model
        model = build_next_activity_model(vocab_size=vocab_size, max_len=max_len, num_heads=4, d_model=64, num_blocks=2)
        model_path = os.path.join(artifacts_dir, "next_activity_transformer.keras")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing model at {model_path}")
    model.load_weights(model_path)

    output_dir = os.path.join(artifacts_dir, "explainability", f"{case_id}_{case_index}")
    os.makedirs(output_dir, exist_ok=True)

    result = {}
    if method.lower() == "shap":
        explainer = SHAPExplainer(model, task=task, label_encoder=label_encoder)
        explainer.initialize_explainer(bg_data)
        explainer.explain_samples(X_sample, num_samples=1, sample_ids=[case_id], sample_indexes=[case_index])
        explainer.plot_explanation(output_dir, sample_idx=0, case_id=case_id, case_index=case_index)
        
        import shutil, glob
        shap_files = glob.glob(os.path.join(output_dir, "shap_explanation_*.png"))
        if shap_files:
            shutil.copy(shap_files[0], os.path.join(output_dir, "shap_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"SHAP Explanation Failed\\nGraph could not be generated", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "shap_summary.png"))
            plt.close()
            
        result["method"] = "shap"
        result["files"] = ["shap_summary.png"]
    elif method.lower() == "lime":
        explainer = LIMEExplainer(model, task=task, label_encoder=label_encoder)
        explainer.initialize_explainer(bg_data)
        explainer.explain_samples(X_sample, num_samples=1, sample_case_ids=[case_id], sample_indexes=[case_index])
        explainer.plot_explanation(output_dir, sample_idx=0, case_id=case_id, case_index=case_index)
        
        import shutil, glob
        lime_files = glob.glob(os.path.join(output_dir, "lime_explanation_*.png"))
        if lime_files:
            shutil.copy(lime_files[0], os.path.join(output_dir, "lime_summary.png"))
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, f"LIME Explanation Failed\\nGraph could not be generated", ha='center', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "lime_summary.png"))
            plt.close()
        
        result["method"] = "lime"
        result["files"] = ["lime_summary.png"]
        
    res_path = os.path.join(output_dir, f"{method}_result.json")
    with open(res_path, "w") as f:
        import json
        json.dump(result, f)
        
    return res_path

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
            with open(os.path.join(output_dir, "gradient_values.json"), "w") as f:
                json.dump(contrib.tolist(), f)
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
    parser.add_argument("--task", default="activity")
    
    args = parser.parse_args()
    
    if args.model_type == "transformer":
        res_path = run_transformer_explainability_on_demand(
            args.run_dir, args.dataset, args.case_id, args.case_index, args.method, args.task
        )
        print(json.dumps({"success": True, "result_file": res_path}))
    elif args.model_type == "gnn":
        res_path = run_gnn_explainability_on_demand(
            args.run_dir, args.dataset, args.case_id, args.case_index, args.method, args.task
        )
        print(json.dumps({"success": True, "result_file": res_path}))
    else:
        print(json.dumps({"success": False, "error": f"Unknown model type {args.model_type}"}))

if __name__ == "__main__":
    main()
