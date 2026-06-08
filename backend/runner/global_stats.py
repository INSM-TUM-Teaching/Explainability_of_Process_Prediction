# backend/runner/global_stats.py
import os
import json
import pandas as pd
from typing import Any, Dict, List
try:
    import pm4py
except ImportError:
    pm4py = None

def get_process_map(dataset_path: str):
    if pm4py is None:
        return {"nodes": [], "edges": [], "error": "pm4py is not installed"}

    try:
        df = pd.read_csv(dataset_path)
        
        case_col = None
        case_patterns = ['CaseID', 'case:id', 'case:concept:name', 'case_id', 'caseid', 'Case ID', 'Case_ID']
        for col in df.columns:
            if col in case_patterns:
                case_col = col
                break
                
        act_col = None
        act_patterns = ['Activity', 'concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
        for col in df.columns:
            if col in act_patterns:
                act_col = col
                break
                
        ts_col = None
        ts_patterns = ['Timestamp', 'Complete Timestamp', 'time:timestamp', 'timestamp', 'time', 'Time', 'start_time', 'StartTime', 'complete_time', 'CompleteTime']
        for col in df.columns:
            if col in ts_patterns:
                ts_col = col
                break

        if not case_col or not act_col or not ts_col:
            return {"nodes": [], "edges": [], "error": f"Required columns not found in dataset. Found: {list(df.columns)}"}
            
        # Standardize for pm4py
        df = pm4py.format_dataframe(df, case_id=case_col, activity_key=act_col, timestamp_key=ts_col)
        
        # Extract DFG (Directly Follows Graph)
        dfg, start_activities, end_activities = pm4py.discover_dfg(df)
        
        nodes_dict = {}
        edges = []
        
        # Add edges and track nodes
        for (source, target), count in dfg.items():
            if source not in nodes_dict:
                nodes_dict[source] = {"id": source, "label": source}
            if target not in nodes_dict:
                nodes_dict[target] = {"id": target, "label": target}
                
            edges.append({
                "source": source,
                "target": target,
                "weight": count
            })
            
        # Ensure start and end nodes are also tracked (sometimes they have no edges if trace length 1)
        for act in start_activities:
            if act not in nodes_dict:
                nodes_dict[act] = {"id": act, "label": act}
        for act in end_activities:
            if act not in nodes_dict:
                nodes_dict[act] = {"id": act, "label": act}
                
        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges,
            "start_activities": start_activities,
            "end_activities": end_activities
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

def calculate_global_metrics(run_dir: str, dataset_path: str):
    artifacts_dir = os.path.join(run_dir, "artifacts")
    summary_path = os.path.join(artifacts_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return {"error": "Run summary not found"}
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    req = summary.get("request", {})
    model_type = req.get("model_type", "transformer")
    task = req.get("task", "next_activity")
    
    if model_type == "gnn":
        pred_file = os.path.join(artifacts_dir, "gnn_predictions.json")
    elif model_type == "transformer":
        pred_file = os.path.join(artifacts_dir, "transformer_predictions.json")
    else:
        pred_file = os.path.join(artifacts_dir, "best_predictions.json")
        
    if not os.path.exists(pred_file):
        return {"error": "Predictions not found"}
        
    with open(pred_file, 'r') as f:
        preds = json.load(f)
        
    # Read original dataset to compute variants
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return {"error": f"Could not read dataset: {str(e)}"}
        
    case_col = None
    case_patterns = ['CaseID', 'case:id', 'case:concept:name', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    for col in df.columns:
        if col in case_patterns:
            case_col = col
            break
            
    act_col = None
    act_patterns = ['Activity', 'concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
    for col in df.columns:
        if col in act_patterns:
            act_col = col
            break

    if not case_col or not act_col:
        return {"error": f"Could not find case or activity column in dataset. Found: {list(df.columns)}"}
    
    # Identify variant for each case
    variants = {} # case_id -> variant signature
    case_groups = df.groupby(case_col)[act_col].apply(list).to_dict()
    for c_id, trace in case_groups.items():
        # Clean case ID to match the format in predictions files
        clean_id = str(c_id).replace("Case ", "").replace("case ", "").replace(" ", "_").strip()
        variants[clean_id] = " -> ".join(trace)
        
    variant_stats = {}
    prefix_stats = {}
    
    overall_correct = 0
    total_preds = len(preds)
    
    for p in preds:
        c_id = str(p.get("case_id"))
        seq = p.get("sequence", "")
        # For GNN, sequence might not be explicitly stored like transformer, it's prefix_length
        # We need to compute prefix length
        if "prefix_length" in p:
            prefix_len = p["prefix_length"]
        else:
            prefix_len = len([x for x in seq.split(",") if x.strip()]) if seq else 1
            
        true_act = p.get("true_next_activity") or p.get("actual_next_activity")
        pred_act = p.get("predicted_next_activity")
        
        is_correct = (true_act == pred_act)
        if is_correct:
            overall_correct += 1
            
        var_sig = variants.get(c_id, "Unknown Variant")
        
        if var_sig not in variant_stats:
            variant_stats[var_sig] = {"total": 0, "correct": 0}
        variant_stats[var_sig]["total"] += 1
        if is_correct:
            variant_stats[var_sig]["correct"] += 1
            
        if prefix_len not in prefix_stats:
            prefix_stats[prefix_len] = {"total": 0, "correct": 0}
        prefix_stats[prefix_len]["total"] += 1
        if is_correct:
            prefix_stats[prefix_len]["correct"] += 1

    # Format output
    variant_list = []
    for var_sig, stats in variant_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        variant_list.append({
            "variant": var_sig,
            "total_cases_in_test": stats["total"],
            "accuracy": acc
        })
    variant_list.sort(key=lambda x: x["total_cases_in_test"], reverse=True)
    
    prefix_list = []
    for plen, stats in prefix_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        prefix_list.append({
            "prefix_length": plen,
            "total_cases": stats["total"],
            "accuracy": acc
        })
    prefix_list.sort(key=lambda x: x["prefix_length"])
    
    # Extract Global Explainability (if available)
    global_explanations = {"shap_features": [], "lime_features": [], "gnn_features": []}
    
    if model_type == "transformer":
        shap_csv = os.path.join(artifacts_dir, "explainability", "shap", "global_importance_data.csv")
        lime_csv = os.path.join(artifacts_dir, "explainability", "lime", "global_importance_data.csv")
        
        if os.path.exists(shap_csv):
            try:
                sdf = pd.read_csv(shap_csv)
                if 'activity' in sdf.columns:
                    val_col = 'Mean_Impact' if 'Mean_Impact' in sdf.columns else 'importance' if 'importance' in sdf.columns else None
                    if val_col:
                        global_explanations["shap_features"] = sdf.rename(columns={'activity': 'feature', val_col: 'importance'}).to_dict(orient="records")
            except Exception:
                pass
        
        if os.path.exists(lime_csv):
            try:
                ldf = pd.read_csv(lime_csv)
                if 'activity' in ldf.columns:
                    val_col = 'Mean_Impact' if 'Mean_Impact' in ldf.columns else 'importance' if 'importance' in ldf.columns else None
                    if val_col:
                        global_explanations["lime_features"] = ldf.rename(columns={'activity': 'feature', val_col: 'importance'}).to_dict(orient="records")
            except Exception:
                pass
    elif model_type == "gnn":
        gnn_task_name = "activity" if task in ["next_activity", "custom_activity", "unified"] else task
        grad_csv = os.path.join(artifacts_dir, "explainability", "gradient", f"gradient_global_{gnn_task_name}.csv")
        lime_csv = os.path.join(artifacts_dir, "explainability", "graphlime", f"graphlime_global_{gnn_task_name}.csv")
        if os.path.exists(grad_csv):
            try:
                gdf = pd.read_csv(grad_csv)
                if 'activity' in gdf.columns and 'importance' in gdf.columns:
                    global_explanations["gnn_features"] = gdf.rename(columns={'activity': 'feature'}).to_dict(orient="records")
            except Exception:
                pass
        if os.path.exists(lime_csv):
            try:
                ldf = pd.read_csv(lime_csv)
                if 'activity' in ldf.columns and 'importance' in ldf.columns:
                    global_explanations["lime_features"] = ldf.rename(columns={'activity': 'feature'}).to_dict(orient="records")
            except Exception:
                pass

    return {
        "overall_accuracy": (overall_correct / total_preds * 100) if total_preds > 0 else 0,
        "total_variants": len(case_groups.keys()),
        "unique_variants": len(set(variants.values())),
        "variants": variant_list,
        "prefix_accuracy": prefix_list,
        "global_explanations": global_explanations
    }
