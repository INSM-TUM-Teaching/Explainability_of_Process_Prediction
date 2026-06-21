# backend/runner/global_stats.py
import os
import json
import pandas as pd
from typing import Any, Dict, List
try:
    import pm4py
except ImportError:
    pm4py = None

def _generate_variant_id(signature: str) -> int:
    """Generates a stable numeric ID for a variant signature."""
    import zlib
    return zlib.crc32(signature.encode()) & 0xffffffff

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
        
        # Calculate node frequencies
        node_counts = df[act_col].value_counts().to_dict()
        total_cases = len(df[case_col].unique())

        # Group by variant to map edges to variants and collect case IDs
        case_groups_raw = df.groupby(case_col)[act_col].apply(list).to_dict()
        variant_data = {} # signature -> {id, count, cases}
        
        for c_id, trace in case_groups_raw.items():
            sig = " -> ".join(trace)
            if sig not in variant_data:
                # Generate a stable numeric ID for the variant
                v_id = _generate_variant_id(sig)
                variant_data[sig] = {"id": v_id, "count": 0, "cases": []}
            variant_data[sig]["count"] += 1
            variant_data[sig]["cases"].append(str(c_id))

        # Map each edge and node to its containing variants
        edge_to_variants = {}
        node_to_variants = {}
        
        for sig, data in variant_data.items():
            v_id = data["id"]
            count = data["count"]
            cases = data["cases"]
            trace = sig.split(" -> ")
            
            # Node mapping
            unique_activities = set(trace)
            for act in unique_activities:
                if act not in node_to_variants:
                    node_to_variants[act] = []
                node_to_variants[act].append({
                    "id": v_id,
                    "signature": sig,
                    "count": count,
                    "cases": cases
                })
            
            # Edge mapping
            for i in range(len(trace) - 1):
                edge = (trace[i], trace[i+1])
                if edge not in edge_to_variants:
                    edge_to_variants[edge] = []
                edge_to_variants[edge].append({
                    "id": v_id,
                    "signature": sig, 
                    "count": count,
                    "cases": cases
                })
            
            # Start/End transitions
            if trace:
                start_edge = ("__START__", trace[0])
                if start_edge not in edge_to_variants:
                    edge_to_variants[start_edge] = []
                edge_to_variants[start_edge].append({
                    "id": v_id,
                    "signature": sig, 
                    "count": count,
                    "cases": cases
                })
                
                end_edge = (trace[-1], "__END__")
                if end_edge not in edge_to_variants:
                    edge_to_variants[end_edge] = []
                edge_to_variants[end_edge].append({
                    "id": v_id,
                    "signature": sig, 
                    "count": count,
                    "cases": cases
                })

        nodes_dict = {}
        edges = []
        
        # Add virtual START node
        nodes_dict["__START__"] = {
            "id": "__START__", 
            "label": "Start", 
            "type": "start",
            "count": total_cases
        }
        for act, count in start_activities.items():
            edges.append({
                "source": "__START__",
                "target": act,
                "weight": count,
                "type": "virtual",
                "variants": sorted(edge_to_variants.get(("__START__", act), []), key=lambda x: x['count'], reverse=True)
            })
            
        # Add edges and track nodes
        for (source, target), count in dfg.items():
            if source not in nodes_dict:
                nodes_dict[source] = {
                    "id": source, 
                    "label": source, 
                    "type": "activity",
                    "count": node_counts.get(source, 0),
                    "variants": sorted(node_to_variants.get(source, []), key=lambda x: x['count'], reverse=True)
                }
            if target not in nodes_dict:
                nodes_dict[target] = {
                    "id": target, 
                    "label": target, 
                    "type": "activity",
                    "count": node_counts.get(target, 0),
                    "variants": sorted(node_to_variants.get(target, []), key=lambda x: x['count'], reverse=True)
                }
                
            edges.append({
                "source": source,
                "target": target,
                "weight": count,
                "type": "regular",
                "variants": sorted(edge_to_variants.get((source, target), []), key=lambda x: x['count'], reverse=True)
            })
            
        # Add virtual END node
        nodes_dict["__END__"] = {
            "id": "__END__", 
            "label": "End", 
            "type": "end",
            "count": total_cases
        }
        for act, count in end_activities.items():
            if act not in nodes_dict:
                nodes_dict[act] = {
                    "id": act, 
                    "label": act, 
                    "type": "activity",
                    "count": node_counts.get(act, 0)
                }
            edges.append({
                "source": act,
                "target": "__END__",
                "weight": count,
                "type": "virtual",
                "variants": sorted(edge_to_variants.get((act, "__END__"), []), key=lambda x: x['count'], reverse=True)
            })
                
        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges,
            "start_activities": list(start_activities.keys()),
            "end_activities": list(end_activities.keys())
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
        # Fallback to CSV (BEST model outputs CSV by default)
        csv_file = pred_file.replace(".json", ".csv")
        if os.path.exists(csv_file):
            try:
                preds_df = pd.read_csv(csv_file)
                # Ensure sequence is treated safely if empty
                if 'sequence' in preds_df.columns:
                    preds_df['sequence'] = preds_df['sequence'].fillna("")
                preds = preds_df.to_dict(orient="records")
            except Exception as e:
                return {"error": f"Could not read predictions CSV: {e}"}
        else:
            return {"error": "Predictions not found"}
    else:
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        
    # Read original dataset to compute variants
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return {"error": f"Could not read dataset: {str(e)}"}
        
    case_col = None
    case_patterns = ['CaseID', 'case:id', 'case:concept:name', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    for pattern in case_patterns:
        if pattern in df.columns:
            case_col = pattern
            break
            
    act_col = None
    act_patterns = ['Activity', 'concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
    for pattern in act_patterns:
        if pattern in df.columns:
            act_col = pattern
            break

    if not case_col or not act_col:
        return {"error": f"Could not find case or activity column in dataset. Found: {list(df.columns)}"}

    # Build a variants dict that can match any likely format.
    # Store both the raw key and the cleaned key so lookups are robust.
    variants = {}  # normalised_case_id -> variant_signature (full trace)
    case_groups = df.groupby(case_col)[act_col].apply(list).to_dict()
    for c_id, trace in case_groups.items():
        sig = " -> ".join(trace)
        raw_id = str(c_id)
        # Store under: raw form, stripped "Case "/"case " prefix, and digits-only variant
        variants[raw_id] = sig
        stripped = raw_id.replace("Case ", "").replace("case ", "").replace(" ", "_").strip()
        variants[stripped] = sig
        # Also store with the original separator style (e.g. "Case_1022")
        variants[raw_id.replace(" ", "_")] = sig

    variant_stats = {}    # variant_sig -> {total_cases: set of case_ids, correct, total_rows}
    prefix_stats = {}
    seen_case_variants: dict = {}  # case_id -> variant_sig (to avoid double-counting)

    overall_correct = 0
    total_preds = len(preds)

    for p in preds:
        c_id = str(p.get("case_id"))

        # Try multiple normalisation forms to find the variant
        var_sig = variants.get(c_id)
        if var_sig is None:
            var_sig = variants.get(c_id.replace("Case ", "").replace("case ", "").replace(" ", "_").strip())
        if var_sig is None:
            var_sig = "Unknown Variant"

        v_id = _generate_variant_id(var_sig)
        p["variant_id"] = v_id

        seq = p.get("sequence", "")
        if "prefix_length" in p:
            prefix_len = p["prefix_length"]
        else:
            try:
                import json as _json
                parsed_seq = _json.loads(seq) if seq else []
                prefix_len = len(parsed_seq) if isinstance(parsed_seq, list) else len([x for x in seq.split(",") if x.strip()])
            except Exception:
                prefix_len = len([x for x in seq.split(",") if x.strip()]) if seq else 1

        true_act = p.get("true_next_activity") or p.get("actual_next_activity")
        pred_act = p.get("predicted_next_activity")

        is_correct = (true_act == pred_act)
        if is_correct:
            overall_correct += 1

        if var_sig not in variant_stats:
            variant_stats[var_sig] = {"case_ids": set(), "correct": 0, "total_rows": 0}
        variant_stats[var_sig]["total_rows"] += 1
        if is_correct:
            variant_stats[var_sig]["correct"] += 1
        # Track unique cases per variant (not rows)
        variant_stats[var_sig]["case_ids"].add(c_id)
        seen_case_variants[c_id] = var_sig

        if prefix_len not in prefix_stats:
            prefix_stats[prefix_len] = {"total": 0, "correct": 0}
        prefix_stats[prefix_len]["total"] += 1
        if is_correct:
            prefix_stats[prefix_len]["correct"] += 1

    # Save updated predictions with variant IDs back to files
    try:
        try:
            with open(pred_file, 'w') as f:
                json.dump(preds, f, indent=2)
        except Exception:
            pass  # pred_file might not exist (CSV-only model)
        # Always update CSV
        csv_path = pred_file.replace(".json", ".csv")
        p_df = pd.DataFrame(preds)
        p_df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to update predictions with variant IDs: {e}")

    # Format output: count unique cases per variant (not rows)
    variant_list = []
    for var_sig, stats in variant_stats.items():
        unique_cases = len(stats["case_ids"])
        # Accuracy = correct rows / total rows (per-step accuracy)
        acc = (stats["correct"] / stats["total_rows"]) * 100 if stats["total_rows"] > 0 else 0
        variant_list.append({
            "id": _generate_variant_id(var_sig),
            "variant": var_sig,
            "total_cases_in_test": unique_cases,
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

    res = {
        "overall_accuracy": (overall_correct / total_preds * 100) if total_preds > 0 else 0,
        "total_dataset_cases": len(case_groups.keys()),
        "unique_variants": len(set(variants.values())),
        "total_test_events": total_preds,
        "total_test_cases": len(seen_case_variants),
        "variants": variant_list,
        "prefix_accuracy": prefix_list,
        "global_explanations": global_explanations
    }

    # Save to file
    try:
        global_res_path = os.path.join(artifacts_dir, "global_results.json")
        with open(global_res_path, 'w') as f:
            json.dump(res, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save global_results.json: {e}")

    return res
