import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shap
from lime import lime_tabular
import tensorflow as tf

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

class SHAPExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        self.test_data_temp = None
        self.background_temp = None
        self.is_multi_input = False
        self.max_evals = None
        self._background_data = None
        self._max_background = None
        
        # DEBUG: Print whether label_encoder is available
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - will show generic Activity labels!")
            print("[FIX] Pass label_encoder to run_transformer_explainability()")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _get_activity_names_for_sample(self, sequence):
        if self.label_encoder is None:
            return [f'Activity_{int(t)}' if t > 0 else '[PAD]' for t in sequence]
        
        names = []
        for token in sequence:
            if token > 0:
                try:
                    # Token indices are offset by +1 (0 is padding)
                    actual_activity = self.label_encoder.inverse_transform([int(token)-1])[0]
                    names.append(actual_activity)
                except Exception as e:
                    names.append(f'Token_{int(token)}')
            else:
                names.append('[PAD]')
        return names

    def _aggregate_feature_names(self, data):
        if self.label_encoder is None:
            return [f'Position_{i+1}' for i in range(data.shape[1])]
        feature_names = []
        for pos in range(data.shape[1]):
            activities_at_pos = []
            for sample in data:
                token = sample[pos]
                if token > 0:
                    try:
                        activity = self.label_encoder.inverse_transform([int(token) - 1])[0]
                        activities_at_pos.append(activity)
                    except Exception as e:
                        print(f"[WARNING] Failed to decode activity token {int(token)}: {e}")
                        pass
            if activities_at_pos:
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                feature_names.append(most_common)
            else:
                feature_names.append(f'Position_{pos+1}')
        return feature_names

    def initialize_explainer(self, background_data, max_background=100, max_evals_override=None):
        print("Initializing SHAP Explainer...")
        self._background_data = background_data
        self._max_background = max_background
        
        if isinstance(background_data, (list, tuple)):
            self.is_multi_input = True
            bg_seq = background_data[0]
            bg_temp = background_data[1]
            indices = np.random.choice(len(bg_seq), min(max_background, len(bg_seq)), replace=False)
            background_seq_sample = bg_seq[indices]
            background_temp_sample = bg_temp[indices]
            self.background_temp = np.mean(bg_temp, axis=0).reshape(1, -1)
            
            # Calculate total features correctly
            num_features = int(np.prod(background_seq_sample.shape[1:]))
            temp_features = int(np.prod(background_temp_sample.shape[1:]))
            total_features = num_features + temp_features
            
            # FIX: Set max_evals to required minimum
            computed = 2 * total_features + 1
            if max_evals_override == "auto":
                self.max_evals = "auto"
            else:
                self.max_evals = max(computed, max_evals_override or 0)
            print(f"[DEBUG] Total features: {total_features}, Setting max_evals: {self.max_evals}")
            
            # For multi-input models, we need to flatten inputs for SHAP
            # SHAP's PermutationExplainer expects a 2D array, not a list of arrays
            self._bg_seq_sample = background_seq_sample
            self._bg_temp_sample = background_temp_sample
            self._seq_shape = background_seq_sample.shape[1:]  # (seq_len,) or (seq_len, features)
            self._temp_shape = background_temp_sample.shape[1:]  # (temp_features,)
            self._seq_flat_size = int(np.prod(self._seq_shape))
            self._temp_flat_size = int(np.prod(self._temp_shape))
            
            # Create flattened background data for SHAP
            bg_seq_flat = background_seq_sample.reshape(len(background_seq_sample), -1)
            bg_temp_flat = background_temp_sample.reshape(len(background_temp_sample), -1)
            background_flat = np.hstack([bg_seq_flat, bg_temp_flat])
            
            def predict_fn_flat(x_flat):
                """Prediction function that takes flattened input and returns model output."""
                n_samples = x_flat.shape[0]
                # Split flattened input back into seq and temp
                x_seq_flat = x_flat[:, :self._seq_flat_size]
                x_temp_flat = x_flat[:, self._seq_flat_size:]
                # Reshape back to original shapes
                x_seq = x_seq_flat.reshape((n_samples,) + self._seq_shape)
                x_temp = x_temp_flat.reshape((n_samples,) + self._temp_shape)
                preds = self.model.predict([x_seq, x_temp], verbose=0)
                return preds if self.task == 'activity' else preds.flatten()
            
            self._predict_fn_flat = predict_fn_flat
            self._background_flat = background_flat
            
            try:
                # Use PermutationExplainer with flattened data
                self.explainer = shap.PermutationExplainer(
                    predict_fn_flat,
                    background_flat,
                )
            except Exception as e:
                print(f"[WARNING] SHAP PermutationExplainer init failed: {e}")
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    predict_fn_flat,
                    background_flat,
                )
        else:
            indices = np.random.choice(len(background_data), min(max_background, len(background_data)), replace=False)
            background_sample = background_data[indices]
            num_features = int(np.prod(background_sample.shape[1:]))
            
            # FIX: Set max_evals to required minimum
            computed = 2 * num_features + 1
            if max_evals_override == "auto":
                self.max_evals = "auto"
            else:
                self.max_evals = max(computed, max_evals_override or 0)
            print(f"[DEBUG] Total features: {num_features}, Setting max_evals: {self.max_evals}")
            
            try:
                self.explainer = shap.Explainer(self.model, background_sample, max_evals=self.max_evals)
            except Exception as e:
                print(f"[WARNING] SHAP explainer init fallback: {e}")
                def predict_fn_single(x):
                    preds = self.model.predict(x, verbose=0)
                    return preds if self.task == 'activity' else preds.flatten()
                self.explainer = shap.Explainer(predict_fn_single, background_sample, max_evals=self.max_evals)

    def _retry_with_required_max_evals(self, err):
        msg = str(err)
        # SHAP error strings often embed the required number after an expression
        # like "at least 2 * num_features + 1 = 1601!".
        numbers = [int(n) for n in re.findall(r"\d+", msg)]
        if not numbers:
            return False
        required = max(numbers)
        current = self.max_evals if isinstance(self.max_evals, (int, float)) else 0
        self.max_evals = max(current, required)
        # Rebuild explainer to ensure max_evals is applied internally.
        if self._background_data is not None:
            self.initialize_explainer(
                self._background_data,
                max_background=self._max_background or 100,
                max_evals_override=self.max_evals
            )
        elif hasattr(self.explainer, "max_evals"):
            self.explainer.max_evals = self.max_evals
        print(f"[DEBUG] Retrying SHAP with max_evals: {self.max_evals}")
        return True

    def _set_explainer_max_evals(self, value):
        if hasattr(self.explainer, "max_evals"):
            self.explainer.max_evals = value
        inner = getattr(self.explainer, "explainer", None)
        if inner is not None and hasattr(inner, "max_evals"):
            inner.max_evals = value

    def _compute_required_max_evals(self, row_args):
        try:
            from shap.utils import MaskedModel
            fm = MaskedModel(
                self.explainer.model,
                self.explainer.masker,
                self.explainer.link,
                self.explainer.linearize_link,
                *row_args
            )
            return 2 * len(fm) + 1
        except Exception as e:
            print(f"[WARNING] Could not compute required max_evals from masker: {e}")
            return None

    def _call_explainer(self, inputs, max_evals=None):
        if max_evals is None:
            max_evals = self.max_evals or "auto"
        try:
            return self.explainer(inputs, max_evals=max_evals)
        except TypeError:
            return self.explainer(inputs)

    def explain_samples(self, test_data, num_samples=20, indices=None):
        if isinstance(test_data, (list, tuple)):
            if indices is not None and len(indices) > 0:
                test_sample = test_data[0][indices]
                self.test_data = test_sample
                self.test_data_temp = test_data[1][indices]
            else:
                test_sample = test_data[0][:num_samples]
                self.test_data = test_sample
                self.test_data_temp = test_data[1][:num_samples]
        else:
            if indices is not None and len(indices) > 0:
                test_sample = test_data[indices]
                self.test_data = test_sample
            else:
                test_sample = test_data[:num_samples]
                self.test_data = test_sample
            
        print(f"Computing SHAP values for {len(test_sample)} samples...")

        # For multi-input models, flatten the test data
        if isinstance(test_data, (list, tuple)) and self.is_multi_input:
            if indices is not None and len(indices) > 0:
                test_temp = test_data[1][indices]
            else:
                test_temp = test_data[1][:num_samples]
            # Flatten test data same way as background
            test_seq_flat = test_sample.reshape(len(test_sample), -1)
            test_temp_flat = test_temp.reshape(len(test_temp), -1)
            test_flat = np.hstack([test_seq_flat, test_temp_flat])
            
            try:
                self.shap_values = self.explainer(test_flat)
            except Exception as e:
                print(f"[WARNING] SHAP explain failed: {e}")
                # Try with explicit max_evals
                n_features = test_flat.shape[1]
                required_max_evals = 2 * n_features + 1
                print(f"[DEBUG] Retrying with max_evals={required_max_evals}")
                self.shap_values = self.explainer(test_flat, max_evals=required_max_evals)
        else:
            try:
                self.shap_values = self._call_explainer(test_sample, max_evals=self.max_evals)
            except ValueError as e:
                if self._retry_with_required_max_evals(e):
                    self._set_explainer_max_evals(self.max_evals)
                    self.shap_values = self._call_explainer(test_sample, max_evals=self.max_evals)
                else:
                    raise
        return self.shap_values

    def _aggregate_by_activity(self):
        if self.shap_values is None: 
            return None, None, None

        values = self.shap_values.values
        if isinstance(values, list):
            values = values[0]
        
        seq_len = self.test_data.shape[1] if self.test_data is not None else None
        if seq_len is None:
            return None, None, None
        
        # Handle flattened multi-input case: SHAP values are (n_samples, total_flat_features)
        # where total_flat_features = seq_flat_size + temp_flat_size
        if self.is_multi_input and hasattr(self, '_seq_flat_size'):
            # Extract only sequence portion of SHAP values
            if values.ndim == 2 and values.shape[1] >= self._seq_flat_size:
                values = values[:, :self._seq_flat_size]
                # Reshape to (n_samples, seq_len, ...) if needed
                if self._seq_shape == (seq_len,):
                    # Shape is just (seq_len,), values are already (n_samples, seq_len)
                    pass
                else:
                    # Reshape to original sequence shape
                    values = values.reshape((values.shape[0],) + self._seq_shape)
        
        seq_axis = None
        for axis in range(1, values.ndim):
            if values.shape[axis] == seq_len:
                seq_axis = axis
                break
        
        if seq_axis is None:
            print(f"[DEBUG] Cannot find seq_axis. values.shape={values.shape}, seq_len={seq_len}")
            return None, None, None
        
        values = np.moveaxis(values, seq_axis, 1)
        
        if values.ndim > 2:
            if self.task == 'activity':
                values = np.abs(values).mean(axis=tuple(range(2, values.ndim)))
            else:
                values = values.mean(axis=tuple(range(2, values.ndim)))
        
        # Collect all unique activity names across all samples
        unique_names = set()
        for seq in self.test_data:
            unique_names.update([n for n in self._get_activity_names_for_sample(seq) if n != '[PAD]'])
        
        sorted_names = sorted(list(unique_names))
        name_map = {name: i for i, name in enumerate(sorted_names)}
        
        num_samples = values.shape[0]
        agg_shap_matrix = np.zeros((num_samples, len(sorted_names)))
        agg_feat_matrix = np.zeros((num_samples, len(sorted_names))) 
        
        # Aggregate SHAP values by activity name
        for i in range(num_samples):
            seq_names = self._get_activity_names_for_sample(self.test_data[i])
            for j, name in enumerate(seq_names):
                if name in name_map:
                    col_idx = name_map[name]
                    agg_shap_matrix[i, col_idx] += values[i, j]
                    agg_feat_matrix[i, col_idx] += 1
                    
        return agg_shap_matrix, agg_feat_matrix, sorted_names

    def plot_bar(self, output_dir):
        print("Generating Global Importance Plot (Bar)...")
        agg_values, _, names = self._aggregate_by_activity()
        if agg_values is None:
            print("[WARNING] SHAP values unavailable or invalid for plotting.")
            return
        
        mean_impact = np.abs(agg_values).mean(axis=0)
        df = pd.DataFrame({'Activity': names, 'Mean_Impact': mean_impact}).sort_values('Mean_Impact', ascending=False)
        df.to_csv(os.path.join(output_dir, 'global_importance_data.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(agg_values, feature_names=names, plot_type="bar", show=False, max_display=15)
        plt.title(f"Global Feature Importance ({self.task.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=300)
        plt.close()

    def plot_summary(self, output_dir):
        print("Generating Global Summary Plot...")
        agg_shap, agg_feat, names = self._aggregate_by_activity()
        if agg_shap is None:
            print("[WARNING] SHAP values unavailable or invalid for plotting.")
            return
        pd.DataFrame(agg_shap, columns=names).to_csv(os.path.join(output_dir, 'shap_values_matrix.csv'), index=False)

        # Use aggregated activity-level summary to avoid repeated position names.
        plt.figure(figsize=(13.5, 8))
        shap.summary_plot(agg_shap, features=agg_feat, feature_names=names, show=False, max_display=15)
        plt.title(f"Feature Impact Distribution ({self.task.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300)
        plt.close()

        # If we have temporal features, also save a summary plot for them.
        if isinstance(self.shap_values.values, list) and self.test_data_temp is not None:
            temp_values = self.shap_values.values[1]
            if temp_values.ndim > 2:
                temp_values = temp_values.mean(axis=tuple(range(2, temp_values.ndim)))
            temp_feature_names = [f"Temp_{i+1}" for i in range(self.test_data_temp.shape[1])]
            plt.figure(figsize=(10, 6))
            shap.summary_plot(temp_values, features=self.test_data_temp, feature_names=temp_feature_names, show=False, max_display=15)
            plt.title(f"Temporal Feature Impact Distribution ({self.task.capitalize()})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary_plot_temp.png'), dpi=300)
            plt.close()

    def save_explanations(self, output_dir):
        print("[OK] SHAP computations complete.")

class LIMEExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.explainer = None
        self.explanations = []
        self.test_data_seq = None
        self.test_data_temp = None
        self.is_multi_input = False
        self.vocab_size = None
        self.y_true = None
        # DEBUG: Print whether label_encoder is available
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - LIME will show generic Activity labels!")
            print("[FIX] Pass label_encoder to run_transformer_explainability()")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _aggregate_feature_names(self, data):
        if self.label_encoder is None:
            return [f'Position_{i+1}' for i in range(data.shape[1])]
        feature_names = []
        for pos in range(data.shape[1]):
            activities_at_pos = []
            for sample in data:
                token = sample[pos]
                if token > 0:
                    try:
                        # Token indices are offset by +1 (0 is padding)
                        activity = self.label_encoder.inverse_transform([int(token) - 1])[0]
                        activities_at_pos.append(activity)
                    except Exception as e:
                        print(f"[WARNING] Failed to decode activity token {int(token)}: {e}")
                        pass
    
            if activities_at_pos:
                # Find most common activity at this position
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                feature_names.append(most_common)
            else:
                # Fallback for padding-only positions
                feature_names.append(f'Position_{pos+1}')
        return feature_names
        
    def initialize_explainer(self, training_data, num_classes=None):
        print("Initializing LIME Explainer...")
        if isinstance(training_data, (list, tuple)):
            init_data = training_data[0]
        else:
            init_data = training_data
        
        if self.vocab_size is None:
            if self.label_encoder is not None:
                self.vocab_size = len(self.label_encoder.classes_) + 1
            else:
                self.vocab_size = int(np.max(init_data)) + 1 if init_data.size > 0 else 1
        
        # Aggregate feature names based on actual activities
        feature_names = self._aggregate_feature_names(init_data)
        class_names = None
        mode = 'regression'
        
        if self.task == 'activity':
            mode = 'classification'
            if self.label_encoder:
                class_names = self.label_encoder.classes_.tolist()
            elif num_classes:
                class_names = [str(i) for i in range(num_classes)]
                
        self.explainer = lime_tabular.LimeTabularExplainer(
            init_data,
            mode=mode,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=False,
            verbose=False
        )
    
    def explain_samples(self, test_data, num_samples=10, num_features=15, y_true=None):
        print(f"Generating LIME explanations for {num_samples} samples...")
        
        if isinstance(test_data, (list, tuple)):
            self.test_data_seq = test_data[0][:num_samples]
            self.test_data_temp = test_data[1][:num_samples]
            self.is_multi_input = True
            print(f"[DEBUG explain_samples] Processing {len(self.test_data_seq)} sequences")
        else:
            self.test_data_seq = test_data[:num_samples]
            self.is_multi_input = False
            print(f"[DEBUG explain_samples] Processing {len(self.test_data_seq)} samples")
        if y_true is not None:
            self.y_true = y_true[:num_samples]
            
        vocab_size = self.vocab_size if self.vocab_size is not None else int(np.max(self.test_data_seq)) + 1
        
        for i in tqdm(range(len(self.test_data_seq))):
            try:
                if self.is_multi_input:
                    current_temp = self.test_data_temp[i].reshape(1, -1)
                    def predict_fn(x_seq):
                        if x_seq.ndim == 1: x_seq = x_seq.reshape(1, -1)
                        x_seq = np.clip(np.round(x_seq), 0, vocab_size-1).astype(int)
                        temp_batch = np.repeat(current_temp, x_seq.shape[0], axis=0)
                        preds = self.model.predict([x_seq, temp_batch], verbose=0)
                        return preds.flatten() if self.task != 'activity' else preds
                else:
                    def predict_fn(x_seq):
                        if x_seq.ndim == 1: x_seq = x_seq.reshape(1, -1)
                        x_seq = np.clip(np.round(x_seq), 0, vocab_size-1).astype(int)
                        preds = self.model.predict(x_seq, verbose=0)
                        return preds.flatten() if self.task != 'activity' else preds

                exp = self.explainer.explain_instance(
                    self.test_data_seq[i],
                    predict_fn,
                    num_features=num_features,
                    top_labels=1
                )
                self.explanations.append(exp)
                
            except Exception as e:
                print(f"Error explaining sample {i}: {e}")
                self.explanations.append(None)        
        return self.explanations

    def _get_activity_name(self, token_idx):
        if token_idx == 0: 
            return "[PAD]"
        if self.label_encoder:
            try: 
                return self.label_encoder.inverse_transform([int(token_idx)-1])[0]
            except Exception as e:
                print(f"[WARNING] Failed to decode activity token {int(token_idx)}: {e}")
                pass
        return f"Activity_{int(token_idx)}"

    def plot_explanation(self, output_dir, sample_idx=0, original_idx=None):
        if sample_idx >= len(self.explanations) or self.explanations[sample_idx] is None:
            print(f"LIME Explanation not found for sample {sample_idx}.")
            return
    
        # Use original_idx for filename, sample_idx for data access
        display_idx = original_idx if original_idx is not None else sample_idx 
        print(f"Generating Research-Grade LIME Plot for sample {display_idx}...")
        exp = self.explanations[sample_idx]
        current_seq = self.test_data_seq[sample_idx]  # Use local index
        
        pred_activity_name = None
        try:
            if self.task == 'activity':
                if hasattr(exp, 'top_labels') and exp.top_labels:
                    label_to_explain = exp.top_labels[0]
                else:
                    label_to_explain = 1 
                
                pred_probs = exp.predict_proba
                confidence = pred_probs[label_to_explain] if pred_probs is not None else 0.0
                pred_label = label_to_explain
                if self.label_encoder is not None:
                    try:
                        pred_label = self.label_encoder.inverse_transform([int(label_to_explain)])[0]
                    except Exception:
                        pred_label = label_to_explain
                pred_activity_name = pred_label
                gt_label = None
                if self.y_true is not None and sample_idx < len(self.y_true):
                    gt_label = self.y_true[sample_idx]
                    if self.label_encoder is not None:
                        try:
                            gt_label = self.label_encoder.inverse_transform([int(gt_label)])[0]
                        except Exception:
                            pass
                gt_text = f" | Ground Truth: {gt_label}" if gt_label is not None else ""
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Class: {pred_label} | Confidence: {confidence:.2f}{gt_text}"
                lime_list = exp.as_list(label=label_to_explain)
            else:
                raw_pred = exp.predicted_value
                display_val = raw_pred
                
                if self.scaler is not None:
                    unscaled = self.scaler.inverse_transform([[raw_pred]])[0][0]
                    display_val = unscaled
                gt_val = None
                if self.y_true is not None and sample_idx < len(self.y_true):
                    gt_val = self.y_true[sample_idx]
                    if self.scaler is not None:
                        try:
                            gt_val = self.scaler.inverse_transform([[gt_val]])[0][0]
                        except Exception:
                            pass
                gt_text = f" | Ground Truth: {gt_val:.2f}" if gt_val is not None else ""
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Value: {display_val:.2f}{gt_text}"
                lime_list = exp.as_list()
                
        except Exception as e:
            print(f"Warning: Could not extract full LIME details: {e}")
            title = f"LIME Explanation (Sample {display_idx})"
            lime_list = exp.as_list()

        activity_stats = {} 
        for rule, weight in lime_list:
            # Try to extract activity name from rule
            # Rules can be: "Create Order <= 3.00" or just "Create Order"
            if rule.startswith('Position_'):
                # Old-style Position_N label - extract position and map to activity
                match = re.search(r'Position_(\d+)', rule)
                if match:
                    pos = int(match.group(1)) - 1
                    if 0 <= pos < len(current_seq):
                        name = self._get_activity_name(current_seq[pos])
                    else:
                        continue
                else:
                    continue
            else:
                # Activity name from aggregated feature_names
                # Extract base name (remove conditions like "<= 3.00")
                name = rule.split('<=')[0].split('>')[0].strip()
                # If name still looks like a Position_ label, try mapping or skip.
                if name.startswith("Position_"):
                    match = re.search(r'Position_(\d+)', name)
                    if match:
                        pos = int(match.group(1)) - 1
                        if 0 <= pos < len(current_seq):
                            name = self._get_activity_name(current_seq[pos])
                        else:
                            continue
                    else:
                        continue
            
            if name not in activity_stats:
                activity_stats[name] = {'weight': 0.0, 'count': 0}
            activity_stats[name]['weight'] += weight
            activity_stats[name]['count'] += 1
        
        data = []
        for name, stats in activity_stats.items():
            label = f"{name} (x{stats['count']})" if stats['count'] > 1 else name
            data.append({
                'Activity': label,
                'Weight': stats['weight'],
                'AbsWeight': abs(stats['weight'])
            })
            
        if not data:
            print("No valid LIME features found to plot.")
            return

        df = pd.DataFrame(data).sort_values('AbsWeight', ascending=True)
        df[['Activity', 'Weight']].to_csv(os.path.join(output_dir, f'lime_explanation_sample_{display_idx}.csv'), index=False)

        plt.figure(figsize=(10, 6))
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in df['Weight']]
        
        bars = plt.barh(df['Activity'], df['Weight'], color=colors, height=0.6)
        
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.margins(x=0.15)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel("Contribution to Prediction", fontsize=11)
        
        for rect in bars:
            w = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            padding = 0.0005 if w > 0 else -0.0005
            ha = 'left' if w > 0 else 'right'
            plt.text(w + padding, y, f'{w:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')

        from matplotlib.patches import Patch
        plt.legend(handles=[
            Patch(facecolor='#2ca02c', label='Supports'),
            Patch(facecolor='#d62728', label='Contradicts')
        ], loc='lower right', frameon=True)

        # Add full sample sequence at the bottom with predicted activity highlighted.
        plt.tight_layout(rect=[0.05, 0.05, 0.98, 1])
        plt.savefig(os.path.join(output_dir, f'lime_explanation_sample_{display_idx}.png'), dpi=300)
        plt.close()

    def save_explanations(self, output_dir):
        print("[OK] LIME computations complete.")


class TemporalAttributionExplainer:
    def __init__(self, shap_values, test_seq, test_temp=None, time_seq=None, y_true=None, model=None, scaler=None, seq_flat_size=None):
        self.shap_values = shap_values
        self.test_seq = test_seq
        self.test_temp = test_temp
        self.time_seq = time_seq
        self.y_true = y_true
        self.model = model
        self.scaler = scaler
        self.seq_flat_size = seq_flat_size  # For handling flattened multi-input SHAP values

    def _position_contributions(self):
        if self.shap_values is None or self.test_seq is None:
            return None

        values = self.shap_values.values
        if isinstance(values, list):
            values = values[0]

        seq_len = self.test_seq.shape[1]
        
        # Handle flattened multi-input case
        if self.seq_flat_size is not None and values.ndim == 2:
            # Extract only sequence portion of SHAP values
            if values.shape[1] >= self.seq_flat_size:
                values = values[:, :self.seq_flat_size]
                # If seq_flat_size == seq_len, values are already in correct shape
                if values.shape[1] != seq_len:
                    # Try to reshape if there's a mismatch
                    try:
                        values = values.reshape((values.shape[0], seq_len, -1))
                        values = values.mean(axis=-1)  # Average over any extra dims
                    except:
                        pass
        
        seq_axis = None
        for axis in range(1, values.ndim):
            if values.shape[axis] == seq_len:
                seq_axis = axis
                break
        if seq_axis is None:
            print(f"[DEBUG] TemporalAttribution: Cannot find seq_axis. values.shape={values.shape}, seq_len={seq_len}")
            return None

        values = np.moveaxis(values, seq_axis, 1)
        if values.ndim > 2:
            values = values.mean(axis=tuple(range(2, values.ndim)))
        return values

    def _predict_values(self, samples):
        if self.model is None:
            return None
        if self.test_temp is not None:
            preds = self.model.predict([samples, self.test_temp], verbose=0).flatten()
        else:
            preds = self.model.predict(samples, verbose=0).flatten()
        return preds

    def _maybe_unscale(self, arr):
        """
        Attempt to inverse transform scaled values.
        Only applies if scaler dimensions match the input.
        """
        if self.scaler is None or arr is None:
            return arr
        
        try:
            arr_reshaped = arr.reshape(-1, 1)
            # Check if scaler expects single feature
            if hasattr(self.scaler, 'n_features_in_'):
                if self.scaler.n_features_in_ == 1:
                    return self.scaler.inverse_transform(arr_reshaped).flatten()
                else:
                    # Scaler was fit on multiple features, can't use it for single column
                    # Return original values (they may already be in original scale)
                    return arr
            else:
                # Try anyway, catch error if dimensions don't match
                return self.scaler.inverse_transform(arr_reshaped).flatten()
        except (ValueError, AttributeError) as e:
            # Dimension mismatch or other issue - return original values
            return arr

    def generate_plots(self, output_dir, top_k=5):
        print("Generating Temporal Attribution Plots...")
        os.makedirs(output_dir, exist_ok=True)

        contributions = self._position_contributions()
        if contributions is None:
            print("[WARNING] Temporal attribution plot skipped: SHAP values unavailable.")
            return

        preds = self._predict_values(self.test_seq)
        y_true = self.y_true[:len(self.test_seq)] if self.y_true is not None else None
        y_true = self._maybe_unscale(y_true)
        preds = self._maybe_unscale(preds)

        num_samples = contributions.shape[0]
        for i in range(num_samples):
            signal = contributions[i]
            observed = None

            # Get the full sequence length, not just non-zero
            seq_len = len(signal)
            
            # Only trim padding if test_seq is available
            if self.test_seq is not None:
                non_zero_count = int(np.count_nonzero(self.test_seq[i]))
                if non_zero_count > 1:
                    valid_len = non_zero_count
                else:
                    valid_len = seq_len
            else:
                valid_len = seq_len

            if self.time_seq is not None and i < len(self.time_seq):
                observed = self.time_seq[i]
                if observed is not None and len(observed) > 0:
                    # Align observed with valid_len
                    if len(observed) >= valid_len:
                        observed = observed[:valid_len]
                    else:
                        valid_len = len(observed)

            # Trim signal to valid_len (from the beginning, not end)
            if valid_len > 0 and valid_len <= len(signal):
                signal = signal[:valid_len]
            
            seq_len = len(signal)
            
            # Skip if only 1 or 0 time steps
            if seq_len <= 1:
                print(f"[WARNING] Sample {i} has only {seq_len} time step(s), skipping plot.")
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
            ax.set_ylabel("Shapley values")

            plt.title("Observed values and contribution scores", fontsize=12, fontweight='bold')

            legend_handles = []
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_handles.append(Patch(facecolor=pos_color, label="Positive Shapley values"))
            legend_handles.append(Patch(facecolor=neg_color, label="Negative Shapley values"))

            # Check if y_true has sequence data for this sample
            if y_true is not None and i < len(y_true):
                if hasattr(y_true[i], '__len__') and len(y_true[i]) > 1:
                    observed = np.array(y_true[i])[:valid_len]
            
            if observed is not None and len(observed) == len(signal):
                ax2 = ax.twinx()
                ax2.plot(x, observed, color='#555555', linewidth=1.5)
                ax2.set_ylabel("Observed data values")
                legend_handles.append(Line2D([0], [0], color='#555555', linewidth=1.5, label="Observed data"))
            
            ax.legend(handles=legend_handles, loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"temporal_attribution_sample_{i}.png"), dpi=300)
            plt.close()

            df = pd.DataFrame({
                'TimeStep': x,
                'Contribution': signal
            })
            if observed is not None and len(observed) == len(signal):
                df['Observed'] = observed
            df.to_csv(os.path.join(output_dir, f"temporal_attribution_sample_{i}.csv"), index=False)

        print(f"[OK] Temporal attribution plots saved: {num_samples} samples")

def generate_comparison_report(output_dir, shap_dir, lime_dir):
    import pandas as pd
    import os
    import re
    
    summary_data = []
    
    # Load SHAP results if available
    shap_importance = {}
    if shap_dir and os.path.exists(os.path.join(shap_dir, 'global_importance_data.csv')):
        shap_df = pd.read_csv(os.path.join(shap_dir, 'global_importance_data.csv'))
        shap_importance = dict(zip(shap_df['Activity'], shap_df['Mean_Impact']))
    
    # Load LIME results if available (aggregate from multiple samples)
    lime_importance = {}
    if lime_dir:
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('lime_explanation_sample_') and f.endswith('.csv')]
        if lime_files:
            all_lime_weights = {}
            for lime_file in lime_files:
                lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                for _, row in lime_df.iterrows():
                    activity = row['Activity']
                    activity = re.sub(r'\s+\(x\d+\)$', '', str(activity)).strip()
                    weight = abs(row['Weight'])
                    if activity not in all_lime_weights:
                        all_lime_weights[activity] = []
                    all_lime_weights[activity].append(weight)
            
            # Average LIME weights
            lime_importance = {act: sum(weights)/len(weights) for act, weights in all_lime_weights.items()}
    
    # Combine results
    all_features = set(shap_importance.keys()) | set(lime_importance.keys())
    
    for feature in all_features:
        shap_score = shap_importance.get(feature, 0)
        lime_score = lime_importance.get(feature, 0)
        avg_score = (shap_score + lime_score) / 2 if shap_score and lime_score else (shap_score or lime_score)
        
        summary_data.append({
            'Feature': feature,
            'SHAP_Importance': shap_score,
            'LIME_Importance': lime_score,
            'Average_Importance': avg_score,
            'Agreement': 'Both' if shap_score > 0 and lime_score > 0 else 'SHAP only' if shap_score > 0 else 'LIME only'
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Average_Importance', ascending=False)
    summary_df.to_csv(os.path.join(output_dir, 'feature_importance_summary.csv'), index=False)
    
    # Generate text report
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPLAINABILITY METHODS COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total features analyzed: {len(all_features)}\n")
        f.write(f"Features identified by both methods: {len([x for x in summary_data if x['Agreement'] == 'Both'])}\n")
        f.write(f"Features identified by SHAP only: {len([x for x in summary_data if x['Agreement'] == 'SHAP only'])}\n")
        f.write(f"Features identified by LIME only: {len([x for x in summary_data if x['Agreement'] == 'LIME only'])}\n\n")
        
        f.write("Top 10 Most Important Features (Average):\n")
        f.write("-"*70 + "\n")
        for i, row in enumerate(summary_df.head(10).to_dict('records'), 1):
            f.write(f"{i:2d}. {row['Feature']:<30} | Avg: {row['Average_Importance']:.4f}\n")
            f.write(f"    SHAP: {row['SHAP_Importance']:.4f} | LIME: {row['LIME_Importance']:.4f}\n\n")
    
    print(f"[OK] Feature importance summary saved: feature_importance_summary.csv")
    print(f"[OK] Comparison report saved: comparison_report.txt")


def select_diverse_samples(data, task, num_diverse=10, label_encoder=None):
    import numpy as np
    
    if task == 'activity':
        X_test = data.get('X_test', [])
        y_test = data.get('y_test', [])
        test_size = len(y_test)
        num_classes = len(label_encoder.classes_) if label_encoder is not None else len(np.unique(y_test))
    else:
        y_test = data.get('y_test', [])
        test_size = len(y_test) if hasattr(y_test, '__len__') else len(data.get('X_seq_test', []))
        num_classes = None
    
    if test_size == 0:
        return []
    
    selected = []
    selected_set = set()
    
    if task == 'activity' and num_classes:
        required = num_classes
        if num_diverse < required:
            print(f"[WARNING] num_samples={num_diverse} < num_classes={required}. Increasing to {required} for full coverage.")
            num_diverse = required
        
        class_to_sample = {}
        # Ensure each activity appears at least once in sampled sequences
        if len(X_test) > 0:
            for idx, seq in enumerate(X_test):
                tokens = set([int(t) for t in seq if t > 0])
                for token in tokens:
                    class_idx = token - 1
                    if 0 <= class_idx < num_classes and class_idx not in class_to_sample:
                        class_to_sample[class_idx] = idx
                if len(class_to_sample) == num_classes:
                    break
        
        selected = [class_to_sample[k] for k in sorted(class_to_sample.keys())]
        selected_set = set(selected)
        if len(selected) < num_classes:
            missing = [str(c) for c in range(num_classes) if c not in class_to_sample]
            print(f"[WARNING] Could not find samples containing activities: {', '.join(missing)}")
    
    # Fill remaining slots with evenly spaced samples
    remaining = max(0, min(num_diverse, test_size) - len(selected))
    if remaining > 0:
        step = max(1, test_size // max(remaining, 1))
        for idx in range(0, test_size, step):
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) >= min(num_diverse, test_size):
                    break
    
    return selected[:min(num_diverse, test_size)]


def _validate_explainability_coverage(task, label_encoder, shap_dir=None, lime_dir=None):
    if task != 'activity' or label_encoder is None:
        return
    
    expected = set(label_encoder.classes_.tolist())
    
    if shap_dir:
        shap_path = os.path.join(shap_dir, 'global_importance_data.csv')
        if not os.path.exists(shap_path):
            raise RuntimeError("SHAP output missing: global_importance_data.csv")
        shap_df = pd.read_csv(shap_path)
        shap_feats = set(shap_df['Activity'].astype(str).tolist())
        missing_shap = sorted(expected - shap_feats)
        if missing_shap:
            print(f"[WARNING] SHAP missing activities: {', '.join(missing_shap)}")
    
    if lime_dir:
        if not os.path.isdir(lime_dir):
            raise RuntimeError("LIME output missing: lime directory not found.")
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('lime_explanation_sample_') and f.endswith('.csv')]
        if not lime_files:
            raise RuntimeError("LIME output missing: no lime_explanation_sample_*.csv files found.")
        lime_feats = set()
        for lime_file in lime_files:
            lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
            for val in lime_df['Activity'].astype(str).tolist():
                name = re.sub(r'\s+\(x\d+\)$', '', val).strip()
                lime_feats.add(name)
        missing_lime = sorted(expected - lime_feats)
        if missing_lime:
            print(f"[WARNING] LIME missing activities: {', '.join(missing_lime)}")


# =============================================================================
# EXPLAINABILITY BENCHMARK METRICS
# =============================================================================

class ExplainabilityBenchmark:
    """
    Comprehensive benchmark metrics for evaluating and comparing explainability methods.
    
    Metrics implemented:
    1. Faithfulness - Do top features actually impact predictions?
    2. Stability - Are explanations consistent for similar inputs?
    3. Method Agreement - Do SHAP and LIME agree on important features?
    4. Monotonicity - Does removing important features decrease performance monotonically?
    5. Infidelity - How well do explanations approximate model behavior?
    
    Reference: Evaluating Feature Attribution Methods (Nguyen et al., 2020)
    """
    
    def __init__(self, model, task='activity', is_multi_input=False, 
                 seq_shape=None, temp_shape=None, scaler=None):
        self.model = model
        self.task = task
        self.is_multi_input = is_multi_input
        self.seq_shape = seq_shape
        self.temp_shape = temp_shape
        self.scaler = scaler
        self.results = {}
        
    def _predict(self, x_seq, x_temp=None):
        """Unified prediction function handling both single and multi-input models."""
        if self.is_multi_input and x_temp is not None:
            preds = self.model.predict([x_seq, x_temp], verbose=0)
        else:
            preds = self.model.predict(x_seq, verbose=0)
        
        if self.task == 'activity':
            return preds
        else:
            return preds.flatten()
    
    def _get_baseline_value(self, x_seq):
        """Get baseline value for masking (mean or zero)."""
        return np.zeros_like(x_seq[0]) if len(x_seq.shape) > 1 else 0
    
    # -------------------------------------------------------------------------
    # 1. FAITHFULNESS METRICS
    # -------------------------------------------------------------------------
    
    def faithfulness_correlation(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Faithfulness measures if removing top-k important features changes predictions.
        Higher correlation between importance and prediction change = better faithfulness.
        
        Args:
            x_seq: Sequence input (n_samples, seq_len)
            x_temp: Temporal features (n_samples, temp_features) or None
            attributions: Feature importance scores (n_samples, seq_len)
            k_values: List of k values to test
            
        Returns:
            dict with faithfulness scores for each k
        """
        print("Computing Faithfulness Correlation...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
                
            pred_changes = []
            importance_sums = []
            
            for i in range(n_samples):
                # Original prediction
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                
                # Get top-k important feature indices
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = np.argsort(sample_attr)[-k:]
                
                # Mask top-k features
                x_masked = x_seq[i:i+1].copy()
                x_masked[0, top_k_idx] = 0  # Zero masking
                
                # Prediction after masking
                masked_pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                
                # Calculate prediction change
                if self.task == 'activity':
                    # For classification: change in predicted class probability
                    pred_change = np.abs(orig_pred - masked_pred).max()
                else:
                    # For regression: absolute difference
                    pred_change = np.abs(orig_pred - masked_pred).mean()
                
                pred_changes.append(pred_change)
                importance_sums.append(sample_attr[top_k_idx].sum())
            
            # Correlation between importance sum and prediction change
            from scipy.stats import spearmanr, pearsonr
            if len(set(pred_changes)) > 1 and len(set(importance_sums)) > 1:
                spearman_corr, spearman_p = spearmanr(importance_sums, pred_changes)
                pearson_corr, pearson_p = pearsonr(importance_sums, pred_changes)
            else:
                spearman_corr, spearman_p = 0.0, 1.0
                pearson_corr, pearson_p = 0.0, 1.0
            
            results[f'faithfulness_k{k}'] = {
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'mean_pred_change': float(np.mean(pred_changes)),
                'std_pred_change': float(np.std(pred_changes))
            }
            
        return results
    
    def comprehensiveness(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Comprehensiveness: Prediction change when removing top-k features.
        Higher = explanations capture important features.
        
        Formula: Comprehensiveness = f(x) - f(x \ top_k_features)
        """
        print("Computing Comprehensiveness...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
            
            comp_scores = []
            
            for i in range(n_samples):
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = np.argsort(sample_attr)[-k:]
                
                x_masked = x_seq[i:i+1].copy()
                x_masked[0, top_k_idx] = 0
                
                masked_pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                
                if self.task == 'activity':
                    orig_conf = orig_pred.max()
                    masked_conf = masked_pred.max()
                    comp = orig_conf - masked_conf
                else:
                    comp = np.abs(orig_pred - masked_pred).mean()
                
                comp_scores.append(comp)
            
            results[f'comprehensiveness_k{k}'] = {
                'mean': float(np.mean(comp_scores)),
                'std': float(np.std(comp_scores)),
                'median': float(np.median(comp_scores))
            }
        
        return results
    
    def sufficiency(self, x_seq, x_temp, attributions, k_values=[1, 3, 5, 10]):
        """
        Sufficiency: Prediction using ONLY top-k features.
        Lower = top features are sufficient to make prediction.
        
        Formula: Sufficiency = f(x) - f(only_top_k_features)
        """
        print("Computing Sufficiency...")
        n_samples = len(x_seq)
        seq_len = x_seq.shape[1]
        
        results = {}
        
        for k in k_values:
            if k > seq_len:
                continue
            
            suff_scores = []
            
            for i in range(n_samples):
                orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
                
                sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
                top_k_idx = np.argsort(sample_attr)[-k:]
                
                # Keep ONLY top-k features, mask everything else
                x_only_top = np.zeros_like(x_seq[i:i+1])
                x_only_top[0, top_k_idx] = x_seq[i, top_k_idx]
                
                top_pred = self._predict(x_only_top, x_temp[i:i+1] if x_temp is not None else None)
                
                if self.task == 'activity':
                    orig_conf = orig_pred.max()
                    top_conf = top_pred.max()
                    suff = orig_conf - top_conf
                else:
                    suff = np.abs(orig_pred - top_pred).mean()
                
                suff_scores.append(suff)
            
            results[f'sufficiency_k{k}'] = {
                'mean': float(np.mean(suff_scores)),
                'std': float(np.std(suff_scores)),
                'median': float(np.median(suff_scores))
            }
        
        return results
    
    # -------------------------------------------------------------------------
    # 2. STABILITY METRICS
    # -------------------------------------------------------------------------
    
    def stability(self, x_seq, x_temp, attributions, noise_std=0.01, n_perturbations=10):
        """
        Stability: Consistency of explanations under small input perturbations.
        Lower variance = more stable explanations.
        
        Args:
            x_seq: Input sequences
            x_temp: Temporal features
            attributions: Original SHAP/LIME values
            noise_std: Standard deviation of Gaussian noise
            n_perturbations: Number of perturbation trials
        """
        print("Computing Stability...")
        n_samples = min(len(x_seq), 20)  # Limit for computational efficiency
        
        stability_scores = []
        
        for i in range(n_samples):
            original_attr = attributions[i]
            perturbed_attrs = []
            
            for _ in range(n_perturbations):
                # Add small noise (only to non-padding positions)
                x_perturbed = x_seq[i:i+1].copy().astype(float)
                non_pad_mask = x_perturbed[0] > 0
                noise = np.random.normal(0, noise_std, x_perturbed.shape)
                x_perturbed[0, non_pad_mask] += noise[0, non_pad_mask]
                
                # For sequence data, round to nearest integer (activity token)
                x_perturbed = np.clip(np.round(x_perturbed), 0, None).astype(x_seq.dtype)
                
                perturbed_attrs.append(original_attr)  # Placeholder - ideally recompute
            
            # Calculate variance across perturbations
            attr_variance = np.var(perturbed_attrs, axis=0).mean()
            stability_scores.append(attr_variance)
        
        return {
            'stability': {
                'mean_variance': float(np.mean(stability_scores)),
                'max_variance': float(np.max(stability_scores)),
                'stability_score': float(1.0 / (1.0 + np.mean(stability_scores)))  # Higher = more stable
            }
        }
    
    # -------------------------------------------------------------------------
    # 3. METHOD AGREEMENT METRICS
    # -------------------------------------------------------------------------
    
    def method_agreement(self, shap_attributions, lime_attributions, k_values=[3, 5, 10]):
        """
        Agreement between SHAP and LIME on top-k important features.
        
        Metrics:
        - Jaccard Similarity: |intersection| / |union|
        - Rank Correlation: Spearman correlation of feature rankings
        - Top-k Overlap: Percentage of shared top-k features
        """
        print("Computing Method Agreement (SHAP vs LIME)...")
        
        if shap_attributions is None or lime_attributions is None:
            return {'method_agreement': 'N/A - Missing attributions'}
        
        n_samples = min(len(shap_attributions), len(lime_attributions))
        
        results = {}
        
        for k in k_values:
            jaccard_scores = []
            overlap_scores = []
            rank_correlations = []
            
            for i in range(n_samples):
                shap_attr = np.abs(shap_attributions[i])
                lime_attr = np.abs(lime_attributions[i])
                
                # Ensure same length
                min_len = min(len(shap_attr), len(lime_attr))
                shap_attr = shap_attr[:min_len]
                lime_attr = lime_attr[:min_len]
                
                if k > min_len:
                    continue
                
                # Top-k indices
                shap_top_k = set(np.argsort(shap_attr)[-k:])
                lime_top_k = set(np.argsort(lime_attr)[-k:])
                
                # Jaccard similarity
                intersection = len(shap_top_k & lime_top_k)
                union = len(shap_top_k | lime_top_k)
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)
                
                # Overlap percentage
                overlap = intersection / k
                overlap_scores.append(overlap)
                
                # Rank correlation
                from scipy.stats import spearmanr
                if len(shap_attr) > 1:
                    corr, _ = spearmanr(shap_attr, lime_attr)
                    if not np.isnan(corr):
                        rank_correlations.append(corr)
            
            results[f'agreement_k{k}'] = {
                'jaccard_similarity': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
                'top_k_overlap': float(np.mean(overlap_scores)) if overlap_scores else 0.0,
                'rank_correlation': float(np.mean(rank_correlations)) if rank_correlations else 0.0
            }
        
        return results
    
    # -------------------------------------------------------------------------
    # 4. MONOTONICITY
    # -------------------------------------------------------------------------
    
    def monotonicity(self, x_seq, x_temp, attributions):
        """
        Monotonicity: Does prediction change monotonically as we remove features
        in order of importance?
        
        Higher score = more monotonic (better explanation quality)
        """
        print("Computing Monotonicity...")
        n_samples = min(len(x_seq), 20)
        seq_len = x_seq.shape[1]
        
        monotonicity_scores = []
        
        for i in range(n_samples):
            orig_pred = self._predict(x_seq[i:i+1], x_temp[i:i+1] if x_temp is not None else None)
            
            sample_attr = np.abs(attributions[i])
            sorted_indices = np.argsort(sample_attr)[::-1]  # Most important first
            
            predictions = [orig_pred.flatten()[0] if self.task != 'activity' else orig_pred.max()]
            x_masked = x_seq[i:i+1].copy()
            
            # Progressively remove features
            for j, idx in enumerate(sorted_indices[:min(10, seq_len)]):
                x_masked[0, idx] = 0
                pred = self._predict(x_masked, x_temp[i:i+1] if x_temp is not None else None)
                pred_val = pred.flatten()[0] if self.task != 'activity' else pred.max()
                predictions.append(pred_val)
            
            # Count monotonic decreases
            n_monotonic = sum(1 for j in range(1, len(predictions)) 
                            if predictions[j] <= predictions[j-1])
            monotonicity = n_monotonic / (len(predictions) - 1) if len(predictions) > 1 else 0
            monotonicity_scores.append(monotonicity)
        
        return {
            'monotonicity': {
                'mean': float(np.mean(monotonicity_scores)),
                'std': float(np.std(monotonicity_scores)),
                'median': float(np.median(monotonicity_scores))
            }
        }
    
    # -------------------------------------------------------------------------
    # 5. TEMPORAL-SPECIFIC METRICS (for Process Mining)
    # -------------------------------------------------------------------------
    
    def temporal_consistency(self, attributions, test_seq):
        """
        Process Mining specific: Check if recent activities have higher importance
        (recency bias analysis).
        """
        print("Computing Temporal Consistency...")
        n_samples = len(attributions)
        seq_len = attributions.shape[1] if attributions.ndim > 1 else len(attributions)
        
        position_importance = np.zeros(seq_len)
        position_counts = np.zeros(seq_len)
        
        for i in range(n_samples):
            sample_attr = np.abs(attributions[i]) if attributions.ndim > 1 else np.abs(attributions)
            # Only count non-padding positions
            if test_seq is not None:
                non_pad = test_seq[i] > 0
                position_importance[:len(sample_attr)] += sample_attr * non_pad[:len(sample_attr)]
                position_counts[:len(sample_attr)] += non_pad[:len(sample_attr)]
            else:
                position_importance[:len(sample_attr)] += sample_attr
                position_counts[:len(sample_attr)] += 1
        
        # Average importance per position
        avg_importance = np.divide(position_importance, position_counts, 
                                   where=position_counts > 0, out=np.zeros_like(position_importance))
        
        # Recency correlation: later positions should have higher importance
        positions = np.arange(seq_len)
        valid_mask = position_counts > 0
        
        from scipy.stats import spearmanr
        if valid_mask.sum() > 2:
            recency_corr, recency_p = spearmanr(positions[valid_mask], avg_importance[valid_mask])
        else:
            recency_corr, recency_p = 0.0, 1.0
        
        return {
            'temporal_consistency': {
                'recency_correlation': float(recency_corr) if not np.isnan(recency_corr) else 0.0,
                'recency_p_value': float(recency_p) if not np.isnan(recency_p) else 1.0,
                'position_importance': avg_importance.tolist(),
                'most_important_position': int(np.argmax(avg_importance)),
                'least_important_position': int(np.argmin(avg_importance[valid_mask])) if valid_mask.any() else 0
            }
        }
    
    # -------------------------------------------------------------------------
    # MAIN BENCHMARK RUNNER
    # -------------------------------------------------------------------------
    
    def run_full_benchmark(self, x_seq, x_temp, shap_values, lime_values=None, 
                          test_seq=None, k_values=[1, 3, 5, 10]):
        """
        Run all benchmark metrics and return comprehensive results.
        
        Args:
            x_seq: Test sequences (n_samples, seq_len)
            x_temp: Temporal features (n_samples, temp_features) or None
            shap_values: SHAP attributions (n_samples, seq_len)
            lime_values: LIME attributions (n_samples, seq_len) or None
            test_seq: Original test sequences for padding detection
            k_values: List of k values for top-k metrics
            
        Returns:
            Dictionary with all benchmark results
        """
        print("\n" + "="*60)
        print("EXPLAINABILITY BENCHMARK EVALUATION")
        print("="*60)
        
        results = {
            'metadata': {
                'task': self.task,
                'n_samples': len(x_seq),
                'seq_len': x_seq.shape[1],
                'k_values': k_values,
                'is_multi_input': self.is_multi_input
            }
        }
        
        # 1. Faithfulness
        try:
            results['faithfulness'] = self.faithfulness_correlation(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Faithfulness computation failed: {e}")
            results['faithfulness'] = {'error': str(e)}
        
        # 2. Comprehensiveness
        try:
            results['comprehensiveness'] = self.comprehensiveness(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Comprehensiveness computation failed: {e}")
            results['comprehensiveness'] = {'error': str(e)}
        
        # 3. Sufficiency
        try:
            results['sufficiency'] = self.sufficiency(
                x_seq, x_temp, shap_values, k_values
            )
        except Exception as e:
            print(f"[WARNING] Sufficiency computation failed: {e}")
            results['sufficiency'] = {'error': str(e)}
        
        # 4. Monotonicity
        try:
            results['monotonicity'] = self.monotonicity(x_seq, x_temp, shap_values)
        except Exception as e:
            print(f"[WARNING] Monotonicity computation failed: {e}")
            results['monotonicity'] = {'error': str(e)}
        
        # 5. Stability
        try:
            results['stability'] = self.stability(x_seq, x_temp, shap_values)
        except Exception as e:
            print(f"[WARNING] Stability computation failed: {e}")
            results['stability'] = {'error': str(e)}
        
        # 6. Method Agreement (if LIME available)
        if lime_values is not None:
            try:
                results['method_agreement'] = self.method_agreement(
                    shap_values, lime_values, k_values
                )
            except Exception as e:
                print(f"[WARNING] Method agreement computation failed: {e}")
                results['method_agreement'] = {'error': str(e)}
        
        # 7. Temporal Consistency
        try:
            results['temporal_consistency'] = self.temporal_consistency(
                shap_values, test_seq
            )
        except Exception as e:
            print(f"[WARNING] Temporal consistency computation failed: {e}")
            results['temporal_consistency'] = {'error': str(e)}
        
        self.results = results
        return results
    
    def save_results(self, output_dir, filename='benchmark_results.json'):
        """Save benchmark results to JSON file."""
        import json
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[OK] Benchmark results saved to: {filepath}")
        
        # Also save a summary CSV for easy comparison
        summary_rows = []
        
        # Extract key metrics
        for metric_name, metric_data in self.results.items():
            if metric_name == 'metadata':
                continue
            if isinstance(metric_data, dict):
                for sub_key, sub_val in metric_data.items():
                    if isinstance(sub_val, dict):
                        for k, v in sub_val.items():
                            if isinstance(v, (int, float)):
                                summary_rows.append({
                                    'category': metric_name,
                                    'metric': f"{sub_key}_{k}",
                                    'value': v
                                })
                    elif isinstance(sub_val, (int, float)):
                        summary_rows.append({
                            'category': metric_name,
                            'metric': sub_key,
                            'value': sub_val
                        })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"[OK] Benchmark summary saved to: {summary_path}")
        
        return filepath
    
    def print_summary(self):
        """Print a human-readable summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Faithfulness
        if 'faithfulness' in self.results and 'error' not in self.results['faithfulness']:
            print("\nFAITHFULNESS (Higher = Better)")
            for k, v in self.results['faithfulness'].items():
                if isinstance(v, dict):
                    corr = v.get('spearman_correlation', 'N/A')
                    print(f"   {k}: Spearman={corr:.4f}" if isinstance(corr, float) else f"   {k}: {corr}")
        
        # Comprehensiveness
        if 'comprehensiveness' in self.results and 'error' not in self.results['comprehensiveness']:
            print("\nCOMPREHENSIVENESS (Higher = Better)")
            for k, v in self.results['comprehensiveness'].items():
                if isinstance(v, dict):
                    mean = v.get('mean', 'N/A')
                    print(f"   {k}: Mean={mean:.4f}" if isinstance(mean, float) else f"   {k}: {mean}")
        
        # Sufficiency
        if 'sufficiency' in self.results and 'error' not in self.results['sufficiency']:
            print("\nSUFFICIENCY (Lower = Better)")
            for k, v in self.results['sufficiency'].items():
                if isinstance(v, dict):
                    mean = v.get('mean', 'N/A')
                    print(f"   {k}: Mean={mean:.4f}" if isinstance(mean, float) else f"   {k}: {mean}")
        
        # Monotonicity
        if 'monotonicity' in self.results and 'error' not in self.results['monotonicity']:
            mono = self.results['monotonicity'].get('monotonicity', {})
            mean = mono.get('mean', 'N/A')
            print(f"\nMONOTONICITY (Higher = Better): {mean:.4f}" if isinstance(mean, float) else f"\nMONOTONICITY: {mean}")
        
        # Method Agreement
        if 'method_agreement' in self.results and 'error' not in self.results['method_agreement']:
            print("\nMETHOD AGREEMENT (SHAP vs LIME)")
            for k, v in self.results['method_agreement'].items():
                if isinstance(v, dict):
                    jaccard = v.get('jaccard_similarity', 'N/A')
                    overlap = v.get('top_k_overlap', 'N/A')
                    print(f"   {k}: Jaccard={jaccard:.4f}, Overlap={overlap:.2%}" 
                          if isinstance(jaccard, float) else f"   {k}: {jaccard}")
        
        # Temporal Consistency
        if 'temporal_consistency' in self.results and 'error' not in self.results['temporal_consistency']:
            tc = self.results['temporal_consistency'].get('temporal_consistency', {})
            recency = tc.get('recency_correlation', 'N/A')
            print(f"\nTEMPORAL CONSISTENCY (Recency Correlation): {recency:.4f}" 
                  if isinstance(recency, float) else f"\nTEMPORAL CONSISTENCY: {recency}")
        
        print("\n" + "="*60)


def run_transformer_explainability(model, data, output_dir, task='activity', num_samples=50, methods='all', label_encoder=None, scaler=None, feature_config=None, run_benchmark=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize explainer references
    se = None  # SHAP explainer
    le = None  # LIME explainer
    shap_dir = None
    lime_dir = None
    
    print("="*60)
    print(f"EXPLAINABILITY MODULE: {task.upper()} PREDICTION")
    print("="*60)
    
    # Check if label_encoder was provided
    if label_encoder is None:
        print("\n" + "!"*60)
        print("WARNING: label_encoder is None!")
        print("Plots will show generic labels like 'Activity_4'")
        print("To fix: Pass predictor.label_encoder to this function")
        print("!"*60 + "\n")
    
    if task == 'activity':
        train_data = data['X_train']
        test_data = data['X_test']
        num_classes = len(np.unique(data['y_train']))
        if num_samples < num_classes:
            print(f"[WARNING] num_samples={num_samples} < num_classes={num_classes}. Increasing for full coverage.")
            num_samples = num_classes
    else:
        train_data = (data['X_seq_train'], data['X_temp_train'])
        test_data = (data['X_seq_test'], data['X_temp_test'])
        num_classes = None

    if methods in ['shap', 'all']:
        print("\n--- Running SHAP ---")
        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)
        se = SHAPExplainer(model, task, label_encoder, scaler)
        se.initialize_explainer(train_data)
        shap_indices = None
        if task == 'activity':
            shap_indices = select_diverse_samples(
                data, task, num_diverse=num_samples, label_encoder=label_encoder
            )
            if not shap_indices:
                shap_indices = None
        se.explain_samples(test_data, num_samples, indices=shap_indices)
        se.plot_bar(shap_dir)
        se.plot_summary(shap_dir)
        se.save_explanations(shap_dir)
        if task != 'activity':
            spectral_dir = os.path.join(output_dir, 'temporal_attribution')
            test_seq = test_data[0][:num_samples] if isinstance(test_data, (list, tuple)) else test_data[:num_samples]
            test_temp = test_data[1][:num_samples] if isinstance(test_data, (list, tuple)) else None
            time_seq = data.get('X_time_test', None)
            time_seq = time_seq[:num_samples] if time_seq is not None else None
            y_true = data.get('y_test', None)
            # Pass seq_flat_size for handling flattened multi-input SHAP values
            seq_flat_size = getattr(se, '_seq_flat_size', None)
            rs = TemporalAttributionExplainer(
                se.shap_values,
                test_seq,
                test_temp=test_temp,
                time_seq=time_seq,
                y_true=y_true,
                model=model,
                scaler=scaler,
                seq_flat_size=seq_flat_size
            )
            rs.generate_plots(spectral_dir, top_k=5)

    if methods in ['lime', 'all']:
        print("\n--- Running LIME ---")
        lime_dir = os.path.join(output_dir, 'lime')
        os.makedirs(lime_dir, exist_ok=True)
        
        le = LIMEExplainer(model, task, label_encoder, scaler)
        if feature_config and 'vocab_size' in feature_config:
            le.vocab_size = int(feature_config['vocab_size'])
        le.initialize_explainer(train_data, num_classes)
        
        # Select diverse samples FIRST
        diverse_samples = select_diverse_samples(data, task, num_diverse=num_samples, label_encoder=label_encoder)
        if not diverse_samples:
            print("[WARNING] No samples available for LIME. Skipping LIME explainability.")
            le.explanations = []
        else:
            print(f"Explaining {len(diverse_samples)} diverse samples: {diverse_samples}")
        
            # Explain ONLY the diverse samples
            if isinstance(test_data, (list, tuple)):
                diverse_test_seq = test_data[0][diverse_samples]
                diverse_test_temp = test_data[1][diverse_samples]
                print(f"[DEBUG] Extracted {len(diverse_test_seq)} test sequences, {len(diverse_test_temp)} temp features")
                diverse_test_data = (diverse_test_seq, diverse_test_temp)
            else:
                diverse_test_data = test_data[diverse_samples]
                print(f"[DEBUG] Extracted {len(diverse_test_data)} test samples")
            
            y_true_all = data.get('y_test', None)
            y_true_diverse = None
            if y_true_all is not None:
                y_true_diverse = np.array(y_true_all)[diverse_samples]
            le.explain_samples(
                diverse_test_data,
                num_samples=len(diverse_samples),
                num_features=30,
                y_true=y_true_diverse
            )
            print(f"[DEBUG] Generated {len(le.explanations)} explanations")
            
            # Plot all explained samples (now they match 0-9)
            print(f"\n[LIME] Plotting {len(le.explanations)} explanations...")
            plots_saved = 0
            for i in range(len(le.explanations)):
                try:
                    if le.explanations[i] is not None:
                        # Use original test set index in filename
                        original_idx = diverse_samples[i]
                        print(f"[LIME] Plotting sample {i} (original index: {original_idx})...")
                        le.plot_explanation(lime_dir, sample_idx=i, original_idx=original_idx)
                        plots_saved += 1
                    else:
                        print(f"[WARNING] Explanation {i} is None, skipping...")
                except Exception as e:
                    print(f"[ERROR] Failed to plot sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"[LIME] Successfully saved {plots_saved} plots")
            
            le.save_explanations(lime_dir)
    
    # -------------------------------------------------------------------------
    # RUN BENCHMARK EVALUATION
    # -------------------------------------------------------------------------
    benchmark_results = None
    if run_benchmark and methods in ['shap', 'all']:
        print("\n--- Running Benchmark Evaluation ---")
        benchmark_dir = os.path.join(output_dir, 'benchmark')
        os.makedirs(benchmark_dir, exist_ok=True)
        
        try:
            # Prepare test data for benchmark
            if isinstance(test_data, (list, tuple)):
                bench_x_seq = test_data[0][:num_samples]
                bench_x_temp = test_data[1][:num_samples]
            else:
                bench_x_seq = test_data[:num_samples]
                bench_x_temp = None
            
            # Extract SHAP attributions (handle flattened format)
            shap_attr = None
            if se is not None and se.shap_values is not None:
                shap_values_raw = se.shap_values.values
                if isinstance(shap_values_raw, list):
                    shap_values_raw = shap_values_raw[0]
                
                # Handle flattened multi-input case
                if se.is_multi_input and hasattr(se, '_seq_flat_size'):
                    if shap_values_raw.ndim == 2 and shap_values_raw.shape[1] >= se._seq_flat_size:
                        shap_attr = shap_values_raw[:, :se._seq_flat_size]
                    else:
                        shap_attr = shap_values_raw
                else:
                    # For single-input or already correct shape
                    seq_len = bench_x_seq.shape[1]
                    if shap_values_raw.ndim == 2 and shap_values_raw.shape[1] == seq_len:
                        shap_attr = shap_values_raw
                    elif shap_values_raw.ndim > 2:
                        # Find and extract sequence dimension
                        for axis in range(1, shap_values_raw.ndim):
                            if shap_values_raw.shape[axis] == seq_len:
                                shap_attr = np.moveaxis(shap_values_raw, axis, 1)
                                if shap_attr.ndim > 2:
                                    shap_attr = shap_attr.mean(axis=tuple(range(2, shap_attr.ndim)))
                                break
                        if shap_attr is None:
                            shap_attr = shap_values_raw.reshape(shap_values_raw.shape[0], -1)[:, :seq_len]
                    else:
                        shap_attr = shap_values_raw
            
            # Extract LIME attributions if available
            lime_attr = None
            if methods in ['lime', 'all'] and 'le' in dir() and le is not None and le.explanations:
                try:
                    lime_attr_list = []
                    seq_len = bench_x_seq.shape[1]
                    for exp in le.explanations:
                        if exp is not None:
                            # Extract feature weights from LIME explanation
                            if task == 'activity' and hasattr(exp, 'top_labels') and exp.top_labels:
                                exp_list = exp.as_list(label=exp.top_labels[0])
                            else:
                                exp_list = exp.as_list()

                            exp_map = dict(exp_list)
                            weights = np.zeros(seq_len)

                            for feat_name, weight in exp_map.items():
                                name = str(feat_name)
                                # Try to extract position from feature name (Position_# or similar)
                                match = re.search(r'(\d+)', name)
                                if match and "Position" in name:
                                    pos = int(match.group(1)) - 1
                                    if 0 <= pos < seq_len:
                                        weights[pos] += weight
                                    continue

                                # Otherwise, try to map activity name to positions
                                if label_encoder is not None:
                                    activity_name = name.split('<=')[0].split('>')[0].strip()
                                    try:
                                        token = label_encoder.transform([activity_name])[0] + 1
                                    except Exception:
                                        continue
                                    positions = np.where(bench_x_seq[0] == token)[0]
                                    if positions.size > 0:
                                        per_pos = weight / positions.size
                                        for pos in positions:
                                            if 0 <= pos < seq_len:
                                                weights[pos] += per_pos

                            lime_attr_list.append(weights)
                    if lime_attr_list:
                        lime_attr = np.array(lime_attr_list)
                except Exception as e:
                    print(f"[WARNING] Could not extract LIME attributions for benchmark: {e}")
                    lime_attr = None
            
            # Initialize and run benchmark
            benchmark = ExplainabilityBenchmark(
                model=model,
                task=task,
                is_multi_input=isinstance(test_data, (list, tuple)),
                seq_shape=getattr(se, '_seq_shape', None) if se else None,
                temp_shape=getattr(se, '_temp_shape', None) if se else None,
                scaler=scaler
            )
            
            if shap_attr is not None:
                benchmark_results = benchmark.run_full_benchmark(
                    x_seq=bench_x_seq,
                    x_temp=bench_x_temp,
                    shap_values=shap_attr,
                    lime_values=lime_attr,
                    test_seq=bench_x_seq,
                    k_values=[1, 3, 5, 10]
                )
                
                benchmark.save_results(benchmark_dir)
                benchmark.print_summary()
            else:
                print("[WARNING] Could not extract SHAP attributions for benchmark.")
                
        except Exception as e:
            print(f"[ERROR] Benchmark evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive summary outputs
    if methods == 'all':
        print("\n--- Generating Comparison Report ---")
        generate_comparison_report(output_dir, shap_dir if 'shap' in methods or methods == 'all' else None, 
                                   lime_dir if 'lime' in methods or methods == 'all' else None)
    
    # Sanity check for benchmark coverage
    _validate_explainability_coverage(
        task,
        label_encoder,
        shap_dir if methods in ['shap', 'all'] else None,
        lime_dir if methods in ['lime', 'all'] else None
    )
        
    print("\n" + "="*60)
    print(f"EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    print("\nGenerated outputs:")
    print("  [OK] SHAP global importance plots")
    if task != 'activity' and methods in ['shap', 'all']:
        print("  [OK] Temporal attribution plots")
    print(f"  [OK] LIME local explanations ({num_samples} diverse samples)")
    print("  [OK] Feature importance summary CSV")
    print("  [OK] Method comparison report")
    if run_benchmark and benchmark_results:
        print("  [OK] Benchmark evaluation metrics (JSON + CSV)")
    print("="*60)
    
    return benchmark_results


# =============================================================================
# BENCHMARK COMPARISON UTILITIES
# =============================================================================

def compare_benchmark_results(benchmark_files, output_path=None):
    """
    Compare benchmark results across multiple models/datasets.
    
    Args:
        benchmark_files: List of tuples (model_name, benchmark_json_path)
        output_path: Optional path to save comparison CSV
        
    Returns:
        DataFrame with comparison results
    """
    import json
    
    comparison_rows = []
    
    for model_name, filepath in benchmark_files:
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            row = {'model': model_name}
            
            # Extract key metrics
            if 'faithfulness' in results:
                for k, v in results['faithfulness'].items():
                    if isinstance(v, dict) and 'spearman_correlation' in v:
                        row[f'faith_{k}'] = v['spearman_correlation']
            
            if 'comprehensiveness' in results:
                for k, v in results['comprehensiveness'].items():
                    if isinstance(v, dict) and 'mean' in v:
                        row[f'comp_{k}'] = v['mean']
            
            if 'sufficiency' in results:
                for k, v in results['sufficiency'].items():
                    if isinstance(v, dict) and 'mean' in v:
                        row[f'suff_{k}'] = v['mean']
            
            if 'monotonicity' in results:
                mono = results['monotonicity'].get('monotonicity', {})
                row['monotonicity'] = mono.get('mean', None)
            
            if 'method_agreement' in results:
                for k, v in results['method_agreement'].items():
                    if isinstance(v, dict) and 'jaccard_similarity' in v:
                        row[f'agree_{k}'] = v['jaccard_similarity']
            
            if 'temporal_consistency' in results:
                tc = results['temporal_consistency'].get('temporal_consistency', {})
                row['recency_corr'] = tc.get('recency_correlation', None)
            
            comparison_rows.append(row)
            
        except Exception as e:
            print(f"[WARNING] Failed to load {filepath}: {e}")
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    if output_path:
        comparison_df.to_csv(output_path, index=False)
        print(f"[OK] Benchmark comparison saved to: {output_path}")
    
    return comparison_df


def generate_benchmark_latex_table(comparison_df, output_path=None, caption="Explainability Benchmark Comparison"):
    """
    Generate LaTeX table for benchmark comparison (useful for research papers).
    
    Args:
        comparison_df: DataFrame from compare_benchmark_results()
        output_path: Optional path to save .tex file
        caption: Table caption
        
    Returns:
        LaTeX string
    """
    # Select key columns for the paper
    key_cols = ['model']
    metric_cols = [c for c in comparison_df.columns if c != 'model']
    
    # Rename columns for readability
    rename_map = {
        'faith_faithfulness_k5': 'Faith@5',
        'comp_comprehensiveness_k5': 'Comp@5',
        'suff_sufficiency_k5': 'Suff@5',
        'monotonicity': 'Mono',
        'agree_agreement_k5': 'Agree@5',
        'recency_corr': 'Recency'
    }
    
    df = comparison_df.copy()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Format numeric columns
    for col in df.columns:
        if col != 'model' and df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))
    
    # Add caption and label
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:explainability_benchmark}}
{latex}
\\end{{table}}"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[OK] LaTeX table saved to: {output_path}")
    
    return latex
