"""
Transformer Explainability Module with Timestep-Level Attribution Support

This module provides SHAP and LIME explainers that can generate:
1. Traditional aggregated explanations (backward compatible)
2. Timestep-level temporal explanations (NEW - for time predictions)

File: explainability/transformers/transformer_explainer.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from tqdm import tqdm
import shap
from lime import lime_tabular
import tensorflow as tf

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})


# ============================================================================
# CONFIGURATION
# ============================================================================
class ExplainabilityConfig:
    """Configuration for explainability behavior"""
    # Set to True to enable timestep-level explanations for time predictions
    ENABLE_TIMESTEP_EXPLANATIONS = True
    
    # Model type detection (will be auto-detected, but can override)
    # Options: 'auto', 'per_timestep', 'original'
    MODEL_TYPE = 'auto'


# ============================================================================
# BASE SHAP EXPLAINER (Original - for Activity Prediction)
# ============================================================================
class SHAPExplainer:
    """Original SHAP Explainer - works for activity prediction"""
    
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        self.background_temp = None
        
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - will show generic Activity labels!")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _get_activity_names_for_sample(self, sequence):
        if self.label_encoder is None:
            return [f'Activity_{int(t)}' if t > 0 else '[PAD]' for t in sequence]
        
        names = []
        for token in sequence:
            if token > 0:
                try:
                    actual_activity = self.label_encoder.inverse_transform([int(token)-1])[0]
                    names.append(actual_activity)
                except Exception as e:
                    names.append(f'Token_{int(token)}')
            else:
                names.append('[PAD]')
        return names

    def initialize_explainer(self, background_data, max_background=100):
        print("Initializing SHAP Explainer...")
        
        if isinstance(background_data, (list, tuple)):
            bg_seq = background_data[0]
            bg_temp = background_data[1]
            indices = np.random.choice(len(bg_seq), min(max_background, len(bg_seq)), replace=False)
            background_sample = bg_seq[indices]
            self.background_temp = np.mean(bg_temp, axis=0).reshape(1, -1)
            
            def predict_fn(x_seq):
                temp_tiled = np.repeat(self.background_temp, x_seq.shape[0], axis=0)
                outputs = self.model.predict([x_seq, temp_tiled], verbose=0)
                # Handle both single and multi-output models
                if isinstance(outputs, list):
                    return outputs[0].flatten()
                return outputs.flatten()
            
            self.explainer = shap.Explainer(predict_fn, background_sample)
        else:
            indices = np.random.choice(len(background_data), min(max_background, len(background_data)), replace=False)
            background_sample = background_data[indices]
            
            try:
                self.explainer = shap.Explainer(self.model, background_sample)
            except:
                def predict_fn_single(x):
                    return self.model.predict(x, verbose=0)
                self.explainer = shap.Explainer(predict_fn_single, background_sample)

    def explain_samples(self, test_data, num_samples=20):
        if isinstance(test_data, (list, tuple)):
            test_sample = test_data[0][:num_samples]
            self.test_data = test_sample
        else:
            test_sample = test_data[:num_samples]
            self.test_data = test_sample
            
        print(f"Computing SHAP values for {len(test_sample)} samples...")
        self.shap_values = self.explainer(test_sample)
        return self.shap_values

    def _aggregate_by_activity(self):
        if self.shap_values is None: 
            return None, None, None

        values = self.shap_values.values
        
        if self.task == 'activity' and values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        
        unique_names = set()
        for seq in self.test_data:
            unique_names.update([n for n in self._get_activity_names_for_sample(seq) if n != '[PAD]'])
        
        sorted_names = sorted(list(unique_names))
        name_map = {name: i for i, name in enumerate(sorted_names)}
        
        num_samples = values.shape[0]
        agg_shap_matrix = np.zeros((num_samples, len(sorted_names)))
        agg_feat_matrix = np.zeros((num_samples, len(sorted_names))) 
        
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
        
        pd.DataFrame(agg_shap, columns=names).to_csv(os.path.join(output_dir, 'shap_values_matrix.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(agg_shap, features=agg_feat, feature_names=names, show=False, max_display=15)
        plt.title(f"Feature Impact Distribution ({self.task.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300)
        plt.close()

    def save_explanations(self, output_dir):
        print("[OK] SHAP computations complete.")


# ============================================================================
# TIMESTEP-LEVEL SHAP EXPLAINER (NEW - for Time Predictions)
# ============================================================================
class TimestepSHAPExplainer(SHAPExplainer):
    """Extended SHAP Explainer with timestep-level attribution support"""
    
    def __init__(self, model, task='time', label_encoder=None, scaler=None):
        super().__init__(model, task, label_encoder, scaler)
        self.model_has_timestep_outputs = self._detect_model_type()
        
        if self.model_has_timestep_outputs:
            print("[OK] Detected timestep-explainable model - will generate temporal plots")
        else:
            print("[INFO] Using original aggregated explanations")
    
    def _detect_model_type(self):
        """Auto-detect if model has timestep outputs"""
        if ExplainabilityConfig.MODEL_TYPE == 'per_timestep':
            return True
        elif ExplainabilityConfig.MODEL_TYPE == 'original':
            return False
        else:  # 'auto'
            # Try to detect by checking number of outputs
            if hasattr(self.model, 'outputs'):
                return len(self.model.outputs) > 1
            return False
    
    def plot_temporal_evolution(self, output_dir, sample_idx=0, show_prediction=True):
        """
        Generate temporal evolution plot with SHAP bars and prediction overlay.
        This matches your target visualization!
        """
        if self.shap_values is None:
            print("No SHAP values computed. Run explain_samples() first.")
            return
        
        print(f"Generating Temporal Evolution Plot for sample {sample_idx}...")
        
        sample_shap = self.shap_values.values[sample_idx]
        sample_sequence = self.test_data[sample_idx]
        activity_names = self._get_activity_names_for_sample(sample_sequence)
        
        # Filter padding
        non_pad_mask = sample_sequence > 0
        filtered_shap = sample_shap[non_pad_mask]
        filtered_activities = [name for name, is_valid in zip(activity_names, non_pad_mask) if is_valid]
        timesteps = np.arange(len(filtered_shap))
        
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(16, 7))
        
        # Plot SHAP bars (positive and negative)
        positive_shap = np.where(filtered_shap > 0, filtered_shap, 0)
        negative_shap = np.where(filtered_shap < 0, filtered_shap, 0)
        
        ax1.bar(timesteps, positive_shap, color='#d62728', alpha=0.8, 
                label='Positive Shapley values', width=0.8)
        ax1.bar(timesteps, negative_shap, color='#1f77b4', alpha=0.8, 
                label='Negative Shapley values', width=0.8)
        
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xlabel('Time steps', fontsize=13, fontweight='bold')
        ax1.set_ylabel('SHAP values (contribution to prediction)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left', fontsize=11)
        
        # Add activity annotations at important timesteps
        abs_shap = np.abs(filtered_shap)
        threshold = np.percentile(abs_shap, 75)  # Top 25%
        
        for i, (ts, act, shap_val) in enumerate(zip(timesteps, filtered_activities, filtered_shap)):
            if abs_shap[i] > threshold and act != '[PAD]':
                y_pos = shap_val + (0.2 if shap_val > 0 else -0.2)
                ax1.text(ts, y_pos, act, ha='center', va='bottom' if shap_val > 0 else 'top',
                        fontsize=9, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Overlay prediction if model supports it
        if show_prediction and self.model_has_timestep_outputs:
            try:
                # Get per-timestep predictions
                temp_input = self.background_temp if self.background_temp is not None else np.zeros((1, 3))
                outputs = self.model.predict([sample_sequence.reshape(1, -1), temp_input], verbose=0)
                
                if isinstance(outputs, list) and len(outputs) > 1:
                    timestep_preds = outputs[1][0]
                    filtered_preds = timestep_preds[non_pad_mask]
                    
                    # Inverse transform if scaler available
                    if self.scaler is not None:
                        filtered_preds = self.scaler.inverse_transform(
                            filtered_preds.reshape(-1, 1)
                        ).flatten()
                    
                    # Plot on secondary y-axis
                    ax2 = ax1.twinx()
                    ax2.plot(timesteps, filtered_preds, color='black', linewidth=2, 
                            label='Predicted remaining time', marker='o', markersize=3)
                    ax2.set_ylabel('Predicted remaining time (days)', fontsize=12, fontweight='bold')
                    ax2.legend(loc='upper right', fontsize=11)
            except Exception as e:
                print(f"Could not add prediction overlay: {e}")
        
        plt.title(f'Transformer Model - SHAP Explainability (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_temporal_evolution_sample_{sample_idx}.png'), dpi=300)
        plt.close()
        
        # Save data
        df = pd.DataFrame({
            'Timestep': timesteps,
            'Activity': filtered_activities,
            'SHAP_Value': filtered_shap
        })
        df.to_csv(os.path.join(output_dir, f'shap_timestep_data_sample_{sample_idx}.csv'), index=False)
    
    def plot_timestep_heatmap(self, output_dir, sample_idx=0):
        """Generate heatmap showing SHAP values across timesteps"""
        if self.shap_values is None:
            return
        
        print(f"Generating Timestep Heatmap for sample {sample_idx}...")
        
        sample_shap = self.shap_values.values[sample_idx]
        sample_sequence = self.test_data[sample_idx]
        activity_names = self._get_activity_names_for_sample(sample_sequence)
        
        non_pad_mask = sample_sequence > 0
        filtered_shap = sample_shap[non_pad_mask]
        filtered_activities = [name for name, is_valid in zip(activity_names, non_pad_mask) if is_valid]
        timesteps = np.arange(len(filtered_shap))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['#d62728' if val < 0 else '#2ca02c' for val in filtered_shap]
        bars = ax.bar(timesteps, filtered_shap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(timesteps)
        ax.set_xticklabels(filtered_activities, rotation=45, ha='right', fontsize=9)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Timestep (Activity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('SHAP Value (Contribution)', fontsize=12, fontweight='bold')
        ax.set_title(f'Timestep-Level SHAP Attribution - Sample {sample_idx}', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Increases Prediction'),
            Patch(facecolor='#d62728', label='Decreases Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_timestep_heatmap_sample_{sample_idx}.png'), dpi=300)
        plt.close()
    
    def plot_global_temporal_importance(self, output_dir):
        """Aggregate SHAP values across samples to show global temporal patterns"""
        if self.shap_values is None:
            return
        
        print("Generating Global Temporal Importance Plot...")
        
        all_shap = self.shap_values.values
        mean_shap_per_timestep = np.mean(np.abs(all_shap), axis=0)
        
        # Get most common activity at each timestep
        activity_labels = []
        for pos in range(all_shap.shape[1]):
            activities_at_pos = []
            for sample in self.test_data:
                if sample[pos] > 0:
                    try:
                        act = self.label_encoder.inverse_transform([int(sample[pos])-1])[0]
                        activities_at_pos.append(act)
                    except:
                        pass
            
            if activities_at_pos:
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                activity_labels.append(most_common)
            else:
                activity_labels.append('[PAD]')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        timesteps = np.arange(len(mean_shap_per_timestep))
        
        ax.bar(timesteps, mean_shap_per_timestep, color='#2ca02c', alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Timestep Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        ax.set_title('Global Timestep Importance (Averaged Across All Samples)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Annotate top timesteps
        top_n = 10
        top_indices = np.argsort(mean_shap_per_timestep)[-top_n:]
        for idx in top_indices:
            ax.text(idx, mean_shap_per_timestep[idx], activity_labels[idx],
                   ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_global_temporal_importance.png'), dpi=300)
        plt.close()
        
        df = pd.DataFrame({
            'Timestep': timesteps,
            'Most_Common_Activity': activity_labels,
            'Mean_Absolute_SHAP': mean_shap_per_timestep
        })
        df.to_csv(os.path.join(output_dir, 'shap_global_temporal_data.csv'), index=False)


# ============================================================================
# LIME EXPLAINER (Enhanced with Timestep Support)
# ============================================================================
class LIMEExplainer:
    """LIME Explainer with timestep-level support"""
    
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
        self.model_has_timestep_outputs = len(model.outputs) > 1 if hasattr(model, 'outputs') else False
        
        if self.label_encoder is None:
            print("[WARNING] label_encoder is None - LIME will show generic Activity labels!")
        else:
            print(f"[OK] label_encoder available with {len(self.label_encoder.classes_)} activities")
        
    def _aggregate_feature_names(self, data):
        """Generate feature names based on activities"""
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
                    except:
                        pass
            
            if activities_at_pos:
                most_common = max(set(activities_at_pos), key=activities_at_pos.count)
                feature_names.append(most_common)
            else:
                feature_names.append(f'Position_{pos+1}')
        
        return feature_names
        
    def initialize_explainer(self, training_data, num_classes=None):
        print("Initializing LIME Explainer...")
        
        if isinstance(training_data, (list, tuple)):
            init_data = training_data[0]
        else:
            init_data = training_data
        
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
    
    def explain_samples(self, test_data, num_samples=10, num_features=15):
        print(f"Generating LIME explanations for {num_samples} samples...")
        
        if isinstance(test_data, (list, tuple)):
            self.test_data_seq = test_data[0][:num_samples]
            self.test_data_temp = test_data[1][:num_samples]
            self.is_multi_input = True
        else:
            self.test_data_seq = test_data[:num_samples]
            self.is_multi_input = False
            
        vocab_size = int(np.max(self.test_data_seq)) + 1
        
        for i in tqdm(range(len(self.test_data_seq))):
            try:
                if self.is_multi_input:
                    current_temp = self.test_data_temp[i].reshape(1, -1)
                    def predict_fn(x_seq):
                        if x_seq.ndim == 1: x_seq = x_seq.reshape(1, -1)
                        x_seq = np.clip(np.round(x_seq), 0, vocab_size-1).astype(int)
                        temp_batch = np.repeat(current_temp, x_seq.shape[0], axis=0)
                        preds = self.model.predict([x_seq, temp_batch], verbose=0)
                        # Handle multi-output models
                        if isinstance(preds, list):
                            return preds[0].flatten()
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
            except: 
                pass
        return f"Activity_{int(token_idx)}"

    def plot_explanation(self, output_dir, sample_idx=0, original_idx=None):
        """Plot LIME explanation for a sample"""
        if sample_idx >= len(self.explanations) or self.explanations[sample_idx] is None:
            print(f"LIME Explanation not found for sample {sample_idx}.")
            return
        
        display_idx = original_idx if original_idx is not None else sample_idx
            
        print(f"Generating LIME Plot for sample {display_idx}...")
        exp = self.explanations[sample_idx]
        current_seq = self.test_data_seq[sample_idx]
        
        try:
            if self.task == 'activity':
                if hasattr(exp, 'top_labels') and exp.top_labels:
                    label_to_explain = exp.top_labels[0]
                else:
                    label_to_explain = 1 
                
                pred_probs = exp.predict_proba
                confidence = pred_probs[label_to_explain] if pred_probs is not None else 0.0
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Class: {label_to_explain} | Confidence: {confidence:.2f}"
                lime_list = exp.as_list(label=label_to_explain)
            else:
                raw_pred = exp.predicted_value
                display_val = raw_pred
                
                if self.scaler is not None:
                    unscaled = self.scaler.inverse_transform([[raw_pred]])[0][0]
                    display_val = unscaled
                    
                title = f"LIME Explanation (Sample {display_idx})\nPredicted Value: {display_val:.2f}"
                lime_list = exp.as_list()
                
        except Exception as e:
            print(f"Warning: Could not extract full LIME details: {e}")
            title = f"LIME Explanation (Sample {display_idx})"
            lime_list = exp.as_list()

        activity_stats = {} 
        for rule, weight in lime_list:
            if rule.startswith('Position_'):
                match = re.search(r'Position_(\d+)', rule)
                if match:
                    pos = int(match.group(1)) - 1
                    if 0 <= pos < len(current_seq):
                        name = self._get_activity_name(current_seq[pos])
                    else:
                        name = rule
                else:
                    name = rule
            else:
                name = rule.split('<=')[0].split('>')[0].strip()
            
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
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel("Contribution to Prediction", fontsize=11)
        
        for rect in bars:
            w = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            padding = 0.0005 if w > 0 else -0.0005
            ha = 'left' if w > 0 else 'right'
            plt.text(w + padding, y, f'{w:.4f}', va='center', ha=ha, fontsize=9, fontweight='bold')

        plt.legend(handles=[
            Patch(facecolor='#2ca02c', label='Supports'),
            Patch(facecolor='#d62728', label='Contradicts')
        ], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_explanation_sample_{display_idx}.png'), dpi=300)
        plt.close()

    def save_explanations(self, output_dir):
        print("[OK] LIME computations complete.")


# ============================================================================
# UNIFIED EXPLAINABILITY RUNNER
# ============================================================================
def run_transformer_explainability(model, data, output_dir, task='activity', 
                                   num_samples=50, methods='all', 
                                   label_encoder=None, scaler=None, 
                                   feature_config=None):
    """
    Run explainability analysis on transformer model.
    Automatically detects if model supports timestep-level explanations.
    
    Args:
        model: Trained transformer model
        data: Dict with train/test splits
        output_dir: Output directory for results
        task: 'activity', 'time', 'event_time', or 'remaining_time'
        num_samples: Number of samples to explain
        methods: 'shap', 'lime', or 'all'
        label_encoder: For activity name mapping
        scaler: For time denormalization
        feature_config: Additional configuration (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect if this is a time prediction task
    is_time_task = task in ['time', 'event_time', 'remaining_time']
    
    print("="*70)
    print(f"EXPLAINABILITY MODULE: {task.upper()} PREDICTION")
    print("="*70)
    
    if label_encoder is None:
        print("\n[WARNING] label_encoder is None! Plots will show generic labels")
    
    # Prepare data
    if task == 'activity':
        train_data = data['X_train']
        test_data = data['X_test']
        num_classes = len(np.unique(data['y_train']))
    else:
        train_data = (data['X_seq_train'], data['X_temp_train'])
        test_data = (data['X_seq_test'], data['X_temp_test'])
        num_classes = None

    # Run SHAP
    if methods in ['shap', 'all']:
        print("\n--- Running SHAP ---")
        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)
        
        # Use timestep-aware SHAP for time tasks
        if is_time_task and ExplainabilityConfig.ENABLE_TIMESTEP_EXPLANATIONS:
            se = TimestepSHAPExplainer(model, task, label_encoder, scaler)
            se.initialize_explainer(train_data)
            se.explain_samples(test_data, num_samples)
            
            # Generate timestep-level visualizations
            if se.model_has_timestep_outputs:
                print("\n[SHAP] Generating timestep-level visualizations...")
                for i in range(min(5, num_samples)):
                    se.plot_temporal_evolution(shap_dir, sample_idx=i, show_prediction=True)
                
                for i in range(min(3, num_samples)):
                    se.plot_timestep_heatmap(shap_dir, sample_idx=i)
                
                se.plot_global_temporal_importance(shap_dir)
            else:
                # Fall back to regular plots
                se.plot_bar(shap_dir)
                se.plot_summary(shap_dir)
            
            se.save_explanations(shap_dir)
        else:
            # Original SHAP for activity prediction
            se = SHAPExplainer(model, task, label_encoder, scaler)
            se.initialize_explainer(train_data)
            se.explain_samples(test_data, num_samples)
            se.plot_bar(shap_dir)
            se.plot_summary(shap_dir)
            se.save_explanations(shap_dir)

    # Run LIME
    if methods in ['lime', 'all']:
        print("\n--- Running LIME ---")
        lime_dir = os.path.join(output_dir, 'lime')
        os.makedirs(lime_dir, exist_ok=True)
        
        le = LIMEExplainer(model, task, label_encoder, scaler)
        le.initialize_explainer(train_data, num_classes)
        
        # Select diverse samples
        diverse_samples = select_diverse_samples(data, task, num_diverse=10)
        print(f"Explaining {len(diverse_samples)} diverse samples: {diverse_samples}")
        
        if isinstance(test_data, (list, tuple)):
            diverse_test_data = (test_data[0][diverse_samples], test_data[1][diverse_samples])
        else:
            diverse_test_data = test_data[diverse_samples]
        
        le.explain_samples(diverse_test_data, num_samples=len(diverse_samples))
        
        # Generate plots
        print(f"\n[LIME] Plotting {len(le.explanations)} explanations...")
        for i in range(len(le.explanations)):
            if le.explanations[i] is not None:
                original_idx = diverse_samples[i]
                le.plot_explanation(lime_dir, sample_idx=i, original_idx=original_idx)
        
        le.save_explanations(lime_dir)
    
    # Generate comparison report
    if methods == 'all':
        print("\n--- Generating Comparison Report ---")
        generate_comparison_report(output_dir, 
                                   shap_dir if 'shap' in methods or methods == 'all' else None, 
                                   lime_dir if 'lime' in methods or methods == 'all' else None)
        
    print("\n" + "="*70)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)


def select_diverse_samples(data, task, num_diverse=10):
    """Select diverse samples from test set"""
    if task == 'activity':
        y_test = data.get('y_test', [])
        test_size = len(y_test)
    else:
        y_test = data.get('y_test', [])
        test_size = len(y_test) if hasattr(y_test, '__len__') else len(data.get('X_seq_test', []))
    
    if test_size == 0:
        return [0]
    
    max_samples = min(num_diverse, test_size)
    step = max(1, test_size // max_samples)
    diverse_indices = list(range(0, test_size, step))[:max_samples]
    
    return diverse_indices


def generate_comparison_report(output_dir, shap_dir, lime_dir):
    """Generate comparison report between SHAP and LIME"""
    summary_data = []
    
    # Load SHAP results
    shap_importance = {}
    if shap_dir and os.path.exists(os.path.join(shap_dir, 'global_importance_data.csv')):
        shap_df = pd.read_csv(os.path.join(shap_dir, 'global_importance_data.csv'))
        shap_importance = dict(zip(shap_df['Activity'], shap_df['Mean_Impact']))
    
    # Load LIME results
    lime_importance = {}
    if lime_dir:
        lime_files = [f for f in os.listdir(lime_dir) if f.startswith('lime_explanation_sample_') and f.endswith('.csv')]
        if lime_files:
            all_lime_weights = {}
            for lime_file in lime_files:
                lime_df = pd.read_csv(os.path.join(lime_dir, lime_file))
                for _, row in lime_df.iterrows():
                    activity = row['Activity']
                    weight = abs(row['Weight'])
                    if activity not in all_lime_weights:
                        all_lime_weights[activity] = []
                    all_lime_weights[activity].append(weight)
            
            lime_importance = {act: sum(weights)/len(weights) for act, weights in all_lime_weights.items()}
    
    # Combine
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
    
    print(f"[OK] Feature importance summary saved")
    print(f"[OK] Comparison report saved")
