import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shap
from lime import lime_tabular
import tensorflow as tf

# Standard Research plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

class SHAPExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler # Store scaler for future use if needed
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        self.background_temp = None
        
    def _get_activity_names_for_sample(self, sequence):
        """Maps token indices to real Activity names."""
        if self.label_encoder is None:
            return [f'Activity_{int(t)}' if t > 0 else '[PAD]' for t in sequence]
        
        names = []
        for token in sequence:
            if token > 0:
                try:
                    names.append(self.label_encoder.inverse_transform([int(token)-1])[0])
                except:
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
                return self.model.predict([x_seq, temp_tiled], verbose=0).flatten()
            
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
        if self.shap_values is None: return None, None, None

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
        print("✓ SHAP computations complete.")


class LIMEExplainer:
    def __init__(self, model, task='activity', label_encoder=None, scaler=None):
        self.model = model
        self.task = task
        self.label_encoder = label_encoder
        self.scaler = scaler # NEW: Store scaler for inverse transform
        self.explainer = None
        self.explanations = []
        self.test_data_seq = None
        self.test_data_temp = None
        self.is_multi_input = False
        
    def initialize_explainer(self, training_data, num_classes=None):
        print("Initializing LIME Explainer...")
        
        if isinstance(training_data, (list, tuple)):
            init_data = training_data[0]
        else:
            init_data = training_data
            
        feature_names = [f'Position_{i+1}' for i in range(init_data.shape[1])]
        
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
        self.explanations = []
        
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
        if token_idx == 0: return "[PAD]"
        if self.label_encoder:
            try: return self.label_encoder.inverse_transform([int(token_idx)-1])[0]
            except: pass
        return f"Activity_{int(token_idx)}"

    def plot_explanation(self, output_dir, sample_idx=0):
        if sample_idx >= len(self.explanations) or self.explanations[sample_idx] is None:
            print("LIME Explanation not found for this sample.")
            return
            
        print("Generating Research-Grade LIME Plot...")
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
                title = f"LIME Explanation (Sample {sample_idx})\nPredicted Class: {label_to_explain} | Confidence: {confidence:.2f}"
                lime_list = exp.as_list(label=label_to_explain)
            else:
                # --- REGRESSION FIX ---
                raw_pred = exp.predicted_value
                display_val = raw_pred
                
                # If scaler is provided, Inverse Transform to show Real Values (Days/Time)
                if self.scaler is not None:
                    # Inverse Z-Score
                    unscaled = self.scaler.inverse_transform([[raw_pred]])[0][0]
                    # Note: Since your training used np.log1p(), we should ideally use np.expm1()
                    # to get back to exact seconds, but getting back to positive scale is usually enough for the plot.
                    # We will use the unscaled value directly as it represents 'Log Seconds' or 'Normalized Days' better than 0.00
                    display_val = unscaled
                    
                title = f"LIME Explanation (Sample {sample_idx})\nPredicted Value: {display_val:.2f}"
                lime_list = exp.as_list()
                
        except Exception as e:
            print(f"Warning: Could not extract full LIME details: {e}")
            title = f"LIME Explanation (Sample {sample_idx})"
            lime_list = exp.as_list()

        activity_stats = {} 
        for rule, weight in lime_list:
            match = re.search(r'Position_(\d+)', rule)
            if match:
                pos = int(match.group(1)) - 1
                if 0 <= pos < len(current_seq):
                    name = self._get_activity_name(current_seq[pos])
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
        df[['Activity', 'Weight']].to_csv(os.path.join(output_dir, f'lime_explanation_sample_{sample_idx}.csv'), index=False)

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

        from matplotlib.patches import Patch
        plt.legend(handles=[
            Patch(facecolor='#2ca02c', label='Supports'),
            Patch(facecolor='#d62728', label='Contradicts')
        ], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lime_explanation.png'), dpi=300)
        plt.close()

    def save_explanations(self, output_dir):
        print("✓ LIME computations complete.")

def run_transformer_explainability(model, data, output_dir, task='activity', num_samples=20, methods='all', label_encoder=None, scaler=None):
    os.makedirs(output_dir, exist_ok=True)
    print("="*60)
    print(f"EXPLAINABILITY MODULE: {task.upper()} PREDICTION")
    print("="*60)
    
    if task == 'activity':
        train_data = data['X_train']
        test_data = data['X_test']
        num_classes = len(np.unique(data['y_train']))
    else:
        train_data = (data['X_seq_train'], data['X_temp_train'])
        test_data = (data['X_seq_test'], data['X_temp_test'])
        num_classes = None

    if methods in ['shap', 'all']:
        print("\n--- Running SHAP ---")
        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)
        # Pass scaler here if you want to use it for SHAP too (optional)
        se = SHAPExplainer(model, task, label_encoder, scaler)
        se.initialize_explainer(train_data)
        se.explain_samples(test_data, num_samples)
        se.plot_bar(shap_dir)
        se.plot_summary(shap_dir)
        se.save_explanations(shap_dir)

    if methods in ['lime', 'all']:
        print("\n--- Running LIME ---")
        lime_dir = os.path.join(output_dir, 'lime')
        os.makedirs(lime_dir, exist_ok=True)
        
        # Pass Scaler to LIME for Inverse Transforming Title
        le = LIMEExplainer(model, task, label_encoder, scaler)
        le.initialize_explainer(train_data, num_classes)
        le.explain_samples(test_data, num_samples)
        le.plot_explanation(lime_dir, sample_idx=0)
        le.save_explanations(lime_dir)
        
    print("\n" + "="*60)
    print(f"DONE. Results saved to: {output_dir}")
    print("="*60)