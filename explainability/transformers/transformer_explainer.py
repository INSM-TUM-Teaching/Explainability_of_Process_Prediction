import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import json
import shap
from lime import lime_tabular


class SHAPExplainer:
    
    def __init__(self, model, task='activity'):
        self.model = model
        self.task = task
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        
    def initialize_explainer(self, background_data, max_background=100):
        background_sample = background_data[np.random.choice(
            background_data.shape[0], 
            min(max_background, background_data.shape[0]), 
            replace=False
        )]
        
        if self.task == 'activity':
            predict_fn = lambda x: self.model.predict(x, verbose=0)
        else:
            predict_fn = lambda x: self.model.predict(x, verbose=0).flatten() if isinstance(x, np.ndarray) else self.model.predict([x[0], x[1]], verbose=0).flatten()
        
        self.explainer = shap.KernelExplainer(predict_fn, background_sample)
    
    def explain_samples(self, test_data, num_samples=20):
        if isinstance(test_data, tuple):
            test_sample = (
                test_data[0][:num_samples],
                test_data[1][:num_samples]
            )
            self.test_data = test_sample
            self.shap_values = self.explainer.shap_values(test_sample[0])
        else:
            test_sample = test_data[:num_samples]
            self.test_data = test_sample
            self.shap_values = self.explainer.shap_values(test_sample)
        
        return self.shap_values
    
    def save_explanations(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        if shap_vals.ndim == 3:
            mean_shap = np.mean(np.abs(shap_vals), axis=(1, 2))
            max_shap = np.max(np.abs(shap_vals), axis=(1, 2))
        else:
            mean_shap = np.mean(np.abs(shap_vals), axis=1)
            max_shap = np.max(np.abs(shap_vals), axis=1)
        
        num_features = shap_vals.shape[1]
        
        summary_data = []
        for i in range(len(shap_vals)):
            mean_val = mean_shap[i]
            max_val = max_shap[i]
            
            if isinstance(mean_val, np.ndarray):
                mean_val = float(mean_val.item()) if mean_val.size == 1 else float(np.mean(mean_val))
            if isinstance(max_val, np.ndarray):
                max_val = float(max_val.item()) if max_val.size == 1 else float(np.max(max_val))
            
            summary_data.append({
                'sample_id': i,
                'mean_shap_value': float(mean_val),
                'max_shap_value': float(max_val),
                'num_features': num_features
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'shap_summary.csv'), index=False)
        
        if shap_vals.ndim == 3:
            feature_importance = np.mean(np.abs(shap_vals), axis=(0, 2))
        else:
            feature_importance = np.mean(np.abs(shap_vals), axis=0)
        
        feature_data = []
        for i in range(len(feature_importance)):
            imp_val = feature_importance[i]
            if isinstance(imp_val, np.ndarray):
                imp_val = float(imp_val.item()) if imp_val.size == 1 else float(np.mean(imp_val))
            
            feature_data.append({
                'feature_index': i,
                'importance': float(imp_val)
            })
        
        feature_df = pd.DataFrame(feature_data)
        feature_df = feature_df.sort_values('importance', ascending=False)
        feature_df.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)
    
    def plot_summary(self, output_dir, max_display=20):
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        feature_names = [f'Pos_{i+1}' for i in range(shap_vals.shape[1])]
        
        try:
            plt.figure(figsize=(12, 8))
            
            if shap_vals.ndim == 3:
                shap_vals_2d = np.mean(np.abs(shap_vals), axis=2)
                test_data_to_use = self.test_data if not isinstance(self.test_data, tuple) else self.test_data[0]
                
                shap.summary_plot(
                    shap_vals_2d, 
                    test_data_to_use,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                    plot_type='bar'
                )
            else:
                shap.summary_plot(
                    shap_vals, 
                    self.test_data if not isinstance(self.test_data, tuple) else self.test_data[0],
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False
                )
            
            plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create SHAP summary plot - {e}")
            plt.close()
        
        try:
            plt.figure(figsize=(10, 6))
            
            if shap_vals.ndim == 3:
                feature_importance = np.mean(np.abs(shap_vals), axis=(0, 2))
            else:
                feature_importance = np.mean(np.abs(shap_vals), axis=0)
            
            top_k = min(15, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-top_k:]
            
            plt.barh(range(top_k), feature_importance[top_indices], color='steelblue')
            plt.yticks(range(top_k), [feature_names[i] for i in top_indices])
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.title(f'Top {top_k} Feature Importance (SHAP)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create feature importance plot - {e}")
            plt.close()
    
    def plot_force(self, output_dir, sample_idx=0):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if isinstance(self.shap_values, list):
                if isinstance(self.test_data, tuple):
                    pred_probs = self.model.predict([self.test_data[0][sample_idx:sample_idx+1], self.test_data[1][sample_idx:sample_idx+1]], verbose=0)
                else:
                    pred_probs = self.model.predict(self.test_data[sample_idx:sample_idx+1], verbose=0)
                predicted_class = np.argmax(pred_probs)
                shap_vals = self.shap_values[predicted_class][sample_idx]
                
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    expected_value = self.explainer.expected_value[predicted_class]
                else:
                    expected_value = self.explainer.expected_value
            else:
                if self.shap_values.ndim == 3:
                    if isinstance(self.test_data, tuple):
                        pred_probs = self.model.predict([self.test_data[0][sample_idx:sample_idx+1], self.test_data[1][sample_idx:sample_idx+1]], verbose=0)
                    else:
                        pred_probs = self.model.predict(self.test_data[sample_idx:sample_idx+1], verbose=0)
                    predicted_class = np.argmax(pred_probs)
                    shap_vals = self.shap_values[sample_idx, :, predicted_class]
                    
                    if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1:
                        expected_value = float(self.explainer.expected_value[predicted_class])
                    else:
                        expected_value = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])
                else:
                    shap_vals = self.shap_values[sample_idx]
                    expected_value = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])
            
            test_sample = self.test_data if not isinstance(self.test_data, tuple) else self.test_data[0]
            feature_names = [f'Pos_{i+1}' for i in range(test_sample.shape[1])]
            
            shap.force_plot(
                expected_value,
                shap_vals,
                test_sample[sample_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Sample {sample_idx}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_force_plot_sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create force plot for sample {sample_idx} - {e}")
            plt.close()
    
    def plot_waterfall(self, output_dir, sample_idx=0):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if isinstance(self.shap_values, list):
                if isinstance(self.test_data, tuple):
                    pred_probs = self.model.predict([self.test_data[0][sample_idx:sample_idx+1], self.test_data[1][sample_idx:sample_idx+1]], verbose=0)
                else:
                    pred_probs = self.model.predict(self.test_data[sample_idx:sample_idx+1], verbose=0)
                predicted_class = np.argmax(pred_probs)
                shap_vals = self.shap_values[predicted_class][sample_idx]
                
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    expected_value = self.explainer.expected_value[predicted_class]
                else:
                    expected_value = self.explainer.expected_value
            else:
                if self.shap_values.ndim == 3:
                    if isinstance(self.test_data, tuple):
                        pred_probs = self.model.predict([self.test_data[0][sample_idx:sample_idx+1], self.test_data[1][sample_idx:sample_idx+1]], verbose=0)
                    else:
                        pred_probs = self.model.predict(self.test_data[sample_idx:sample_idx+1], verbose=0)
                    predicted_class = np.argmax(pred_probs)
                    shap_vals = self.shap_values[sample_idx, :, predicted_class]
                    
                    if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1:
                        expected_value = float(self.explainer.expected_value[predicted_class])
                    else:
                        expected_value = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])
                else:
                    shap_vals = self.shap_values[sample_idx]
                    expected_value = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])
            
            test_sample = self.test_data if not isinstance(self.test_data, tuple) else self.test_data[0]
            
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=expected_value,
                data=test_sample[sample_idx],
                feature_names=[f'Pos_{i+1}' for i in range(test_sample.shape[1])]
            )
            
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_waterfall_sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create waterfall plot for sample {sample_idx} - {e}")
            plt.close()



class LIMEExplainer:
    
    def __init__(self, model, task='activity'):
        self.model = model
        self.task = task
        self.explainer = None
        self.explanations = []
        
    def initialize_explainer(self, training_data, num_classes=None):
        if isinstance(training_data, tuple):
            train_sample = training_data[0]
        else:
            train_sample = training_data
        
        feature_names = [f'Position_{i+1}' for i in range(train_sample.shape[1])]
        
        if self.task == 'activity':
            class_names = [f'Activity_{i}' for i in range(num_classes)] if num_classes else None
            mode = 'classification'
        else:
            class_names = None
            mode = 'regression'
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            train_sample,
            mode=mode,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=False
        )
    
    def explain_samples(self, test_data, num_samples=10, num_features=10):
        explanations = []
        
        if isinstance(test_data, tuple):
            test_sample = test_data[0][:num_samples]
            temp_features = test_data[1][:num_samples]
            vocab_size = int(np.max(test_sample)) + 1
            
            def predict_fn(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                X_clipped = np.clip(X.round().astype(int), 0, vocab_size - 1)
                temp_expanded = np.repeat(temp_features[:1], len(X), axis=0) if len(X) > 1 else temp_features[:1]
                return self.model.predict([X_clipped, temp_expanded], verbose=0)
        else:
            test_sample = test_data[:num_samples]
            vocab_size = int(np.max(test_sample)) + 1
            
            def predict_fn(X):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                X_clipped = np.clip(X.round().astype(int), 0, vocab_size - 1)
                return self.model.predict(X_clipped, verbose=0)
        
        for idx in tqdm(range(len(test_sample)), desc="Generating LIME explanations"):
            try:
                exp = self.explainer.explain_instance(
                    test_sample[idx],
                    predict_fn,
                    num_features=num_features,
                    top_labels=3 if self.task == 'activity' else 1
                )
                explanations.append(exp)
            except Exception as e:
                print(f"Warning: Could not generate LIME explanation for sample {idx} - {e}")
                continue
        
        self.explanations = explanations
        return explanations
    
    def save_explanations(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        exp_data = []
        for idx, exp in enumerate(self.explanations):
            labels = exp.available_labels()
            exp_dict = {'sample_id': idx, 'labels': {}}
            
            for label in labels:
                exp_list = exp.as_list(label=label)
                exp_dict['labels'][str(label)] = [
                    {'feature': feat, 'weight': float(weight)} 
                    for feat, weight in exp_list
                ]
            exp_data.append(exp_dict)
        
        with open(os.path.join(output_dir, 'lime_explanations.json'), 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        summary_data = []
        for idx, exp in enumerate(self.explanations):
            labels = exp.available_labels()
            for label in labels:
                exp_list = exp.as_list(label=label)
                avg_weight = np.mean([abs(w) for _, w in exp_list])
                summary_data.append({
                    'sample_id': idx,
                    'label': label,
                    'avg_feature_weight': avg_weight,
                    'num_features': len(exp_list)
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'lime_summary.csv'), index=False)
    
    def plot_explanation(self, output_dir, sample_idx=0, num_features=10):
        os.makedirs(output_dir, exist_ok=True)
        
        if sample_idx >= len(self.explanations):
            print(f"Warning: Sample {sample_idx} not available. Only {len(self.explanations)} samples.")
            return
        
        exp = self.explanations[sample_idx]
        labels = exp.available_labels()
        
        for label in labels[:3]:
            try:
                exp_list = exp.as_list(label=label)
                
                if not exp_list:
                    print(f"Warning: No explanation data for sample {sample_idx}, class {label}")
                    continue
                
                features = [f for f, w in exp_list[:num_features]]
                weights = [w for f, w in exp_list[:num_features]]
                
                colors = ['green' if w > 0 else 'red' for w in weights]
                
                plt.figure(figsize=(10, 6))
                y_pos = np.arange(len(features))
                plt.barh(y_pos, weights, color=colors, alpha=0.7)
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance', fontsize=12)
                plt.title(f'LIME Explanation - Sample {sample_idx}, Class {label}', 
                         fontsize=14, fontweight='bold')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                
                output_path = os.path.join(output_dir, f'lime_explanation_sample_{sample_idx}_class_{label}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not plot LIME for sample {sample_idx}, class {label} - {e}")
                plt.close()
    
    def plot_feature_importance(self, output_dir, top_k=15):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            feature_weights = {}
            
            for exp in self.explanations:
                labels = exp.available_labels()
                for label in labels:
                    try:
                        exp_list = exp.as_list(label=label)
                        for feature, weight in exp_list:
                            if feature not in feature_weights:
                                feature_weights[feature] = []
                            feature_weights[feature].append(abs(weight))
                    except Exception as e:
                        continue
            
            if not feature_weights:
                print("Warning: No feature importance data available for LIME")
                return
            
            avg_weights = {feat: np.mean(weights) for feat, weights in feature_weights.items()}
            sorted_features = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            if not sorted_features:
                print("Warning: No features to plot for LIME")
                return
            
            plt.figure(figsize=(12, 6))
            features, weights = zip(*sorted_features)
            plt.barh(range(len(features)), weights, color='forestgreen')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Average |Weight|', fontsize=12)
            plt.title(f'Top {top_k} Feature Importance (LIME)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'lime_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create LIME feature importance plot - {e}")
            plt.close()


def run_transformer_explainability(model, data, output_dir, task='activity', num_samples=20, methods='all'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRANSFORMER EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    if task == 'activity':
        train_data = data['X_train']
        test_data = data['X_test']
        num_classes = len(np.unique(data['y_train']))
    else:
        train_data = (data['X_seq_train'], data['X_temp_train'])
        test_data = (data['X_seq_test'], data['X_temp_test'])
        num_classes = None
    
    run_shap = methods in ['shap', 'all']
    run_lime = methods in ['lime', 'all']
    
    if run_shap:
        print("\n[SHAP Method]")
        print("-"*70)
        shap_explainer = SHAPExplainer(model, task=task)
        shap_explainer.initialize_explainer(
            train_data if not isinstance(train_data, tuple) else train_data[0],
            max_background=100
        )
        
        shap_explainer.explain_samples(test_data, num_samples=num_samples)
        
        shap_dir = os.path.join(output_dir, 'shap')
        shap_explainer.save_explanations(shap_dir)
        shap_explainer.plot_summary(shap_dir)
        
        for i in range(min(3, num_samples)):
            shap_explainer.plot_force(shap_dir, sample_idx=i)
            shap_explainer.plot_waterfall(shap_dir, sample_idx=i)
        
        print(f"✓ SHAP results saved to: {shap_dir}")
    
    if run_lime:
        print("\n[LIME Method]")
        print("-"*70)
        lime_explainer = LIMEExplainer(model, task=task)
        lime_explainer.initialize_explainer(train_data, num_classes=num_classes)
        
        lime_explainer.explain_samples(test_data, num_samples=min(10, num_samples), num_features=10)
        
        lime_dir = os.path.join(output_dir, 'lime')
        lime_explainer.save_explanations(lime_dir)
        lime_explainer.plot_feature_importance(lime_dir)
        
        for i in range(min(3, len(lime_explainer.explanations))):
            lime_explainer.plot_explanation(lime_dir, sample_idx=i)
        
        print(f"✓ LIME results saved to: {lime_dir}")
    
    print("\n" + "="*70)
    print("TRANSFORMER EXPLAINABILITY COMPLETE")
    print("="*70)
    
    results = {}
    if run_shap:
        results['shap'] = shap_explainer.shap_values
    if run_lime:
        results['lime'] = lime_explainer.explanations
    
    return results