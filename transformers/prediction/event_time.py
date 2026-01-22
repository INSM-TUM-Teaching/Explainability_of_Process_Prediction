import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from ..model import build_time_prediction_model


class EventTimePredictor:
    def __init__(self, max_len=16, d_model=64, num_heads=4, num_blocks=2, 
                 dropout_rate=0.1):
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.vocab_size = None
        self.history = None
    
    def prepare_data(self, df, test_size=0.3, val_split=0.5):
        print("Preparing data for Event Time Prediction...")
        
        required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) != len(required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {available_cols}")
        
        df = df[required_cols].copy()
        
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        
        df = self._calculate_temporal_features(df)
        
        self.label_encoder.fit(df['concept:name'])
        df['activity_encoded'] = self.label_encoder.transform(df['concept:name'])
        
        grouped = df.groupby('case:concept:name')
        
        sequences = []
        temporal_features = []
        next_event_times = []
        
        for case_id, group in grouped:
            activities = group['activity_encoded'].values
            timestamps = group['time:timestamp'].values
            fvt1_vals = group['fvt1'].values
            fvt2_vals = group['fvt2'].values
            fvt3_vals = group['fvt3'].values
            
            for i in range(1, len(activities)):
                seq = activities[:i]
                temp_feat = [fvt1_vals[i], fvt2_vals[i], fvt3_vals[i]]
                time_to_next = (timestamps[i] - timestamps[i-1]).astype('timedelta64[s]').astype(float) / 86400
                
                sequences.append(seq)
                temporal_features.append(temp_feat)
                next_event_times.append(time_to_next)
        
        print(f"Total training samples: {len(sequences):,}")
        print(f"Example sequence length range: {min(map(len, sequences))} to {max(map(len, sequences))}")
        
        X_seq = keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_len, padding='pre', value=0
        )
        
        X_seq = X_seq + 1
        X_temp = np.array(temporal_features)
        y_event_time = np.array(next_event_times)
        
        X_temp_scaled = self.scaler.fit_transform(X_temp)
        
        self.vocab_size = len(self.label_encoder.classes_) + 2
        
        print(f"\nSequence shape: {X_seq.shape}")
        print(f"Temporal features shape: {X_temp_scaled.shape}")
        print(f"Event time targets shape: {y_event_time.shape}")
        print(f"Vocabulary size: {self.vocab_size}")

        X_seq_train, X_seq_temp, X_temp_train, X_temp_temp, y_train, y_temp = train_test_split(
            X_seq, X_temp_scaled, y_event_time, test_size=test_size, random_state=42
        )
        
        X_seq_val, X_seq_test, X_temp_val, X_temp_test, y_val, y_test = train_test_split(
            X_seq_temp, X_temp_temp, y_temp, test_size=val_split, random_state=42
        )
        
        print(f"\nDataset splits:")
        print(f"Train: {len(X_seq_train):,} samples")
        print(f"Validation: {len(X_seq_val):,} samples")
        print(f"Test: {len(X_seq_test):,} samples")
        
        return {
            'X_seq_train': X_seq_train, 'X_temp_train': X_temp_train, 'y_train': y_train,
            'X_seq_val': X_seq_val, 'X_temp_val': X_temp_val, 'y_val': y_val,
            'X_seq_test': X_seq_test, 'X_temp_test': X_temp_test, 'y_test': y_test
        }
    
    def _calculate_temporal_features(self, df):
        def calculate_features(group):
            group = group.sort_values('time:timestamp').reset_index(drop=True)
            
            group['fvt1'] = group['time:timestamp'].diff().dt.total_seconds() / 86400
            group['fvt1'].fillna(0, inplace=True)
            
            group['fvt2'] = (group['time:timestamp'] - group['time:timestamp'].shift(2)).dt.total_seconds() / 86400
            group['fvt2'].fillna(0, inplace=True)
            
            group['fvt3'] = (group['time:timestamp'] - group['time:timestamp'].iloc[0]).dt.total_seconds() / 86400
            
            return group
        
        df = df.groupby('case:concept:name', group_keys=False).apply(calculate_features)
        print("Temporal features created.")
        return df
    
    def build_model(self, use_timestep_explainability=True):
        """
        Build Event Time Prediction Model.
        
        Args:
            use_timestep_explainability: If True (default), enables timestep-level 
                                        explanations for SHAP/LIME
        """
        print("\nBuilding Event Time Prediction Model...")
        print(f"Timestep explainability: {'ENABLED' if use_timestep_explainability else 'DISABLED'}")
        
        self.model = build_time_prediction_model(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            use_timestep_explainability=use_timestep_explainability  # NEW!
        )

        # Handle compilation based on model type
        if len(self.model.outputs) > 1:
            # Timestep-explainable model (2 outputs)
            print("Model type: Timestep-explainable (2 outputs)")
            # Compile with loss only on the first output (time_output)
            # The second output (timestep predictions) is for explainability only
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=['mae', None],  # Loss only on first output, None for second
                metrics={'time_output': ['mae']}
            )
        else:
            # Original model (1 output)
            print("Model type: Original (1 output)")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mae',
                metrics=['mae']
            )
        
        print("\nModel Summary:")
        self.model.summary()
    
    def train(self, data, epochs=50, batch_size=128, patience=10):
        print(f"\nTraining model for {epochs} epochs...")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Handle training based on model type
        if len(self.model.outputs) > 1:
            # Timestep-explainable model - provide targets as list
            self.history = self.model.fit(
                [data['X_seq_train'], data['X_temp_train']], 
                [data['y_train'], None],  # Target for first output, None for second
                validation_data=(
                    [data['X_seq_val'], data['X_temp_val']], 
                    [data['y_val'], None]  # Target for first output, None for second
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            # Original model
            self.history = self.model.fit(
                [data['X_seq_train'], data['X_temp_train']], 
                data['y_train'],
                validation_data=([data['X_seq_val'], data['X_temp_val']], data['y_val']),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self, data):
        print("\nEvaluating on test set...")
        
        # Handle evaluation based on model type
        if len(self.model.outputs) > 1:
            # Timestep-explainable model
            results = self.model.evaluate(
                [data['X_seq_test'], data['X_temp_test']], 
                [data['y_test'], None],  # Target for first output, None for second
                verbose=0
            )
            # results is [total_loss, output1_loss, output1_mae]
            test_loss = results[1] if len(results) > 1 else results[0]
            test_mae = results[2] if len(results) > 2 else results[-1]
        else:
            # Original model
            test_loss, test_mae = self.model.evaluate(
                [data['X_seq_test'], data['X_temp_test']], 
                data['y_test'], 
                verbose=0
            )
        
        print(f"Test MAE: {test_mae:.4f} days")
        print(f"Test Loss: {test_loss:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_mae': test_mae
        }
    
    def predict(self, data):
        """
        Generate predictions.
        
        Returns:
            y_pred: Final time predictions (n_samples,)
            timestep_preds: Per-timestep predictions (n_samples, max_len) or None
        """
        print("\nGenerating predictions...")
        
        outputs = self.model.predict(
            [data['X_seq_test'], data['X_temp_test']], 
            verbose=0
        )
        
        # Handle both single and multi-output models
        if isinstance(outputs, list) and len(outputs) > 1:
            # Timestep-explainable model
            y_pred = outputs[0].flatten()  # Final prediction
            timestep_preds = outputs[1]     # Per-timestep predictions
            print(f"Predictions generated for {len(data['X_seq_test']):,} test samples.")
            print(f"  - Final predictions shape: {y_pred.shape}")
            print(f"  - Timestep predictions shape: {timestep_preds.shape}")
            return y_pred, timestep_preds
        else:
            # Original model
            y_pred = outputs.flatten() if hasattr(outputs, 'flatten') else outputs
            print(f"Predictions generated for {len(data['X_seq_test']):,} test samples.")
            return y_pred, None
    
    def save_results(self, data, y_pred, output_dir):
        print("\nSaving results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle both tuple and single return from predict()
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Extract final predictions
        
        results = pd.DataFrame({
            'actual_event_time_days': data['y_test'],
            'predicted_event_time_days': y_pred,
            'absolute_error_days': np.abs(data['y_test'] - y_pred)
        })
        
        output_path = os.path.join(output_dir, "event_time_predictions.csv")
        results.to_csv(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
    
    def plot_predictions(self, data, y_pred, output_dir):
        print("\nPlotting predictions...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle both tuple and single return from predict()
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Extract final predictions
        
        plt.figure(figsize=(10, 6))
        plt.scatter(data['y_test'], y_pred, alpha=0.5, s=10)
        plt.plot([data['y_test'].min(), data['y_test'].max()], 
                 [data['y_test'].min(), data['y_test'].max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Event Time (days)', fontsize=12)
        plt.ylabel('Predicted Event Time (days)', fontsize=12)
        
        mae = np.mean(np.abs(data['y_test'] - y_pred))
        plt.title(f'Event Time Prediction\nMAE: {mae:.4f} days', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "event_time_predictions_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions plot saved to: {output_path}")
    
    def plot_training_history(self, output_dir):
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        print("\nPlotting training history...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle different history key formats
        if 'loss' in self.history.history:
            train_loss_key = 'loss'
            val_loss_key = 'val_loss'
        elif 'time_output_loss' in self.history.history:
            train_loss_key = 'time_output_loss'
            val_loss_key = 'val_time_output_loss'
        else:
            print("Could not find loss keys in history")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history[train_loss_key], label="Training Loss", linewidth=2)
        plt.plot(self.history.history[val_loss_key], label="Validation Loss", linewidth=2)
        plt.title("Event Time Prediction - Loss Over Time", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MAE)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "event_time_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Training history plot saved to: {output_path}")
        
        print("\nTraining metrics:")
        print(f"Final training loss: {self.history.history[train_loss_key][-1]:.4f}")
        print(f"Final validation loss: {self.history.history[val_loss_key][-1]:.4f}")
        print(f"Best validation loss: {min(self.history.history[val_loss_key]):.4f}")
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "event_time_transformer.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")