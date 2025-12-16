import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ..model import build_next_activity_model


class NextActivityPredictor:
    
    def __init__(self, max_len=16, d_model=64, num_heads=4, num_blocks=2, 
                 dropout_rate=0.1):
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.label_encoder = LabelEncoder()
        self.model = None
        self.vocab_size = None
        self.history = None
    
    def prepare_data(self, df, test_size=0.3, val_split=0.5):
        print("Preparing data for Next Activity Prediction...")
        process_data = df[['case:id', 'concept:name', 'time:timestamp']].copy()
        process_data = process_data.rename(columns={
            'case:id': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp'
        })
        process_data['timestamp'] = pd.to_datetime(process_data['timestamp'])
        process_data = process_data.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        sequences, next_activities, metadata = self._create_sequences_with_prefixes(process_data)
        
        print(f"Total sequences: {len(sequences):,}")
        print(f"Max sequence length: {metadata['max_len']}")
        
        self.label_encoder.fit(process_data['activity'])
        
        X_encoded = []
        for seq in sequences:
            encoded_seq = self.label_encoder.transform(seq)
            X_encoded.append(encoded_seq)
        
        y_encoded = self.label_encoder.transform(next_activities)
        
        X = keras.preprocessing.sequence.pad_sequences(
            X_encoded, maxlen=self.max_len, padding='pre', value=0
        )
        
        X = X + 1
        y = y_encoded + 1
        self.vocab_size = len(self.label_encoder.classes_) + 2  # +1 for encoding, +1 for padding
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Number of unique activities: {len(self.label_encoder.classes_)}")
        # Split data: train, validation, test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=42
        )
        print(f"\nDataset splits:")
        print(f"Train: {len(X_train):,} samples")
        print(f"Validation: {len(X_val):,} samples")
        print(f"Test: {len(X_test):,} samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def _create_sequences_with_prefixes(self, data):
        sequences = []
        next_activities = []
        case_ids_list = []
        prefix_lengths = []
        
        grouped = data.groupby('case_id')
        for case_id, group in grouped:
            activities = group['activity'].values
            for i in range(1, len(activities)):
                prefix = activities[:i]
                next_activity = activities[i]
                
                sequences.append(prefix)
                next_activities.append(next_activity)
                case_ids_list.append(case_id)
                prefix_lengths.append(len(prefix))
        
        max_len = max(len(seq) for seq in sequences)
        
        metadata = {
            'case_ids': case_ids_list,
            'prefix_lengths': prefix_lengths,
            'max_len': max_len
        }
        return sequences, next_activities, metadata
    
    def build_model(self):
        print("\nBuilding Next Activity Prediction Model...")
        
        self.model = build_next_activity_model(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate
        )
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Summary:")
        self.model.summary()
    
    def train(self, data, epochs=50, batch_size=128, patience=10):
        print(f"\nTraining model for {epochs} epochs...")
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self, data):
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(
            data['X_test'], data['y_test'], verbose=0
        )
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test loss: {test_loss:.4f}")
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def predict(self, data):
        print("\nGenerating predictions...")
        
        y_pred_probs = self.model.predict(data['X_test'], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        print(f"Predictions generated for {len(data['X_test']):,} test samples.")
        return y_pred, y_pred_probs
    
    def save_results(self, data, y_pred, y_pred_probs, output_dir):
        print("\nSaving results...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results
        results = []
        for i in range(len(data['X_test'])):
            seq = data['X_test'][i]
            seq = seq[seq > 0]  # Remove padding
            seq = seq - 1  # Undo encoding offset
            decoded_seq = self.label_encoder.inverse_transform(seq)
            
            true_decoded = self.label_encoder.inverse_transform([data['y_test'][i] - 1])[0]
            pred_decoded = self.label_encoder.inverse_transform([y_pred[i] - 1])[0]
            confidence = y_pred_probs[i][y_pred[i]] * 100
            
            results.append({
                "sequence": ", ".join(decoded_seq),
                "true_next_activity": true_decoded,
                "predicted_next_activity": pred_decoded,
                "confidence_percent": round(confidence, 2)
            })
        
        results_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, "next_activity_predictions.csv")
        results_df.to_csv(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
    
    def plot_training_history(self, output_dir):
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        print("\nPlotting training history...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.history.history["accuracy"], label="Training Accuracy", linewidth=2)
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax1.set_title("Model Accuracy Over Time", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history["loss"], label="Training Loss", linewidth=2)
        ax2.plot(self.history.history["val_loss"], label="Validation Loss", linewidth=2)
        ax2.set_title("Model Loss Over Time", fontsize=14)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "next_activity_training_history.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Training history plot saved to: {output_path}")
        
        # training metrics
        print("\nTraining metrics:")
        print(f"Final training accuracy: {self.history.history['accuracy'][-1] * 100:.2f}%")
        print(f"Final validation accuracy: {self.history.history['val_accuracy'][-1] * 100:.2f}%")
        print(f"Best validation accuracy: {max(self.history.history['val_accuracy']) * 100:.2f}%")
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "next_activity_transformer.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")