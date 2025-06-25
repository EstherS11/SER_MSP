import os
import json
import torch
import torchaudio
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from train_msp_podcast import WavLMECAPAClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SER_Inference')

class EnhancedEmotionPredictor:
    """Enhanced emotion predictor with long audio handling"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = WavLMECAPAClassifier(num_classes=10).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
        
        # Configuration
        self.emotion_map = {
            0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fear",
            5: "disgust", 6: "surprise", 7: "contempt", 8: "other", 9: "unknown"
        }
        
        self.sample_rate = 16000
        self.window_length = 10.0  # seconds
        self.window_samples = int(self.window_length * self.sample_rate)
        self.hop_length = 5.0  # seconds for sliding window
        self.hop_samples = int(self.hop_length * self.sample_rate)
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        logger.debug(f"Loading audio: {audio_path}")
        
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            logger.debug(f"Resampled from {sr}Hz to {self.sample_rate}Hz")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug("Converted to mono")
        
        return waveform.squeeze(0)
    
    def predict(self, audio_path: str) -> Dict:
        """Predict emotion for a single audio file"""
        waveform = self.preprocess_audio(audio_path)
        
        # Handle variable length audio
        if waveform.shape[0] <= self.window_samples:
            # Short audio: pad to window length
            return self._predict_single_window(waveform)
        else:
            # Long audio: use sliding window
            return self.predict_long_audio(audio_path)
    
    def _predict_single_window(self, waveform: torch.Tensor) -> Dict:
        """Predict emotion for a single window"""
        # Pad if necessary
        if waveform.shape[0] < self.window_samples:
            pad_length = self.window_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:self.window_samples]
        
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(waveform)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        return {
            'emotion': self.emotion_map[pred_idx],
            'confidence': confidence,
            'probabilities': {
                self.emotion_map[i]: probs[0, i].item() 
                for i in range(10)
            }
        }
    
    def predict_long_audio(self, audio_path: str, 
                          aggregation: str = 'weighted_avg') -> Dict:
        """
        Predict emotion for long audio using sliding window
        
        Args:
            audio_path: Path to audio file
            aggregation: Method to aggregate predictions
                - 'weighted_avg': Weighted average by confidence
                - 'avg': Simple average
                - 'voting': Majority voting
        """
        logger.info(f"Processing long audio with {aggregation} aggregation")
        
        waveform = self.preprocess_audio(audio_path)
        audio_length = waveform.shape[0] / self.sample_rate
        logger.info(f"Audio length: {audio_length:.2f} seconds")
        
        # Extract windows
        windows = []
        window_times = []
        
        for start in range(0, len(waveform) - self.window_samples + 1, self.hop_samples):
            end = start + self.window_samples
            window = waveform[start:end]
            windows.append(window)
            window_times.append((start / self.sample_rate, end / self.sample_rate))
        
        # If last window doesn't cover the end, add one more
        if len(waveform) > end:
            start = len(waveform) - self.window_samples
            window = waveform[start:]
            windows.append(window)
            window_times.append((start / self.sample_rate, len(waveform) / self.sample_rate))
        
        logger.info(f"Extracted {len(windows)} windows")
        
        # Predict for each window
        window_predictions = []
        all_probs = []
        
        for i, window in enumerate(windows):
            pred = self._predict_single_window(window)
            window_predictions.append(pred)
            all_probs.append([pred['probabilities'][self.emotion_map[j]] for j in range(10)])
            logger.debug(f"Window {i+1}: {pred['emotion']} (conf: {pred['confidence']:.3f})")
        
        # Aggregate predictions
        all_probs = np.array(all_probs)
        
        if aggregation == 'weighted_avg':
            # Weight by confidence
            confidences = np.array([p['confidence'] for p in window_predictions])
            weights = confidences / confidences.sum()
            final_probs = np.sum(all_probs * weights[:, np.newaxis], axis=0)
        
        elif aggregation == 'avg':
            # Simple average
            final_probs = np.mean(all_probs, axis=0)
        
        elif aggregation == 'voting':
            # Majority voting
            votes = [np.argmax(probs) for probs in all_probs]
            emotion_counts = np.bincount(votes, minlength=10)
            final_probs = emotion_counts / len(votes)
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Final prediction
        pred_idx = np.argmax(final_probs)
        confidence = final_probs[pred_idx]
        
        result = {
            'emotion': self.emotion_map[pred_idx],
            'confidence': float(confidence),
            'probabilities': {
                self.emotion_map[i]: float(final_probs[i]) 
                for i in range(10)
            },
            'window_predictions': window_predictions,
            'window_times': window_times,
            'aggregation_method': aggregation,
            'num_windows': len(windows),
            'audio_length': audio_length
        }
        
        return result
    
    def visualize_long_audio_prediction(self, result: Dict, save_path: str = None):
        """Visualize predictions across time for long audio"""
        if 'window_predictions' not in result:
            logger.warning("No window predictions to visualize")
            return
        
        # Extract data
        window_times = result['window_times']
        window_preds = result['window_predictions']
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Emotion predictions over time
        ax1 = axes[0]
        emotions = [p['emotion'] for p in window_preds]
        confidences = [p['confidence'] for p in window_preds]
        window_centers = [(t[0] + t[1]) / 2 for t in window_times]
        
        # Create emotion to color mapping
        emotion_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        emotion_to_color = {self.emotion_map[i]: emotion_colors[i] for i in range(10)}
        
        # Plot each window
        for i, (center, emotion, conf) in enumerate(zip(window_centers, emotions, confidences)):
            ax1.scatter(center, emotion, 
                       c=[emotion_to_color[emotion]], 
                       s=conf*500,  # Size proportional to confidence
                       alpha=0.7)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Predicted Emotion')
        ax1.set_title('Emotion Predictions Over Time (size = confidence)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Probability distribution over time
        ax2 = axes[1]
        
        # Create probability matrix
        prob_matrix = np.array([[p['probabilities'][self.emotion_map[j]] 
                                for j in range(10)] 
                               for p in window_preds]).T
        
        # Plot heatmap
        im = ax2.imshow(prob_matrix, aspect='auto', 
                       extent=[window_centers[0], window_centers[-1], -0.5, 9.5],
                       cmap='YlOrRd')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Emotion')
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(list(self.emotion_map.values()))
        ax2.set_title('Emotion Probability Distribution Over Time')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Probability')
        
        # Add final prediction
        fig.suptitle(f"Final Prediction: {result['emotion']} (conf: {result['confidence']:.3f})")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def predict_batch(self, audio_paths: List[str], 
                     batch_size: int = 32,
                     use_sliding_window: bool = True) -> List[Dict]:
        """Batch prediction with progress bar"""
        results = []
        
        logger.info(f"Processing {len(audio_paths)} audio files")
        
        for audio_path in tqdm(audio_paths, desc="Processing audio files"):
            try:
                if use_sliding_window:
                    result = self.predict(audio_path)
                else:
                    waveform = self.preprocess_audio(audio_path)
                    result = self._predict_single_window(waveform)
                
                result['file'] = audio_path
                result['status'] = 'success'
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {str(e)}")
                result = {
                    'file': audio_path,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        return results

def evaluate_on_test_set(model_path: str, test_json: str, data_root: str, 
                        output_dir: str = 'test_results'):
    """Evaluate model on test set with enhanced metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = EnhancedEmotionPredictor(model_path)
    
    # Load test data
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    emotion_map = {
        'N': 0, 'H': 1, 'S': 2, 'A': 3, 'F': 4,
        'D': 5, 'U': 6, 'C': 7, 'O': 8, 'X': 9
    }
    
    # Prepare data
    audio_paths = []
    true_labels = []
    file_keys = []
    
    for key, item in test_data.items():
        audio_path = os.path.join(data_root, item['wav'])
        audio_paths.append(audio_path)
        true_labels.append(emotion_map[item['emo']])
        file_keys.append(key)
    
    # Batch prediction
    logger.info(f"Evaluating on {len(audio_paths)} test samples...")
    predictions = predictor.predict_batch(audio_paths, use_sliding_window=False)
    
    # Filter successful predictions
    successful_predictions = [p for p in predictions if p['status'] == 'success']
    failed_predictions = [p for p in predictions if p['status'] == 'error']
    
    if failed_predictions:
        logger.warning(f"{len(failed_predictions)} predictions failed")
        with open(output_dir / 'failed_predictions.json', 'w') as f:
            json.dump(failed_predictions, f, indent=4)
    
    # Extract results
    pred_labels = []
    confidences = []
    valid_indices = []
    
    for i, pred in enumerate(predictions):
        if pred['status'] == 'success':
            pred_emotion = pred['emotion']
            pred_idx = [k for k, v in predictor.emotion_map.items() if v == pred_emotion][0]
            pred_labels.append(pred_idx)
            confidences.append(pred['confidence'])
            valid_indices.append(i)
    
    # Filter true labels
    true_labels_filtered = [true_labels[i] for i in valid_indices]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(true_labels_filtered, pred_labels)
    macro_f1 = f1_score(true_labels_filtered, pred_labels, average='macro')
    weighted_f1 = f1_score(true_labels_filtered, pred_labels, average='weighted')
    
    logger.info(f"\nTest Results:")
    logger.info(f"Samples evaluated: {len(pred_labels)}/{len(audio_paths)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Average Confidence: {np.mean(confidences):.4f}")
    
    # Detailed classification report
    emotion_names = list(predictor.emotion_map.values())
    report = classification_report(true_labels_filtered, pred_labels, 
                                 target_names=emotion_names, 
                                 output_dict=True)
    
    # Print report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels_filtered, pred_labels, 
                              target_names=emotion_names))
    
    # Save results
    results_df = pd.DataFrame({
        'file_key': [file_keys[i] for i in valid_indices],
        'true_emotion': [predictor.emotion_map[label] for label in true_labels_filtered],
        'pred_emotion': [pred['emotion'] for pred in successful_predictions],
        'confidence': confidences,
        'correct': [t == p for t, p in zip(true_labels_filtered, pred_labels)]
    })
    
    # Add probability columns
    for emotion in emotion_names:
        results_df[f'prob_{emotion}'] = [
            pred['probabilities'][emotion] for pred in successful_predictions
        ]
    
    results_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'test_predictions.csv'}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels_filtered, pred_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title(f'Confusion Matrix (Macro F1: {macro_f1:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # Save summary
    summary = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'avg_confidence': float(np.mean(confidences)),
            'samples_evaluated': len(pred_labels),
            'total_samples': len(audio_paths)
        },
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"\nAll results saved to: {output_dir}")
    
    return results_df, report

def main():
    parser = argparse.ArgumentParser(description='Enhanced inference for emotion recognition')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'batch', 'test', 'long_audio'], 
                       default='test')
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--audio_list', type=str)
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--aggregation', type=str, 
                       choices=['weighted_avg', 'avg', 'voting'], 
                       default='weighted_avg')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        predictor = EnhancedEmotionPredictor(args.model_path)
        result = predictor.predict(args.audio_path)
        
        print(f"\nPrediction for: {args.audio_path}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.4f})")
        
        if 'num_windows' in result:
            print(f"Audio length: {result['audio_length']:.2f}s")
            print(f"Number of windows: {result['num_windows']}")
            print(f"Aggregation method: {result['aggregation_method']}")
        
        print("\nAll probabilities:")
        for emotion, prob in sorted(result['probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")
        
        if args.visualize and 'window_predictions' in result:
            save_path = os.path.join(args.output_dir, 'prediction_timeline.png')
            os.makedirs(args.output_dir, exist_ok=True)
            predictor.visualize_long_audio_prediction(result, save_path)
    
    elif args.mode == 'long_audio':
        predictor = EnhancedEmotionPredictor(args.model_path)
        result = predictor.predict_long_audio(args.audio_path, 
                                             aggregation=args.aggregation)
        
        print(f"\nLong audio prediction for: {args.audio_path}")
        print(f"Audio length: {result['audio_length']:.2f} seconds")
        print(f"Number of windows: {result['num_windows']}")
        print(f"Aggregation method: {result['aggregation_method']}")
        print(f"\nFinal prediction: {result['emotion']} (confidence: {result['confidence']:.4f})")
        
        if args.visualize:
            save_path = os.path.join(args.output_dir, 'long_audio_timeline.png')
            os.makedirs(args.output_dir, exist_ok=True)
            predictor.visualize_long_audio_prediction(result, save_path)
    
    elif args.mode == 'batch':
        with open(args.audio_list, 'r') as f:
            audio_paths = [line.strip() for line in f]
        
        predictor = EnhancedEmotionPredictor(args.model_path)
        results = predictor.predict_batch(audio_paths)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'batch_predictions.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Batch predictions saved to: {args.output_dir}/batch_predictions.json")
    
    elif args.mode == 'test':
        if os.path.exists('/data/user_data/esthers/SER_MSP'):
            data_root = '/data/user_data/esthers/SER_MSP'
        else:
            data_root = '/Users/esthersun/Desktop/SER/SER_MSP'
        
        test_json = os.path.join(data_root, 'msp_test_10class.json')
        evaluate_on_test_set(args.model_path, test_json, data_root, args.output_dir)

if __name__ == "__main__":
    main()