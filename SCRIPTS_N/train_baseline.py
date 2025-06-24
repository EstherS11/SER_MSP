#!/usr/bin/env python3
"""
Improved MSP-PODCAST emotion classification training script
Using WavLM + ECAPA-TDNN
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimpleMSPEmotionBrain(sb.Brain):
    """Improved MSP-PODCAST emotion classification Brain"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize metrics
        self.error_metrics = self.hparams.error_stats()
        # For macro-F1 calculation
        self.predictions = []
        self.targets = []
        
    def compute_forward(self, batch, stage):
        """计算前向传播"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # 使用WavLM提取特征
        with torch.no_grad():
            feats = self.modules.ssl_model(wavs)
            
            # WavLM返回多个输出时的处理
            if isinstance(feats, tuple):
                feats = feats[0]
        
        # 保持时序信息！feats shape: (batch, time, features)
        # 投影层处理每个时间步的特征
        feats = self.modules.feature_projection(feats)
        
        # 计算特征序列的实际长度
        wav_lens_ratio = wav_lens / wavs.shape[1]
        feat_lens = torch.round(wav_lens_ratio * feats.shape[1])
        
        # ECAPA-TDNN处理完整的特征序列
        # ECAPA-TDNN会在内部进行时序建模和池化
        embeddings = self.modules.embedding_model(feats, feat_lens)
        
        # 分类器接收ECAPA-TDNN输出的固定维度嵌入
        outputs = self.modules.classifier(embeddings)
        outputs = self.hparams.log_softmax(outputs)
        
        return outputs
    
    def length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """将长度转换为掩码"""
        assert len(length.shape) == 1
        
        if max_len is None:
            max_len = length.max().long().item()
        
        if device is None:
            device = length.device
            
        mask = torch.arange(max_len, device=device, dtype=length.dtype)\
                    .expand(len(length), max_len) < length.unsqueeze(1)
        
        if dtype is not None:
            mask = mask.to(dtype)
        
        return mask
    
    def compute_objectives(self, predictions, batch, stage):
        """Compute loss and metrics"""
        emo_ids = batch.emo_id
        
        # Compute loss
        loss = self.hparams.compute_cost(predictions, emo_ids)
        
        # For evaluation stages, compute error rate and collect predictions
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emo_ids)
            
            # Collect predictions for macro-F1
            pred_labels = predictions.argmax(dim=-1)
            self.predictions.extend(pred_labels.cpu().numpy())
            self.targets.extend(emo_ids.cpu().numpy())
        
        # Log training statistics
        if stage == sb.Stage.TRAIN and hasattr(self, 'step') and self.step % 100 == 0:
            with torch.no_grad():
                pred_labels = predictions.argmax(dim=-1)
                acc = (pred_labels == emo_ids).float().mean()
                logger.info(f"Step {self.step} - Loss: {loss:.4f}, Acc: {acc:.2%}")
        
        return loss
    
    def on_stage_start(self, stage, epoch):
        """Stage start processing"""
        if stage != sb.Stage.TRAIN:
            # Reset metrics
            self.error_metrics = self.hparams.error_stats()
            self.predictions = []
            self.targets = []
            
        # Set model state
        if stage == sb.Stage.TRAIN:
            self.modules.train()
            # SSL model always in eval mode
            self.modules.ssl_model.eval()
        else:
            self.modules.eval()
            
        logger.info(f"Starting {stage} stage, Epoch {epoch}")
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Stage end processing"""
        # Store statistics
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # Calculate error rate (keep for compatibility)
            stage_stats["error_rate"] = self.error_metrics.summarize("average")
            
            # Calculate detailed metrics
            if len(self.predictions) > 0:
                # Macro F1
                stage_stats["macro_f1"] = f1_score(
                    self.targets, 
                    self.predictions, 
                    average='macro'
                )
                
                # Weighted F1 (accounts for class imbalance)
                stage_stats["weighted_f1"] = f1_score(
                    self.targets,
                    self.predictions,
                    average='weighted'
                )
                
                # Per-class metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    self.targets,
                    self.predictions,
                    average=None,
                    zero_division=0
                )
                
                # Store per-class F1 scores
                stage_stats["per_class_f1"] = f1
                stage_stats["per_class_support"] = support
                
                # Calculate macro precision and recall
                stage_stats["macro_precision"] = precision.mean()
                stage_stats["macro_recall"] = recall.mean()
                
                # Calculate minority class performance (classes with < 10% of data)
                total_samples = len(self.targets)
                minority_mask = support < (0.1 * total_samples)
                if minority_mask.any():
                    stage_stats["minority_f1"] = f1[minority_mask].mean()
                else:
                    stage_stats["minority_f1"] = 0.0
                    
            else:
                stage_stats["macro_f1"] = 0.0
                stage_stats["weighted_f1"] = 0.0
                stage_stats["macro_precision"] = 0.0
                stage_stats["macro_recall"] = 0.0
                stage_stats["minority_f1"] = 0.0
            
        # Validation: perform learning rate scheduling
        if stage == sb.Stage.VALID:
            # Use macro-F1 for scheduling (higher is better, so use negative)
            metric_for_scheduler = -stage_stats["macro_f1"]
            old_lr, new_lr = self.hparams.lr_annealing_model(metric_for_scheduler)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            
            # Log statistics
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            
            # Save checkpoint
            self.checkpointer.save_and_keep_only(
                meta={
                    "macro_f1": stage_stats["macro_f1"],
                    "weighted_f1": stage_stats["weighted_f1"],
                    "epoch": epoch
                }, 
                min_keys=["error_rate"],
                max_keys=["macro_f1", "weighted_f1"],
                num_to_keep=3
            )
            
            # Log meaningful metrics
            logger.info(f"Valid - Epoch {epoch}")
            logger.info(f"  Loss: {stage_loss:.4f}")
            logger.info(f"  Macro F1: {stage_stats['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {stage_stats['weighted_f1']:.4f}")
            logger.info(f"  Macro Precision: {stage_stats['macro_precision']:.4f}")
            logger.info(f"  Macro Recall: {stage_stats['macro_recall']:.4f}")
            if stage_stats['minority_f1'] > 0:
                logger.info(f"  Minority Classes F1: {stage_stats['minority_f1']:.4f}")
        
        # Test: report results with detailed breakdown
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            logger.info("="*60)
            logger.info("TEST RESULTS")
            logger.info("="*60)
            logger.info(f"Loss: {stage_loss:.4f}")
            logger.info(f"Macro F1: {stage_stats['macro_f1']:.4f}")
            logger.info(f"Weighted F1: {stage_stats['weighted_f1']:.4f}")
            logger.info(f"Macro Precision: {stage_stats['macro_precision']:.4f}")
            logger.info(f"Macro Recall: {stage_stats['macro_recall']:.4f}")
            
            if stage_stats['minority_f1'] > 0:
                logger.info(f"Minority Classes F1: {stage_stats['minority_f1']:.4f}")
            
            # Print per-class performance
            if "per_class_f1" in stage_stats:
                logger.info("\nPer-class F1 scores:")
                emotion_names = ['Neutral', 'Happy', 'Excited', 'Angry', 'Sad',
                               'Surprise', 'Contempt', 'Other', 'Disgust', 'Fear']
                for i, (f1, support) in enumerate(zip(stage_stats["per_class_f1"], 
                                                     stage_stats["per_class_support"])):
                    if i < len(emotion_names):
                        logger.info(f"  {emotion_names[i]:10s}: F1={f1:.3f} (n={int(support)})")
                
                # Print confusion matrix
                logger.info("\nClassification Report:")
                report = classification_report(
                    self.targets,
                    self.predictions,
                    target_names=emotion_names[:len(np.unique(self.targets))],
                    digits=3
                )
                logger.info("\n" + report)
            
            logger.info("="*60)
    
    def create_optimizers(self):
        """Create optimizers"""
        # Get all parameters to optimize
        params = []
        for module in self.modules.values():
            # Skip frozen SSL model
            if module == self.modules.ssl_model:
                continue
            params.extend(module.parameters())
        
        # Create optimizer
        if hasattr(self.hparams, "model_opt_class"):
            self.optimizer = self.hparams.model_opt_class(params)
        else:
            self.optimizer = self.hparams.opt_class(params)
            
        logger.info(f"Optimizer created with {len(params)} parameter groups")
    
    def on_fit_start(self):
        """Processing before training starts"""
        super().on_fit_start()
        
        # Ensure SSL model is in eval mode and parameters are frozen
        self.modules.ssl_model.eval()
        for param in self.modules.ssl_model.parameters():
            param.requires_grad = False
            
        logger.info("SSL model frozen")
        
        # Print model information
        total_params = sum(p.numel() for p in self.modules.parameters())
        trainable_params = sum(p.numel() for p in self.modules.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

def dataio_prepare(hparams):
    """Prepare data loaders"""
    
    # Audio processing pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        try:
            # Read audio
            sig = sb.dataio.dataio.read_audio(wav)
            
            # Ensure tensor is not empty
            if sig.shape[0] == 0:
                logger.warning(f"Empty audio file: {wav}")
                sig = torch.zeros(int(hparams["sample_rate"]))  # 1 second of silence
            
            # Check duration
            duration = sig.shape[0] / hparams["sample_rate"]
            min_duration = 0.5
            max_duration = 30
            
            if duration < min_duration:
                # Pad short audio
                pad_length = int(hparams["sample_rate"] * min_duration) - sig.shape[0]
                sig = torch.nn.functional.pad(sig, (0, pad_length), mode='constant', value=0)
            elif duration > max_duration:
                # Truncate long audio
                sig = sig[:int(hparams["sample_rate"] * max_duration)]
            
            # Normalize
            max_val = sig.abs().max()
            if max_val > 0:
                sig = sig / max_val * 0.95
            
            # Ensure 2D tensor (batch_size=1, time)
            if len(sig.shape) == 1:
                sig = sig.unsqueeze(0)
                
            return sig
            
        except Exception as e:
            logger.error(f"Failed to read audio {wav}: {str(e)}")
            # Return 1 second of silence with proper shape
            return torch.zeros(1, int(hparams["sample_rate"]))
    
    # Label processing pipeline
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo_id")
    def label_pipeline(emo):
        try:
            # 10-class emotion mapping
            emo_map = {
                'N': 0,  # Neutral
                'H': 1,  # Happy
                'X': 2,  # Excited
                'A': 3,  # Angry
                'S': 4,  # Sad
                'U': 5,  # Surprise
                'C': 6,  # Contempt
                'O': 7,  # Other
                'D': 8,  # Disgust
                'F': 9   # Fear
            }
            
            if emo not in emo_map:
                logger.warning(f"Unknown emotion label: {emo}, defaulting to neutral")
                return torch.tensor(0, dtype=torch.long)
                
            return torch.tensor(emo_map[emo], dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Failed to process emotion label {emo}: {str(e)}")
            return torch.tensor(0, dtype=torch.long)
    
    # Create datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    for dataset in data_info:
        logger.info(f"Loading {dataset} dataset...")
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_id"],
        )
        logger.info(f"{dataset} dataset size: {len(datasets[dataset])}")
    
    # Note: Sorting can cause issues with variable length sequences in some cases
    # If you encounter batching errors, comment out the sorting code
    sorting = hparams.get("sorting", "random")
    if sorting == "ascending":
        try:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration",  # Use 'duration' instead of 'length'
                reverse=False
            )
            hparams["train_dataloader_opts"]["shuffle"] = False
            logger.info("Training set sorted by duration")
        except Exception as e:
            logger.warning(f"Could not sort dataset: {e}. Using random order.")
            hparams["train_dataloader_opts"]["shuffle"] = True
    
    return datasets

def check_data_integrity(datasets, hparams):
    """Check data integrity"""
    emotion_names = {
        0: 'Neutral', 1: 'Happy', 2: 'Excited', 3: 'Angry', 4: 'Sad',
        5: 'Surprise', 6: 'Contempt', 7: 'Other', 8: 'Disgust', 9: 'Fear'
    }
    
    for split_name, dataset in datasets.items():
        logger.info(f"Checking {split_name} dataset...")
        
        # Statistics
        label_counts = {}
        error_count = 0
        
        # Sample check
        sample_size = min(100, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for idx in tqdm(indices, desc=f"Checking {split_name}"):
            try:
                item = dataset[int(idx)]
                
                # Check required fields
                assert "sig" in item, "Missing audio signal"
                assert "emo_id" in item, "Missing emotion label"
                assert item["sig"].shape[0] > 0, "Empty audio signal"
                
                # Count labels
                label = item["emo_id"].item()
                label_counts[label] = label_counts.get(label, 0) + 1
                
            except Exception as e:
                logger.error(f"{split_name} dataset item {idx} has issues: {e}")
                error_count += 1
        
        # Report statistics
        logger.info(f"{split_name} dataset statistics:")
        logger.info(f"  - Total samples: {len(dataset)}")
        logger.info(f"  - Checked samples: {sample_size}")
        logger.info(f"  - Error samples: {error_count}")
        logger.info(f"  - Label distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = count / sample_size * 100
            logger.info(f"    {emotion_names[label]}: {count} ({percentage:.1f}%)")
        
        if error_count > sample_size * 0.1:  # Error rate > 10%
            logger.warning(f"{split_name} dataset has high error rate!")

def main():
    """Main training function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    
    logger.info("="*60)
    logger.info("MSP-PODCAST Emotion Classification")
    logger.info("="*60)
    
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Setup device
    if run_opts.get("device") is None:
        if torch.cuda.is_available():
            run_opts["device"] = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            run_opts["device"] = "cpu"
            logger.warning("CUDA not available, using CPU for training")
    
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Prepare datasets
    try:
        datasets = dataio_prepare(hparams)
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return
    
    # Check data integrity
    check_data_integrity(datasets, hparams)
    
    # Initialize Brain
    emotion_brain = SimpleMSPEmotionBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Create optimizers
    emotion_brain.create_optimizers()
    
    # Train
    try:
        emotion_brain.fit(
            epoch_counter=emotion_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    # Test
    logger.info("\nStarting final evaluation...")
    test_stats = emotion_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    
    logger.info("\nTraining completed successfully!")

if __name__ == "__main__":
    main()