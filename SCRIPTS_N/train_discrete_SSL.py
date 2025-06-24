#!/usr/bin/env python3
"""
Ultimate training script for MSP-PODCAST emotion recognition using discrete SSL
Fusion of IEMOCAP official implementation + advanced features + MSP-PODCAST adaptations

Version: 1.1.0
Last Updated: 2025-06-24
Changes:
- Added support for non-discrete SSL mode
- Enhanced error handling for feature dimension issues
- Improved compatibility with different embedding model implementations

Key Features:
- IEMOCAP official attention computation (matrix multiplication)
- Proper codec calling with tokenizer_config
- Enhanced audio preprocessing with resampling
- Comprehensive analysis and logging
- MSP-PODCAST specific adaptations
- Robust error handling and validation

Usage:
    python train_discrete_SSL.py hparams/train_discrete_SSL.yaml
    python train_discrete_SSL.py hparams/train_discrete_SSL.yaml --attention_mode=iemocap
    python train_discrete_SSL.py hparams/train_discrete_SSL.yaml --data_folder=./data

Authors:
 * [Your Name]
 * Based on IEMOCAP official + enhanced features
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import custom_model
import numpy as np
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class MSPPodcastUltimateBrain(sb.Brain):
    """
    Ultimate Brain class combining IEMOCAP official implementation with enhancements
    
    Key improvements from IEMOCAP official:
    - Proper attention computation using matrix multiplication
    - Correct codec calling convention
    - Enhanced logging and analysis
    - MSP-PODCAST specific adaptations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # For comprehensive analysis
        self.attention_stats = []
        self.training_stats = {'epoch_losses': [], 'epoch_accs': []}
        self.best_valid_acc = 0.0
        self.codec_cache = None  # Cache codec for efficiency
        
    def compute_forward(self, batch, stage):
        """
        Forward computation using IEMOCAP official approach + enhancements
    
        Key differences from my previous version:
        1. Uses IEMOCAP's matrix multiplication for attention
        2. Proper codec calling with tokenizer_config
        3. More efficient computation pipeline
        4. Supports both discrete SSL and direct SSL modes
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
    
        # 检查是否使用 discrete_ssl
        use_discrete = hasattr(self.hparams, 'use_discrete_ssl') and self.hparams.use_discrete_ssl and hasattr(self.hparams, 'codec')
    
        if use_discrete:
            # Extract discrete tokens using IEMOCAP official approach
            with torch.no_grad():
                # Move codec to device and set to eval (IEMOCAP official approach)
                self.hparams.codec.to(self.device).eval()
            
                # Call codec with tokenizer_config (IEMOCAP official way)
                tokens, _, _ = self.hparams.codec(
                    wavs, wav_lens, **self.hparams.tokenizer_config
                )
        
            # Embed discrete tokens (our enhanced embedding layer)
            embeddings = self.modules.discrete_embedding_layer(tokens)
        
            # IEMOCAP official attention computation (THE KEY DIFFERENCE!)
            att_w = self.modules.attention_mlp(embeddings)
        
            # IEMOCAP's elegant matrix multiplication approach
            # att_w: [batch, time, num_codebooks, 1]
            # embeddings: [batch, time, num_codebooks, emb_dim]
            # Result: [batch, time, emb_dim]
            feats = torch.matmul(att_w.transpose(2, -1), embeddings).squeeze(-2)
        else:
            # 直接使用 SSL 模型路径
            with torch.no_grad():
                # 提取 SSL 特征 - 处理不同的返回格式
                ssl_outputs = self.modules.ssl_model(wavs)
                
                # 记录 SSL 输出类型和形状，用于调试
                if isinstance(ssl_outputs, tuple):
                    logger.debug(f"SSL model output is tuple with length {len(ssl_outputs)}")
                    if len(ssl_outputs) > 0:
                        logger.debug(f"First element shape: {ssl_outputs[0].shape}")
                else:
                    logger.debug(f"SSL model output shape: {ssl_outputs.shape}")
            
                # 根据返回类型处理特征
                if isinstance(ssl_outputs, tuple):
                    if len(ssl_outputs) == 2:
                        feats, _ = ssl_outputs
                    else:
                        # 取第一个输出作为特征
                        feats = ssl_outputs[0]
                else:
                    # 直接使用返回值作为特征
                    feats = ssl_outputs
            
                # 使用特征投影
                if hasattr(self.modules, 'feature_projection'):
                    feats = self.modules.feature_projection(feats)
            
                # 检查特征的维度并处理
                if feats.dim() != 3:
                    # 如果不是 [batch, time, features]，尝试重塑
                    if feats.dim() == 4:  # [batch, channel, time, features]
                        # 合并通道和特征维度
                        b, c, t, f = feats.size()
                        feats = feats.permute(0, 2, 1, 3).reshape(b, t, c*f)
                    elif feats.dim() == 2:  # [batch, features]
                        # 添加时间维度
                        feats = feats.unsqueeze(1)
            
                # 确保特征是3维的 [batch, time, features]
                if feats.dim() != 3:
                    raise ValueError(f"SSL model output has unexpected shape: {feats.shape}")
                
            # 创建虚拟注意力权重用于兼容性
            # 注意：这里使用 feats 的时间维度来创建正确形状的注意力权重
            att_w = torch.ones((feats.shape[0], feats.shape[1], 1, 1), device=feats.device)
    
        # 打印出一些调试信息
        logger.info(f"Features shape: {feats.shape}, lengths shape: {wav_lens.shape}")
    
        # 调整 wav_lens 以确保它与 feats 兼容
        if wav_lens.dim() == 1:
            # 确保长度是相对于特征时间步长的比例
            wav_lens = (wav_lens * feats.shape[1] / wavs.shape[1]).long()
    
        # 确保长度值不超过特征的时间步长
        wav_lens = torch.clamp(wav_lens, max=feats.shape[1])
        
        try:
            # ECAPA-TDNN processing
            embeddings = self.modules.embedding_model(feats, wav_lens)
        except TypeError as e:
            if 'unexpected keyword argument' in str(e):
                # 如果 embedding_model 不接受 lengths 参数，直接传递特征
                logger.warning("Embedding model does not accept lengths parameter, using only features")
                embeddings = self.modules.embedding_model(feats)
            else:
                raise
    
        # Final classification
        outputs = self.modules.classifier(embeddings)
        outputs = self.hparams.log_softmax(outputs)
    
        # Store attention statistics for analysis (our enhancement)
        if stage != sb.Stage.TRAIN and hasattr(self, 'store_attention_stats') and self.store_attention_stats:
            self._store_attention_stats(att_w, batch.id)
    
        return outputs, att_w
   
    def compute_objectives(self, predictions, batch, stage):
        """
        Compute loss using IEMOCAP official approach + MSP-PODCAST multi-class adaptations
        """
        if isinstance(predictions, tuple):
            predictions, attention_weights = predictions
        
        # Handle emotion labels (MSP-PODCAST 10-class specific)
        emo_labels, _ = batch.emo_encoded
        
        # Convert string labels to indices for 10-class MSP-PODCAST
        if isinstance(emo_labels[0], str):
            # MSP-PODCAST 10-class emotion mapping
            emotion_to_idx = {
                'N': 0, 'H': 1, 'X': 2, 'A': 3, 'S': 4,
                'U': 5, 'C': 6, 'O': 7, 'D': 8, 'F': 9
            }
            emo_labels = torch.tensor([emotion_to_idx.get(emo, 0) for emo in emo_labels], 
                                     device=predictions.device)
        
        # Apply class weights if configured
        loss = self._compute_weighted_loss(predictions, emo_labels)
        
        # Compute multi-class metrics for non-training stages
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emo_labels)
        
        return loss
    
    def _compute_weighted_loss(self, predictions, targets):
        """Compute loss with optional class weighting for imbalanced classes"""
        if hasattr(self.hparams, 'use_class_weights') and self.hparams.use_class_weights:
            # Use class weights for imbalanced MSP-PODCAST data
            class_weights = torch.tensor(
                self.hparams.get('class_weights', [1.0] * 10),
                device=predictions.device,
                dtype=torch.float32
            )
            
            # Weighted NLL loss
            loss_fn = torch.nn.NLLLoss(weight=class_weights)
            loss = loss_fn(predictions, targets)
        else:
            # Standard NLL loss
            loss = self.hparams.compute_cost(predictions, targets)
        
        return loss
    
    def on_stage_start(self, stage, epoch=None):
        """
        Stage initialization using IEMOCAP official approach + enhancements
        """
        # Set up loss tracking (IEMOCAP official)
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        
        # Set up error metrics for evaluation stages (IEMOCAP official)
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.attention_stats = []  # Reset attention stats
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Stage completion using IEMOCAP official structure + multi-class metrics
        """
        # Store training loss (IEMOCAP official)
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.training_stats['epoch_losses'].append(stage_loss)
            
        else:
            # Compute multi-class statistics
            primary_metric_value = self.error_metrics.summarize("average")
            
            stats = {
                "loss": stage_loss,
                "primary_metric": primary_metric_value,  # macro_f1 or configured metric
            }
            
            # For backward compatibility, also compute error rate
            if hasattr(self.error_metrics, 'predictions') and self.error_metrics.predictions:
                predictions = np.array(self.error_metrics.predictions)
                targets = np.array(self.error_metrics.targets)
                accuracy = (predictions == targets).mean()
                stats["accuracy"] = accuracy
                stats["error_rate"] = 1 - accuracy  # For compatibility
            else:
                stats["error_rate"] = 1 - primary_metric_value  # Approximation
                stats["accuracy"] = primary_metric_value
            
            if stage == sb.Stage.VALID:
                self.training_stats['epoch_accs'].append(stats["accuracy"])
                
                # Track best performance using primary metric
                if primary_metric_value > self.best_valid_acc:
                    self.best_valid_acc = primary_metric_value
                    primary_metric_name = getattr(self.hparams, 'eval_metrics', {}).get('primary_metric', 'macro_f1')
                    logger.info(f"🏆 New best {primary_metric_name}: {self.best_valid_acc:.4f}")
            
            # Log attention analysis (our enhancement)
            if self.attention_stats:
                self._log_attention_analysis(stage, stats)
        
        # Validation stage processing (IEMOCAP official + multi-class adaptations)
        if stage == sb.Stage.VALID:
            # Learning rate scheduling based on primary metric
            metric_for_lr = stats.get("primary_metric", stats.get("error_rate", 0.0))
            
            # Handle different LR scheduler types
            if hasattr(self.hparams, 'lr_scheduling_metric'):
                lr_metric = stats.get(self.hparams.lr_scheduling_metric, metric_for_lr)
                if self.hparams.get('lr_scheduling_mode', 'min') == 'max':
                    # For metrics like macro_f1 that should be maximized
                    old_lr, new_lr = self.hparams.lr_annealing_model(1 - lr_metric)
                else:
                    # For metrics like error_rate that should be minimized
                    old_lr, new_lr = self.hparams.lr_annealing_model(lr_metric)
            else:
                # Fallback to error rate (IEMOCAP style)
                old_lr, new_lr = self.hparams.lr_annealing_model(stats["error_rate"])
            
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr)
            
            # Enhanced logging with multi-class info
            log_stats = {
                "Epoch": epoch, 
                "lr": old_lr,
                "best_primary_metric": self.best_valid_acc,
            }
            
            # Add primary metric name for clarity
            if hasattr(self.hparams, 'eval_metrics'):
                primary_metric_name = self.hparams.eval_metrics.get('primary_metric', 'macro_f1')
                log_stats[f"best_{primary_metric_name}"] = self.best_valid_acc
            
            self.hparams.train_logger.log_stats(
                log_stats,
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            
            # Checkpointing with primary metric (IEMOCAP official + enhancement)
            checkpoint_meta = stats.copy()
            # Ensure we have both metrics for compatibility
            if "error_rate" not in checkpoint_meta:
                checkpoint_meta["error_rate"] = 1 - stats.get("accuracy", 0.0)
            
            self.checkpointer.save_and_keep_only(
                meta=checkpoint_meta, 
                min_keys=["error_rate"]  # Keep for compatibility
            )
        
        # Test stage processing (IEMOCAP official + multi-class enhancements)
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            
            # Export comprehensive multi-class analysi
            self._export_final_analysis(stats)
            
            # Enhanced final results logging
            logger.info(f"🎯 Final Multi-class Test Results:")
            logger.info(f"   Test Accuracy: {stats['accuracy']:.4f}")
            
            primary_metric_name = 'macro_f1'
            if hasattr(self.hparams, 'eval_metrics'):
                primary_metric_name = self.hparams.eval_metrics.get('primary_metric', 'macro_f1')
            logger.info(f"   Test {primary_metric_name}: {stats['primary_metric']:.4f}")
            logger.info(f"   Best Validation {primary_metric_name}: {self.best_valid_acc:.4f}")
            
            # Log confusion matrix and classification report if available
            if hasattr(self.error_metrics, 'get_confusion_matrix'):
                cm = self.error_metrics.get_confusion_matrix()
                report = self.error_metrics.get_classification_report()
                if cm is not None and report is not None:
                    logger.info(f"📊 Detailed Classification Report:")
                    logger.info(f"\n{report}")
                    
                    # Save confusion matrix and report to files
                    analysis_dir = os.path.join(self.hparams.output_folder, 'analysis')
                    os.makedirs(analysis_dir, exist_ok=True)
                    
                    # Save confusion matrix
                    np.savetxt(os.path.join(analysis_dir, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')
                    
                    # Save classification report
                    with open(os.path.join(analysis_dir, 'classification_report.txt'), 'w') as f:
                        f.write(report)
    
    def init_optimizers(self):
        """
        Initialize optimizers using IEMOCAP official approach
        """
        # IEMOCAP official optimizer initialization
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        
        # Optimizer dictionary (IEMOCAP official)
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        
        # Checkpointer integration (IEMOCAP official)
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
    
    def _store_attention_stats(self, attention_weights, batch_ids):
        """Store attention statistics for analysis (our enhancement)"""
        # 确保 attention_weights 有适当的维度
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(-1)
        elif attention_weights.dim() < 3:
            # 如果维度太少，添加必要的维度
            while attention_weights.dim() < 3:
                attention_weights = attention_weights.unsqueeze(-1)
        
        # 检查注意力权重的形状是否合理
        if hasattr(self.hparams, 'num_codebooks') and attention_weights.shape[-1] != self.hparams.num_codebooks:
            # 如果不匹配，创建一个虚拟的注意力权重
            logger.warning(f"Attention weights shape {attention_weights.shape} is not compatible with num_codebooks {self.hparams.num_codebooks}. Using uniform weights.")
            # 假设形状是 [batch, time, ?]
            b, t = attention_weights.shape[0], attention_weights.shape[1]
            attention_weights = torch.ones((b, t, self.hparams.num_codebooks), device=attention_weights.device) / self.hparams.num_codebooks
        
        # Compute mean attention across time
        mean_attention = attention_weights.mean(dim=1)  # [batch, num_codebooks]
        
        for i, batch_id in enumerate(batch_ids):
            # 确保索引在有效范围内
            if i < mean_attention.shape[0]:
                # 确保我们可以安全地计算熵和获取主要的代码本
                if mean_attention.shape[1] > 0:
                    self.attention_stats.append({
                        'id': batch_id,
                        'attention_weights': mean_attention[i].cpu().numpy(),
                        'attention_entropy': self._compute_entropy(mean_attention[i:i+1]).item(),
                        'dominant_codebook': mean_attention[i].argmax().item()
                    })
                else:
                    # 如果没有多个代码本，使用占位符
                    self.attention_stats.append({
                        'id': batch_id,
                        'attention_weights': np.array([1.0]),
                        'attention_entropy': 0.0,
                        'dominant_codebook': 0
                    })
    
    def _compute_entropy(self, attention_weights):
        """Compute entropy of attention distribution"""
        epsilon = 1e-8
        # 确保权重是正数并且总和为1
        norm_weights = torch.nn.functional.softmax(attention_weights, dim=-1) + epsilon
        log_weights = torch.log(norm_weights)
        entropy = -(norm_weights * log_weights).sum(dim=-1)
        return entropy
    
    def _log_attention_analysis(self, stage, stats):
        """Log detailed attention analysis"""
        if not self.attention_stats:
            return
        
        try:
            attention_arrays = np.array([s['attention_weights'] for s in self.attention_stats])
            entropies = [s['attention_entropy'] for s in self.attention_stats]
            
            # Compute statistics
            mean_attention = attention_arrays.mean(axis=0)
            std_attention = attention_arrays.std(axis=0)
            mean_entropy = np.mean(entropies)
            
            logger.info(f"\n📊 {stage.upper()} Attention Analysis:")
            logger.info(f"   Mean entropy: {mean_entropy:.4f}")
            
            # Log SSL layer importance if compatible
            if hasattr(self.hparams, 'ssl_layer_num'):
                ssl_layers = self.hparams.ssl_layer_num
                # 确保我们有足够的层来记录
                if len(ssl_layers) == mean_attention.shape[0]:
                    for i, (layer, attention) in enumerate(zip(ssl_layers, mean_attention)):
                        logger.info(f"   Layer {layer:2d}: {attention:.4f} ± {std_attention[i]:.4f}")
                else:
                    logger.warning(f"Number of SSL layers ({len(ssl_layers)}) doesn't match attention dimensions ({mean_attention.shape[0]})")
        except Exception as e:
            logger.error(f"Error in attention analysis: {e}")
    
    def _export_final_analysis(self, test_stats):
        """Export comprehensive analysis to files"""
        analysis_dir = os.path.join(self.hparams.output_folder, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Export comprehensive results
        results_file = os.path.join(analysis_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            export_data = {
                'test_stats': test_stats,
                'training_stats': self.training_stats,
                'best_valid_accuracy': float(self.best_valid_acc),
                'model_config': {
                    'ssl_model_type': self.hparams.ssl_model_type,
                    'attention_mode': getattr(self.hparams, 'attention_mode', 'iemocap'),
                    'num_codebooks': self.hparams.num_codebooks,
                    'num_clusters': self.hparams.num_clusters,
                    'dataset': 'MSP-PODCAST'
                }
            }
            
            # Add attention analysis if available
            if self.attention_stats:
                attention_data = []
                for stats in self.attention_stats:
                    attention_data.append({
                        'id': stats['id'],
                        'attention_weights': stats['attention_weights'].tolist(),
                        'attention_entropy': stats['attention_entropy'],
                        'dominant_codebook': stats['dominant_codebook']
                    })
                export_data['attention_analysis'] = attention_data
            
            json.dump(export_data, f, indent=2)
        
        logger.info(f"💾 Comprehensive analysis exported to {results_file}")


def dataio_prepare(hparams):
    """
    Enhanced data preparation combining IEMOCAP official + MSP-PODCAST specifics
    """
    
    # IEMOCAP official audio pipeline + enhancements
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """
        Load and process audio with resampling (IEMOCAP official approach)
        """
        # Load signal
        sig = sb.dataio.dataio.read_audio(wav)
        
        # Get audio info for resampling (IEMOCAP official)
        info = torchaudio.info(wav)
        
        # Resample if needed (IEMOCAP official approach)
        if info.sample_rate != hparams["sample_rate"]:
            resampled = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"]
            )(sig)
            return resampled
        else:
            return sig
    
    # Label encoder initialization (IEMOCAP official)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    
    # IEMOCAP official label pipeline + MSP-PODCAST adaptation
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded
    
    # Dataset creation (IEMOCAP official structure)
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )
    
    # Label encoder setup (IEMOCAP official)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )
    
    # Enhanced logging
    total_samples = sum(len(dataset) for dataset in datasets.values())
    logger.info(f"📊 MSP-PODCAST Dataset Loaded:")
    for dataset_name, dataset in datasets.items():
        logger.info(f"   {dataset_name:5s}: {len(dataset):5d} samples")
    logger.info(f"   Total: {total_samples:5d} samples")
    
    return datasets


def validate_configuration(hparams):
    """Enhanced configuration validation"""
    logger.info("🔍 Validating configuration...")
    
    # Check and set data folder
    if hparams["data_folder"] == "!PLACEHOLDER":
        hparams["data_folder"] = "."
        logger.info("   Data folder set to current directory")
    
    # Validate required files
    required_files = [
        hparams["train_annotation"],
        hparams["valid_annotation"],
        hparams["test_annotation"]
    ]
    
    for file_path in required_files:
        full_path = os.path.join(hparams["data_folder"], os.path.basename(file_path))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Required file not found: {full_path}")
        logger.info(f"   ✅ Found: {os.path.basename(file_path)}")
    
    # Validate model configuration
    ssl_type = hparams.get('ssl_model_type', 'wavlm')
    attention_mode = hparams.get('attention_mode', 'iemocap')
    
    logger.info(f"   SSL Model: {ssl_type}")
    logger.info(f"   Attention Mode: {attention_mode}")
    logger.info(f"   Batch Size: {hparams['batch_size']}")
    logger.info(f"   Learning Rate: {hparams['lr']}")
    logger.info(f"   Epochs: {hparams['number_of_epochs']}")
    
    logger.info("✅ Configuration validation passed!")


def main():
    """
    Main training function using IEMOCAP official structure + enhancements
    """
    # 设置详细的日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger.info("🚀 Starting Ultimate MSP-PODCAST Discrete SSL Training")
    logger.info("   Based on IEMOCAP official implementation + enhancements")
    
    # Parse arguments (IEMOCAP official)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Initialize distributed training (IEMOCAP official)
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters (IEMOCAP official)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Validate configuration
    validate_configuration(hparams)
    
    # Create experiment directory (IEMOCAP official)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Import data preparation function (MSP-PODCAST adaptation)
    # Note: We use our enhanced dataio_prepare instead of IEMOCAP's prepare_data
    # since MSP-PODCAST data is already prepared
    
    # Skip data preparation for MSP-PODCAST (already done)
    if not hparams.get("skip_prep", True):
        logger.info("📋 Data preparation skipped for MSP-PODCAST (already prepared)")
    
    # Prepare datasets
    logger.info("📋 Loading datasets...")
    datasets = dataio_prepare(hparams)
    
    # Initialize Ultimate Brain (IEMOCAP official structure + our enhancements)
    logger.info("🧠 Initializing Ultimate MSP-PODCAST Brain...")
    emo_id_brain = MSPPodcastUltimateBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Add tokenizer configuration (IEMOCAP official)
    emo_id_brain.tokenizer = hparams.get('tokenizer_config', {})
    
    # Enable attention statistics for analysis
    emo_id_brain.store_attention_stats = True
    
    # Training loop (IEMOCAP official)
    logger.info("🎓 Starting training...")
    logger.info(f"   Device: {run_opts['device']}")
    logger.info(f"   Output: {hparams['output_folder']}")
    
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    
    # Final evaluation (IEMOCAP official)
    logger.info("📊 Starting final evaluation...")
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"💾 Results saved in: {hparams['output_folder']}")
    logger.info(f"📈 Best validation accuracy: {emo_id_brain.best_valid_acc:.4f}")
    logger.info(f"🎯 Final test accuracy: {1 - test_stats['error_rate']:.4f}")


if __name__ == "__main__":
    main()