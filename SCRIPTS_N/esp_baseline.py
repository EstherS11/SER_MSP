#!/usr/bin/env python3
# Copyright 2024 [Your Name]
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
ESPnet-style WavLM + ECAPA-TDNN for Speech Emotion Recognition.

This script defines a configurable model architecture, intended to be used
within an ESPnet recipe. It incorporates advanced features like Focal Loss,
multi-task learning, and a highly configurable ECAPA-TDNN backend,
all controlled via a YAML configuration file.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.model_summary import model_summary
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

# --- Loss Functions ---

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    This is a re-implementation of Focal Loss, which can be enabled via config.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Advanced ECAPA-TDNN Components (from user's original script) ---

class SERes2NetBlock(nn.Module):
    """Squeeze-Excitation Res2Net block, configurable for ECAPA."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
                 res2net_scale, se_channels, dropout):
        super().__init__()
        self.res2net_scale = res2net_scale
        width = out_channels // res2net_scale
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, dilation=dilation,
                      padding=(kernel_size - 1) * dilation // 2)
            for _ in range(res2net_scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(res2net_scale - 1)])
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, se_channels, 1),
            nn.ReLU(),
            nn.Conv1d(se_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.conv_expand = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = self.conv_expand(x)
        
        xs = torch.chunk(residual, self.res2net_scale, dim=1)
        ys = []
        for i in range(self.res2net_scale):
            if i == 0:
                ys.append(xs[i])
            else:
                y = self.convs[i-1](xs[i] + ys[-1])
                y = self.bns[i-1](y)
                y = self.relu(y)
                ys.append(y)
        
        out = torch.cat(ys, dim=1)
        
        se_weight = self.se(out)
        out = out * se_weight
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + residual

class AttentiveStatisticsPooling(nn.Module):
    """Attention-based statistics pooling"""
    def __init__(self, in_dim, attention_channels, global_context=True):
        super().__init__()
        self.global_context = global_context
        
        if global_context:
            # Global context requires concatenating mean and std, tripling the channels
            tdnn_in_dim = in_dim * 3
        else:
            tdnn_in_dim = in_dim
            
        self.tdnn = nn.Conv1d(tdnn_in_dim, attention_channels, 1)
        self.tanh = nn.Tanh()
        self.gating = nn.Conv1d(attention_channels, in_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Time)
        # mask: (Batch, 1, Time)
        
        if self.global_context:
            mean = x.mean(dim=2, keepdim=True).expand_as(x)
            std = x.std(dim=2, keepdim=True).expand_as(x)
            x_ctx = torch.cat([x, mean, std], dim=1)
        else:
            x_ctx = x
            
        alpha = self.tanh(self.tdnn(x_ctx))
        alpha = self.gating(alpha)
        alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=2)
        
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x.pow(2), dim=2) - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-8))
        
        return torch.cat([mean, std], dim=1)

class AdvancedEcapaTdnn(nn.Module):
    """A highly configurable ECAPA-TDNN based on the user's advanced config."""
    def __init__(
        self,
        input_size: int,
        channels: List[int],
        kernel_sizes: List[int],
        dilations: List[int],
        attention_channels: int,
        res2net_scale: int,
        se_channels: int,
        global_context: bool,
        lin_neurons: int,
        dropout: float,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # The first Conv1d block
        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(input_size, channels[0], kernel_sizes[0], 
                          padding=kernel_sizes[0]//2, dilation=dilations[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        )
        
        # SE-Res2Net blocks
        for i in range(1, len(channels)-1): # The last channel is for the final projection
            self.blocks.append(
                SERes2NetBlock(
                    channels[i-1], channels[i], kernel_sizes[i], 
                    dilations[i], res2net_scale, se_channels, dropout
                )
            )
        
        # Attention Statistics Pooling
        self.pooling = AttentiveStatisticsPooling(
            channels[-2], attention_channels, global_context
        )
        
        # Final layers
        # The input to the final FC layer is the output of the pooling layer (mean+std)
        # which has size channels[-2] * 2.
        self.fc = nn.Sequential(
            nn.Linear(channels[-2] * 2, lin_neurons),
            nn.BatchNorm1d(lin_neurons),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, F, T)
        
        for block in self.blocks:
            x = block(x)
            
        mask = make_pad_mask(lengths).unsqueeze(1).to(x.device)
        x = self.pooling(x, mask)
        
        x = self.fc(x)
        return x

# --- Main ESPnet Model ---

class EmotionModel(AbsESPnetModel):
    """
    WavLM-based Speech Emotion Recognition model for ESPnet.
    This model is now highly configurable via a YAML file.
    """
    def __init__(
        self,
        num_class: int,
        wavlm_model_name: str = "microsoft/wavlm-large",
        wavlm_freeze: bool = True,
        
        # Loss configuration
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.1,
        
        # ECAPA-TDNN configuration
        ecapa_conf: dict = None,
        
        # Multi-task learning configuration
        use_multitask: bool = False,
        multitask_conf: Optional[dict] = None,
        
        # Mixup augmentation configuration
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        super().__init__()
        
        # Save configuration
        self.num_class = num_class
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_multitask = use_multitask
        
        # 1. WavLM Frontend
        logging.info(f"Loading WavLM model: {wavlm_model_name}")
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        if wavlm_freeze:
            logging.info("Freezing WavLM feature encoder.")
            self.wavlm.freeze_feature_encoder()
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        wavlm_output_size = self.wavlm.config.hidden_size
        
        # 2. Advanced ECAPA-TDNN Backend
        if ecapa_conf is None:
            raise ValueError("ecapa_conf must be provided in the config.")
        
        self.encoder = AdvancedEcapaTdnn(
            input_size=wavlm_output_size,
            **ecapa_conf
        )
        
        # 3. Classifier Heads
        self.emotion_classifier = nn.Linear(ecapa_conf["lin_neurons"], num_class)
        
        if self.use_multitask:
            if multitask_conf is None:
                raise ValueError("multitask_conf must be provided for multi-task learning.")
            self.arousal_weight = multitask_conf.get("arousal_weight", 0.3)
            self.valence_weight = multitask_conf.get("valence_weight", 0.3)
            self.arousal_classifier = nn.Linear(ecapa_conf["lin_neurons"], 3) # 3 classes: low, med, high
            self.valence_classifier = nn.Linear(ecapa_conf["lin_neurons"], 3) # 3 classes: neg, neu, pos
        
        # 4. Loss Functions
        weights = torch.tensor(class_weights) if class_weights else None
        
        if loss_type == "focal":
            logging.info("Using Focal Loss")
            self.emotion_criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        else:
            logging.info("Using Cross Entropy Loss")
            self.emotion_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        
        if self.use_multitask:
            self.arousal_criterion = nn.CrossEntropyLoss()
            self.valence_criterion = nn.CrossEntropyLoss()

    def _mixup_data(self, x, y_a, y_b, lam):
        """Applies mixup to a single tensor."""
        return lam * y_a + (1 - lam) * y_b

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        emotion: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        if not self.wavlm.training:
            self.wavlm.eval()
        
        # --- Feature Extraction ---
        wavlm_output = self.wavlm(speech, output_hidden_states=True)
        feats = wavlm_output.hidden_states[-1]
        feats_lengths = self._get_feat_lengths(speech_lengths)

        # --- Mixup Augmentation (if training) ---
        lam = 1.0
        if self.training and self.use_mixup and np.random.rand() < 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = feats.size(0)
            index = torch.randperm(batch_size).to(feats.device)
            
            # Mixup features
            feats = self._mixup_data(feats, feats, feats[index], lam)
            
            # Mixup labels
            emotion_a, emotion_b = emotion, emotion[index]
            if self.use_multitask:
                arousal_a, arousal_b = kwargs["arousal"], kwargs["arousal"][index]
                valence_a, valence_b = kwargs["valence"], kwargs["valence"][index]

        # --- Encoding and Classification ---
        embedding = self.encoder(feats, feats_lengths)
        emotion_logits = self.emotion_classifier(embedding)
        
        # --- Loss Calculation ---
        total_loss = 0
        stats = {}
        
        # Emotion Loss
        if lam < 1.0: # Mixed labels
            emotion_loss = lam * self.emotion_criterion(emotion_logits, emotion_a) + \
                           (1 - lam) * self.emotion_criterion(emotion_logits, emotion_b)
        else: # Standard labels
            emotion_loss = self.emotion_criterion(emotion_logits, emotion.long())

        total_loss += emotion_loss
        stats["emotion_loss"] = emotion_loss.detach()
        stats["acc"] = (emotion_logits.argmax(dim=-1) == emotion).float().mean()
        
        # Multi-task Loss
        if self.use_multitask and "arousal" in kwargs and "valence" in kwargs:
            arousal_logits = self.arousal_classifier(embedding)
            valence_logits = self.valence_classifier(embedding)
            
            if lam < 1.0: # Mixed labels
                arousal_loss = lam * self.arousal_criterion(arousal_logits, arousal_a) + \
                               (1 - lam) * self.arousal_criterion(arousal_logits, arousal_b)
                valence_loss = lam * self.valence_criterion(valence_logits, valence_a) + \
                               (1 - lam) * self.valence_criterion(valence_logits, valence_b)
            else: # Standard labels
                arousal_loss = self.arousal_criterion(arousal_logits, kwargs["arousal"].long())
                valence_loss = self.valence_criterion(valence_logits, kwargs["valence"].long())

            total_loss += self.arousal_weight * arousal_loss + self.valence_weight * valence_loss
            stats.update({
                'arousal_loss': arousal_loss.detach(),
                'valence_loss': valence_loss.detach(),
                'arousal_acc': (arousal_logits.argmax(dim=-1) == kwargs["arousal"]).float().mean(),
                'valence_acc': (valence_logits.argmax(dim=-1) == kwargs["valence"]).float().mean(),
            })
            
        stats["loss"] = total_loss.detach()
        weight = torch.tensor(speech.shape[0], device=total_loss.device)
        return total_loss, stats, weight

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            wavlm_output = self.wavlm(speech, output_hidden_states=True)
            feats = wavlm_output.hidden_states[-1]
            feats_lengths = self._get_feat_lengths(speech_lengths)
            embedding = self.encoder(feats, feats_lengths)
        return {"feats": embedding}

    def _get_feat_lengths(self, speech_lengths: torch.Tensor) -> torch.Tensor:
        """Calculates the sequence length of WavLM features."""
        def _conv_out_length(in_len, kernel, stride, padding):
            return torch.floor((in_len + 2 * padding - (kernel - 1) - 1) / stride + 1)

        l = speech_lengths
        l = _conv_out_length(l, 10, 5, 0)
        l = _conv_out_length(l, 3, 2, 0)
        l = _conv_out_length(l, 3, 2, 0)
        l = _conv_out_length(l, 3, 2, 0)
        l = _conv_out_length(l, 3, 2, 0)
        l = _conv_out_length(l, 2, 2, 0)
        l = _conv_out_length(l, 2, 2, 0)
        return l.long()
