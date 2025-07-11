#!/usr/bin/env python3
# espnet_ser_model.py - ESP-net兼容的WavLM + ECAPA-TDNN模型

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel
from sklearn.metrics import f1_score, classification_report
import numpy as np

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.class_choices import ClassChoices
# 需要额外导入
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer

# 安全导入CommonCollateFn
try:
    from espnet2.train.collate_fn import CommonCollateFn
except ImportError:
    try:
        from espnet2.train.dataset import CommonCollateFn
    except ImportError:
        # 如果都导入失败，提供一个简单的实现
        class CommonCollateFn:
            def __init__(self, float_pad_value=0.0, int_pad_value=-1):
                self.float_pad_value = float_pad_value
                self.int_pad_value = int_pad_value
            
            def __call__(self, batch):
                # 简单的collate function实现
                return batch

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

# 如果make_pad_mask不可用，提供备选实现
def make_pad_mask_fallback(lengths, max_len=None):
    """备选的pad mask实现"""
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    seq_range = torch.arange(0, max_len, dtype=lengths.dtype, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
    
    return seq_range_expand >= seq_length_expand

# 尝试导入，如果失败则使用备选
try:
    from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
except ImportError:
    make_pad_mask = make_pad_mask_fallback

class SERes2NetBlock(nn.Module):
    """ECAPA-TDNN的SE-Res2Net块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, scale=8, se_channels=128):
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        
        # Res2Net卷积分支
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, dilation=dilation,
                      padding=(kernel_size - 1) * dilation // 2)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        
        # SE (Squeeze-and-Excitation) 模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, se_channels, 1),
            nn.ReLU(),
            nn.Conv1d(se_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 残差连接的投影层
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = self.residual_conv(x)
        
        # Res2Net处理
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                ys.append(xs[i])
            else:
                y = self.convs[i-1](xs[i] + ys[-1])
                y = self.bns[i-1](y)
                y = self.relu(y)
                ys.append(y)
        
        out = torch.cat(ys, dim=1)
        
        # SE注意力
        se_weight = self.se(out)
        out = out * se_weight
        
        # 残差连接和标准化
        out = self.bn(out + residual)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out

class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化"""
    def __init__(self, in_dim, bottleneck_dim=128, global_context=True):
        super().__init__()
        self.global_context = global_context
        
        if global_context:
            # 全局上下文: 拼接 [x, global_mean, global_std]
            context_dim = in_dim * 3
        else:
            context_dim = in_dim
            
        self.attention_conv = nn.Conv1d(context_dim, bottleneck_dim, 1)
        self.tanh = nn.Tanh()
        self.context_conv = nn.Conv1d(bottleneck_dim, in_dim, 1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T) - 输入特征
            mask: (B, T) - padding mask
        Returns:
            (B, C*2) - 拼接的均值和标准差
        """
        if self.global_context:
            # 计算全局统计
            if mask is not None:
                lengths = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
                mean = (x * mask.unsqueeze(1)).sum(dim=2, keepdim=True) / lengths.unsqueeze(1)  # (B, C, 1)
                var = ((x - mean).pow(2) * mask.unsqueeze(1)).sum(dim=2, keepdim=True) / lengths.unsqueeze(1)
                std = torch.sqrt(var.clamp(min=1e-8))  # (B, C, 1)
            else:
                mean = x.mean(dim=2, keepdim=True)  # (B, C, 1)
                std = x.std(dim=2, keepdim=True)   # (B, C, 1)
            
            # 扩展到所有时间步
            mean_expanded = mean.expand_as(x)      # (B, C, T)
            std_expanded = std.expand_as(x)        # (B, C, T)
            
            # 拼接上下文
            context = torch.cat([x, mean_expanded, std_expanded], dim=1)  # (B, C*3, T)
        else:
            context = x
        
        # 计算注意力权重
        alpha = self.tanh(self.attention_conv(context))  # (B, bottleneck_dim, T)
        alpha = self.context_conv(alpha)                 # (B, C, T)
        
        # 应用mask
        if mask is not None:
            alpha = alpha.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        alpha = F.softmax(alpha, dim=2)  # (B, C, T)
        
        # 加权统计
        mean = torch.sum(alpha * x, dim=2)  # (B, C)
        
        # 计算加权方差和标准差
        var = torch.sum(alpha * (x - mean.unsqueeze(2)).pow(2), dim=2)  # (B, C)
        std = torch.sqrt(var.clamp(min=1e-8))  # (B, C)
        
        # 拼接均值和标准差
        pooled = torch.cat([mean, std], dim=1)  # (B, C*2)
        
        return pooled

class WavLMECAPAModel(AbsESPnetModel):
    """ESP-net兼容的WavLM + ECAPA-TDNN模型"""
    
    def __init__(
        self,
        num_class: int,
        wavlm_model_name: str = "microsoft/wavlm-base",  # 改为base版本
        wavlm_freeze: bool = True,
        
        # ECAPA-TDNN配置
        ecapa_channels: List[int] = None,
        ecapa_kernels: List[int] = None,
        ecapa_dilations: List[int] = None,
        context_dim: int = 512,  # 减小参数
        embedding_dim: int = 128,  # 减小参数
        
        # 损失和优化配置
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        
        # macro-F1优化
        save_macro_f1: bool = True,
    ):
        super().__init__()
        
        # 默认值 - 适用于快速测试
        if ecapa_channels is None:
            ecapa_channels = [256, 256]  # 减少层数
        if ecapa_kernels is None:
            ecapa_kernels = [3, 3]
        if ecapa_dilations is None:
            ecapa_dilations = [1, 2]
        
        self.num_class = num_class
        self.save_macro_f1 = save_macro_f1
        
        # 用于macro-F1计算
        self.all_predictions = []
        self.all_targets = []
        
        # WavLM前端
        logging.info(f"Loading WavLM model: {wavlm_model_name}")
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        
        if wavlm_freeze:
            logging.info("Freezing WavLM parameters")
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        wavlm_output_size = self.wavlm.config.hidden_size  # 1024 for Large
        
        # ECAPA-TDNN编码器
        self.layers = nn.ModuleList()
        
        # 第一层
        in_channels = wavlm_output_size
        for i, (out_channels, kernel, dilation) in enumerate(zip(ecapa_channels, ecapa_kernels, ecapa_dilations)):
            if i == 0:
                # 第一层使用普通卷积
                self.layers.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel, padding=kernel//2, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ))
            else:
                # 后续层使用SE-Res2Net块
                self.layers.append(SERes2NetBlock(in_channels, out_channels, kernel, dilation))
            in_channels = out_channels
        
        # 上下文聚合层
        total_channels = sum(ecapa_channels)  # 聚合所有ECAPA层的输出
        self.context_conv = nn.Sequential(
            nn.Conv1d(total_channels, context_dim, 1),
            nn.BatchNorm1d(context_dim),
            nn.ReLU()
        )
        
        # 注意力池化
        self.pooling = AttentiveStatisticsPooling(context_dim, bottleneck_dim=128)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(context_dim * 2, embedding_dim),  # *2 因为mean+std
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_class)
        )
        
        # 损失函数
        if loss_type == "focal":
            weights = torch.tensor(class_weights) if class_weights else None
            self.criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        else:
            weights = torch.tensor(class_weights) if class_weights else None
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        emotion: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        # 确保WavLM在推理模式（如果冻结）
        if not self.wavlm.training:
            self.wavlm.eval()
        
        # WavLM特征提取
        with torch.no_grad() if not any(p.requires_grad for p in self.wavlm.parameters()) else torch.enable_grad():
            wavlm_output = self.wavlm(speech)
            features = wavlm_output.last_hidden_state  # (B, T, 1024)
        
        # 计算特征长度（考虑WavLM的下采样）
        feats_lengths = self._get_feat_lengths(speech_lengths)
        
        # 转换为卷积格式 (B, C, T)
        x = features.transpose(1, 2)  # (B, 1024, T)
        
        # ECAPA-TDNN处理
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        
        # 特征聚合
        x_cat = torch.cat(layer_outputs, dim=1)  # (B, total_channels, T)
        x_context = self.context_conv(x_cat)     # (B, context_dim, T)
        
        # 创建mask用于池化
        mask = make_pad_mask(feats_lengths).to(x_context.device)  # (B, T)
        
        # 注意力池化
        pooled = self.pooling(x_context, mask)  # (B, context_dim*2)
        
        # 分类
        logits = self.classifier(pooled)  # (B, num_class)
        
        # 损失计算
        loss = self.criterion(logits, emotion.long())
        
        # 统计信息
        stats = {}
        stats["loss"] = loss.detach()
        stats["acc"] = (logits.argmax(dim=-1) == emotion).float().mean()
        
        # 收集预测用于macro-F1计算
        if not self.training and self.save_macro_f1:
            predictions = logits.argmax(dim=-1).cpu().numpy()
            targets = emotion.cpu().numpy()
            self.all_predictions.extend(predictions)
            self.all_targets.extend(targets)
            
            # 计算当前batch的macro-F1
            if len(set(targets)) > 1:  # 避免单类别batch的警告
                try:
                    batch_macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
                    stats["macro_f1"] = torch.tensor(batch_macro_f1)
                except:
                    stats["macro_f1"] = torch.tensor(0.0)
            else:
                stats["macro_f1"] = torch.tensor(0.0)
        
        # ESP-net要求的返回格式
        weight = torch.tensor(speech.shape[0], device=loss.device)
        return loss, stats, weight
    
    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """特征提取方法（ESP-net要求）"""
        self.eval()
        with torch.no_grad():
            # WavLM特征
            wavlm_output = self.wavlm(speech)
            features = wavlm_output.last_hidden_state
            
            # ECAPA-TDNN处理
            x = features.transpose(1, 2)
            layer_outputs = []
            for layer in self.layers:
                x = layer(x)
                layer_outputs.append(x)
            
            x_cat = torch.cat(layer_outputs, dim=1)
            x_context = self.context_conv(x_cat)
            
            # 池化
            feats_lengths = self._get_feat_lengths(speech_lengths)
            mask = make_pad_mask(feats_lengths).to(x_context.device)
            pooled = self.pooling(x_context, mask)
            
        return {"feats": pooled}
    
    def get_epoch_macro_f1(self) -> float:
        """获取整个epoch的macro-F1"""
        if len(self.all_predictions) == 0:
            return 0.0
        
        try:
            macro_f1 = f1_score(self.all_targets, self.all_predictions, average='macro', zero_division=0)
            
            # 生成详细报告（可选）
            if len(set(self.all_targets)) > 1:
                emotion_names = [f"emotion_{i}" for i in range(self.num_class)]
                report = classification_report(
                    self.all_targets, 
                    self.all_predictions,
                    target_names=emotion_names,
                    zero_division=0
                )
                logging.info(f"Epoch Classification Report:\n{report}")
        except Exception as e:
            logging.warning(f"Error calculating macro-F1: {e}")
            macro_f1 = 0.0
        
        # 清空用于下一个epoch
        self.all_predictions = []
        self.all_targets = []
        
        return macro_f1
    
    def _get_feat_lengths(self, speech_lengths: torch.Tensor) -> torch.Tensor:
        """计算WavLM特征的序列长度"""
        # WavLM的下采样比例计算
        def _conv_out_length(in_len, kernel, stride, padding):
            return torch.floor((in_len + 2 * padding - (kernel - 1) - 1) / stride + 1)
        
        # WavLM-Large的卷积层配置（根据实际模型调整）
        l = speech_lengths.float()
        l = _conv_out_length(l, 10, 5, 0)    # 第1层
        l = _conv_out_length(l, 3, 2, 0)     # 第2层
        l = _conv_out_length(l, 3, 2, 0)     # 第3层
        l = _conv_out_length(l, 3, 2, 0)     # 第4层
        l = _conv_out_length(l, 3, 2, 0)     # 第5层
        l = _conv_out_length(l, 2, 2, 0)     # 第6层
        l = _conv_out_length(l, 2, 2, 0)     # 第7层
        
        return l.long()


# ============================================================================
# 完全修复的SER任务类 - ESP-net集成
# ============================================================================

class SERTask(AbsTask):
    """完全兼容ESP-net的语音情感识别任务"""
    
    # ESP-net任务必需属性
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list = [
        ClassChoices(
            name="model",
            classes=dict(wavlm_ecapa=WavLMECAPAModel),
            type_check=AbsESPnetModel,
            default="wavlm_ecapa",
        ),
        ClassChoices(
            name="preprocessor", 
            classes=dict(default=CommonPreprocessor),
            type_check=CommonPreprocessor,
            default="default",
        ),
    ]
    
    @classmethod
    def add_task_arguments(cls, parser):
        """添加SER任务特定参数 - 只添加SER独有的参数"""
        group = parser.add_argument_group("SER task related")
        
        # 只添加SER特有的参数，避免与ASR任务参数冲突
        group.add_argument("--num_class", type=int, default=10, help="Number of emotion classes")
        group.add_argument("--wavlm_model_name", type=str, default="microsoft/wavlm-base")
        group.add_argument("--wavlm_freeze", type=bool, default=True)
        group.add_argument("--ecapa_channels", type=int, nargs="+", default=[256, 256])
        group.add_argument("--ecapa_kernels", type=int, nargs="+", default=[3, 3])
        group.add_argument("--ecapa_dilations", type=int, nargs="+", default=[1, 2])
        group.add_argument("--context_dim", type=int, default=512)
        group.add_argument("--embedding_dim", type=int, default=128)
        group.add_argument("--loss_type", type=str, default="cross_entropy")
        group.add_argument("--focal_gamma", type=float, default=2.0)
        group.add_argument("--label_smoothing", type=float, default=0.0)
        group.add_argument("--save_macro_f1", type=bool, default=True)
    
    @classmethod
    def build_collate_fn(cls, args, train: bool):
        """构建数据整理函数"""
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
    
    @classmethod
    def build_preprocess_fn(cls, args, train: bool):
        """构建预处理函数"""
        if hasattr(args, 'preprocessor_conf'):
            preprocessor = cls.class_choices_list[1].get_class(args.preprocessor)
            return preprocessor(**args.preprocessor_conf)
        return None
    
    @classmethod
    def build_model(cls, args):
        """构建模型"""
        return WavLMECAPAModel(
            num_class=getattr(args, 'num_class', 10),
            wavlm_model_name=getattr(args, 'wavlm_model_name', "microsoft/wavlm-base"),
            wavlm_freeze=getattr(args, 'wavlm_freeze', True),
            ecapa_channels=getattr(args, 'ecapa_channels', [256, 256]),
            ecapa_kernels=getattr(args, 'ecapa_kernels', [3, 3]),
            ecapa_dilations=getattr(args, 'ecapa_dilations', [1, 2]),
            context_dim=getattr(args, 'context_dim', 512),
            embedding_dim=getattr(args, 'embedding_dim', 128),
            loss_type=getattr(args, 'loss_type', "cross_entropy"),
            focal_gamma=getattr(args, 'focal_gamma', 2.0),
            label_smoothing=getattr(args, 'label_smoothing', 0.0),
            save_macro_f1=getattr(args, 'save_macro_f1', True),
        )
    
    @classmethod
    def required_data_names(cls, inference: bool = False):
        """必需的数据名称"""
        if not inference:
            return ("speech", "emotion")
        else:
            return ("speech",)
    
    @classmethod
    def optional_data_names(cls, inference: bool = False):
        """可选的数据名称"""
        return ()


# ============================================================================
# Focal Loss实现
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
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


# ============================================================================
# 模型选择注册
# ============================================================================

ser_model_choices = ClassChoices(
    name="ser_model",
    classes=dict(
        wavlm_ecapa=WavLMECAPAModel,
    ),
    type_check=AbsESPnetModel,
    default="wavlm_ecapa",
)