#!/usr/bin/env python3
"""
简化的MSP-PODCAST情感分类训练脚本
使用WavLM + ECAPA-TDNN
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

class SimpleMSPEmotionBrain(sb.Brain):
    """简化的MSP-PODCAST情感分类Brain"""
    
    def compute_forward(self, batch, stage):
        """计算前向传播"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # 使用WavLM提取特征
        with torch.no_grad():
            wav_lens_ratio = wav_lens / wavs.shape[1]
            feats = self.modules.ssl_model(wavs)
            if isinstance(feats, tuple):
                feats = feats[0]
            
            # 投影
            feats = self.modules.feature_projection(feats)
        
        # 确保长度正确
        emb_lens = torch.round(wav_lens_ratio * feats.shape[1]).long()
        emb_lens = torch.clamp(emb_lens, min=1, max=feats.shape[1])
        
        # ECAPA-TDNN处理
        embeddings = self.modules.embedding_model(feats)
        
        # 分类
        outputs = self.modules.classifier(embeddings)
        outputs = self.hparams.log_softmax(outputs)
        
        return outputs
    
    def compute_objectives(self, predictions, batch, stage):
        """计算损失"""
        emo_ids = batch.emo_id
        
        # 计算损失
        loss = self.hparams.compute_cost(predictions, emo_ids)
        
        # 评估阶段计算错误率
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emo_ids)
            
        return loss

def dataio_prepare(hparams):
    """准备数据加载器"""
    # 音频处理管道
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        try:
            sig = sb.dataio.dataio.read_audio(wav)
            # 确保音频不是空的
            if sig.shape[0] == 0:
                print(f"警告: 空音频文件 {wav}，使用默认音频")
                sig = torch.zeros(hparams["sample_rate"])
            return sig
        except Exception as e:
            print(f"警告: 无法读取音频文件 {wav}: {str(e)}")
            # 返回一个短的默认音频
            return torch.zeros(hparams["sample_rate"])
 
   
    
    # 标签处理管道
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo_id")
    def label_pipeline(emo):
        try:
        # 10分类情感映射
            emo_map = {'N': 0, 'H': 1, 'X': 2, 'A': 3, 'S': 4,
                   'U': 5, 'C': 6, 'O': 7, 'D': 8, 'F': 9}
            return torch.tensor(emo_map.get(emo, 0), dtype=torch.long)
        except Exception as e:
            print(f"警告: 无法处理情感标签 {emo}: {str(e)}")
            return torch.tensor(0, dtype=torch.long)  # 默认为中性
    
    # 创建数据集
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
            output_keys=["id", "sig", "emo_id"],
        )
    
    # 根据长度排序训练集
    if hparams.get("sorting", "ascending") == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False
    
    return datasets

def main():
    """主训练函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger.info("开始MSP-PODCAST情感分类训练")
    
    # 解析参数
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # 加载配置
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # 创建输出目录
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # 准备数据集
    datasets = dataio_prepare(hparams)
    
    # 初始化Brain
    emotion_brain = SimpleMSPEmotionBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # 训练
    emotion_brain.fit(
        epoch_counter=emotion_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    
    # 测试
    emotion_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()