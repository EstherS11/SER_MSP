"""
MSP-PODCAST data preparation for K-means clustering
Adapted from IEMOCAP preparation script
"""

import json
import logging
import torch
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)

def dataio_prepare(hparams):
    """
    Prepare the data for MSP-PODCAST emotion recognition training
    """
    
    # Define audio pipeline
    @speechbrain.utils.data_pipeline.takes("wav")
    @speechbrain.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load audio"""
        sig = read_audio(wav)
        return sig
    
    # Define emotion pipeline  
    @speechbrain.utils.data_pipeline.takes("emo")
    @speechbrain.utils.data_pipeline.provides("emo_encoded")
    def emotion_pipeline(emo):
        """Encode emotion labels"""
        yield emo
    
    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"], 
        "test": hparams["test_json"]
    }
    
    for dataset in data_info:
        datasets[dataset] = speechbrain.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, emotion_pipeline],
            output_keys=["id", "sig", "emo_encoded"]
        )
    
    # Sort training set by length for efficient batching
    if hparams.get("sorting", "ascending") == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams.get("sorting", "ascending") == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="length", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams.get("sorting", "ascending") == "random":
        pass
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")
    
    return datasets


def prepare_msp_podcast_datasets(
    train_json,
    valid_json, 
    test_json,
    sample_rate=16000
):
    """
    Prepare MSP-PODCAST datasets for K-means training
    
    Args:
        train_json: Path to training JSON file
        valid_json: Path to validation JSON file  
        test_json: Path to test JSON file
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary containing train/valid/test datasets
    """
    
    # Import speechbrain modules
    import speechbrain as sb
    from speechbrain.dataio.dataset import DynamicItemDataset
    from speechbrain.dataio.dataio import read_audio
    import speechbrain.utils.data_pipeline
    
    # Audio processing pipeline
    @speechbrain.utils.data_pipeline.takes("wav")
    @speechbrain.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load and process audio"""
        sig = read_audio(wav)
        return sig
    
    # Emotion encoding pipeline
    @speechbrain.utils.data_pipeline.takes("emo") 
    @speechbrain.utils.data_pipeline.provides("emo_encoded")
    def emotion_pipeline(emo):
        """Process emotion labels"""
        yield emo
    
    # Load datasets
    datasets = {}
    json_files = {
        "train": train_json,
        "valid": valid_json,
        "test": test_json
    }
    
    for split, json_file in json_files.items():
        logger.info(f"Loading {split} dataset from {json_file}")
        
        datasets[split] = DynamicItemDataset.from_json(
            json_path=json_file,
            dynamic_items=[audio_pipeline, emotion_pipeline],
            output_keys=["id", "sig", "emo_encoded", "length"]
        )
        
        logger.info(f"{split} dataset loaded: {len(datasets[split])} samples")
    
    return datasets


if __name__ == "__main__":
    # Test data loading
    datasets = prepare_msp_podcast_datasets(
        train_json="msp_train_minimal.json",
        valid_json="msp_valid_minimal.json", 
        test_json="msp_test_minimal.json"
    )
    
    print("Datasets loaded successfully!")
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} samples")
        
    # Test loading one sample
    train_sample = datasets["train"][0]
    print(f"Sample keys: {train_sample.keys()}")
    print(f"Audio shape: {train_sample['sig'].shape}")
    print(f"Emotion: {train_sample['emo_encoded']}")