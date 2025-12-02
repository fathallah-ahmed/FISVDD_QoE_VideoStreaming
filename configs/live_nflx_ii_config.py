# configs/live_nflx_ii_config.py
"""
Configuration for LIVE-Netflix-II dataset.
"""
from .base_config import BaseDatasetConfig

class LiveNflxIIConfig(BaseDatasetConfig):
    """Configuration for LIVE-Netflix-II dataset."""
    
    # Dataset identification
    DATASET_NAME = "LIVE_NFLX_II"
    DATASET_DESCRIPTION = "LIVE-Netflix-II Video QoE Dataset (2018)"
    
    # Feature columns (same as original)
    FEATURE_COLUMNS = [
        "vmaf_mean",
        "vmaf_std", 
        "vmaf_mad",
        "bitrate_mean",
        "stall_ratio",
        "tsl_end"
    ]
    
    TARGET_COLUMN = "QoE_win"
    
    # Preprocessing (same as original common_features.py)
    CLIP_FEATURES = ["bitrate_mean", "stall_ratio", "tsl_end"]
    LOG_TRANSFORM_FEATURES = ["bitrate_mean", "stall_ratio", "tsl_end"]
    
    # Data files
    TRAIN_FILE = "LIVE_NFLX_II_FISVDD_train.csv"
    TEST_FILE = "LIVE_NFLX_II_windows_minimal.csv"
    
    # FISVDD parameters (tuned for this dataset)
    THRESHOLD_QUANTILE = 0.95
    SIGMA_METHOD = "median_heuristic"
    
    # Incremental learning parameters
    NORMAL_BUFFER_MAX = 500
    REFIT_EVERY = 100
    MIN_REFIT_SIZE = 50
    REFIT_SAMPLE_SIZE = 200
    EWMA_ALPHA = 0.1
    MARGIN = 0.0
