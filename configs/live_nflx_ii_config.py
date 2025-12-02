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
    
    # Data files - Fresh split from .mat files
    # Train: Good windows from 57 training videos
    # Test: All windows from 3 held-out videos
    TRAIN_FILE = "LIVE_NFLX_II_train.csv"
    TEST_FILE = "LIVE_NFLX_II_test.csv"
    
    # FISVDD parameters (optimized via grid search)
    THRESHOLD_QUANTILE = 0.90
    SIGMA_METHOD = "fixed"
    SIGMA_VALUE = 2.081761
    
    # Batch learning parameters (optimized)
    INITIAL_BATCH_SIZE = 100
    
    # Incremental learning parameters
    NORMAL_BUFFER_MAX = 500
    REFIT_EVERY = 100
    MIN_REFIT_SIZE = 50
    REFIT_SAMPLE_SIZE = 200
    EWMA_ALPHA = 0.1
    MARGIN = 0.0
