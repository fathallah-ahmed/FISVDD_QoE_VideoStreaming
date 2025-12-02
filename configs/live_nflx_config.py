"""
Configuration for LIVE_NFLX Dataset
"""
from .base_config import BaseDatasetConfig

class LiveNflxConfig(BaseDatasetConfig):
    """Configuration for LIVE_NFLX (Original) Dataset"""
    
    # Dataset identification
    DATASET_NAME = "LIVE_NFLX"
    DESCRIPTION = "LIVE-Netflix Video QoE Dataset (Original)"
    
    # Features
    FEATURE_COLUMNS = [
        "vmaf_mean",
        "vmaf_std",
        "vmaf_mad",
        "ssim",
        "stall_count",
        "tsl_end"
    ]
    
    # Target column
    TARGET_COLUMN = "QoE_win"
    
    # Preprocessing
    CLIP_FEATURES = ["stall_count", "tsl_end"]
    LOG_TRANSFORM_FEATURES = ["stall_count", "tsl_end"]
    
    # Data files
    TRAIN_FILE = "LIVE_NFLX_train.csv"
    TEST_FILE = "LIVE_NFLX_test.csv"
    
    # FISVDD parameters  
    THRESHOLD_QUANTILE = 0.90
    SIGMA_METHOD = "fixed"
    SIGMA_VALUE = 1.5  # Will tune this
    
    # Batch learning
    INITIAL_BATCH_SIZE = 100
    
    # Anomaly definition
    @staticmethod
    def is_anomaly(qoe_score):
        """QoE < 0 is considered anomaly"""
        return 1 if qoe_score < 0 else 0
