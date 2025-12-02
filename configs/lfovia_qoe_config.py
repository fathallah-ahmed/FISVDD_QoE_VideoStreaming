# configs/lfovia_qoe_config.py
"""
Configuration for LFOVIA_QoE dataset.
"""
from .base_config import BaseDatasetConfig

class LfoviaQoeConfig(BaseDatasetConfig):
    """Configuration for LFOVIA_QoE dataset."""
    
    # Dataset identification
    DATASET_NAME = "LFOVIA_QoE"
    DATASET_DESCRIPTION = "LFOVIA QoE Dataset"
    
    # Feature columns from .mat files
    # Using streaming-specific + visual quality features
    # Based on correlation analysis:
    #   - TSL: 0.53 (time since last rebuffer - higher is better)
    #   - Nrebuffers: -0.48 (number of rebuffers - lower is better)
    #   - NIQE: -0.28 (no-reference visual quality - lower is better)
    #   - SSIM: 0.18 (structural similarity - higher is better)
    FEATURE_COLUMNS = [
        "TSL",           # Time since last rebuffer
        "Nrebuffers",    # Number of rebuffering events
        "NIQE",          # No-reference image quality (lower = better)
        "SSIM"           # Structural similarity (higher = better)
    ]
    
    TARGET_COLUMN = "score_continuous"
    
    # Preprocessing
    # Normalize features so "lower is better" for anomaly detection consistency
    CLIP_FEATURES = ["TSL", "Nrebuffers"]  # Clip streaming metrics to avoid outliers
    LOG_TRANSFORM_FEATURES = ["TSL", "Nrebuffers"]  # Log transform for heavy-tailed distributions
    INVERT_FEATURES = ["TSL", "SSIM"]  # Invert so lower values = worse quality
    
    # Data files
    TRAIN_FILE = "LFOVIA_QoE_train.csv"
    TEST_FILE = "LFOVIA_QoE_test.csv"
    
    # FISVDD parameters (optimized via grid search)
    THRESHOLD_QUANTILE = 0.90
    SIGMA_METHOD = "fixed"
    SIGMA_VALUE = 1.655873
    
    # Batch learning parameters (optimized)
    INITIAL_BATCH_SIZE = 100
    
    # Incremental learning parameters
    NORMAL_BUFFER_MAX = 500
    REFIT_EVERY = 100
    MIN_REFIT_SIZE = 50
    REFIT_SAMPLE_SIZE = 200
    EWMA_ALPHA = 0.1
    MARGIN = 0.0

    @staticmethod
    def is_anomaly(y_value: float) -> int:
        """
        LFOVIA score_continuous values are 0-100.
        We define anomaly (bad quality) as score < 50.
        """
        return 1 if y_value < 50 else 0
