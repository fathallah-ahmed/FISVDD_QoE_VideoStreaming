# configs/base_config.py
"""
Base configuration class for all datasets.
Each dataset-specific config should inherit from this.
"""
import os
from typing import List, Optional

class BaseDatasetConfig:
    """Base configuration for dataset-specific settings."""
    
    # Dataset identification
    DATASET_NAME: str = "base"
    DATASET_DESCRIPTION: str = "Base configuration"
    
    # Paths (relative to project root)
    RESOURCES_DIR: str = "resources"
    ARTIFACTS_DIR: str = "artifacts"
    RESULTS_DIR: str = "results"
    
    # Feature columns
    FEATURE_COLUMNS: List[str] = []
    TARGET_COLUMN: Optional[str] = None
    
    # Preprocessing parameters
    CLIP_FEATURES: List[str] = []
    LOG_TRANSFORM_FEATURES: List[str] = []
    INVERT_FEATURES: List[str] = []
    
    # Training parameters
    TRAIN_FILE: Optional[str] = None
    TEST_FILE: Optional[str] = None
    THRESHOLD_QUANTILE: float = 0.95
    
    # FISVDD parameters
    SIGMA_METHOD: str = "median_heuristic"  # or "fixed"
    SIGMA_VALUE: Optional[float] = None
    
    # Batch learning parameters
    INITIAL_BATCH_SIZE: int = 100  # Default batch size for initial training
    INCREMENTAL_BATCH_SIZE: int = 50  # Batch size for incremental updates
    CHECKPOINT_EVERY_N_BATCHES: int = 10  # Save checkpoint frequency
    ENABLE_BATCH_CHECKPOINTS: bool = False  # Whether to save intermediate checkpoints
    
    # Incremental learning parameters
    NORMAL_BUFFER_MAX: int = 500
    REFIT_EVERY: int = 100
    MIN_REFIT_SIZE: int = 50
    REFIT_SAMPLE_SIZE: int = 200
    EWMA_ALPHA: float = 0.1
    MARGIN: float = 0.0
    
    @staticmethod
    def is_anomaly(y_value: float) -> int:
        """
        Determine if a target value represents an anomaly (1) or normal (0).
        Default: value <= 0 is anomaly.
        """
        return 1 if y_value <= 0 else 0

    @classmethod
    def get_resource_path(cls, filename: str) -> str:
        """Get full path to a resource file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, cls.RESOURCES_DIR, cls.DATASET_NAME, filename)
    
    @classmethod
    def get_artifact_path(cls, filename: str) -> str:
        """Get full path to an artifact file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, cls.ARTIFACTS_DIR, filename)
    
    @classmethod
    def get_results_path(cls, filename: str) -> str:
        """Get full path to a results file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, cls.RESULTS_DIR, cls.DATASET_NAME, filename)
    
    @classmethod
    def get_model_artifact_name(cls) -> str:
        """Get the model artifact filename for this dataset."""
        return f"{cls.DATASET_NAME}_fisvdd.joblib"
