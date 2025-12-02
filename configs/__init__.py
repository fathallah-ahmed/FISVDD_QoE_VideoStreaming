# configs/__init__.py
"""
Dataset configuration management.
"""
from .base_config import BaseDatasetConfig
from .live_nflx_ii_config import LiveNflxIIConfig
from .lfovia_qoe_config import LfoviaQoeConfig

# Registry of available datasets
DATASET_CONFIGS = {
    "LIVE_NFLX_II": LiveNflxIIConfig,
    "LFOVIA_QoE": LfoviaQoeConfig,
}

def get_config(dataset_name: str) -> BaseDatasetConfig:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "LIVE_NFLX_II", "LFOVIA_QoE")
        
    Returns:
        Dataset configuration class
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    return DATASET_CONFIGS[dataset_name]

__all__ = [
    "BaseDatasetConfig",
    "LiveNflxIIConfig", 
    "LfoviaQoeConfig",
    "DATASET_CONFIGS",
    "get_config",
]
