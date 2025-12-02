# common_features.py
"""
Feature Preprocessing Utilities for Video QoE Anomaly Detection.

This module provides preprocessing functions for transforming raw video QoE features
before feeding them to the FISVDD model. It supports both legacy (LIVE_NFLX_II specific)
and generic (multi-dataset) preprocessing workflows.

Key Transformations:
- **Clipping**: Cap outliers to prevent extreme values
- **Log Transform**: Apply log(1 + x) to compress dynamic range and handle heavy tails
- **Inversion**: Flip features where 'higher is better' to 'lower is better'

Usage:
    For new datasets, use transform_df_generic() with dataset-specific configs.
    Legacy functions (transform_df, transform_row) are kept for backward compatibility.

Example:
    >>> from common_features import transform_df_generic
    >>> from configs import get_config
    >>> config = get_config("LFOVIA_QoE")
    >>> preprocessed_df = transform_df_generic(
    ...     df,
    ...     clip_features=config.CLIP_FEATURES,
    ...     log_features=config.LOG_TRANSFORM_FEATURES,
    ...     invert_features=config.INVERT_FEATURES
    ... )
"""
import numpy as np
import pandas as pd
from typing import List, Union, Optional

# Legacy feature list for LIVE_NFLX_II (backward compatibility)
FEATS = ["vmaf_mean", "vmaf_std", "vmaf_mad", "bitrate_mean", "stall_ratio", "tsl_end"]

# Legacy constants (for backward compatibility with LIVE_NFLX_II)
# These were previously in config.py which has been deprecated
BITRATE_CAP = 20000.0  # Maximum bitrate in kbps
TSL_CAP = 300.0        # Maximum time since last stall in seconds

# ============================================================================
# LEGACY FUNCTIONS (for backward compatibility with LIVE_NFLX_II)
# ============================================================================

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of windows (LIVE_NFLX_II specific):
      1. Clip bitrate and TSL to caps.
      2. Apply log1p to bitrate and TSL.
    
    NOTE: This is kept for backward compatibility. 
    Use transform_df_generic for new datasets.
    """
    X = df.copy()
    # clip extrêmes (robust against spikes)
    X["bitrate_mean"] = X["bitrate_mean"].clip(0, BITRATE_CAP)
    X["tsl_end"]      = X["tsl_end"].clip(0, TSL_CAP)
    X["stall_ratio"]  = X["stall_ratio"].clip(0, 1.0)
    # log1p to shrink dynamic range
    X["bitrate_mean"] = np.log1p(X["bitrate_mean"])
    X["tsl_end"]      = np.log1p(X["tsl_end"])
    X["stall_ratio"]  = np.log1p(X["stall_ratio"])
    return X

def transform_row(vals: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Preprocess a single row (list or array of 6 floats).
    Order must be: [vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl]
    
    NOTE: This is kept for backward compatibility.
    Use transform_row_generic for new datasets.
    """
    vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl = map(float, vals)
    
    bitrate = np.log1p(min(max(bitrate, 0.0), BITRATE_CAP))
    stall   = np.log1p(min(max(stall,   0.0), 1.0))
    tsl     = np.log1p(min(max(tsl,     0.0), TSL_CAP))
    
    return np.array([vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl], dtype=float)



# ============================================================================
# GENERIC FUNCTIONS (for multi-dataset support)
# ============================================================================

def transform_df_generic(
    df: pd.DataFrame,
    clip_features: Optional[List[str]] = None,
    log_features: Optional[List[str]] = None,
    invert_features: Optional[List[str]] = None,
    clip_min: float = 0.0,
    clip_max: Optional[float] = None
) -> pd.DataFrame:
    """
    Generic preprocessing pipeline for any video QoE dataset.
    
    This function applies a sequence of transformations to normalize and prepare
    features for the FISVDD model. Transformations are applied in order:
    1. Clipping (handle outliers)
    2. Log transform (compress dynamic range)
    3. Inversion (align feature semantics)
    
    Args:
        df: Input DataFrame with raw features
        clip_features: List of feature names to clip (e.g., ['bitrate', 'stall_ratio'])
        log_features: List of feature names to apply log(1 + x) transform
        invert_features: List of features to invert (max - x), used when
                        'higher is better' but model expects 'lower is better'
        clip_min: Minimum value for clipping (default: 0.0)
        clip_max: Maximum value for clipping (default: None = no upper bound)
    
    Returns:
        Preprocessed DataFrame with same structure as input
    
    Example:
        >>> df = pd.DataFrame({'TSL': [0.5, 10.2, 100], 'NIQE': [5.3, 6.1, 7.8]})
        >>> preprocessed = transform_df_generic(
        ...     df,
        ...     clip_features=['TSL'],
        ...     log_features=['TSL'],
        ...     invert_features=['TSL']  # Higher TSL = better quality
        ... )
    
    Notes:
        - Inversion uses batch max, suitable for offline training/testing
        - All transformations preserve NaN values
        - Returns a copy of the DataFrame, original is unchanged
    """
    X = df.copy()
    
    # Apply clipping
    if clip_features:
        for feat in clip_features:
            if feat in X.columns:
                if clip_max is not None:
                    X[feat] = X[feat].clip(clip_min, clip_max)
                else:
                    X[feat] = X[feat].clip(lower=clip_min)
    
    # Apply log1p transform
    if log_features:
        for feat in log_features:
            if feat in X.columns:
                X[feat] = np.log1p(X[feat])

    # Apply inversion (max - x)
    # Note: This uses the max of the current batch (df). 
    # For strict correctness in online inference, we should use a fixed max,
    # but for batch training/testing this is usually sufficient as StandardScaler follows.
    if invert_features:
        for feat in invert_features:
            if feat in X.columns:
                X[feat] = X[feat].max() - X[feat]
    
    return X


def transform_row_generic(
    vals: Union[List[float], np.ndarray],
    feature_names: List[str],
    clip_features: Optional[List[str]] = None,
    log_features: Optional[List[str]] = None,
    clip_min: float = 0.0,
    clip_max: Optional[float] = None
) -> np.ndarray:
    """
    Generic preprocessing for a single row.
    
    Args:
        vals: Feature values
        feature_names: Names of features (must match length of vals)
        clip_features: Features to clip
        log_features: Features to apply log1p
        clip_min: Minimum clipping value
        clip_max: Maximum clipping value
        
    Returns:
        Preprocessed feature array
    """
    if len(vals) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(vals)} values but {len(feature_names)} feature names"
        )
    
    result = np.array(vals, dtype=float)
    
    # Apply clipping
    if clip_features:
        for i, feat in enumerate(feature_names):
            if feat in clip_features:
                if clip_max is not None:
                    result[i] = np.clip(result[i], clip_min, clip_max)
                else:
                    result[i] = max(result[i], clip_min)
    
    # Apply log1p
    if log_features:
        for i, feat in enumerate(feature_names):
            if feat in log_features:
                result[i] = np.log1p(result[i])
    
    return result

