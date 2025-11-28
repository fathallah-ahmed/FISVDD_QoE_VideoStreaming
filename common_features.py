# common_features.py
import numpy as np
import pandas as pd
from typing import List, Union
import config

FEATS = ["vmaf_mean", "vmaf_std", "vmaf_mad", "bitrate_mean", "stall_ratio", "tsl_end"]

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of windows:
      1. Clip bitrate and TSL to caps.
      2. Apply log1p to bitrate and TSL.
    """
    X = df.copy()
    # clip extrêmes (robust against spikes)
    X["bitrate_mean"] = X["bitrate_mean"].clip(0, config.BITRATE_CAP)
    X["tsl_end"]      = X["tsl_end"].clip(0, config.TSL_CAP)
    # log1p to shrink dynamic range
    X["bitrate_mean"] = np.log1p(X["bitrate_mean"])
    X["tsl_end"]      = np.log1p(X["tsl_end"])
    return X

def transform_row(vals: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Preprocess a single row (list or array of 6 floats).
    Order must be: [vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl]
    """
    vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl = map(float, vals)
    
    bitrate = np.log1p(min(max(bitrate, 0.0), config.BITRATE_CAP))
    tsl     = np.log1p(min(max(tsl,     0.0), config.TSL_CAP))
    
    return np.array([vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl], dtype=float)

