# common_features.py
import numpy as np
import pandas as pd

FEATS = ["vmaf_mean","vmaf_std","vmaf_mad","bitrate_mean","stall_ratio","tsl_end"]

# caps chosen to tame heavy tails; adjust if you know your ranges better
BITRATE_CAP = 8000.0   # kbps
TSL_CAP     = 60.0     # seconds

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # clip extrêmes (robust against spikes)
    X["bitrate_mean"] = X["bitrate_mean"].clip(0, BITRATE_CAP)
    X["tsl_end"]      = X["tsl_end"].clip(0, TSL_CAP)
    # log1p to shrink dynamic range
    X["bitrate_mean"] = np.log1p(X["bitrate_mean"])
    X["tsl_end"]      = np.log1p(X["tsl_end"])
    return X

def transform_row(vals):
    vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl = map(float, vals)
    bitrate = np.log1p(min(max(bitrate, 0.0), BITRATE_CAP))
    tsl     = np.log1p(min(max(tsl,     0.0), TSL_CAP))
    return np.array([vmaf_mean, vmaf_std, vmaf_mad, bitrate, stall, tsl], dtype=float)
