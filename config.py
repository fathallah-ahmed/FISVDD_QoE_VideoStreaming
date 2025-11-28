# config.py

# Feature extraction constants
BITRATE_CAP = 8000.0   # kbps
TSL_CAP     = 60.0     # seconds

# Model incremental learning constants
NORMAL_BUFFER_MAX = 4000
REFIT_EVERY       = 500
EWMA_ALPHA        = 0.10
MARGIN            = 0.02
MIN_REFIT_SIZE    = 50
REFIT_SAMPLE_SIZE = 2000
THRESHOLD_QUANTILE = 0.95

# File paths (relative to project root usually, but defined here for clarity if needed)
ARTIFACTS_FILENAME = "fisvdd_artifacts.joblib"
