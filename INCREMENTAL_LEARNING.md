# Incremental Learning Quick Reference

## Default Behavior (NEW)

**FISVDD now uses incremental batch learning by default!** 🎉

### Basic Usage

```bash
# Train with incremental learning (default batch size from config)
python train_fisvdd.py --dataset LIVE_NFLX_II

# Train with custom batch size
python train_fisvdd.py --dataset LIVE_NFLX_II --batch-size 200

# Train LFOVIA dataset
python train_fisvdd.py --dataset LFOVIA_QoE --batch-size 100
```

### Standard Mode (If Needed)

If you need the old behavior (all data at once):
```bash
python train_fisvdd.py --dataset LIVE_NFLX_II --standard-mode
```

## Update Existing Models

Add new data to trained models:
```bash
# First, create or prepare your new data CSV file
# Then update the model:
python update_model_incremental.py --dataset LIVE_NFLX_II --new-data path/to/new_data.csv
```

## Key Advantages

✅ **Memory Efficient**: Process data in batches, not all at once  
✅ **True Incremental**: Model learns continuously as new data arrives  
✅ **Progress Tracking**: See batch-by-batch progress  
✅ **Flexible**: Adjust batch sizes based on your needs  
✅ **Backward Compatible**: Old standard mode still available with `--standard-mode`

## Default Batch Sizes (from config)

- `INITIAL_BATCH_SIZE = 100`: Default for training
- `INCREMENTAL_BATCH_SIZE = 50`: Default for updates

Override with `--batch-size` argument.

## Example Workflow

```bash
# Day 1: Initial training
python train_fisvdd.py --dataset LIVE_NFLX_II --batch-size 100

# Day 2: New data arrives - update model
python update_model_incremental.py --dataset LIVE_NFLX_II --new-data day2_samples.csv

# Day 3: More new data - keep learning
python update_model_incremental.py --dataset LIVE_NFLX_II --new-data day3_samples.csv

# Model continuously evolves with new data!
```

## Testing

```bash
# After training, test the model
python test_fisvdd.py --dataset LIVE_NFLX_II

# Benchmark with k-fold validation
python benchmark_fisvdd.py --dataset LIVE_NFLX_II
```

---

**Note**: All existing scripts and API remain backward compatible. Old models can be updated incrementally after training with `update_model_incremental.py`.
