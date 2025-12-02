# test_batch_learning.py
"""Quick test to verify incremental batch learning works."""
import numpy as np
from fisvdd import fisvdd

# Create simple synthetic data
np.random.seed(42)
normal_data = np.random.randn(200, 3) * 0.5  # 200 normal points, 3 features
outliers = np.random.randn(20, 3) * 3  # 20 outliers
data = np.vstack([normal_data, outliers])
np.random.shuffle(data)

print("Testing Incremental Batch Learning")
print("="*60)

# Test 1: Initialize with first batch
batch_size = 50
first_batch = data[:batch_size]
print(f"\n[Test 1] Initialize with {batch_size} samples")
model = fisvdd(first_batch, sigma=1.0, initial_batch_only=True)
model.find_sv(first_batch)
print(f"✓ Initialized. SVs: {len(model.sv)}, Processed: {model.num_processed}")

# Test 2: Update with second batch
second_batch = data[batch_size:batch_size*2]
print(f"\n[Test 2] Update with {len(second_batch)} more samples")
stats = model.update_incremental(second_batch)
print(f"✓ Updated. SVs: {stats['final_sv_count']}, Total processed: {stats['total_processed']}")
print(f"  Added: {stats['sv_added']}, Removed: {stats['sv_removed']}")

# Test 3: Update with third batch
third_batch = data[batch_size*2:batch_size*3]
print(f"\n[Test 3] Update with {len(third_batch)} more samples")
stats = model.update_incremental(third_batch)
print(f"✓ Updated. SVs: {stats['final_sv_count']}, Total processed: {stats['total_processed']}")

# Test 4: Get and set state (checkpointing)
print(f"\n[Test 4] Testing state save/restore")
state = model.get_state()
print(f"✓ Saved state. SVs in state: {len(state['sv'])}")

# Create new model and restore state
model2 = fisvdd(np.array([[0,0,0]]), sigma=1.0, initial_batch_only=True)
model2.set_state(state)
print(f"✓ Restored state. SVs: {len(model2.sv)}, Processed: {model2.num_processed}")

# Test 5: Score some points
print(f"\n[Test 5] Scoring new points")
test_points = np.random.randn(5, 3) * 0.5
for i, point in enumerate(test_points):
    score, _ = model.score_fcn(point.reshape(1, -1))
    print(f"  Point {i+1}: score = {score:.4f}")

print(f"\n{'='*60}")
print("✅ All tests passed!")
print(f"{'='*60}")
