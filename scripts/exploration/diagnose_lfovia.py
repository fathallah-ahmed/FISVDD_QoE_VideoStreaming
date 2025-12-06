import pandas as pd
import numpy as np

# Load LFOVIA data
train = pd.read_csv('resources/LFOVIA_QoE/LFOVIA_QoE_train.csv')
test = pd.read_csv('resources/LFOVIA_QoE/LFOVIA_QoE_test.csv')

print("="*60)
print("LFOVIA DATASET ANALYSIS")
print("="*60)

print("\n1. TRAINING DATA:")
print(f"   Total samples: {len(train)}")
print(f"   Good (QoE>=50): {len(train[train['QoE']>=50])}")
print(f"   Bad (QoE<50): {len(train[train['QoE']<50])}")
print(f"   QoE range: {train['QoE'].min():.1f} to {train['QoE'].max():.1f}")
print(f"   QoE mean: {train['QoE'].mean():.1f}")

print("\n2. TEST DATA:")
print(f"   Total samples: {len(test)}")
print(f"   Good (QoE>=50): {len(test[test['QoE']>=50])}")
print(f"   Bad (QoE<50): {len(test[test['QoE']<50])}")

print("\n3. FEATURE-QoE CORRELATIONS:")
corr = train.corr()['QoE'].drop('QoE')
print(corr.sort_values(ascending=False))

print("\n4. DIAGNOSIS:")
if corr['TSL'].abs() < 0.1 and corr['Nrebuffers'].abs() < 0.1:
    print("   ⚠️  WEAK CORRELATIONS - Features barely predict QoE")
if len(train[train['QoE']>=50]) < 100:
    print("   ⚠️  TOO FEW GOOD SAMPLES - Insufficient training data")
if corr.min() < -0.5:
    print(f"   ⚠️  STRONG NEGATIVE CORRELATION: {corr.idxmin()} = {corr.min():.3f}")
if corr.max() < 0.3:
    print("   ⚠️  ALL CORRELATIONS WEAK - May need better features")

print("\n" + "="*60)
