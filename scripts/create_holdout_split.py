"""
Create Hold-out Test Set for Rakuten Project

This script creates a proper train/test split ONCE at the project level.
The hold-out test set will NEVER be used during development, only for final evaluation.

Usage:
    python scripts/create_holdout_split.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Configuration
DATA_DIR = Path("data")
HOLDOUT_SIZE = 0.15
RANDOM_STATE = 42

def create_holdout_split():
    """Create and save hold-out test set."""

    print("=" * 80)
    print("CREATING HOLD-OUT TEST SET")
    print("=" * 80)
    print(f"Hold-out size: {HOLDOUT_SIZE} ({HOLDOUT_SIZE*100:.0f}%)")
    print(f"Random state: {RANDOM_STATE}")
    print()

    # Load data
    print("Loading data...")
    X_train = pd.read_csv(DATA_DIR / "X_train_update.csv", index_col=0)
    Y_train = pd.read_csv(DATA_DIR / "Y_train_CVw08PX.csv", index_col=0)

    df_full = X_train.join(Y_train, how='inner')
    print(f"✓ Total samples: {len(df_full):,}")
    print(f"  Classes: {df_full['prdtypecode'].nunique()}")

    # Split
    print(f"\nCreating stratified split...")
    train_dev, test_holdout = train_test_split(
        df_full,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full['prdtypecode']
    )

    print(f"✓ Development set: {len(train_dev):,} samples ({100*(1-HOLDOUT_SIZE):.0f}%)")
    print(f"✓ Hold-out test:   {len(test_holdout):,} samples ({HOLDOUT_SIZE*100:.0f}%)")

    # Verify stratification
    print("\nVerifying stratification...")
    dev_classes = train_dev['prdtypecode'].nunique()
    test_classes = test_holdout['prdtypecode'].nunique()
    total_classes = df_full['prdtypecode'].nunique()

    assert dev_classes == total_classes, f"Development set missing classes: {dev_classes}/{total_classes}"
    assert test_classes == total_classes, f"Hold-out set missing classes: {test_classes}/{total_classes}"
    print(f"✓ All {total_classes} classes present in both splits")

    # Save
    print("\nSaving splits...")
    train_dev.to_csv(DATA_DIR / "train_development.csv")
    test_holdout.to_csv(DATA_DIR / "test_holdout.csv")

    print(f"✓ Development set saved: {DATA_DIR / 'train_development.csv'}")
    print(f"✓ Hold-out test saved:   {DATA_DIR / 'test_holdout.csv'}")

    # Save metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'holdout_size': HOLDOUT_SIZE,
        'random_state': RANDOM_STATE,
        'total_samples': len(df_full),
        'dev_samples': len(train_dev),
        'test_samples': len(test_holdout),
        'num_classes': int(total_classes),
        'warning': 'DO NOT USE test_holdout.csv DURING DEVELOPMENT!'
    }

    with open(DATA_DIR / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved: {DATA_DIR / 'split_metadata.json'}")

    # Warning
    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT WARNING")
    print("=" * 80)
    print("The file 'test_holdout.csv' contains the hold-out test set.")
    print("DO NOT use this file during development!")
    print("Only load it for final model evaluation.")
    print("=" * 80)

    return train_dev, test_holdout

if __name__ == "__main__":
    create_holdout_split()
