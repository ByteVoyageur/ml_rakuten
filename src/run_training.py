"""
Production Training Script for Rakuten Image Classification

This script implements the EXACT training pipeline required for team benchmark consistency.

CRITICAL FEATURES (Read before running):
1. Global Label Encoding: Uses LabelEncoder on ENTIRE dataset BEFORE splitting
   - Ensures exactly 27 classes (0-26) globally
   - Prevents shape mismatch errors between train/val

2. Strict Splitting Strategy: Uses team's exact train_test_split method
   - test_size=0.15 (85% train, 15% val)
   - random_state=42 for reproducibility
   - Stratified on encoded labels

3. Class Weights: Computed from y_train frequencies only (sklearn's compute_class_weight)
   - Handles class imbalance properly
   - Based on training set only (no data leakage)

4. Model Configuration:
   - ResNet50 with 27 output classes
   - Frozen backbone (transfer learning) by default
   - Custom classification head with dropout

5. Hardware Optimizations:
   - Batch size 32 (safe for RTX 3060 Ti 8GB)
   - Automatic Mixed Precision (AMP) enabled
   - 4 DataLoader workers

Usage:
    # Default training (recommended):
    python src/run_training.py

    # Custom configuration:
    python src/run_training.py --epochs 20 --batch_size 64 --lr 5e-5

    # See all options:
    python src/run_training.py --help

Hardware: Optimized for NVIDIA RTX 3060 Ti (8GB VRAM)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import rakuten_image library
from src.rakuten_image.datasets import RakutenImageDataset
from src.rakuten_image.models import create_model
from src.rakuten_image.transforms import get_train_transforms, get_val_transforms
from src.rakuten_image.train import train
from src.rakuten_image.utils import set_seed


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Rakuten Image Classification Model (Production)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--x_train_csv', type=str,
                        default='data/X_train_update.csv',
                        help='Path to X_train CSV file')
    parser.add_argument('--y_train_csv', type=str,
                        default='data/Y_train_CVw08PX.csv',
                        help='Path to Y_train CSV file')
    parser.add_argument('--image_dir', type=str,
                        default='data/images/image_train',
                        help='Directory containing training images')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/resnet50_full',
                        help='Directory to save model checkpoints')

    # Model configuration
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vit', 'fusion', 'fusion_lite'],
                        help='Model architecture to train')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone weights (transfer learning)')
    parser.add_argument('--no_freeze_backbone', dest='freeze_backbone',
                        action='store_false',
                        help='Train backbone end-to-end')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (32 is safe for GTX 3060 Ti)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4,
                        dest='learning_rate',
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio (0.15 = 15%, team benchmark standard)')

    # Hardware optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable AMP (not recommended for 8GB VRAM)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer')

    # Early stopping and scheduling
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='LR scheduler patience (epochs)')

    # Data augmentation
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (224 for ImageNet models)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable training data augmentation')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--no_class_weights', dest='use_class_weights',
                        action='store_false',
                        help='Disable class weights')

    return parser.parse_args()


def load_and_merge_data(x_train_csv, y_train_csv):
    """
    Load and merge X_train and Y_train CSV files.

    Args:
        x_train_csv: Path to X_train CSV
        y_train_csv: Path to Y_train CSV

    Returns:
        pd.DataFrame: Merged dataframe
    """
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)

    # Load CSV files
    print(f"Loading {x_train_csv}...")
    X_train = pd.read_csv(x_train_csv)
    print(f"  Shape: {X_train.shape}")

    print(f"Loading {y_train_csv}...")
    Y_train = pd.read_csv(y_train_csv)
    print(f"  Shape: {Y_train.shape}")

    # Merge
    df = pd.concat([X_train, Y_train], axis=1)
    print(f"\n✓ Merged dataset: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

    # Verify required columns
    required_cols = ['imageid', 'productid', 'prdtypecode']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def encode_labels(df, label_col='prdtypecode'):
    """
    Encode labels using LabelEncoder to ensure consistent 0-N mapping.

    CRITICAL: This prevents shape mismatch errors by ensuring we have
    exactly num_classes classes mapped to 0, 1, 2, ..., num_classes-1.

    This MUST be done on the ENTIRE dataset BEFORE splitting to ensure
    consistent label encoding across train/val splits.

    Args:
        df: DataFrame with labels
        label_col: Name of label column

    Returns:
        df: DataFrame with encoded labels
        label_encoder: Fitted LabelEncoder
        num_classes: Number of unique classes
        label_mapping: Dictionary mapping original to encoded labels
    """
    print("\n" + "=" * 70)
    print("Global Label Encoding (CRITICAL - BEFORE Splitting)")
    print("=" * 70)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform on ENTIRE dataset
    df['encoded_label'] = label_encoder.fit_transform(df[label_col])

    num_classes = len(label_encoder.classes_)

    print(f"✓ Label encoding complete")
    print(f"  Original labels (unique): {len(df[label_col].unique())}")
    print(f"  Encoded labels: 0 to {num_classes - 1}")
    print(f"  Number of classes: {num_classes}")

    # CRITICAL VALIDATION: Ensure exactly 27 classes (0-26)
    unique_encoded = sorted(df['encoded_label'].unique())
    expected_classes = 27

    if num_classes != expected_classes:
        raise ValueError(
            f"Expected exactly {expected_classes} classes, but found {num_classes}! "
            f"This will cause shape mismatch errors during training."
        )

    if min(unique_encoded) != 0 or max(unique_encoded) != expected_classes - 1:
        raise ValueError(
            f"Expected labels in range [0, {expected_classes - 1}], "
            f"but found [{min(unique_encoded)}, {max(unique_encoded)}]"
        )

    print(f"  ✅ VALIDATION PASSED: Exactly {expected_classes} classes (0-{expected_classes - 1})")

    # Create mapping dictionary
    label_mapping = {
        original: encoded
        for original, encoded in zip(label_encoder.classes_, range(num_classes))
    }

    print(f"\nClass mapping (first 10):")
    for orig, enc in list(label_mapping.items())[:10]:
        count = (df[label_col] == orig).sum()
        print(f"  {orig} → {enc} ({count} samples)")

    if len(label_mapping) > 10:
        print(f"  ... and {len(label_mapping) - 10} more classes")

    return df, label_encoder, num_classes, label_mapping


def stratified_split(df, val_split=0.15, random_state=42):
    """
    Perform stratified train/validation split using EXACT team benchmark method.

    CRITICAL: This uses the exact splitting strategy required for team consistency:
    - Split on indices (X = df.index, y = df['encoded_label'])
    - Use test_size, random_state, stratify parameters
    - Create train_df and val_df based on returned indices

    Args:
        df: DataFrame with encoded labels
        val_split: Validation split ratio
        random_state: Random seed

    Returns:
        train_df, val_df: Split dataframes
        y_train, y_val: Label arrays for class weight computation
    """
    print("\n" + "=" * 70)
    print("Stratified Train/Val Split (Team Benchmark Method)")
    print("=" * 70)

    # EXACT splitting code as per user requirements
    X = df.index  # Use indices
    y = df['encoded_label']  # Use encoded labels

    # Perform stratified split
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_split,
        random_state=random_state,
        stratify=y,
    )

    # Create train_df and val_df based on these indices
    train_df = df.loc[X_train_idx].reset_index(drop=True)
    val_df = df.loc[X_val_idx].reset_index(drop=True)

    print(f"✓ Split complete")
    print(f"  Train samples: {len(train_df)} ({100 * (1 - val_split):.0f}%)")
    print(f"  Val samples: {len(val_df)} ({100 * val_split:.0f}%)")

    # Verify stratification
    print(f"\nClass distribution verification:")
    train_dist = train_df['encoded_label'].value_counts().sort_index()
    val_dist = val_df['encoded_label'].value_counts().sort_index()

    print(f"  All classes in train: {len(train_dist) == df['encoded_label'].nunique()}")
    print(f"  All classes in val: {len(val_dist) == df['encoded_label'].nunique()}")

    return train_df, val_df, y_train, y_val


def compute_class_weights(y_train, num_classes):
    """
    Compute class weights for handling imbalanced datasets.

    CRITICAL: Uses sklearn's compute_class_weight on y_train frequencies only.

    Args:
        y_train: Training labels (from train_test_split)
        num_classes: Number of classes

    Returns:
        torch.Tensor: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    print("\n" + "=" * 70)
    print("Computing Class Weights (from y_train frequencies)")
    print("=" * 70)

    # Compute class weights using sklearn
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = torch.tensor(class_weights_array, dtype=torch.float32)

    print(f"✓ Class weights computed from training set")
    print(f"  Shape: {class_weights.shape}")
    print(f"  Min weight: {class_weights.min():.4f}")
    print(f"  Max weight: {class_weights.max():.4f}")
    print(f"  Mean weight: {class_weights.mean():.4f}")

    # Show most imbalanced classes
    weight_sorted_idx = torch.argsort(class_weights, descending=True)
    print(f"\nMost imbalanced classes (highest weights - top 5):")
    for i in range(min(5, len(weight_sorted_idx))):
        idx = weight_sorted_idx[i].item()
        count = (y_train == idx).sum()
        print(f"  Class {idx}: {count} samples, weight {class_weights[idx]:.4f}")

    return class_weights


def save_training_metadata(checkpoint_dir, label_mapping, args, num_classes):
    """
    Save training metadata for later inference.

    Args:
        checkpoint_dir: Checkpoint directory
        label_mapping: Dictionary mapping original labels to encoded labels
        args: Parsed arguments
        num_classes: Number of classes
    """
    metadata = {
        'num_classes': num_classes,
        'label_mapping': {int(k): int(v) for k, v in label_mapping.items()},
        'model': args.model,
        'img_size': args.img_size,
        'freeze_backbone': args.freeze_backbone,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'use_amp': args.use_amp,
        }
    }

    metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Training metadata saved to {metadata_path}")


def main():
    """Main training pipeline."""

    # Parse arguments
    args = parse_arguments()

    # Print configuration
    print("\n" + "=" * 70)
    print("RAKUTEN IMAGE CLASSIFICATION - PRODUCTION TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Validation Split: {args.val_split}")
    print(f"  Freeze Backbone: {args.freeze_backbone}")
    print(f"  Use AMP: {args.use_amp}")
    print(f"  Use Class Weights: {args.use_class_weights}")
    print(f"  Random Seed: {args.seed}")

    # Set random seed
    set_seed(args.seed)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'=' * 70}")
    print("Hardware Configuration")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        warnings.warn("⚠ No GPU detected! Training will be very slow.")

    # ========================================================================
    # STEP 1: Load and Merge Data
    # ========================================================================

    df = load_and_merge_data(args.x_train_csv, args.y_train_csv)

    # ========================================================================
    # STEP 2: Encode Labels (CRITICAL for avoiding shape mismatch)
    # ========================================================================

    df, label_encoder, num_classes, label_mapping = encode_labels(df, 'prdtypecode')

    # ========================================================================
    # STEP 3: Stratified Train/Val Split (EXACT team benchmark method)
    # ========================================================================

    train_df, val_df, y_train, y_val = stratified_split(
        df, val_split=args.val_split, random_state=args.seed
    )

    # ========================================================================
    # STEP 4: Compute Class Weights (from y_train frequencies)
    # ========================================================================

    if args.use_class_weights:
        class_weights = compute_class_weights(y_train, num_classes)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
        print("\n⚠ Class weights disabled (may hurt performance on imbalanced data)")

    # ========================================================================
    # STEP 5: Create Datasets and DataLoaders
    # ========================================================================

    print("\n" + "=" * 70)
    print("Creating Datasets and DataLoaders")
    print("=" * 70)

    # Get transforms
    augment = not args.no_augment
    train_transform = get_train_transforms(img_size=args.img_size, augment=augment)
    val_transform = get_val_transforms(img_size=args.img_size)

    print(f"Train augmentation: {'Enabled' if augment else 'Disabled'}")

    # IMPORTANT: Use 'encoded_label' as the label column
    # We already have the encoded labels, so we tell the dataset NOT to remap
    # by setting the label_col to 'encoded_label' which is already 0-indexed

    # Create training dataset
    train_dataset = RakutenImageDataset(
        dataframe=train_df,
        image_dir=args.image_dir,
        transform=train_transform,
        label_col='encoded_label',  # Use pre-encoded labels
        verify_images=True,
        remove_missing=True
    )

    # Create validation dataset
    val_dataset = RakutenImageDataset(
        dataframe=val_df,
        image_dir=args.image_dir,
        transform=val_transform,
        label_col='encoded_label',  # Use pre-encoded labels
        verify_images=True,
        remove_missing=True
    )

    print(f"\n✓ Datasets created")
    print(f"  Train samples (after filtering): {len(train_dataset)}")
    print(f"  Val samples (after filtering): {len(val_dataset)}")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and (device == 'cuda'),
        drop_last=True  # For BatchNorm stability
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and (device == 'cuda')
    )

    print(f"\n✓ DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Workers: {args.num_workers}")

    # ========================================================================
    # STEP 6: Create Model
    # ========================================================================

    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)

    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=0.3
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # ========================================================================
    # STEP 7: Create Checkpoint Directory and Save Metadata
    # ========================================================================

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_training_metadata(args.checkpoint_dir, label_mapping, args, num_classes)

    # ========================================================================
    # STEP 8: Train Model
    # ========================================================================

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        use_amp=args.use_amp,
        scheduler_patience=args.scheduler_patience,
        class_weights=class_weights
    )

    # ========================================================================
    # STEP 9: Save Training History
    # ========================================================================

    print("\n" + "=" * 70)
    print("Saving Training History")
    print("=" * 70)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.checkpoint_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to {history_path}")

    # Print final results
    best_epoch = history_df['val_loss'].idxmin()
    best_val_loss = history_df.loc[best_epoch, 'val_loss']
    best_val_acc = history_df.loc[best_epoch, 'val_acc']

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"  - best_model.pth")
    print(f"  - training_metadata.json")
    print(f"  - training_history.csv")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        raise
