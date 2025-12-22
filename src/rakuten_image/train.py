"""
Training pipeline for Rakuten image classification.

Includes:
- Training and evaluation loops
- Automatic Mixed Precision (AMP) for memory efficiency
- Early stopping and model checkpointing
- Learning rate scheduling
- Progress tracking and metrics logging
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import custom modules
from .datasets import RakutenImageDataset
from .models import create_model
from .transforms import get_train_transforms, get_val_transforms
from .utils import set_seed, save_checkpoint, EarlyStopping


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Train the model for one epoch with Automatic Mixed Precision.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        scaler: GradScaler for mixed precision training
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision (default: True)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (CPU or no AMP)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Calculate metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        current_acc = 100.0 * correct_predictions / total_samples
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    # Calculate epoch metrics
    epoch_loss = running_loss / total_samples
    epoch_acc = 100.0 * correct_predictions / total_samples

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: Optional[int] = None,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Evaluate the model on validation/test set.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        epoch: Current epoch number (optional, for logging)
        use_amp: Whether to use automatic mixed precision (default: True)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Progress bar
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Evaluation"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Mixed precision inference
            if use_amp and device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Calculate metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            current_acc = 100.0 * correct_predictions / total_samples
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    # Calculate metrics
    avg_loss = running_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples

    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    checkpoint_dir: str = 'checkpoints',
    early_stopping_patience: int = 7,
    use_amp: bool = True,
    scheduler_patience: int = 3,
    class_weights: Optional[torch.Tensor] = None
):
    """
    Complete training pipeline with early stopping and checkpointing.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        use_amp: Use automatic mixed precision (default: True)
        scheduler_patience: Patience for learning rate scheduler
        class_weights: Optional class weights for imbalanced datasets

    Returns:
        dict: Training history with losses and accuracies
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Loss function with optional class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✓ Using weighted CrossEntropyLoss for class imbalance")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # Learning rate scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-7
    )

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == 'cuda'))

    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001, mode='min')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mixed Precision (AMP): {use_amp and device == 'cuda'}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print("=" * 70 + "\n")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            use_amp=use_amp
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                filepath=best_model_path,
                scaler=scaler
            )

        # Early stopping check
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\n✓ Training completed with early stopping at epoch {epoch}")
            break

    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model Saved: {best_model_path}")
    print("=" * 70 + "\n")

    return history


# ============================================================================
# Main execution block
# ============================================================================

if __name__ == "__main__":
    """
    Example training script that can be run directly from the terminal.

    Usage:
        python -m src.rakuten_image.train
    """

    print("=" * 70)
    print("Rakuten Image Classification - Training Pipeline")
    print("=" * 70)

    # ========================================================================
    # CONFIGURATION (Modify these settings as needed)
    # ========================================================================

    # Paths
    CSV_FILE = "data/X_train.csv"  # Path to your CSV file
    IMAGE_DIR = "data/images/image_train"  # Path to image directory
    CHECKPOINT_DIR = "checkpoints/image_models"

    # Model settings
    MODEL_NAME = "resnet50"  # Options: 'resnet50', 'vit', 'fusion'
    NUM_CLASSES = 27  # Number of product categories
    FREEZE_BACKBONE = True  # Freeze backbone initially (transfer learning)

    # Training hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 32  # Adjust based on your 8GB VRAM (32-64 recommended)
    LEARNING_RATE = 1e-4  # AdamW learning rate
    VAL_SPLIT = 0.2  # 80% train, 20% validation
    EARLY_STOPPING_PATIENCE = 7
    IMG_SIZE = 224  # Input image size

    # Hardware settings
    USE_AMP = True  # Automatic Mixed Precision (recommended for 3060 Ti)
    NUM_WORKERS = 4  # DataLoader workers (adjust based on CPU cores)
    RANDOM_SEED = 42

    # ========================================================================
    # SETUP
    # ========================================================================

    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected, using CPU (training will be slow)")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    print("\n" + "-" * 70)
    print("Loading Data...")
    print("-" * 70)

    # Load CSV
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} samples from {CSV_FILE}")

    # Get transforms
    train_transform = get_train_transforms(img_size=IMG_SIZE, augment=True)
    val_transform = get_val_transforms(img_size=IMG_SIZE)

    # Create full dataset
    full_dataset = RakutenImageDataset(
        dataframe=df,
        image_dir=IMAGE_DIR,
        transform=None,  # Will set per split
        verify_images=True
    )

    # Split into train and validation
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Apply transforms to splits
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == 'cuda'),
        drop_last=True  # Drop incomplete batches for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == 'cuda')
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Get class weights for imbalanced datasets (optional)
    class_weights = full_dataset.get_class_weights()
    print(f"✓ Class weights computed (range: {class_weights.min():.3f} - {class_weights.max():.3f})")

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================

    print("\n" + "-" * 70)
    print("Initializing Model...")
    print("-" * 70)

    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        freeze_backbone=FREEZE_BACKBONE,
        dropout_rate=0.3
    )

    # Move model to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_amp=USE_AMP,
        scheduler_patience=3,
        class_weights=class_weights  # Use weighted loss for imbalanced data
    )

    # ========================================================================
    # SAVE TRAINING HISTORY
    # ========================================================================

    history_df = pd.DataFrame(history)
    history_path = os.path.join(CHECKPOINT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to {history_path}")

    print("\n" + "=" * 70)
    print("Training pipeline completed successfully!")
    print("=" * 70)
