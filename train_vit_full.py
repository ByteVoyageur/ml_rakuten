# ==============================================================================
# 🚀 PRODUCTION ViT TRAINING - Google/vit-base-patch16-224
# ==============================================================================
# Script complet pour entraîner le modèle ViT sur l'ensemble du dataset Rakuten
# Avec validation, sauvegarde de checkpoints, et early stopping
# ==============================================================================

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import json
from datetime import datetime

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CONFIG = {
    # Paths
    "data_dir": Path("/workspace/data"),
    "img_dir": Path("/workspace/data/images/image_train"),
    "checkpoint_dir": Path("/workspace/checkpoints/vit_production"),

    # Model
    "model_name": "google/vit-base-patch16-224",
    "img_size": 224,

    # Training
    "batch_size": 16,  # Augmenté pour plus de vitesse (testé: max 96 possible)
    "num_epochs": 20,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,

    # Data split
    "train_ratio": 0.8,
    "val_ratio": 0.2,

    # Early stopping
    "early_stopping_patience": 5,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
}

print("=" * 80)
print("🚀 PRODUCTION ViT TRAINING")
print("=" * 80)
print(f"Device: {CONFIG['device']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------------
try:
    from transformers import ViTForImageClassification, get_scheduler
    from torch.optim import AdamW
    print("✓ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Add src to path
project_root = Path("/workspace")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.rakuten_image.datasets import RakutenImageDataset
    print("✓ RakutenImageDataset imported successfully")
except ImportError as e:
    print(f"❌ Cannot import RakutenImageDataset: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------
print("\n📂 Loading data...")
X_train = pd.read_csv(CONFIG["data_dir"] / "X_train_update.csv", index_col=0)
Y_train = pd.read_csv(CONFIG["data_dir"] / "Y_train_CVw08PX.csv", index_col=0)
df = X_train.join(Y_train, how="inner")
print(f"✓ Loaded {len(df)} samples")

# ------------------------------------------------------------------
# 3. Create Datasets
# ------------------------------------------------------------------
print("\n🔧 Creating datasets...")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create full dataset
full_dataset = RakutenImageDataset(
    dataframe=df,
    image_dir=CONFIG["img_dir"],
    transform=train_transform,
    label_col="prdtypecode",
    verify_images=False
)

print(f"✓ Full dataset: {len(full_dataset)} samples, {full_dataset.num_classes} classes")

# Split train/val
train_size = int(CONFIG["train_ratio"] * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset_temp = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create validation dataset with val_transform
val_indices = val_dataset_temp.indices
val_df = df.iloc[val_indices].reset_index(drop=True)
val_dataset = RakutenImageDataset(
    dataframe=val_df,
    image_dir=CONFIG["img_dir"],
    transform=val_transform,
    label_col="prdtypecode",
    verify_images=False
)

print(f"✓ Train set: {len(train_dataset)} samples")
print(f"✓ Val set: {len(val_dataset)} samples")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")

# ------------------------------------------------------------------
# 4. Create Model
# ------------------------------------------------------------------
print("\n🏗️ Creating model...")
model = ViTForImageClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=full_dataset.num_classes,
    ignore_mismatched_sizes=True
)
model.to(CONFIG["device"])
print(f"✓ Model loaded: {full_dataset.num_classes} classes")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ------------------------------------------------------------------
# 5. Setup Training
# ------------------------------------------------------------------
print("\n⚙️ Setting up training...")

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

# Scheduler
total_steps = len(train_loader) * CONFIG["num_epochs"]
num_warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
)

print(f"✓ Optimizer: AdamW (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})")
print(f"✓ Scheduler: Linear warmup ({num_warmup_steps}/{total_steps} steps)")

# Create checkpoint directory
CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
print(f"✓ Checkpoint dir: {CONFIG['checkpoint_dir']}")

# Save config
with open(CONFIG["checkpoint_dir"] / "config.json", "w") as f:
    json.dump({k: str(v) for k, v in CONFIG.items()}, f, indent=2)

# ------------------------------------------------------------------
# 6. Training Loop
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

best_val_acc = 0.0
best_val_loss = float('inf')
patience_counter = 0
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(CONFIG["num_epochs"]):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
    print(f"{'='*80}")

    # -------------------- Training --------------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_pbar = tqdm(train_loader, desc=f"Training", unit="batch")
    for images, labels in train_pbar:
        images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])

        # Forward pass
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Metrics
        train_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        # Update progress bar
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * train_correct / train_total:.2f}%'
        })

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100.0 * train_correct / train_total

    # -------------------- Validation --------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_pbar = tqdm(val_loader, desc=f"Validation", unit="batch")
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * val_correct / val_total:.2f}%'
            })

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total

    # -------------------- Log Epoch Results --------------------
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_accuracy)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_accuracy)

    print(f"\n📊 Epoch {epoch + 1} Results:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")

    # -------------------- Save Best Model --------------------
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_loss = avg_val_loss
        patience_counter = 0

        checkpoint_path = CONFIG["checkpoint_dir"] / "best_model.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_accuracy,
            'val_loss': avg_val_loss,
            'train_acc': train_accuracy,
            'train_loss': avg_train_loss,
        }, checkpoint_path)

        print(f"  ✅ Best model saved! (Val Acc: {val_accuracy:.2f}%)")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

    # -------------------- Early Stopping --------------------
    if patience_counter >= CONFIG["early_stopping_patience"]:
        print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
        break

# ------------------------------------------------------------------
# 7. Final Evaluation
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 FINAL EVALUATION")
print("=" * 80)

# Load best model
checkpoint = torch.load(CONFIG["checkpoint_dir"] / "best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✅ Best Model Stats:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val Accuracy: {checkpoint['val_acc']:.2f}%")
print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

# Full validation evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Final evaluation"):
        images = images.to(CONFIG["device"])
        outputs = model(pixel_values=images)
        predictions = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print("\n📋 Classification Report:")
print("=" * 80)
print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

# Save history
history_path = CONFIG["checkpoint_dir"] / "training_history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

print(f"\n✅ Training history saved to: {history_path}")
print(f"✅ Best model saved to: {CONFIG['checkpoint_dir'] / 'best_model.pth'}")
print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETED!")
print("=" * 80)
