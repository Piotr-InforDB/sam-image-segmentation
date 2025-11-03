import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt


DATASET_DIR = "datasets"
TEMP_DIR = "temp_dataset"
IMG_SIZE = 256
SPLIT_RATIO = 0.8
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"


class RoofClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def calculate_metrics(tp, fp, tn, fn):
    """Calculate precision, recall, and F1 score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return precision, recall, f1, accuracy


def prepare_dataset():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    classes = ["roof", "other"]

    for cls in classes:
        src_dir = Path(DATASET_DIR) / cls
        imgs = list(src_dir.glob("*.png"))
        random.shuffle(imgs)
        split_idx = int(len(imgs) * SPLIT_RATIO)
        split_sets = {"train": imgs[:split_idx], "val": imgs[split_idx:]}

        for set_name, subset in split_sets.items():
            dst_dir = Path(TEMP_DIR) / set_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(subset, desc=f"{cls}-{set_name}"):
                try:
                    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    img.save(dst_dir / img_path.name)
                except Exception as e:
                    print(f"Error {img_path}: {e}")


def classify_image(img_path, model_path=MODEL_PATH):
    """Classify a single image using the trained model"""
    model = RoofClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        probs = torch.softmax(pred, dim=1)
        label = pred.argmax(1).item()
        confidence = probs[0][label].item()

    classes = ['other', 'roof']
    return classes[label], confidence


if __name__ == '__main__':
    prepare_dataset()

    # -----------------------
    # TRANSFORMS
    # -----------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(TEMP_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(TEMP_DIR, "val"), transform=val_transform)

    # -----------------------
    # SAMPLER
    # -----------------------
    class_counts = [len([x for x, y in train_dataset.samples if y == i]) for i in range(len(train_dataset.classes))]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[y] for _, y in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Training on: {DEVICE}")

    # -----------------------
    # MODEL SETUP
    # -----------------------
    model = RoofClassifier(num_classes=len(train_dataset.classes)).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------
    # TRAINING SETUP
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=7)

    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    early_stop_patience = 25

    history = {
        'epoch': [],
        # 'train_acc': [],
        # 'val_acc': [],
        # 'val_f1': [],
        # 'val_precision': [],
        # 'val_recall': []
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / total:.3f}'})

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation with detailed metrics
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0

        # Initialize confusion matrix components for each class
        num_classes = len(train_dataset.classes)
        tp = [0] * num_classes
        fp = [0] * num_classes
        tn = [0] * num_classes
        fn = [0] * num_classes

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

                # Calculate confusion matrix components for each class
                for c in range(num_classes):
                    tp[c] += ((preds == c) & (labels == c)).sum().item()
                    fp[c] += ((preds == c) & (labels != c)).sum().item()
                    tn[c] += ((preds != c) & (labels != c)).sum().item()
                    fn[c] += ((preds != c) & (labels == c)).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # Calculate per-class metrics
        print(f"\nEpoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        class_metrics = []
        for c in range(num_classes):
            precision, recall, f1, _ = calculate_metrics(tp[c], fp[c], tn[c], fn[c])
            class_name = train_dataset.classes[c]
            print(f"  {class_name:>5} - Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            class_metrics.append((precision, recall, f1))

        # Calculate macro-averaged F1
        macro_f1 = sum(f1 for _, _, f1 in class_metrics) / num_classes

        # Learning rate scheduling
        scheduler.step(val_acc)

        avg_precision = sum(p for p, _, _ in class_metrics) / num_classes
        avg_recall = sum(r for _, r, _ in class_metrics) / num_classes
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)

        # Save best model (based on F1 score)
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✓ Saved best model with F1: {best_val_f1:.3f}, Acc: {best_val_acc:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best validation F1 score: {best_val_f1:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()