"""
Train CNN Image Classifier for Speaker Identification using Spectrogram Images
Model: EfficientNet-B0 pretrained + fine-tuning
Input: Spectrogram images (224x224 RGB)
Output: Speaker ID (0-40, 41 classes)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ========================= CONFIGURATION =========================
DATA_DIR = r"D:\Master DS\Intro_to_DS\data_spectrograms"
MODEL_DIR = r"D:\Master DS\Intro_to_DS\models"
OUTPUT_DIR = r"D:\Master DS\Intro_to_DS\models"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# Model config
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 41  # 41 speakers
IMAGE_SIZE = 224

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Speaker names (from app.py)
SPEAKER_NAMES = [
    "Bao_Hue", "Chau_Anh", "Cong_Phuong", "Danh_Son", "Duc", "Hai_Phan",
    "Hai_Yen", "Hoang_Hiep", "Hung", "Huong_Ly", "Huyen_Trang", "Jenifer_Smith",
    "Khanh_Huyen", "Le_Ha", "Le_Nghi", "Lien", "Linh_Dao", "Long_Hai",
    "Ly_Minh", "Ly_Van_Sac", "Minh_Tam", "Nam_Minh", "Nghi_Le", "Nguyen_Lan",
    "Nguyen_Luc", "Nguyen_Thuy", "Phan_Van_Tai_Em", "Phong_Nghi", "Phuong_Anh",
    "The_Hao", "The_Son", "Thomas_Williams", "Thu_Huyen", "Tieu_Thu",
    "Tran_Quyen", "Tran_Van_Phuong", "Trung", "Tuan_Anh", "Van_Anh",
    "Van_Son", "Xuan_Son"
]

# ========================= DATASET CLASS =========================
class SpectrogramDataset(Dataset):
    """Custom Dataset for loading spectrogram images"""
    
    def __init__(self, image_dir, split='train', transform=None):
        """
        Args:
            image_dir: Path to spectrogram images directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.image_dir = os.path.join(image_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for speaker_idx, speaker_name in enumerate(SPEAKER_NAMES):
            speaker_dir = os.path.join(self.image_dir, speaker_name)
            if os.path.exists(speaker_dir):
                for img_file in os.listdir(speaker_dir):
                    if img_file.endswith('.png'):
                        self.images.append(os.path.join(speaker_dir, img_file))
                        self.labels.append(speaker_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ========================= MODEL SETUP =========================
def create_model(num_classes, pretrained=True):
    """Create EfficientNet-B0 model for classification"""
    
    # Load pretrained model
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Replace classification head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def get_data_transforms():
    """Image preprocessing and augmentation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


# ========================= TRAINING FUNCTIONS =========================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_classifier(data_dir=DATA_DIR, model_dir=OUTPUT_DIR, num_epochs=NUM_EPOCHS):
    """Main training function"""
    
    print("="*70)
    print("🚀 TRAINING CNN IMAGE CLASSIFIER FOR SPEAKER IDENTIFICATION")
    print("="*70)
    print(f"📊 Dataset: {data_dir}")
    print(f"💾 Model output: {model_dir}")
    print(f"🎮 Device: {DEVICE}")
    print(f"🏗️ Model: {MODEL_NAME} (pretrained={True})")
    print(f"📈 Hyperparameters: batch={BATCH_SIZE}, epochs={num_epochs}, lr={LEARNING_RATE}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # ============ LOAD DATA ============
    print("\n" + "="*70)
    print("STEP 1: Loading data...")
    print("="*70)
    
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = SpectrogramDataset(data_dir, split='train', transform=train_transform)
    val_dataset = SpectrogramDataset(data_dir, split='val', transform=val_transform)
    test_dataset = SpectrogramDataset(data_dir, split='test', transform=val_transform)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ============ CREATE MODEL ============
    print("\n" + "="*70)
    print("STEP 2: Creating model...")
    print("="*70)
    
    model = create_model(NUM_CLASSES, pretrained=True)
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ============ TRAINING SETUP ============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ============ TRAINING LOOP ============
    print("\n" + "="*70)
    print("STEP 3: Training...")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, DEVICE)
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Metrics - Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(model_dir, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best model saved: {model_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    elapsed = time.time() - start_time
    print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")
    
    # ============ EVALUATION ON TEST SET ============
    print("\n" + "="*70)
    print("STEP 4: Evaluating on test set...")
    print("="*70)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{MODEL_NAME}_best.pth")))
    
    test_loss, test_acc, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\n📊 TEST SET RESULTS:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # ============ SAVE RESULTS ============
    results = {
        'model': MODEL_NAME,
        'timestamp': datetime.now().isoformat(),
        'num_classes': NUM_CLASSES,
        'speaker_names': SPEAKER_NAMES,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': epoch + 1,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
        },
        'metrics': {
            'best_val_accuracy': float(best_val_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
        },
        'training_history': history
    }
    
    results_path = os.path.join(model_dir, f"{MODEL_NAME}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # ============ PLOT TRAINING HISTORY ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plot_path = os.path.join(model_dir, f"{MODEL_NAME}_training_history.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"✓ Training plot saved: {plot_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    train_classifier(
        data_dir=DATA_DIR,
        model_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS
    )
