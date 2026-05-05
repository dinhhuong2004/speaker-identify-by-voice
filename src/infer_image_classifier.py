"""
Inference script for Speaker Identification using Image Classifier
Load trained model and predict speaker from audio input
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import json

# ========================= CONFIGURATION =========================
MODEL_DIR = r"D:\Master DS\Intro_to_DS\models"
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Verify model exists
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pth")

# Speaker names
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

NUM_CLASSES = len(SPEAKER_NAMES)

# Audio config
TARGET_SR = 16000
TARGET_DURATION = 4.5
N_MELS = 128
FMAX = 8000

# ========================= HELPER FUNCTIONS =========================
def normalize_audio_length(y, sr, target_duration=4.5):
    """Normalize audio to target duration"""
    target_samples = int(sr * target_duration)
    current_samples = len(y)
    
    if current_samples < target_samples:
        padding = target_samples - current_samples
        y = np.pad(y, (0, padding), mode='constant', constant_values=0)
    elif current_samples > target_samples:
        y = y[:target_samples]
    
    return y


def audio_to_spectrogram_image(audio_path, target_sr=16000, target_duration=4.5, 
                               image_size=(224, 224), n_mels=128, fmax=8000):
    """Convert audio to spectrogram image"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20, ref=np.max)
        
        # Normalize length
        y = normalize_audio_length(y, sr, target_duration=target_duration)
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to 0-255
        S_normalized = ((S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10) * 255).astype(np.uint8)
        
        # Resize
        spec_image = cv2.resize(S_normalized, image_size[::-1], interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        spec_rgb = np.stack([spec_image, spec_image, spec_image], axis=2)
        
        return spec_rgb
    
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


# ========================= MODEL LOADING =========================
def create_model(num_classes, pretrained=True):
    """Create EfficientNet model"""
    model = models.efficientnet_b0(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    return model


class ImageClassifierInference:
    """Inference wrapper for speaker identification"""
    
    def __init__(self, model_path=None, device=DEVICE):
        """
        Initialize inference
        
        Args:
            model_path: Path to trained model weights
            device: torch device (cuda/cpu)
        """
        self.device = device
        self.model = None
        self.transform = None
        
        # Initialize model
        self._setup_model(model_path)
    
    def _setup_model(self, model_path):
        """Load model"""
        if model_path is None:
            model_path = MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nExpected at: {MODEL_PATH}")
        
        # Create model
        self.model = create_model(NUM_CLASSES, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded: {model_path}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_from_audio(self, audio_path, return_probs=False):
        """
        Predict speaker from audio file
        
        Args:
            audio_path: Path to audio file
            return_probs: Return all probabilities
        
        Returns:
            (speaker_id, speaker_name, confidence, probs) or (speaker_id, speaker_name, confidence)
        """
        # Convert audio to spectrogram
        spec_image = audio_to_spectrogram_image(
            audio_path,
            target_sr=TARGET_SR,
            target_duration=TARGET_DURATION,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            n_mels=N_MELS,
            fmax=FMAX
        )
        
        if spec_image is None:
            return None, None, 0.0
        
        # Convert to tensor
        image_pil = Image.fromarray(spec_image)
        image_tensor = self.transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, speaker_id = torch.max(probs, 0)
        
        speaker_id = speaker_id.item()
        confidence = confidence.item()
        speaker_name = SPEAKER_NAMES[speaker_id]
        
        if return_probs:
            return speaker_id, speaker_name, confidence, probs.cpu().numpy()
        else:
            return speaker_id, speaker_name, confidence
    
    def predict_from_spectrogram(self, image_array, return_probs=False):
        """
        Predict speaker from spectrogram image array
        
        Args:
            image_array: Numpy array (H, W, 3) or PIL Image
            return_probs: Return all probabilities
        
        Returns:
            (speaker_id, speaker_name, confidence)
        """
        if isinstance(image_array, np.ndarray):
            image_pil = Image.fromarray(image_array.astype('uint8'))
        else:
            image_pil = image_array
        
        # Transform and predict
        image_tensor = self.transform(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, speaker_id = torch.max(probs, 0)
        
        speaker_id = speaker_id.item()
        confidence = confidence.item()
        speaker_name = SPEAKER_NAMES[speaker_id]
        
        if return_probs:
            return speaker_id, speaker_name, confidence, probs.cpu().numpy()
        else:
            return speaker_id, speaker_name, confidence


# ========================= TESTING =========================
if __name__ == "__main__":
    print("Testing Image Classifier Inference...")
    
    try:
        # Initialize
        classifier = ImageClassifierInference()
        
        # Example usage (if you have test audio)
        # speaker_id, speaker_name, confidence = classifier.predict_from_audio("test_audio.wav")
        # print(f"Predicted: {speaker_name} (ID: {speaker_id}, Confidence: {confidence:.4f})")
        
        print("✅ Inference module ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
