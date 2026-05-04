"""
Convert audio files to Mel-Spectrogram images
Organize by speaker with train/val/test split
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# ========================= CONFIGURATION =========================
RAW_DATA_DIR = r"D:\Master DS\Intro_to_DS\data_real"
OUTPUT_DIR = r"D:\Master DS\Intro_to_DS\data_spectrograms"
IMAGE_SIZE = (224, 224)  # (width, height)
N_MELS = 128
FMAX = 8000

# Audio normalization
TARGET_SR = 16000  # Normalize all audio to 16kHz (standard for speech processing)
TARGET_DURATION = 4.5  # Normalize all audio to 4.5 seconds

# Train/Val/Test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ========================= HELPER FUNCTIONS =========================
def normalize_audio_length(y, sr, target_duration=4.5):
    """
    Normalize audio to target duration
    - If shorter: pad with silence at the end
    - If longer: trim from the end
    """
    target_samples = int(sr * target_duration)
    current_samples = len(y)
    
    if current_samples < target_samples:
        # Pad with zeros (silence)
        padding = target_samples - current_samples
        y = np.pad(y, (0, padding), mode='constant', constant_values=0)
    elif current_samples > target_samples:
        # Trim from end
        y = y[:target_samples]
    
    return y


def get_speaker_folders(data_dir):
    """Get all speaker folders"""
    return [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]


def get_all_audio_files(data_dir):
    """Get all audio files organized by speaker"""
    speakers_data = {}
    
    for speaker in get_speaker_folders(data_dir):
        speaker_path = os.path.join(data_dir, speaker)
        audio_files = []
        
        for root, _, files in os.walk(speaker_path):
            for file in files:
                if file.lower().endswith((".wav", ".mp3")):
                    audio_files.append(os.path.join(root, file))
        
        if audio_files:
            speakers_data[speaker] = audio_files
    
    return speakers_data


def create_output_structure(output_dir, speakers):
    """Create folder structure for train/val/test"""
    folders = ['train', 'val', 'test']
    
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        for speaker in speakers:
            speaker_folder = os.path.join(folder_path, speaker)
            os.makedirs(speaker_folder, exist_ok=True)


def audio_to_spectrogram_image(audio_path, target_sr=16000, target_duration=4.5, image_size=(224, 224), n_mels=128, fmax=8000):
    """
    Convert audio file to mel-spectrogram image
    1. Load audio with normalized sample rate (16000 Hz)
    2. Normalize audio to target duration
    3. Compute mel-spectrogram
    4. Return: numpy array (224, 224, 3) or None if error
    """
    try:
        from PIL import Image
        import io
        
        # Load audio with normalized sample rate
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Normalize audio length
        y = normalize_audio_length(y, sr, target_duration=target_duration)
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create image without displaying
        fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
        img = librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Convert to numpy array using savefig to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        image = image.resize(image_size, Image.LANCZOS)
        
        plt.close(fig)
        
        return np.array(image)
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def convert_and_save_spectrograms(raw_data_dir, output_dir, target_sr=16000, target_duration=4.5, image_size=(224, 224)):
    """Convert all audio files to spectrogram images and split into train/val/test"""
    
    print("="*60)
    print("STEP 0: Audio Normalization Settings")
    print("="*60)
    print(f"Target Sample Rate: {target_sr} Hz")
    print(f"Target Duration: {target_duration}s")
    print("Audio Processing:")
    print("- Resample all audio to 16000 Hz (standard for speech)")
    print("- Audio shorter than target: Pad with silence")
    print("- Audio longer than target: Trim from end")
    
    print("\n" + "="*60)
    print("STEP 1: Getting all audio files...")
    print("="*60)
    
    speakers_data = get_all_audio_files(raw_data_dir)
    speakers = sorted(list(speakers_data.keys()))
    
    print(f"Found {len(speakers)} speakers:")
    for speaker in speakers:
        print(f"  - {speaker}: {len(speakers_data[speaker])} files")
    
    print("\n" + "="*60)
    print("STEP 2: Creating output folder structure...")
    print("="*60)
    
    create_output_structure(output_dir, speakers)
    print("Folder structure created")
    
    print("\n" + "="*60)
    print("STEP 3: Converting audio to spectrograms & splitting...")
    print("="*60)
    
    total_converted = 0
    
    for speaker in speakers:
        print(f"\nProcessing speaker: {speaker}")
        audio_files = speakers_data[speaker]
        
        # Split into train/val/test
        train_files, temp_files = train_test_split(
            audio_files, test_size=(VAL_RATIO + TEST_RATIO), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42
        )
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_output_dir = os.path.join(output_dir, split_name, speaker)
            
            for idx, audio_file in enumerate(tqdm(files, desc=f"  {split_name}")):
                # Convert to spectrogram (with sample rate and duration normalization)
                spec_image = audio_to_spectrogram_image(audio_file, target_sr=target_sr, target_duration=target_duration, image_size=image_size)
                
                if spec_image is not None:
                    # Save image
                    filename = f"{speaker}_{split_name}_{idx:04d}.png"
                    output_path = os.path.join(split_output_dir, filename)
                    
                    from PIL import Image
                    Image.fromarray(spec_image).save(output_path)
                    total_converted += 1
    
    print("\n" + "="*60)
    print(f"COMPLETED! Total spectrograms created: {total_converted}")
    print("="*60)
    
    # Print statistics
    print("\nFinal folder structure:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        total_images = sum(len(f) for _, _, files in os.walk(split_path) for f in files if f.endswith('.png'))
        print(f"  {split}: {total_images} images")


if __name__ == "__main__":
    print("Starting audio to spectrogram conversion with audio normalization...")
    print("Audio Normalization: Sample Rate (16kHz) + Duration (4.5s)\n")
    convert_and_save_spectrograms(RAW_DATA_DIR, OUTPUT_DIR, target_sr=TARGET_SR, target_duration=TARGET_DURATION, image_size=IMAGE_SIZE)
