
"""
Speaker Identification Streamlit Application
Ứng dụng Streamlit để nhận dạng người nói bằng cách so sánh embedding âm thanh với FAISS vector database.
Sử dụng pre-trained speech models (WavLM, Wav2Vec2, HuBERT, UniSpeech-SAT) để trích xuất embeddings
và FAISS để tìm người nói tương tự nhất trong cơ sở dữ liệu huấn luyện.
"""

import streamlit as st
import os
import numpy as np
from pathlib import Path
import librosa
import faiss
import torch
from transformers import AutoFeatureExtractor, AutoModel

st.set_page_config(page_title="Speaker Identification Monitor", layout="wide")

st.title("🎙️ Speaker Identification Monitor & Test")

# Get data directory relative to app location
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path("D:\\Master DS\\Intro_to_DS\\src\\data_test").resolve()
DATA_PROCESSED_DIR = (APP_DIR.parent / "data_processed").resolve()  # Absolute path

# Debug: Show paths
print(f"🔍 APP_DIR: {APP_DIR}")
print(f"🔍 DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}")
print(f"🔍 DATA_PROCESSED_DIR exists: {DATA_PROCESSED_DIR.exists()}")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

audio_path = st.sidebar.text_input(
    "Audio Folder Path:", 
    value=str(DATA_DIR),
    help="Folder containing test audio files (.wav, .mp3, .ogg, .flac)"
)

models_available = ["wavlm", "wav2vec2", "hubert", "unispeech"]
selected_model = st.sidebar.selectbox("Select Model:", models_available, index=0)

# Model mappings
MODEL_NAMES = {
    "wavlm": "microsoft/wavlm-base",
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "unispeech": "microsoft/unispeech-sat-base"
}

# Speaker ID to Name mapping
SPEAKER_NAMES = [
    "Bao_Hue", "Chau_Anh", "Cong_Phuong", "Danh_Son", "Duc",
    "Hai_Phan", "Hai_Yen", "Hoang_Hiep", "Hung", "Huong_Ly",
    "Huyen_Trang", "Jenifer_Smith", "Khanh_Huyen", "Le_Ha", "Le_Nghi",
    "Lien", "Linh_Dao", "Long_Hai", "Ly_Minh", "Ly_Van_Sac",
    "Minh_Tam", "Nam_Minh", "Nghi_Le", "Nguyen_Lan", "Nguyen_Luc",
    "Nguyen_Thuy", "Phan_Van_Tai_Em", "Phong_Nghi", "Phuong_Anh", "The_Hao",
    "The_Son", "Thomas_Williams", "Thu_Huyen", "Tieu_Thu", "Tran_Quyen",
    "Tran_Van_Phuong", "Trung", "Tuan_Anh", "Van_Anh", "Van_Son",
    "Xuan_Son"
]

# Load FAISS database and models
@st.cache_resource
def load_faiss_and_model(model_key):
    """Load FAISS index, labels, and speech model"""
    result = {
        "index": None, 
        "labels": None,
        "model": None,
        "feature_extractor": None,
        "error": None
    }
    
    try:
        # Resolve paths inside function to avoid cache issues
        data_dir = (Path(__file__).parent.parent / "data_processed").resolve()
        
        # Load FAISS index
        faiss_file = data_dir / f"faiss_{model_key}.index"
        labels_file = data_dir / f"faiss_labels_{model_key}.npy"
        
        print(f"\n🔍 Loading {model_key}:")
        print(f"   Trying FAISS: {faiss_file}")
        print(f"   File exists: {faiss_file.exists()}")
        
        if not faiss_file.exists():
            result["error"] = f"FAISS index not found: {faiss_file}"
            return result
        
        if not labels_file.exists():
            result["error"] = f"Labels file not found: {labels_file}"
            return result
        
        print(f"   ✅ Loading index from: {faiss_file}")
        result["index"] = faiss.read_index(str(faiss_file))
        result["labels"] = np.load(str(labels_file))
        
        # Load model and feature extractor
        model_name = MODEL_NAMES[model_key]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with st.spinner(f"Loading {model_key} model..."):
            result["feature_extractor"] = AutoFeatureExtractor.from_pretrained(model_name)
            result["model"] = AutoModel.from_pretrained(model_name).to(device)
            result["model"].eval()
        
    except Exception as e:
        result["error"] = f"Error loading model/index: {str(e)}"
    
    return result

# Get audio files from folder
def get_audio_files(folder_path):
    """Get list of audio files from folder"""
    if not os.path.exists(folder_path):
        return []
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    try:
        files = [f for f in os.listdir(folder_path) 
                 if Path(f).suffix.lower() in audio_extensions]
        return sorted(files)
    except Exception as e:
        st.error(f"Error reading audio folder: {e}")
        return []

# Extract audio embedding using speech model
def extract_embedding(audio_file_path, model_data):
    """Extract embedding from audio using pre-trained speech model"""
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        # Extract features
        inputs = model_data["feature_extractor"](
            audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        device = next(model_data["model"].parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model_data["model"](**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
            
        # L2 normalize
        embedding = embedding / (torch.norm(embedding, p=2, dim=1, keepdim=True) + 1e-7)
        
        return embedding.cpu().numpy()[0]
    
    except Exception as e:
        st.error(f"Error extracting embedding: {str(e)}")
        return None

# Compare with database using FAISS
def classify_audio(embedding, model_data):
    """Search FAISS index for similar speakers"""
    if model_data["index"] is None or model_data["labels"] is None:
        return None, None, 0.0
    
    try:
        # Reshape for FAISS (1, 768)
        query_embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index (k=1 for top match)
        scores, indices = model_data["index"].search(query_embedding, k=1)
        
        best_idx = indices[0][0]
        confidence = scores[0][0]  # Inner product score
        speaker_id = int(model_data["labels"][best_idx])
        
        # Get speaker name from mapping
        speaker_name = SPEAKER_NAMES[speaker_id] if speaker_id < len(SPEAKER_NAMES) else f"Unknown_{speaker_id}"
        print(f"🎤 Found: ID={speaker_id}, Speaker={speaker_name}, Confidence={confidence:.4f}")
        
        return speaker_id, speaker_name, float(confidence)
    
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None, None, 0.0

# Main interface
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.subheader("📂 Select Audio File")
    audio_files = get_audio_files(audio_path)
    
    if audio_files:
        selected_file = st.selectbox("Choose audio file:", audio_files)
        audio_file_path = os.path.join(audio_path, selected_file)
        
        if os.path.exists(audio_file_path):
            st.audio(audio_file_path, format="audio/wav")
        else:
            st.warning(f"File not found: {audio_file_path}")
            selected_file = None
    else:
        st.warning(f"❌ No audio files found in {audio_path}")
        selected_file = None

with col2:
    st.subheader("🔍 Classification Results")
    
    # Load model
    model_data = load_faiss_and_model(selected_model)
    
    if model_data["error"]:
        st.error(f"❌ {model_data['error']}")
    elif selected_file and model_data["model"] is not None:
        if st.button("🎯 Identify Speaker", key="classify_btn"):
            with st.spinner("Processing audio..."):
                embedding = extract_embedding(audio_file_path, model_data)
                
                if embedding is not None:
                    speaker_id, speaker_name, confidence = classify_audio(embedding, model_data)
                    
                    if speaker_id is not None:
                        # Display results
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("🔢 ID", f"{speaker_id}")
                        with col_res2:
                            st.metric("🏷️ Label", f"{speaker_name}")
                        with col_res3:
                            st.metric("✅ Confidence", f"{confidence:.4f}")
                        
                        # Show summary
                        st.divider()
                        summary = f"""
### 📋 **SUMMARY**

**Model Used:** {selected_model.upper()}  
**Output:** ID: {speaker_id} | Label: **{speaker_name}**  
**Accuracy (Confidence Score):** {confidence:.4f}
"""
                        st.markdown(summary)
                    else:
                        st.error("❌ Failed to classify audio")
    else:
        if not selected_file:
            st.info("👆 Select an audio file first")
        else:
            st.info("⏳ Model loading...")

with col3:
    # Speaker ID Mapping - displayed on the right side
    st.subheader("📊 Speaker ID Mapping")
    
    # Create 2 columns for the mapping tables
    map_col1, map_col2 = st.columns(2)
    
    # First half of speakers
    mid_point = len(SPEAKER_NAMES) // 2
    
    with map_col1:
        st.write("**List 1**")
        table_data_left = ""
        table_data_left += "| ID | Name |\n|:---:|:---:|\n"
        for i in range(mid_point):
            table_data_left += f"| {i} | {SPEAKER_NAMES[i]} |\n"
        st.markdown(table_data_left)
    
    with map_col2:
        st.write("**List 2**")
        table_data_right = ""
        table_data_right += "| ID | Name |\n|:---:|:---:|\n"
        for i in range(mid_point, len(SPEAKER_NAMES)):
            table_data_right += f"| {i} | {SPEAKER_NAMES[i]} |\n"
        st.markdown(table_data_right)

# Sidebar: Database info
st.sidebar.divider()
st.sidebar.subheader("📊 Database Info")

if (model_data := load_faiss_and_model(selected_model)):
    if model_data["labels"] is not None:
        st.sidebar.metric("📈 Total Training Samples", len(model_data["labels"]))
        st.sidebar.metric("🗣️ Unique Speakers", len(SPEAKER_NAMES))
        st.sidebar.metric("🧠 Embedding Dimension", 768)
        st.sidebar.info(f"**Model**: {MODEL_NAMES[selected_model]}")
        st.sidebar.info(f"**Index Type**: IndexFlatIP (Inner Product)")

# Footer
st.sidebar.divider()
st.sidebar.caption("🎙️ Speaker Identification System | Powered by WavLM, FAISS & Streamlit")
