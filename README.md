# Identify Person by Voice

## Overview
This project focuses on researching, experimenting, and training machine learning models to identify and authenticate users based on voice characteristics.

## Objectives
- Research voice identification techniques and algorithms
- Experiment with different feature extraction methods
- Build and train speaker identification models
- Evaluate model performance and accuracy

## Key Topics
- Voice signal processing
- Feature extraction (MFCC, spectrograms, log-Mel)
- Speaker embedding models
- Deep learning architectures for voice recognition

## Dataset
- Hugging Face dataset (100 speakers):
	https://huggingface.co/datasets/thucdangvan020999/speaker_identification_100_speakers_

## Pipeline
```text
[DATA]
	↓
Raw audio
	↓
[PREPROCESS]
(resample, trim, normalize, VAD)
	↓
[FEATURE / EMBEDDING]
WavLM / MFCC / ECAPA
	↓
[REGISTER PHASE]
User registration -> embedding -> vector DB
	↓
[INFERENCE PHASE]
New audio -> embedding -> similarity search
	↓
[OUTPUT]
Speaker ID + confidence score
```

## Project Structure
```
. 
├── data/
├── models/
├── notebooks/
├── src/
└── README.md
```

## Getting Started
1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt` (if provided).
4. Prepare dataset under `data/`.
5. Run preprocessing, training, and evaluation scripts from `src/`.

## References
- Add relevant papers, blog posts, and official documentation here.
