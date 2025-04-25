import torch
import librosa
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)

# 1. Point to your local model directory (the one with config.json and model.safetensors)
local_model_dir = (
    "./wav2vec2_emotion_local__/"
    "models--firdhokk--speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53/"
    "snapshots/611e6db8ee667aa07fe66596f9fc761e036ff5b9"
)

# 2. Load the model and feature extractor from that folder
model = Wav2Vec2ForSequenceClassification.from_pretrained(local_model_dir)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_dir)

# 3. Load an audio file for inference (16kHz is typical for Wav2Vec2 models)
audio_file = "a13.wav"
audio, sr = librosa.load(audio_file, sr=16000)

# 4. Preprocess audio
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

# 5. Run inference
with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = torch.argmax(logits, dim=-1).item()

# 6. (Optional) Retrieve label names
# Many models store a label mapping in the config as `model.config.id2label` if available.
if hasattr(model.config, "id2label"):
    predicted_label = model.config.id2label[predicted_id]
    print(f"Predicted emotion: {predicted_label}")
else:
    print(f"Predicted label index: {predicted_id}")
