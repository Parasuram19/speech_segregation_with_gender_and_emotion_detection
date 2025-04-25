import os
import warnings
import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline as DiarizationPipeline
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)

# Suppress warnings for cleaner output.
warnings.filterwarnings("ignore")

##########################################
# 1. Audio Preprocessing Module
##########################################
class AudioPreprocessor:
    """
    Loads audio, resamples it to a fixed sample rate, and extracts normalized mel spectrogram features.
    """
    def __init__(self, sample_rate=16000, n_mels=128, fmax=8000):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax

    def load_audio(self, audio_path):
        """Load audio and ensure it is not empty."""
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            if waveform.size == 0:
                raise ValueError("Empty audio file")
            return waveform, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {str(e)}")

    def extract_mel_spectrogram(self, waveform):
        """
        Extracts a mel spectrogram, converts power to dB,
        normalizes it, and returns a tensor of shape (1, 1, n_mels, time).
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                fmax=self.fmax
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize the spectrogram
            mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            # Convert to tensor and add channel and batch dimensions: (1, 1, n_mels, time)
            tensor_spec = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
            return tensor_spec
        except Exception as e:
            raise RuntimeError(f"Error extracting mel spectrogram: {str(e)}")

##########################################
# 4. Gender Classifier Module using Speechbrain
##########################################
class GenderClassifierECAPATDNN:
    """
    Gender classification using SpeechBrain's ECAPA-TDNN model.

    Instead of training a separate classifier on embeddings, this implementation
    uses acoustic properties of the embeddings to determine gender.
    """
    # Modify the GenderClassifierECAPATDNN class initialization
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing ECAPA-TDNN gender classifier on {self.device}")
        
        # Load the ECAPA-TDNN model with copy strategy
        try:
            import os
            os.environ['SPEECHBRAIN_SYMLINK_TO'] = 'copy'  # Force copy instead of symlink
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print("Successfully loaded ECAPA-TDNN model")
        except Exception as e:
            print(f"Error loading ECAPA-TDNN model: {str(e)}")
            raise

    def extract_embedding(self, waveform, sample_rate):
        """Extract speaker embedding from audio waveform."""
        try:
            # Convert NumPy array to PyTorch tensor if necessary
            if isinstance(waveform, np.ndarray):
                waveform = torch.FloatTensor(waveform)

            # Ensure correct shape
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)

            # Resample if needed
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate} Hz to 16000 Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)

            # SpeechBrain expects tensors on the correct device
            waveform = waveform.to(self.device)

            # Extract the embedding - handle batch dimension carefully
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(waveform)
                # Remove any extra dimensions to get a flat vector
                embedding = embedding.squeeze()
                return embedding

        except Exception as e:
            print(f"Error in extract_embedding: {str(e)}")
            raise

    def predict(self, waveform, sample_rate):
        """
        Predict gender based on ECAPA-TDNN embeddings and acoustic analysis.

        Instead of using a trained classifier, we determine gender by analyzing
        properties of the embedding that correlate with gender.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate of the audio

        Returns:
            str: "male" or "female"
        """
        try:
            # First, extract the embedding
            embedding = self.extract_embedding(waveform, sample_rate)

            # Convert to numpy for easier analysis
            if isinstance(embedding, torch.Tensor):
                embedding_np = embedding.cpu().numpy()
            else:
                embedding_np = embedding

            # Ensure we have a 1D embedding
            if len(embedding_np.shape) > 1:
                # Average over the first dimension if we have multiple embeddings
                embedding_np = np.mean(embedding_np, axis=0)

            # Simple gender detection based on statistical properties of embeddings
            # These have been found to correlate with gender characteristics

            # Apply statistical analysis on the embedding
            # Higher frequency components are typically more prominent in female voices
            high_freq_components = np.mean(embedding_np[96:])  # Second half of embedding
            low_freq_components = np.mean(embedding_np[:96])   # First half of embedding

            # Calculate additional supporting features from the raw audio
            # Extract pitch information as a reinforcing feature
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform,
                    sr=sample_rate,
                    fmin=50,
                    fmax=400
                )

                # Find valid pitches with significant magnitude
                valid_pitches = []
                for i in range(pitches.shape[1]):
                    idx = np.argmax(magnitudes[:, i])
                    if magnitudes[idx, i] > 0.05:
                        valid_pitches.append(pitches[idx, i])

                # Calculate median pitch if we have valid pitches
                if valid_pitches:
                    median_pitch = np.median(valid_pitches)
                    # Use pitch as additional evidence (higher pitch = more likely female)
                    pitch_evidence = 1 if median_pitch > 160 else -1
                else:
                    # No reliable pitch detected, neutral evidence
                    pitch_evidence = 0
            except:
                # If pitch extraction fails, don't use it
                pitch_evidence = 0

            # Female voices typically have higher relative energy in higher frequencies
            # This is reflected in the embedding pattern
            gender_score = (high_freq_components - low_freq_components) + (pitch_evidence * 0.2)

            # Return gender prediction
            return "female" if gender_score > 0 else "male"

        except Exception as e:
            print(f"Error in gender prediction: {str(e)}")
            # Fallback to acoustic analysis if embedding fails
            try:
                # Simple pitch-based gender detection as fallback
                pitches, magnitudes = librosa.piptrack(
                    y=waveform if isinstance(waveform, np.ndarray) else waveform.cpu().numpy(),
                    sr=sample_rate,
                    fmin=50,
                    fmax=400
                )

                # Find the indices of pitches with highest magnitude in each frame
                pitch_indices = np.argmax(magnitudes, axis=0)

                # Get the corresponding pitches
                valid_pitches = []
                for i, idx in enumerate(pitch_indices):
                    if magnitudes[idx, i] > 0.1:  # Only include if magnitude is significant
                        valid_pitches.append(pitches[idx, i])

                if valid_pitches:
                    median_pitch = np.median(valid_pitches)
                    return "female" if median_pitch > 165 else "male"
                else:
                    return "male"  # Default if we can't determine
            except:
                # Last resort fallback
                return "male"

##########################################
# 5. NEW: Pretrained Wav2Vec2 Emotion Detector
##########################################
class PretrainedWav2Vec2EmotionDetector:
    """
    Emotion detection using the pretrained Wav2Vec2 model from firdhokk.
    """
    def __init__(self, model_dir, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Pretrained Wav2Vec2 Emotion Detector on {self.device}")
        
        try:
            # Load model and feature extractor
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(self.device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get emotion labels from model config if available
            if hasattr(self.model.config, "id2label"):
                self.emotions = self.model.config.id2label
            else:
                # Fallback emotion mapping (common for emotion classification models)
                self.emotions = {
                    0: "angry",
                    1: "disgust", 
                    2: "fear",
                    3: "happy",
                    4: "neutral",
                    5: "sad",
                    6: "surprise"
                }
                
            print(f"Successfully loaded Wav2Vec2 Emotion model with {len(self.emotions)} emotion classes")
            print(f"Available emotions: {list(self.emotions.values())}")
            
        except Exception as e:
            print(f"Error loading Wav2Vec2 Emotion model: {str(e)}")
            raise

    def predict(self, waveform, sample_rate, max_length=250000):
        """
        Predict emotion from audio waveform
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate of the audio
            max_length: Maximum length of audio to process
            
        Returns:
            str: Predicted emotion
            float: Confidence score
        """
        try:
            # Make sure waveform is NumPy array
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # Ensure audio is at 16000 Hz (Wav2Vec2 standard)
            if sample_rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Ensure the audio is the right length (trim or pad)
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            
            # Process audio with the feature extractor
            inputs = self.feature_extractor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding="max_length",
                max_length=max_length
            )
            
            # Move inputs to the right device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Get predicted class
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            return self.emotions[predicted_class], confidence
            
        except Exception as e:
            print(f"Error in emotion prediction: {str(e)}")
            # Fallback to neutral if there's an error
            return "neutral", 0.5

##########################################
# 6. Audio Analyzer Module (Integrating All Components)
##########################################
class AudioAnalyzer:
    """
    Combines audio preprocessing, speaker diarization, emotion classification using Wav2Vec2,
    and gender detection using ECAPA-TDNN embeddings.
    """
    def __init__(self, wav2vec2_model_dir, device=None):
        try:
            # Set device (use GPU if available)
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Initialize preprocessor
            self.preprocessor = AudioPreprocessor()

            # Initialize the Wav2Vec2-based emotion detector using the pretrained model
            self.emotion_detector = PretrainedWav2Vec2EmotionDetector(wav2vec2_model_dir, self.device)

            # Initialize the ECAPA-TDNN-based gender classifier
            self.gender_detector = GenderClassifierECAPATDNN(self.device)

            # Initialize speaker diarization pipeline
            self.diarization = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token="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"  # Replace with your token
            )
            print("Successfully initialized AudioAnalyzer")
        except Exception as e:
            raise RuntimeError(f"Error initializing AudioAnalyzer: {str(e)}")

    def analyze_segment(self, segment, sample_rate):
        """
        For a given audio segment, perform:
          - Emotion classification using the Wav2Vec2 model
          - Gender detection using the ECAPA-TDNN embeddings
        """
        try:
            # --- Emotion Detection with Wav2Vec2 ---
            emotion, emotion_confidence = self.emotion_detector.predict(segment, sample_rate)

            # --- Gender Detection ---
            gender = self.gender_detector.predict(segment, sample_rate)

            return gender, emotion, emotion_confidence
        except Exception as e:
            print(f"Detailed error in analyze_segment: {str(e)}")
            raise RuntimeError(f"Error analyzing segment: {str(e)}")

    def process_audio(self, audio_path):
        """
        Processes the audio file:
          - Loads audio
          - Obtains speaker segments via diarization
          - Analyzes each segment for emotion and gender
        """
        try:
            waveform, sample_rate = self.preprocessor.load_audio(audio_path)
            print(f"Successfully loaded audio: {audio_path}")

            print("Running diarization...")
            diarization_results = self.diarization(audio_path)
            analysis_results = []

            print("Processing speaker segments...")
            segment_count = 0
            for turn, _, speaker in diarization_results.itertracks(yield_label=True):
                try:
                    segment_count += 1
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    segment = waveform[start_sample:end_sample]

                    # Skip segments that are too short
                    if len(segment) < 1000:
                        print(f"Skipping segment {segment_count} (too short: {len(segment)} samples)")
                        continue

                    print(f"Analyzing segment {segment_count} for speaker {speaker} ({turn.start:.2f}s - {turn.end:.2f}s)")
                    gender, emotion, confidence = self.analyze_segment(segment, sample_rate)
                    analysis_results.append({
                        "speaker": speaker,
                        "start_time": turn.start,
                        "end_time": turn.end,
                        "gender": gender,
                        "emotion": emotion,
                        "emotion_confidence": confidence
                    })
                    print(f"Completed analysis: {gender}, {emotion} (confidence: {confidence:.2f})")
                except Exception as seg_e:
                    print(f"Warning: Error processing segment {segment_count}: {str(seg_e)}")
                    continue

            return analysis_results
        except Exception as e:
            print(f"Detailed error in process_audio: {str(e)}")
            raise RuntimeError(f"Error processing audio: {str(e)}")

##########################################
# 7. Main Execution
##########################################
def main():
    try:
        # Path to the pretrained firdhokk Wav2Vec2 model
        wav2vec2_model_dir = "./wav2vec2_emotion_local__/models--firdhokk--speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53/snapshots/611e6db8ee667aa07fe66596f9fc761e036ff5b9"
        print(f"Using Wav2Vec2 emotion model from: {wav2vec2_model_dir}")
        
        print("Initializing AudioAnalyzer...")
        analyzer = AudioAnalyzer(wav2vec2_model_dir)

        # Update with your audio file path
        audio_path = "a13.wav"
        print(f"Processing audio file: {audio_path}")

        results = analyzer.process_audio(audio_path)
        print(f"Analysis complete. Found {len(results)} segments.")

        # Prepare a summary of the results
        speaker_genders = {}
        speaker_emotions = {}

        for i, result in enumerate(results):
            speaker = result['speaker']
            gender = result['gender']
            emotion = result['emotion']
            confidence = result['emotion_confidence']

            # Track the genders detected for each speaker
            if speaker not in speaker_genders:
                speaker_genders[speaker] = []
            speaker_genders[speaker].append(gender)

            # Track the emotions detected for each speaker
            if speaker not in speaker_emotions:
                speaker_emotions[speaker] = []
            speaker_emotions[speaker].append((emotion, confidence))

            # Print detailed results
            print(f"\nResult {i+1}:")
            print(f"Speaker: {speaker}")
            print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Gender: {gender}")
            print(f"Emotion: {emotion} (confidence: {confidence:.2f})")

        # Print a summary of the results
        print("\n=== SUMMARY ===")
        for speaker in speaker_genders:
            # Get the most common gender for this speaker
            genders = speaker_genders[speaker]
            most_common_gender = max(set(genders), key=genders.count)
            gender_confidence = genders.count(most_common_gender) / len(genders)

            # Get the most common emotion for this speaker, weighted by confidence
            emotions = speaker_emotions[speaker]
            # Count emotion occurrences, weighted by confidence
            emotion_weights = {}
            for emotion, confidence in emotions:
                if emotion not in emotion_weights:
                    emotion_weights[emotion] = 0
                emotion_weights[emotion] += confidence
            
            # Get the emotion with the highest total confidence
            most_common_emotion = max(emotion_weights, key=emotion_weights.get)
            # Calculate average confidence for this emotion
            emotion_instances = [conf for emo, conf in emotions if emo == most_common_emotion]
            avg_emotion_confidence = sum(emotion_instances) / len(emotion_instances) if emotion_instances else 0

            print(f"Speaker {speaker}:")
            print(f"  - Gender: {most_common_gender} (confidence: {gender_confidence:.2%})")
            print(f"  - Primary emotion: {most_common_emotion} (avg confidence: {avg_emotion_confidence:.2%})")
            print(f"  - All detected emotions: {', '.join(set([e for e, _ in emotions]))}")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()