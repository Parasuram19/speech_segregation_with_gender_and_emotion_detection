import os
import warnings
import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline as DiarizationPipeline
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

# Install pyctcdecode if not already installed
# try:
#     import pyctcdecode
#     st.success("pyctcdecode library is already installed")
# except ImportError:
#     st.warning("Installing required dependency: pyctcdecode")
#     os.system("pip install pyctcdecode")
#     st.success("Successfully installed pyctcdecode")

# Now import the transformers libraries
from transformers import AutoProcessor, AutoModelForAudioClassification

# Suppress warnings for cleaner output
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
    Gender classification using JaesungHuh's voice-gender-classifier based on ECAPA-TDNN.
    The model is defined in a separate file and imported.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Initializing voice-gender-classifier on {self.device}")
        
        try:
            # Import the model class from separate file
            from model import ECAPA_gender
            
            # Load the model from the huggingface model hub
            self.model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
            self.model.eval()  # Set model to evaluation mode
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            st.success(f"Successfully loaded voice-gender-classifier model")
            
        except Exception as e:
            st.error(f"Error loading voice-gender-classifier model: {str(e)}")
            raise

    def predict(self, waveform, sample_rate):
        """
        Predict gender based on audio waveform using the voice-gender-classifier.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate of the audio

        Returns:
            str: "male" or "female"
        """
        try:
            # Save the waveform to a temporary file to use with the model's predict function
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
                
                # Convert waveform to appropriate format for saving
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                
                # Ensure audio is at 16000 Hz (model requirement)
                if sample_rate != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # Save waveform to temporary file
                sf.write(temp_file_path, waveform, sample_rate)
            
            # Use the model's predict function directly with the file path
            with torch.no_grad():
                output = self.model.predict(temp_file_path, device=self.device)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Convert output to lowercase string if it's not already
            if not isinstance(output, str):
                output = str(output).lower()
            
            return output.lower()  # Return the gender prediction
            
        except Exception as e:
            st.error(f"Error in gender prediction: {str(e)}")
            # Fallback to acoustic analysis if model prediction fails
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
    
    def extract_embedding(self, waveform, sample_rate):
        """
        For compatibility with the previous implementation.
        This method is kept for backward compatibility.
        """
        st.warning("The extract_embedding method is deprecated with the new gender classifier model.")
        return None
##########################################
# 5. Pretrained Wav2Vec2 Emotion Detector
##########################################

from transformers import pipeline
import torch
import numpy as np
import librosa
import streamlit as st

class HatmanEmotionDetector:
    """
    Emotion detection using pipeline approach for audio emotion detection.
    """
    def __init__(self, device=None):
        # Properly handle device parameter
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        st.info(f"Initializing Audio Emotion Detector on {self.device}")
        
        try:
            # Initialize the pipeline with proper device handling
            device_id = 0 if str(self.device) == "cuda" else -1  # 0 for GPU, -1 for CPU
            
            self.pipeline = pipeline(
                "audio-classification", 
                model="Hatman/audio-emotion-detection",
                device=device_id
            )
            
            # Get emotion labels from the pipeline's model config
            self.emotions = self.pipeline.model.config.id2label
                
            st.success(f"Successfully loaded Emotion pipeline with {len(self.emotions)} emotion classes")
            st.info(f"Available emotions: {list(self.emotions.values())}")
            
        except Exception as e:
            st.error(f"Error loading Emotion pipeline: {str(e)}")
            # Fallback to a simpler emotion classification
            st.warning("Falling back to a simpler emotion classifier")
            self.pipeline = None
            # Default emotion mapping for fallback
            self.emotions = {
                0: "angry", 1: "happy", 2: "sad", 3: "neutral",
                4: "fear", 5: "disgust", 6: "surprise"
            }

    def predict(self, waveform, sample_rate):
        """
        Predict emotion from audio waveform
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            str: Predicted emotion
            float: Confidence score
        """
        # If pipeline failed to load, use fallback
        if self.pipeline is None:
            return self._fallback_predict(waveform, sample_rate)
            
        try:
            # Make sure waveform is NumPy array
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # Ensure audio is resampled to the model's expected sample rate (16000 Hz for most audio models)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
            
            # Run emotion prediction
            with torch.no_grad():
                # Pipeline API handles all preprocessing internally
                results = self.pipeline({"sampling_rate": sample_rate, "raw": waveform})
            
            # Get the top predicted emotion and score
            top_result = results[0]  # Pipeline returns a list of results sorted by score
            emotion = top_result["label"]
            confidence = top_result["score"]
            
            return emotion, confidence
            
        except Exception as e:
            st.error(f"Error in emotion prediction: {str(e)}")
            # Fallback to acoustic analysis if model prediction fails
            return self._fallback_predict(waveform, sample_rate)
    
    def _fallback_predict(self, waveform, sample_rate):
        """Simple fallback emotion prediction based on acoustic features"""
        try:
            # Make sure waveform is NumPy array
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
                
            # Extract features
            # 1. Energy (volume)
            energy = np.mean(librosa.feature.rms(y=waveform))
            
            # 2. Pitch
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
            # Find the indices of pitches with highest magnitude in each frame
            pitch_indices = np.argmax(magnitudes, axis=0)
            # Get the corresponding pitches
            valid_pitches = []
            for i, idx in enumerate(pitch_indices):
                if magnitudes[idx, i] > 0.1:  # Only include if magnitude is significant
                    valid_pitches.append(pitches[idx, i])
            pitch = np.median(valid_pitches) if valid_pitches else 0
            
            # 3. Tempo
            tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)[0]
            
            # 4. Spectral contrast
            contrast = np.mean(librosa.feature.spectral_contrast(y=waveform, sr=sample_rate))
            
            # Simple rule-based classification
            if energy > 0.1 and pitch > 180 and tempo > 120:
                return "happy", 0.7
            elif energy > 0.1 and pitch < 150:
                return "angry", 0.6
            elif energy < 0.05 and pitch < 180:
                return "sad", 0.6
            elif energy < 0.08 and pitch > 180:
                return "fear", 0.5
            elif contrast > 10:
                return "surprise", 0.5
            elif energy > 0.08 and pitch > 160:
                return "disgust", 0.4
            else:
                return "neutral", 0.8
                
        except Exception as e:
            st.error(f"Error in fallback emotion prediction: {str(e)}")
            return "neutral", 0.5
        
        ##########################################
# 6. Audio Analyzer Module (Integrating All Components)
##########################################

class AudioAnalyzer:
    """
    Combines audio preprocessing, speaker diarization, emotion classification,
    and gender detection.
    """
    def __init__(self, use_auth_token, device=None):
        try:
            # Set device (use GPU if available)
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
                
            st.info(f"Using device: {self.device}")

            # Initialize preprocessor
            self.preprocessor = AudioPreprocessor()

            # Initialize the emotion detector with proper device
            self.emotion_detector = HatmanEmotionDetector(self.device)

            # Initialize the gender classifier with proper device
            self.gender_detector = GenderClassifierECAPATDNN(self.device)

            # Initialize speaker diarization pipeline with token
            self.diarization = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=use_auth_token
            )
            st.success("Successfully initialized AudioAnalyzer")
        except Exception as e:
            st.error(f"Error initializing AudioAnalyzer: {str(e)}")
            raise RuntimeError(f"Error initializing AudioAnalyzer: {str(e)}")
    # The rest of the class remains the same
    def analyze_segment(self, segment, sample_rate):
        """
        For a given audio segment, perform:
          - Emotion classification using the Hatman model
          - Gender detection using the ECAPA-TDNN embeddings
        """
        try:
            # --- Emotion Detection with Hatman model ---
            emotion, emotion_confidence = self.emotion_detector.predict(segment, sample_rate)

            # --- Gender Detection ---
            gender = self.gender_detector.predict(segment, sample_rate)

            return gender, emotion, emotion_confidence
        except Exception as e:
            st.error(f"Detailed error in analyze_segment: {str(e)}")
            raise RuntimeError(f"Error analyzing segment: {str(e)}")

    # The process_audio method stays the sames
    # The process_audio method stays the same
    # The process_audio method stays the same
    def process_audio(self, audio_path, progress_bar=None):
        """
        Processes the audio file:
          - Loads audio
          - Obtains speaker segments via diarization
          - Analyzes each segment for emotion and gender
        """
        try:
            waveform, sample_rate = self.preprocessor.load_audio(audio_path)
            st.success(f"Successfully loaded audio")

            with st.spinner("Running speaker diarization..."):
                diarization_results = self.diarization(audio_path)
            
            analysis_results = []
            
            # Count total segments for progress tracking
            total_segments = sum(1 for _ in diarization_results.itertracks(yield_label=True))
            
            if progress_bar is None:
                progress_bar = st.progress(0)
                
            segment_count = 0
            
            st.info(f"Found {total_segments} speaker segments to analyze")
            status_text = st.empty()
            
            for turn, _, speaker in diarization_results.itertracks(yield_label=True):
                try:
                    segment_count += 1
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    segment = waveform[start_sample:end_sample]

                    # Skip segments that are too short
                    if len(segment) < 1000:
                        status_text.text(f"Skipping segment {segment_count}/{total_segments} (too short)")
                        continue

                    status_text.text(f"Analyzing segment {segment_count}/{total_segments} for speaker {speaker} ({turn.start:.2f}s - {turn.end:.2f}s)")
                    gender, emotion, confidence = self.analyze_segment(segment, sample_rate)
                    analysis_results.append({
                        "speaker": speaker,
                        "start_time": turn.start,
                        "end_time": turn.end,
                        "gender": gender,
                        "emotion": emotion,
                        "emotion_confidence": confidence
                    })
                    
                    # Update progress bar
                    progress_bar.progress(segment_count / total_segments)
                    
                except Exception as seg_e:
                    st.warning(f"Error processing segment {segment_count}: {str(seg_e)}")
                    continue

            status_text.text("Analysis complete!")
            progress_bar.progress(1.0)
            
            return analysis_results
        except Exception as e:
            st.error(f"Detailed error in process_audio: {str(e)}")
            raise RuntimeError(f"Error processing audio: {str(e)}")

##########################################
# Streamlit UI Functions
##########################################

def plot_timeline(results):
    """Create a timeline plot of speaker segments with emotion and gender info"""
    if not results:
        st.warning("No results to display")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    speakers = sorted(set(r['speaker'] for r in results))
    speaker_map = {speaker: i for i, speaker in enumerate(speakers)}
    
    # Define emotion colors - ensure all lowercase keys to match model output
    colors = {
        "angry": "red",
        "anger": "red",
        "disgust": "brown", 
        "fear": "purple",
        "happy": "green",
        "happiness": "green",
        "neutral": "gray",
        "sad": "blue",
        "sadness": "blue",
        "surprise": "orange"
    }
    
    gender_marker = {"male": "o", "female": "^"}
    
    # Keep track of emotions we encounter for the legend
    encountered_emotions = set()
    
    for result in results:
        speaker_idx = speaker_map[result['speaker']]
        start = result['start_time']
        end = result['end_time']
        duration = end - start
        
        # Normalize emotion string to lowercase for consistent matching
        emotion = result['emotion'].lower()
        encountered_emotions.add(emotion)
        
        gender = result['gender']
        confidence = result['emotion_confidence']
        
        # Get color for this emotion, with fallback to gray if not found
        color = colors.get(emotion, "gray")
        
        ax.barh(
            y=speaker_idx, 
            width=duration, 
            left=start, 
            color=color,
            alpha=confidence,
            edgecolor="black",
            linewidth=1
        )
        
        # Add gender marker at the middle of the segment
        mid_point = start + duration/2
        ax.scatter(mid_point, speaker_idx, marker=gender_marker.get(gender, "s"), color="black", s=50)
    
    # Create legend for emotions that were actually encountered
    encountered_colors = {emotion: colors.get(emotion, "gray") for emotion in encountered_emotions}
    emotion_patches = [plt.Rectangle((0,0), 1, 1, color=color) for emotion, color in encountered_colors.items()]
    emotion_labels = list(encountered_emotions)
    
    # Create legend for gender
    gender_markers = [plt.Line2D([0], [0], marker=marker, color="black", linestyle="none", markersize=8) 
                     for marker in gender_marker.values()]
    gender_labels = list(gender_marker.keys())
    
    # Combine legends
    ax.legend(
        emotion_patches + gender_markers, 
        emotion_labels + gender_labels,
        loc="upper right"
    )
    
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Timeline with Emotion and Gender")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    
    st.pyplot(fig)

def display_results_table(results):
    """Display results in a Streamlit table"""
    if not results:
        st.warning("No results to display")
        return
    
    # Create a DataFrame for display
    df = pd.DataFrame(results)
    
    # Format times to 2 decimal places
    df['start_time'] = df['start_time'].apply(lambda x: f"{x:.2f}s")
    df['end_time'] = df['end_time'].apply(lambda x: f"{x:.2f}s")
    
    # Format confidence as percentage
    df['emotion_confidence'] = df['emotion_confidence'].apply(lambda x: f"{x:.1%}")
    
    # Reorder columns for better display
    df = df[['speaker', 'start_time', 'end_time', 'gender', 'emotion', 'emotion_confidence']]
    
    # Rename columns for display
    df.columns = ['Speaker', 'Start Time', 'End Time', 'Gender', 'Emotion', 'Confidence']
    
    st.dataframe(df)

def create_summary(results):
    """Create a summary of the analysis results"""
    if not results:
        st.warning("No results to display")
        return
    
    # Prepare a summary of the results
    speaker_genders = {}
    speaker_emotions = {}

    for result in results:
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
    
    # Display summary for each speaker
    st.subheader("Analysis Summary")
    
    for speaker in speaker_genders:
        st.markdown(f"**Speaker {speaker}:**")
        
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
        
        # All detected emotions
        all_emotions = set([e for e, _ in emotions])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gender", most_common_gender, f"{gender_confidence:.1%} confidence")
        with col2:
            st.metric("Primary Emotion", most_common_emotion, f"{avg_emotion_confidence:.1%} confidence")
        
        # Display emotion distribution
        emotion_counts = {}
        for emotion, _ in emotions:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        # Create emotion distribution chart
        fig, ax = plt.subplots(figsize=(8, 4))
        emotions_list = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        ax.bar(emotions_list, counts)
        ax.set_ylabel('Count')
        ax.set_title(f'Emotion Distribution for Speaker {speaker}')
        st.pyplot(fig)
        
        st.markdown("---")

##########################################
# Main Streamlit App
##########################################
def main():
    st.set_page_config(
        page_title="Audio Analysis App",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎤 Audio Analysis App")
    st.markdown("""
    This app analyzes audio recordings to identify:
    - Speaker diarization (who speaks when)
    - Gender detection for each speaker
    - Emotion recognition for each speech segment
    
    Upload an audio file to get started!
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # HuggingFace token input
    hf_token = st.sidebar.text_input(
        "HuggingFace Token (for speaker diarization)",
        value="",
        type="password",
        help="Enter your HuggingFace token for accessing the pyannote/speaker-diarization model"
    )
    
    # Model directory input
    model_dir = st.sidebar.text_input(
        "Wav2Vec2 Emotion Model Directory",
        value="./wav2vec2_emotion_local__/models--firdhokk--speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53/snapshots/611e6db8ee667aa07fe66596f9fc761e036ff5b9",
        help="Path to the pretrained emotion model"
    )
    
    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    
    if audio_file is not None:
        # Display audio player
        st.audio(audio_file)
        
        # Create a button to start analysis
        if st.button("Start Analysis"):
            if not hf_token:
                st.error("Please enter your HuggingFace token in the sidebar")
                return
                
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    audio_path = tmp_file.name
                
                # Show a status message during initialization
                # In the main function, where you call AudioAnalyzer:
                with st.spinner("Initializing audio analyzer..."):
                    # Corrected order of parameters - auth token first, no device parameter specified
                    analyzer = AudioAnalyzer(use_auth_token=hf_token)
                
                # Create a progress bar for the analysis process
                progress_bar = st.progress(0)
                
                # Process the audio
                with st.spinner("Analyzing audio..."):
                    results = analyzer.process_audio(audio_path, progress_bar)
                
                # Create tabs for different views of the results
                tab1, tab2, tab3 = st.tabs(["Timeline", "Detailed Results", "Summary"])
                
                with tab1:
                    plot_timeline(results)
                
                with tab2:
                    display_results_table(results)
                
                with tab3:
                    create_summary(results)
                
                # Clean up the temporary file
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()