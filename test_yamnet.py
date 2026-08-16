import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

print("TensorFlow Version:", tf.__version__)
print("TensorFlow imported successfully!")

print("Loading YAMNet model .....")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet model loaded successfully!")
audio_file = "audio_file/1-64398-B-41.wav" 
print("Audio file:", audio_file)
waveform, sample_rate = librosa.load(audio_file, sr=16000)
print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)
scores, embeddings, spectrogram = model(waveform)
class_names=pd.read_csv("yamnet_class_map.csv")
print( class_names.head())
average_scores = scores.numpy().mean(axis=0)
print("Average Scores Shape:", average_scores.shape)
top_class = average_scores.argmax()
top_score = average_scores[top_class]

print("Top Class Index:", top_class)
top_sound = class_names.iloc[top_class]["display_name"]
print("Predicted Sound:", top_sound)
print("Confidence: {:.2f}%".format(top_score * 100))
print("Scores shape:", scores.shape)
print("Embeddings shape:", embeddings.shape)
print("Spectrogram shape:", spectrogram.shape)
