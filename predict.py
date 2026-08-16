import tensorflow_hub as hub
import librosa
import numpy as np
import joblib

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading trained classifier...")
classifier = joblib.load("illegal_logging_classifier.pkl")

# Change this to the audio you want to test
audio_file = "audio_file/1-64398-B-41.wav"

print("\nTesting:", audio_file)

# Load audio
waveform, sample_rate = librosa.load(
    audio_file,
    sr=16000,
    mono=True
)

# Get YAMNet embeddings
scores, embeddings, spectrogram = yamnet(waveform)

# Average embeddings
feature = embeddings.numpy().mean(axis=0)

# Predict
prediction = classifier.predict([feature])[0]

# Probability/confidence
probabilities = classifier.predict_proba([feature])[0]

classes = classifier.classes_

confidence = probabilities[np.argmax(probabilities)] * 100

print("\n==============================")
print("       PREDICTION")
print("==============================")
print("Detected Sound :", prediction)
print("Confidence     : {:.2f}%".format(confidence))

# Logging decision
if prediction == "chainsaw":

    if confidence >= 80:
        status = "🚨 ILLEGAL LOGGING DETECTED - HIGH CONFIDENCE"

    elif confidence >= 60:
        status = "⚠️ SUSPICIOUS LOGGING ACTIVITY - VERIFY"

    elif confidence >= 40:
        status = "🟡 POSSIBLE LOGGING SOUND - MONITOR"

    else:
        status = "✅ NORMAL / UNCERTAIN"

elif prediction == "vehicle":

    status = "⚠️ VEHICLE SOUND DETECTED - MONITOR"

else:

    status = "✅ NORMAL FOREST SOUND"

print("Status         :", status)