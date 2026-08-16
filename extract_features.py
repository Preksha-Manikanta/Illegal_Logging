import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import librosa

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

DATASET_PATH = "dataset"

features = []
labels = []

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print("\nProcessing:", label)

    for file in os.listdir(folder):

        if file.lower().endswith((".wav", ".mp3")):

            filepath = os.path.join(folder, file)

            try:
                waveform, sr = librosa.load(
                    filepath,
                    sr=16000,
                    mono=True
                )

                scores, embeddings, spectrogram = yamnet(waveform)

                # Average YAMNet embeddings
                embedding = embeddings.numpy().mean(axis=0)

                features.append(embedding)
                labels.append(label)

                print("  Done:", file)

            except Exception as e:
                print("  ERROR:", file)
                print(" ", e)

X = np.array(features)
y = np.array(labels)

print("\nFeature extraction completed!")
print("Number of audio files:", len(X))
print("Feature shape:", X.shape)
print("Classes:", np.unique(y))

# Save features
np.save("X.npy", X)
np.save("y.npy", y)

print("\nSaved:")
print("X.npy")
print("y.npy")