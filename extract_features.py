import os
import numpy as np
import librosa
import tensorflow_hub as hub


# ============================================================
# LOAD YAMNET
# ============================================================

print("Loading YAMNet...")

yamnet = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

DATASET_PATH = "dataset"


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_from_folder(folder_path, label_mapping):
    """
    Extract YAMNet embeddings from audio files.

    The function searches through the folder and
    all of its subfolders.

    The TOP-LEVEL folder determines the class.
    """

    features = []
    labels = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            # ------------------------------------------------
            # Only process audio files
            # ------------------------------------------------

            if not file.lower().endswith(
                (".wav", ".mp3")
            ):
                continue

            filepath = os.path.join(
                root,
                file
            )

            # ------------------------------------------------
            # Find TOP-LEVEL folder
            # ------------------------------------------------

            relative_path = os.path.relpath(
                root,
                folder_path
            )

            top_level_folder = relative_path.split(
                os.sep
            )[0]

            # ------------------------------------------------
            # Convert folder name into required label
            # ------------------------------------------------

            if top_level_folder in label_mapping:

                label = label_mapping[
                    top_level_folder
                ]

            else:

                label = top_level_folder.lower()

            try:

                # ------------------------------------------------
                # Load audio
                # ------------------------------------------------

                waveform, sample_rate = librosa.load(
                    filepath,
                    sr=16000,
                    mono=True
                )

                # ------------------------------------------------
                # Run YAMNet
                # ------------------------------------------------

                scores, embeddings, spectrogram = yamnet(
                    waveform
                )

                # ------------------------------------------------
                # Average YAMNet embeddings
                #
                # Result = 1024-dimensional feature vector
                # ------------------------------------------------

                embedding = embeddings.numpy().mean(
                    axis=0
                )

                features.append(
                    embedding
                )

                labels.append(
                    label
                )

                print("  Done:", filepath)
                print("       Label:", label)

            except Exception as e:

                print("  ERROR:", filepath)
                print(" ", e)

    return (
        np.array(features),
        np.array(labels)
    )


# ============================================================
# MAIN CLASSIFIER
# ============================================================

print("\n================================")
print("EXTRACTING MAIN CLASSIFIER DATA")
print("================================")


main_mapping = {

    "bird": "bird",

    "animal": "animal",

    "chainsaw": "chainsaw",

    "rain": "rain",

    "vehicle": "vehicle"
}


main_path = os.path.join(
    DATASET_PATH,
    "main"
)


main_X, main_y = extract_from_folder(
    main_path,
    main_mapping
)


# ------------------------------------------------------------
# Save main features
# ------------------------------------------------------------

np.save(
    "main_X.npy",
    main_X
)

np.save(
    "main_y.npy",
    main_y
)


print("\nMAIN DATASET")

print(
    "Features:",
    main_X.shape
)

print(
    "Labels:",
    main_y.shape
)

print(
    "Classes:",
    np.unique(main_y)
)


# ============================================================
# BIRD CLASSIFIER
# ============================================================

print("\n================================")
print("EXTRACTING BIRD CLASSIFIER DATA")
print("================================")


bird_mapping = {

    "Peacock": "peacock",

    "Parrot": "parrot",

    "Sparrow": "sparrow",

    "OtherBird": "normal_bird"
}


bird_path = os.path.join(
    DATASET_PATH,
    "bird"
)


bird_X, bird_y = extract_from_folder(
    bird_path,
    bird_mapping
)


# ------------------------------------------------------------
# Save bird features
# ------------------------------------------------------------

np.save(
    "bird_X.npy",
    bird_X
)

np.save(
    "bird_y.npy",
    bird_y
)


print("\nBIRD DATASET")

print(
    "Features:",
    bird_X.shape
)

print(
    "Labels:",
    bird_y.shape
)

print(
    "Classes:",
    np.unique(bird_y)
)


# ============================================================
# ANIMAL CLASSIFIER
# ============================================================

print("\n================================")
print("EXTRACTING ANIMAL CLASSIFIER DATA")
print("================================")


animal_mapping = {

    "Elephant": "elephant",

    "Lion": "lion",

    "Monkey": "monkey",

    "OtherAnimal": "other_animal"
}


animal_path = os.path.join(
    DATASET_PATH,
    "animal"
)


animal_X, animal_y = extract_from_folder(
    animal_path,
    animal_mapping
)


# ------------------------------------------------------------
# Save animal features
# ------------------------------------------------------------

np.save(
    "animal_X.npy",
    animal_X
)

np.save(
    "animal_y.npy",
    animal_y
)


print("\nANIMAL DATASET")

print(
    "Features:",
    animal_X.shape
)

print(
    "Labels:",
    animal_y.shape
)

print(
    "Classes:",
    np.unique(animal_y)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n================================")
print("ALL FEATURE EXTRACTION COMPLETE")
print("================================")

print("\nSaved files:")

print("main_X.npy")
print("main_y.npy")

print("bird_X.npy")
print("bird_y.npy")

print("animal_X.npy")
print("animal_y.npy")

print("\nDone!")