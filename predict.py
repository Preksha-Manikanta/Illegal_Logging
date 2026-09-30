import tensorflow_hub as hub
import librosa
import numpy as np
import joblib


# ========================================
# LOAD YAMNET
# ========================================

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")


# ========================================
# LOAD TRAINED CLASSIFIERS
# ========================================

print("Loading trained classifiers...")

main_classifier = joblib.load("main_classifier.pkl")
bird_classifier = joblib.load("bird_classifier.pkl")
animal_classifier = joblib.load("animal_classifier.pkl")

print("All classifiers loaded successfully!")


# ========================================
# AUDIO FILE
# ========================================

audio_file = "dataset/main/animal/Monkey/Monkey_1.wav" 

print("\nTesting:", audio_file)


# ========================================
# LOAD AUDIO
# ========================================

waveform, sample_rate = librosa.load(
    audio_file,
    sr=16000,
    mono=True
)


# ========================================
# YAMNET FEATURE EXTRACTION
# ========================================

scores, embeddings, spectrogram = yamnet(waveform)

feature = embeddings.numpy().mean(axis=0)


# ========================================
# MAIN CLASSIFIER
# ========================================

main_prediction = main_classifier.predict([feature])[0]

main_probabilities = main_classifier.predict_proba([feature])[0]

main_index = np.argmax(main_probabilities)

main_confidence = main_probabilities[main_index] * 100


print("\n================================")
print("        MAIN CLASSIFIER")
print("================================")

print("Detected Category :", main_prediction)
print("Confidence         : {:.2f}%".format(main_confidence))


# ========================================
# DEFAULT FINAL RESULT
# ========================================

final_prediction = main_prediction
final_confidence = main_confidence


# ========================================
# BIRD CLASSIFIER
# ========================================

if main_prediction == "bird":

    bird_prediction = bird_classifier.predict([feature])[0]

    bird_probabilities = bird_classifier.predict_proba([feature])[0]

    bird_index = np.argmax(bird_probabilities)

    bird_confidence = bird_probabilities[bird_index] * 100

    final_prediction = bird_prediction
    final_confidence = bird_confidence

    print("\n================================")
    print("        BIRD CLASSIFIER")
    print("================================")

    print("Detected Bird :", bird_prediction)
    print("Confidence    : {:.2f}%".format(bird_confidence))


# ========================================
# ANIMAL CLASSIFIER
# ========================================

elif main_prediction == "animal":

    animal_prediction = animal_classifier.predict([feature])[0]

    animal_probabilities = animal_classifier.predict_proba([feature])[0]

    animal_index = np.argmax(animal_probabilities)

    animal_confidence = animal_probabilities[animal_index] * 100

    final_prediction = animal_prediction
    final_confidence = animal_confidence

    print("\n================================")
    print("        ANIMAL CLASSIFIER")
    print("================================")

    print("Detected Animal :", animal_prediction)
    print("Confidence      : {:.2f}%".format(animal_confidence))


# ========================================
# FINAL RESULT
# ========================================

print("\n================================")
print("          FINAL RESULT")
print("================================")

print("Detected Sound :", final_prediction)
print("Confidence     : {:.2f}%".format(final_confidence))


# ========================================
# STATUS MESSAGE
# ========================================

if main_prediction == "chainsaw":

    if main_confidence >= 80:

        status = "🚨 POSSIBLE LOGGING ACTIVITY - HIGH CONFIDENCE"

    elif main_confidence >= 60:

        status = "⚠️ SUSPICIOUS CHAINSAW SOUND - VERIFY"

    elif main_confidence >= 40:

        status = "🟡 POSSIBLE CHAINSAW SOUND - MONITOR"

    else:

        status = "✅ LOW-CONFIDENCE CHAINSAW SOUND"


elif main_prediction == "vehicle":

    if main_confidence >= 60:

        status = "⚠️ VEHICLE SOUND DETECTED - MONITOR"

    else:

        status = "✅ LOW-CONFIDENCE VEHICLE SOUND"


elif main_prediction == "bird":

    if final_prediction == "peacock":

        status = "🦚 PEACOCK SOUND DETECTED - NORMAL FOREST SOUND"

    elif final_prediction == "parrot":

        status = "🦜 PARROT SOUND DETECTED - NORMAL FOREST SOUND"

    elif final_prediction == "sparrow":

        status = "🐦 SPARROW SOUND DETECTED - NORMAL FOREST SOUND"

    else:

        status = "🐦 BIRD DETECTED - NORMAL FOREST SOUND"


elif main_prediction == "animal":

    if final_prediction == "elephant":

        status = "🐘 ELEPHANT SOUND DETECTED - NORMAL FOREST SOUND"

    elif final_prediction == "lion":

        status = "🦁 LION SOUND DETECTED - NORMAL FOREST SOUND"

    elif final_prediction == "monkey":

        status = "🐒 MONKEY SOUND DETECTED - NORMAL FOREST SOUND"

    else:

        status = "🐾 ANIMAL DETECTED - NORMAL FOREST SOUND"


elif main_prediction == "rain":

    status = "🌧️ RAIN DETECTED - NATURAL FOREST SOUND"


else:

    status = "🌲 NORMAL / UNKNOWN FOREST SOUND"


# ========================================
# PRINT STATUS
# ========================================

print("Status         :", status)

print("\n================================")
print("       PREDICTION COMPLETE")
print("================================")