from flask import Flask, render_template, send_from_directory
import tensorflow_hub as hub
import librosa
import numpy as np
import joblib
import os
import webbrowser
import threading

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

AUDIO_FOLDER = "audio_file"

# Change this filename whenever you want to test another sound

AUDIO_FILENAME = "1-29561-A-10.wav"
#AUDIO_FILENAME = "1-64398-B-41.wav"
#AUDIO_FILENAME = "59588__morgantj__carsandwalking.mp3"
#AUDIO_FILENAME = "12.-Tiger-Roar-Power-I-Feel-the-Wild-Jungle-Vibes.mp3"
AUDIO_PATH = os.path.join(
    AUDIO_FOLDER,
    AUDIO_FILENAME
)

MODEL_FILE = "illegal_logging_classifier.pkl"


# ============================================================
# LOAD YAMNET
# ============================================================

print("Loading YAMNet...")

yamnet = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet loaded successfully!")


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

print("Loading trained classifier...")

classifier = joblib.load(
    MODEL_FILE
)

print("Trained classifier loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sound(audio_path):

    print("\n======================================")
    print("Testing audio:", audio_path)
    print("======================================")

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    print("Sample rate:", sample_rate)
    print("Audio samples:", len(waveform))

    # --------------------------------------------------------
    # Run YAMNet
    # --------------------------------------------------------

    scores, embeddings, spectrogram = yamnet(
        waveform
    )

    # --------------------------------------------------------
    # Create feature vector
    # --------------------------------------------------------

    feature = embeddings.numpy().mean(axis=0)

    # --------------------------------------------------------
    # Predict class
    # --------------------------------------------------------

    prediction = classifier.predict(
        [feature]
    )[0]

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    probabilities = classifier.predict_proba(
        [feature]
    )[0]

    best_index = np.argmax(
        probabilities
    )

    confidence = (
        probabilities[best_index] * 100
    )

    # ========================================================
    # DECISION LOGIC
    # ========================================================

    if prediction == "chainsaw":

        if confidence >= 80:

            status = (
                "🚨 ILLEGAL LOGGING DETECTED "
                "-       MONITOR "
            )

            status_type = "danger"

        elif confidence >= 60:

            status = (
                "⚠️ SUSPICIOUS LOGGING ACTIVITY "
                "- VERIFY"
            )

            status_type = "warning"

        elif confidence >= 40:

            status = (
                "🟡 POSSIBLE LOGGING SOUND "
                "- MONITOR"
            )

            status_type = "warning"

        else:

            status = (
                "✅ NORMAL / UNCERTAIN"
            )

            status_type = "normal"


    elif prediction == "vehicle":

        if confidence >= 60:

            status = (
                "⚠️ VEHICLE SOUND DETECTED "
                "- MONITOR"
            )

            status_type = "warning"

        else:

            status = (
                "✅ LOW-CONFIDENCE VEHICLE SOUND"
            )

            status_type = "normal"


    elif prediction == "bird":

        status = (
            "🐦 NORMAL FOREST SOUND "
            "- BIRD DETECTED"
        )

        status_type = "normal"


    elif prediction == "animal":

        status = (
            "🦌 NORMAL FOREST SOUND "
            "- ANIMAL DETECTED"
        )

        status_type = "normal"


    elif prediction == "rain":

        status = (
            "🌧️ NATURAL SOUND "
            "- RAIN DETECTED"
        )

        status_type = "normal"


    else:

        status = (
            "🌲 NORMAL / UNKNOWN "
            "FOREST SOUND"
        )

        status_type = "normal"


    # --------------------------------------------------------
    # Print result in terminal
    # --------------------------------------------------------

    print("\n======================================")
    print("           PREDICTION RESULT")
    print("======================================")

    print(
        "Detected Sound :",
        prediction
    )

    print(
        "Confidence     : {:.2f}%".format(
            confidence
        )
    )

    print(
        "Status         :",
        status
    )

    print("======================================\n")


    return (
        prediction,
        confidence,
        status,
        status_type
    )


# ============================================================
# SERVE AUDIO FILE TO BROWSER
# ============================================================

@app.route("/audio/<path:filename>")
def serve_audio(filename):

    return send_from_directory(
        AUDIO_FOLDER,
        filename
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

@app.route("/")
def home():

    # Check if audio exists

    if not os.path.exists(AUDIO_PATH):

        return render_template(
            "index.html",
            error=(
                "Audio file not found: "
                + AUDIO_PATH
            )
        )

    try:

        # Run prediction

        prediction, confidence, status, status_type = (
            predict_sound(AUDIO_PATH)
        )

        # Send result to webpage

        return render_template(

            "index.html",

            prediction=prediction,

            confidence=confidence,

            status=status,

            status_type=status_type,

            audio_file=AUDIO_FILENAME
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# AUTOMATICALLY OPEN BROWSER
# ============================================================

def open_browser():

    webbrowser.open_new(
        "http://127.0.0.1:5000"
    )


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==============================================")
    print("       FOREST MONITORING SYSTEM")
    print("==============================================")

    print(
        "Test Audio:",
        AUDIO_PATH
    )

    print("----------------------------------------------")
    print("Starting Flask server...")
    print("Browser will open automatically.")
    print("----------------------------------------------")

    print(
        "Website: "
        "http://127.0.0.1:5000"
    )

    print("==============================================")
    print("\n")


    # Open browser automatically
    threading.Timer(
        2,
        open_browser
    ).start()


    # Start Flask
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )