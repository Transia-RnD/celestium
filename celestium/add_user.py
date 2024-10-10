#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import speech_recognition as speechsr
import librosa
import time
import pyaudio
import wave
from glob import glob
import noisereduce as nr
import numpy as np
from sklearn.mixture import GaussianMixture
import soundfile as sf
import hashlib
from xahau.wallet import Wallet
from sklearn.preprocessing import StandardScaler

from celestium.utils import (
    speak,
    extract_features,
    generate_secret_key,
    encrypt_data,
    write_encryption,
)


# Function to add a new user
def add_user(name: str):

    # Check if the user already exists in the database
    db_file = "./user_db/embedding.pickle"
    if os.path.exists(db_file):
        with open(db_file, "rb") as database:
            db = pickle.load(database)
            if name in db:
                print("Name Already Exists! Try Another Name...")
                speak("Name Already Exists! Try Another Name...")
                return
    else:
        # Create a new database if it doesn't exist
        os.makedirs("./user_db", exist_ok=True)
        db = []

    # Voice recording settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 4

    source = os.path.join("./voice_database", name)
    os.makedirs(source, exist_ok=True)

    # Number of recordings to collect
    num_recordings = 16

    for i in range(num_recordings):
        audio = pyaudio.PyAudio()

        if i == 0:
            # Countdown before the first recording
            for j in range(3, 0, -1):
                time.sleep(1.0)
                os.system("cls" if os.name == "nt" else "clear")
                print(f"Speak any random sentence in {j} seconds")
                speak(f"Speak any random sentence in {j} seconds")
        elif i < num_recordings - 1:
            time.sleep(2.0)
            print("Speak any random sentence one more time")
            speak("Speak any random sentence one more time")
            time.sleep(0.8)
        else:
            time.sleep(2.0)
            print("Speak your password")
            speak("Speak your password")
            time.sleep(0.8)

        # Start recording
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        print("Recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio
        wave_output_filename = os.path.join(source, f"{i + 1}.wav")
        with wave.open(wave_output_filename, "wb") as waveFile:
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(pyaudio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b"".join(frames))
        print("Done")
        speak("Done")

    # Process the recordings
    clean_voice_dir = os.path.join(source, "clean_voice")
    os.makedirs(clean_voice_dir, exist_ok=True)

    # Remove background noise and silence from each recording
    for path in glob(os.path.join(source, "*.wav")):
        filename = os.path.basename(path)
        print(f"Processing: {filename}")
        input_signal, sample_rate = librosa.load(path, sr=RATE)
        input_signal = input_signal.flatten()

        # Collect background noise sample (first 0.5 seconds)
        noise_sample = input_signal[: int(sample_rate * 0.5)]  # First 0.5 seconds

        # Reduce the background noise from the input signal
        print(f"Removing background noise for: {filename}")
        reduced_noise = nr.reduce_noise(
            y=input_signal,
            y_noise=noise_sample,
            prop_decrease=1.0,
            sr=RATE,
        )

        # Remove silence from the reduced noise signal
        print(f"Removing silence from: {filename}")
        yt, _ = librosa.effects.trim(
            reduced_noise, top_db=20
        )  # Adjust top_db as needed

        # Normalize the audio
        if np.max(np.abs(yt)) > 0:
            yt = yt / np.max(np.abs(yt))

        # Save the cleaned audio
        clean_file_path = os.path.join(clean_voice_dir, filename)
        sf.write(clean_file_path, yt, RATE, subtype="PCM_16")

    # Extract features and train the Gaussian Mixture Model
    dest = "./gmm_models/"
    os.makedirs(dest, exist_ok=True)
    features = []

    clean_voice_files = glob(os.path.join(clean_voice_dir, "*.wav"))
    for path in clean_voice_files:
        print(f"Extracting features from: {path}")
        audio_signal, sr = librosa.load(path, sr=RATE)

        # Extract features using the provided utility function
        vector = extract_features(audio_signal, sr)
        features.append(vector)

    # Stack all the feature vectors vertically
    features = np.vstack(features)

    # Normalize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Save the scaler for future use
    scaler_path = os.path.join(dest, f"{name}_scaler.pkl")
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Train the Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=32, max_iter=200, covariance_type="diag", n_init=5, random_state=42
    )
    gmm.fit(features)

    db.append(name)
    with open(db_file, "wb") as dbf:
        pickle.dump(db, dbf)

    # Save the trained model
    model_path = os.path.join(dest, f"{name}.gmm")
    with open(model_path, "wb") as model_file:
        pickle.dump(gmm, model_file)
    print(f"{name} added successfully")
    # speak(f"{name} has registered successfully")

    # Store the user's password
    # Assuming the last recording is the password
    r = speechsr.Recognizer()
    password_audio_file = os.path.join(source, f"{num_recordings}.wav")
    print(f"Processing password from: {password_audio_file}")
    with speechsr.AudioFile(password_audio_file) as source_audio:
        audio_data = r.record(source_audio)

    try:
        password_text: str = r.recognize_google(audio_data)
    except sr.UnknownValueError:
        print("Could not understand the password audio")
        speak("Could not understand the password audio, please try again.")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speak(
            "Could not request results from Speech Recognition service, please try again."
        )
        return

    # Hash the password using SHA-256
    hashed_password = hashlib.sha256(password_text.encode()).hexdigest()
    password_file_path = os.path.join(source, "password.txt")
    with open(password_file_path, "w") as f:
        f.write(hashed_password)

    # Extract features from the password audio and generate secret key
    audio_signal, sr = librosa.load(password_audio_file, sr=RATE)
    vector = extract_features(audio_signal, sr)
    vector = scaler.transform(vector)

    # Generate secret key
    wallet = Wallet.create()
    secret_key = generate_secret_key(password_text.encode())
    nonce, encrypted, tag = encrypt_data(wallet.seed, secret_key)
    write_encryption(name, nonce.hex(), encrypted.hex(), tag.hex())
    print("Password stored successfully")
    # speak("Password stored successfully")
