#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import json
import time
import wave
import pyaudio
import speech_recognition as speechsr
import librosa
import noisereduce as nr
from glob import glob
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hashlib
from celestium.utils import (
    speak,
    extract_features,
    generate_secret_key,
    read_encryption,
    decrypt_data,
)
from xahau.wallet import Wallet

from typing import Dict, Any
from pydub import AudioSegment
from pydub.silence import split_on_silence
from celestium.xahau import XahauBot
from xahau.models import Transaction


def validate_identity(filename: str, name: str) -> bool:
    modelpath = "./gmm_models/"

    # Paths to the user's GMM model and scaler
    gmm_model_file = os.path.join(modelpath, f"{name}.gmm")
    scaler_file = os.path.join(modelpath, f"{name}_scaler.pkl")

    # Check if the model and scaler exist
    if not os.path.exists(gmm_model_file) or not os.path.exists(scaler_file):
        print(f"No model or scaler found for user '{name}'.")
        return False

    # Load the GMM model
    with open(gmm_model_file, "rb") as model_file:
        gmm: GaussianMixture = pickle.load(model_file)

    # Load the scaler
    with open(scaler_file, "rb") as scaler_file:
        scaler: StandardScaler = pickle.load(scaler_file)

    # Load and process the audio file
    audio, sr = librosa.load(filename, sr=16000)
    vector = extract_features(audio, sr)

    # Check if feature extraction was successful
    if vector is None or vector.size == 0:
        print("Failed to extract features from the audio file.")
        return False

    # Scale the features
    vector = scaler.transform(vector)

    # Compute the log likelihood of the features under the GMM
    scores = gmm.score_samples(vector)
    log_likelihood = np.sum(scores)

    # Set a threshold (this value should be adjusted based on validation data)
    threshold = -5000  # Example threshold, adjust based on your data

    print(f"Log Likelihood: {log_likelihood}")
    print(f"Threshold: {threshold}")

    if log_likelihood >= threshold:
        print(f"User verified successfully.")
        return True
    else:
        print(f"User verification failed.")
        return False


def record_audio(filename, trigger, record_seconds=4):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    time.sleep(1.0)
    speak(trigger)
    frames = []

    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, "wb") as waveFile:
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b"".join(frames))


def remove_background_noise(path: str, clean_voice_dir: str, rate: int = 16000):
    print("Removing background noise...")
    filename = os.path.basename(path)
    print(f"Processing: {filename}")
    input_signal, _ = librosa.load(path, sr=rate)
    input_signal = input_signal.flatten()

    noise_signals = []
    for noise_file in glob("./background_noise/*.wav"):
        noise_signal, _ = librosa.load(noise_file, sr=rate)
        noise_signal = noise_signal.flatten()
        if len(noise_signal) >= len(input_signal):
            noise_signal = noise_signal[: len(input_signal)]
        else:
            padding = len(input_signal) - len(noise_signal)
            noise_signal = np.pad(noise_signal, (0, padding), "constant")
            noise_signals.append(noise_signal)

    combined_noise = np.mean(noise_signals, axis=0)

    print(f"Removing background noise for: {filename}")
    reduced_noise = nr.reduce_noise(
        y=input_signal,
        y_noise=combined_noise,
        prop_decrease=1.0,
        sr=rate,
    )

    print(f"Removing silence from: {filename}")

    # Convert reduced_noise numpy array to AudioSegment
    reduced_noise_int16 = (reduced_noise * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        reduced_noise_int16.tobytes(),
        frame_rate=rate,
        sample_width=reduced_noise_int16.dtype.itemsize,
        channels=1,
    )

    # Use pydub to remove silence
    chunks = split_on_silence(
        audio_segment,
        min_silence_len=500,  # Adjust this value as needed
        silence_thresh=audio_segment.dBFS - 16,
        keep_silence=100,
    )

    print(f"Number of chunks after silence removal: {len(chunks)}")

    # Concatenate chunks back together
    if chunks:
        processed_audio = chunks[0]
        for chunk in chunks[1:]:
            processed_audio += chunk
    else:
        processed_audio = audio_segment  # If no chunks detected, use the original

    # Save the processed audio
    clean_file_path = os.path.join(clean_voice_dir, filename)
    processed_audio.export(clean_file_path, format="wav")


def check_transaction_exists() -> Dict[str, Any]:
    try:
        with open(f"./transaction.json", "r") as file:
            transaction = json.load(file)
            return transaction
    except FileNotFoundError:
        print("Transaction file not found.")
        speak("Transaction file not found.")
        return None


def get_transaction() -> Dict[str, Any]:
    FILENAME = "./transaction_approval.wav"
    record_audio(FILENAME, "Approve or Deny?")

    r = speechsr.Recognizer()
    with speechsr.AudioFile(FILENAME) as source:
        audio_data = r.record(source)

    approval_command: str = r.recognize_google(audio_data)
    print(f"Approval Command: {approval_command}")

    if "approve" not in approval_command.lower():
        print("Transaction denied.")
        speak("Transaction denied.")
        # delete txn file
        os.remove("./transaction.json")
        return None

    tx_json: Dict[str, Any] = check_transaction_exists()
    if tx_json is not None:
        return tx_json
    else:
        speak(f"Transaction does not exist.")
        return None


def verify_password(name, preapproved_txn):

    FILENAME = "./password_verification.wav"
    CLEAN_FILENAME = f"./voice_database/{name}/clean_voice/password_verification.wav"
    record_audio(FILENAME, "What Is Your Callsign?")
    remove_background_noise(FILENAME, f"./voice_database/{name}/clean_voice")

    r = speechsr.Recognizer()
    with speechsr.AudioFile(FILENAME) as source:
        audio_data = r.record(source)

    try:
        test_password: str = r.recognize_google(audio_data)
        hash_password: str = hashlib.sha256(test_password.encode()).hexdigest()
    except speechsr.UnknownValueError:
        print("Could not understand audio")
        speak("Could not understand audio")
        return False
    except speechsr.RequestError as e:
        print(f"Could not request results; {e}")
        speak("Speech recognition error")
        return False

    # For testing purposes, we'll print the recognized password
    print(f"Recognized password: {hash_password}")

    with open(f"./voice_database/{name}/password.txt", "r") as f:
        user_password = f.read().strip()

    print("Stored Password:", user_password)
    print("Spoken Password:", hash_password)

    if not validate_identity(CLEAN_FILENAME, name):
        print("Invalid Identity. Please try again.")
        speak("Invalid Identity. Please try again.")
        return False

    if user_password.lower() == hash_password.lower():
        print("Transaction approved and user authenticated.")
        wallet: Wallet = get_wallet(name, test_password)
        print(wallet.classic_address)
        XahauBot().submit(wallet, Transaction.from_xrpl(preapproved_txn))
        return True
    else:
        print(f"Wrong Password for {name}")
        speak(f"Wrong Password for {name}")
        return False


def get_wallet(name: str, passcode: str) -> Wallet:
    try:
        encrypted_json = read_encryption(name)
        secret_key = generate_secret_key(passcode.encode())
        plain_text = decrypt_data(
            bytes.fromhex(encrypted_json["nonce"]),
            bytes.fromhex(encrypted_json["ciphertext"]),
            bytes.fromhex(encrypted_json["tag"]),
            secret_key,
        )
        return Wallet.from_seed(plain_text)
    except Exception as e:
        print(e)
        speak("Error decrypting wallet.")


def approve_transaction(name: str):

    preapproved_txn = get_transaction()
    if preapproved_txn is not None:
        if verify_password(name, preapproved_txn):
            print("Transaction approved.")
            speak("Transaction approved.")
            return
        else:
            print("Transaction denied due to incorrect password.")
    else:
        print("Invalid approval command. Please try again.")
        speak("Invalid approval command. Please try again.")
