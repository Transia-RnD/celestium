#!/usr/bin/env python
# coding: utf-8

import json
import pyttsx3
import numpy as np
from sklearn import preprocessing
import python_speech_features as psf
import warnings
import hashlib
from Crypto.Cipher import AES

from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")

# Initialize the text-to-speech engine
engine = pyttsx3.init("nsss")  # Use 'nsss' for macOS


def speak(text):
    """Convert text to speech."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(e)


def calculate_delta(array):
    """Calculate and return the delta of the given feature vector matrix."""
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (
            array[index[0][0]]
            - array[index[0][1]]
            + (2 * (array[index[1][0]] - array[index[1][1]]))
        ) / 10
    return deltas


def extract_features(audio, rate) -> np.ndarray:
    """Convert audio to MFCC features."""
    mfcc_feat = psf.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined


def generate_secret_key(bytes: bytes) -> bytes:
    secret_key = hashlib.sha256(bytes).digest()
    return secret_key


def encrypt_data(data: str, key: bytes) -> Tuple[bytes, bytes, bytes]:
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return cipher.nonce, ciphertext, tag


def write_encryption(name: str, nonce: str, ciphertext: str, tag: str):
    try:
        data = {"nonce": nonce, "ciphertext": ciphertext, "tag": tag}
        with open(f"voice_database/{name}/encrypted.json", "w") as json_file:
            json.dump(data, json_file)
    except FileNotFoundError:
        return None


def decrypt_data(nonce: bytes, ciphertext: bytes, tag: bytes, key: bytes) -> str:
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
    return decrypted_data.decode()


def read_encryption(name: str) -> Dict[str, Any]:
    try:
        with open(f"voice_database/{name}/encrypted.json") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return None
