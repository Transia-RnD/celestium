"""
Configuration module for Celestium
Loads environment variables and provides configuration settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration settings for Celestium"""

    # HuggingFace Token for Speaker Diarization
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

    # Device for PyTorch (cpu or cuda)
    DEVICE = os.getenv('DEVICE', 'cpu')

    # Audio Recording Settings
    AUDIO_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK = 1024
    RECORD_SECONDS = 4

    # GMM Model Settings
    GMM_COMPONENTS = 32
    GMM_COVARIANCE_TYPE = 'diag'
    GMM_MAX_ITERATIONS = 200

    # Voice Verification Settings
    GMM_THRESHOLD = -5000
    SPEAKER_DIARIZATION_THRESHOLD = 0.90  # 90% of audio must be primary speaker

    # Directory Paths
    BASE_DIR = Path(__file__).parent.parent
    GMM_MODELS_DIR = BASE_DIR / 'gmm_models'
    VOICE_DATABASE_DIR = BASE_DIR / 'voice_database'
    BACKGROUND_NOISE_DIR = BASE_DIR / 'background_noise'

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.HUGGINGFACE_TOKEN:
            print("⚠️  Warning: HUGGINGFACE_TOKEN not set in .env file")
            print("   Speaker diarization will not work without it.")
            print("   Get a token at: https://huggingface.co/settings/tokens")
            return False
        return True

    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("=" * 60)
        print("Celestium Configuration")
        print("=" * 60)
        print(f"HuggingFace Token: {'✓ Set' if cls.HUGGINGFACE_TOKEN else '✗ Not Set'}")
        print(f"Device: {cls.DEVICE}")
        print(f"Audio Rate: {cls.AUDIO_RATE} Hz")
        print(f"GMM Threshold: {cls.GMM_THRESHOLD}")
        print(f"Speaker Diarization Threshold: {cls.SPEAKER_DIARIZATION_THRESHOLD}")
        print("=" * 60)


# Validate configuration on import
config = Config()
