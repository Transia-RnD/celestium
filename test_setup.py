#!/usr/bin/env python
"""
Test script to verify Celestium setup and speaker diarization configuration
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    modules = [
        ('speech_recognition', 'Speech Recognition'),
        ('librosa', 'Librosa'),
        ('pyaudio', 'PyAudio'),
        ('pyttsx3', 'Text-to-Speech'),
        ('noisereduce', 'Noise Reduction'),
        ('pydub', 'PyDub'),
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('pyannote.audio', 'PyAnnote Audio'),
        ('whisper', 'OpenAI Whisper'),
        ('dotenv', 'Python DotEnv'),
    ]

    failed = []
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:30s} - OK")
        except ImportError as e:
            print(f"✗ {display_name:30s} - FAILED: {e}")
            failed.append(display_name)

    print()
    return len(failed) == 0, failed


def test_config():
    """Test configuration setup"""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)

    try:
        from celestium.config import Config

        Config.print_config()

        if not Config.HUGGINGFACE_TOKEN:
            print("\n⚠️  WARNING: HuggingFace token not set!")
            print("   Set HUGGINGFACE_TOKEN in .env file")
            print("   Get token at: https://huggingface.co/settings/tokens")
            return False
        else:
            print("\n✓ Configuration valid")
            return True

    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_speaker_verification():
    """Test speaker verification module"""
    print("\n" + "=" * 60)
    print("Testing Speaker Verification Module")
    print("=" * 60)

    try:
        from celestium.speaker_verification import SpeakerVerifier

        verifier = SpeakerVerifier()
        print("✓ SpeakerVerifier created")

        # Try to load pipeline (this will fail without token)
        try:
            verifier.load_diarization_pipeline()
            print("✓ Diarization pipeline loaded successfully")
            print("✓ Speaker verification fully operational")
            return True
        except ValueError as e:
            if "auth token required" in str(e):
                print("⚠️  Diarization pipeline requires HuggingFace token")
                print("   Set HUGGINGFACE_TOKEN in .env file")
                return False
            else:
                raise

    except Exception as e:
        print(f"✗ Speaker verification error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_file():
    """Check if .env file exists"""
    print("\n" + "=" * 60)
    print("Checking Environment Setup")
    print("=" * 60)

    env_path = Path('.env')
    env_example_path = Path('.env.example')

    if env_path.exists():
        print(f"✓ .env file exists")
        return True
    else:
        print(f"✗ .env file not found")
        if env_example_path.exists():
            print(f"  Copy .env.example to .env and configure:")
            print(f"  cp .env.example .env")
        return False


def test_directories():
    """Check if required directories exist"""
    print("\n" + "=" * 60)
    print("Checking Required Directories")
    print("=" * 60)

    dirs = [
        'gmm_models',
        'voice_database',
        'background_noise',
    ]

    all_exist = True
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name:30s} - exists")
        else:
            print(f"⚠️  {dir_name:30s} - not found (will be created)")
            all_exist = False

    return all_exist


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CELESTIUM SETUP VERIFICATION")
    print("=" * 60)
    print()

    results = []

    # Test imports
    success, failed = test_imports()
    results.append(('Module Imports', success))
    if not success:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        print("   Run: poetry install")

    # Test env file
    success = test_env_file()
    results.append(('Environment File', success))

    # Test config
    success = test_config()
    results.append(('Configuration', success))

    # Test directories
    success = test_directories()
    results.append(('Directories', success))

    # Test speaker verification
    success = test_speaker_verification()
    results.append(('Speaker Verification', success))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("🎉 All tests passed! Celestium is ready to use.")
        print()
        print("Next steps:")
        print("1. Add a user: python -m celestium.add_user")
        print("2. Approve transaction: python playground.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("1. Install dependencies: poetry install")
        print("2. Create .env file: cp .env.example .env")
        print("3. Set HuggingFace token in .env")
        print("4. Accept user agreement: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return 1


if __name__ == '__main__':
    sys.exit(main())
