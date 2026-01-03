# Celestium Upgrade Guide

## What's New

Celestium has been upgraded with advanced speaker diarization and the latest speech recognition libraries for enhanced security and performance.

## Upgrade Summary

### 📦 Updated Libraries

| Library | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| speechrecognition | 3.10.4 | 3.14.3 | Latest 2025 release with Whisper support |
| python | ^3.10.0 | ^3.10.0,<3.13 | Added upper bound for compatibility |

### 🆕 New Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| openai-whisper | ^20231117 | Enhanced speech-to-text quality |
| pyannote-audio | ^3.0.0 | Speaker diarization (who spoke when) |
| torch | ^2.0.0 | PyTorch ML framework |
| torchaudio | ^2.0.0 | Audio processing for PyTorch |
| python-dotenv | ^1.0.0 | Environment variable management |
| soundfile | ^0.12.0 | Audio file I/O |

### 🔒 New Security Features

1. **Speaker Diarization**
   - Detects number of speakers in audio
   - Rejects authentication if multiple speakers present
   - Prevents coerced or influenced transactions

2. **Multi-Layer Voice Authentication**
   ```
   Layer 1: Speaker Count Verification (NEW!)
   Layer 2: Voice Biometric Matching (GMM)
   Layer 3: Spoken Password Verification
   ```

## Quick Start

### 1. Update Dependencies

```bash
cd /Users/darkmatter/projects/transia/celestium
poetry install
```

### 2. Setup Environment

```bash
# Copy example file
cp .env.example .env

# Edit and add your HuggingFace token
nano .env
```

Get your token:
1. Visit https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Accept user agreement: https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Verify Setup

```python
from celestium.config import Config

Config.print_config()
# Should show: HuggingFace Token: ✓ Set
```

### 4. Test Speaker Diarization

```python
from celestium.speaker_verification import validate_authorized_speaker

# Test with a recording
is_valid, message = validate_authorized_speaker("test.wav")
print(message)
```

## What Changed in Your Code

### approval.py

**Added**: Speaker diarization checks in two places

#### 1. Transaction Approval (`get_transaction`)

```python
# NEW: Verify single speaker before processing
print("Verifying speaker identity...")
is_valid, message = validate_authorized_speaker(FILENAME, threshold=0.90)
if not is_valid:
    print(f"❌ {message}")
    speak("Multiple speakers detected. Please ensure you are alone.")
    return None
```

#### 2. Password Verification (`verify_password`)

```python
# NEW: Verify speaker count before authentication
print("Verifying speaker count...")
is_valid, message = validate_authorized_speaker(FILENAME, threshold=0.90)
if not is_valid:
    print(f"❌ {message}")
    speak("Multiple speakers detected. Authentication failed.")
    return False
```

### New Files

```
celestium/
├── speaker_verification.py   # NEW: Speaker diarization module
├── config.py                  # NEW: Configuration management
├── .env.example               # NEW: Environment template
├── SPEAKER_DIARIZATION.md     # NEW: Detailed documentation
└── UPGRADE_GUIDE.md          # NEW: This file
```

## Migration Path

### If You Have Existing Users

Good news! No migration needed for existing users:

- ✅ Existing GMM models still work
- ✅ Existing voice recordings unchanged
- ✅ Existing password hashes compatible
- ✅ Existing encrypted wallets work

**New feature is additive** - it adds speaker verification on top of existing authentication.

### If You Want to Disable Speaker Diarization

Comment out the validation calls in `approval.py`:

```python
# Disable speaker diarization temporarily
# is_valid, message = validate_authorized_speaker(FILENAME)
# if not is_valid:
#     return False
```

## Performance Considerations

### First Run

- Downloads ~300MB pyannote model (one-time)
- Takes 30-60 seconds to initialize
- Models cached locally for future use

### Subsequent Runs

- **CPU**: ~5-10 seconds per verification
- **GPU**: ~1-2 seconds per verification
- **Memory**: ~2GB RAM for model

### Optimization Tips

1. **Use GPU** (10x faster)
   ```bash
   # In .env file
   DEVICE=cuda
   ```

2. **Reuse verifier instance** (already implemented)
   - Singleton pattern avoids reloading model
   - First call loads model, subsequent calls are fast

3. **Adjust threshold** for speed/security tradeoff
   ```python
   # More lenient = faster (fewer rejections)
   validate_authorized_speaker(audio, threshold=0.85)

   # More strict = slower (more rejections)
   validate_authorized_speaker(audio, threshold=0.95)
   ```

## Compatibility

### Python Version

- **Supported**: Python 3.10, 3.11, 3.12
- **Not Supported**: Python 3.13 (Whisper incompatibility)

### Operating Systems

- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ Windows 10/11 (with PyTorch)

### Hardware

- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU
- **Optimal**: 8GB+ RAM, NVIDIA GPU with CUDA

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'pyannote'"

**Solution**: Run `poetry install` to install new dependencies

#### "HuggingFace auth token required"

**Solution**: Set token in `.env` file
```bash
HUGGINGFACE_TOKEN=hf_xxxxx
```

#### "You need to accept the user agreement"

**Solution**: Visit and accept:
https://huggingface.co/pyannote/speaker-diarization-3.1

#### Performance is slow

**Solutions**:
1. Use GPU: `DEVICE=cuda` in `.env`
2. Lower threshold: `threshold=0.85`
3. Disable temporarily during development

### Getting Help

1. Check [SPEAKER_DIARIZATION.md](./SPEAKER_DIARIZATION.md) for detailed docs
2. Review error messages for specific guidance
3. Test speaker diarization independently:
   ```python
   from celestium.speaker_verification import get_speaker_verifier
   verifier = get_speaker_verifier()
   ```

## Rollback Instructions

If you need to rollback to previous version:

```bash
# Restore old pyproject.toml
git checkout HEAD~1 pyproject.toml

# Remove new files
rm celestium/speaker_verification.py
rm celestium/config.py
rm .env.example

# Restore old approval.py
git checkout HEAD~1 celestium/approval.py

# Reinstall old dependencies
poetry install
```

## Testing Checklist

After upgrading, verify:

- [ ] `poetry install` completes successfully
- [ ] HuggingFace token is set in `.env`
- [ ] User agreement accepted at HuggingFace
- [ ] Config validation passes: `Config.validate()`
- [ ] Existing users can still authenticate
- [ ] Speaker diarization detects single speaker
- [ ] Speaker diarization rejects multiple speakers
- [ ] Transaction approval flow works end-to-end

## What's Next

Future enhancements planned:

- [ ] Voice enrollment with speaker diarization
- [ ] Continuous authentication during long sessions
- [ ] Voice liveness detection (anti-spoofing)
- [ ] Multi-language support for commands
- [ ] Real-time speaker tracking

## Feedback

Questions or issues? Check the documentation or review the code:

- `celestium/speaker_verification.py` - Core implementation
- `celestium/approval.py` - Integration points
- `SPEAKER_DIARIZATION.md` - Detailed guide
