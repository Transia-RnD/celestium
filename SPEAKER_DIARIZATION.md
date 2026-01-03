# Speaker Diarization Integration

Celestium now includes advanced speaker diarization capabilities to enhance security by ensuring only the authorized user's voice is present during authentication and transaction approval.

## Overview

Speaker diarization answers "who spoke when" by:
- Detecting the number of unique speakers in audio
- Identifying when different speakers are talking
- Rejecting audio with multiple speakers present

This adds an extra layer of security beyond voice biometrics (GMM) and password verification.

## Security Flow

```
1. User speaks command
   ↓
2. Speaker Diarization Check ← NEW!
   - Count speakers in audio
   - Verify single speaker (≥90% of audio)
   - Reject if multiple speakers detected
   ↓
3. Voice Biometric Verification (GMM)
   - Match voice against trained model
   ↓
4. Password Verification
   - Verify spoken passphrase
   ↓
5. Transaction Approved ✓
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/darkmatter/projects/transia/celestium
poetry install
```

This will install:
- `openai-whisper` - Enhanced speech recognition
- `pyannote-audio` - Speaker diarization
- `torch` & `torchaudio` - ML framework
- `python-dotenv` - Environment variable management
- `soundfile` - Audio file I/O

### 2. Get HuggingFace Token

Speaker diarization requires a HuggingFace API token:

1. **Create account**: https://huggingface.co/join
2. **Get token**: https://huggingface.co/settings/tokens
3. **Accept user agreement**: https://huggingface.co/pyannote/speaker-diarization-3.1
   - This is required to download the diarization model

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your token
nano .env
```

Add your token:
```
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
DEVICE=cpu  # or "cuda" if you have NVIDIA GPU
```

### 4. Verify Configuration

```python
from celestium.config import Config

Config.print_config()
# Should show: HuggingFace Token: ✓ Set
```

## Usage

### Automatic Integration

Speaker diarization is now **automatically enabled** in:

1. **Transaction Approval** (`approval.py:get_transaction()`)
   - Validates speaker count when recording "Approve or Deny?"
   - Threshold: 90% of audio must be primary speaker

2. **Password Verification** (`approval.py:verify_password()`)
   - Validates speaker count when recording callsign
   - Threshold: 90% of audio must be primary speaker

### Manual Usage

You can also use speaker verification independently:

```python
from celestium.speaker_verification import validate_authorized_speaker

# Quick validation
is_valid, message = validate_authorized_speaker(
    "audio.wav",
    threshold=0.90  # 90% primary speaker required
)

if is_valid:
    print(f"✓ {message}")
else:
    print(f"❌ {message}")
```

### Advanced Usage

```python
from celestium.speaker_verification import SpeakerVerifier

# Create verifier instance
verifier = SpeakerVerifier()
verifier.load_diarization_pipeline()

# Detailed speaker analysis
result = verifier.detect_speakers("audio.wav")
print(f"Speakers detected: {result['num_speakers']}")
print(f"Speaker times: {result['speaker_times']}")

# Verify single speaker with custom threshold
is_valid, details = verifier.verify_single_speaker(
    "audio.wav",
    threshold=0.95  # 95% required
)

# Extract only primary speaker's audio
verifier.extract_primary_speaker_audio(
    "noisy_audio.wav",
    "cleaned_audio.wav"
)
```

## How It Works

### Speaker Diarization Process

1. **Audio Input**: Recorded audio file (16kHz, mono)
2. **Feature Extraction**: PyAnnote extracts speaker embeddings
3. **Clustering**: Groups similar embeddings into speakers
4. **Timeline Analysis**: Maps speakers to time segments
5. **Validation**: Checks if single speaker dominates (≥90%)

### Thresholds

- **90% threshold** (default): Allows brief background noise or interruptions
- **95% threshold** (strict): Requires near-perfect isolation
- **100% threshold** (very strict): No other speakers allowed

### Performance

- **CPU**: ~5-10 seconds per 4-second audio clip
- **GPU (CUDA)**: ~1-2 seconds per 4-second audio clip
- **Memory**: ~2GB RAM for model loading

## Security Benefits

### Before Speaker Diarization
❌ Someone could stand next to you and influence your speech
❌ Hidden microphones could capture background voices
❌ Coerced authentication possible with others present

### After Speaker Diarization
✅ Detects if multiple people are speaking
✅ Rejects audio with background voices
✅ Ensures you are physically alone during authentication
✅ Adds protection against coercion scenarios

## Troubleshooting

### Error: "HuggingFace auth token required"

**Solution**: Set `HUGGINGFACE_TOKEN` in your `.env` file

```bash
# Check if token is set
python -c "from celestium.config import Config; Config.validate()"
```

### Error: "Failed to load diarization pipeline"

**Possible causes**:
1. Haven't accepted pyannote user agreement
   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Invalid token
   - Verify at: https://huggingface.co/settings/tokens
3. Network issues
   - Check internet connection

### Warning: "Multiple speakers detected"

**Expected behavior** when:
- Someone else is talking nearby
- TV/radio playing in background
- Phone call audio
- Echo/reverb causing false detection

**Solutions**:
- Find quiet location
- Use noise-cancelling microphone
- Reduce threshold to 0.85 (more lenient)
- Turn off background audio sources

### Performance Issues

If speaker diarization is too slow:

1. **Use GPU**: Set `DEVICE=cuda` in `.env`
   - Requires NVIDIA GPU with CUDA support
   - 5-10x faster than CPU

2. **Disable temporarily**: Comment out validation calls in `approval.py`
   ```python
   # is_valid, message = validate_authorized_speaker(FILENAME)
   ```

## Configuration Options

Edit `celestium/config.py` to customize:

```python
class Config:
    # Speaker diarization threshold (0.0 - 1.0)
    SPEAKER_DIARIZATION_THRESHOLD = 0.90

    # Device for PyTorch
    DEVICE = 'cpu'  # or 'cuda'

    # Audio settings
    AUDIO_RATE = 16000
    RECORD_SECONDS = 4
```

## Testing

Test speaker diarization with sample audio:

```python
from celestium.speaker_verification import get_speaker_verifier

verifier = get_speaker_verifier()

# Test with your voice recording
result = verifier.detect_speakers("test_audio.wav")
print(f"Detected {result['num_speakers']} speaker(s)")

# Test with multi-speaker audio (should detect 2+)
result = verifier.detect_speakers("conversation.wav")
print(f"Detected {result['num_speakers']} speaker(s)")
```

## Model Information

**PyAnnote Speaker Diarization 3.1**
- Released: 2024
- Architecture: Neural speaker embeddings + clustering
- Languages: Language-agnostic (works with any language)
- License: MIT License (with user agreement)
- Paper: https://arxiv.org/abs/2012.01477

## Further Reading

- [PyAnnote Documentation](https://github.com/pyannote/pyannote-audio)
- [Speaker Diarization Overview](https://www.assemblyai.com/blog/what-is-speaker-diarization-and-how-does-it-work)
- [HuggingFace Model Card](https://huggingface.co/pyannote/speaker-diarization-3.1)
