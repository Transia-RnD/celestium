"""
Speaker Verification and Diarization Module

This module provides speaker diarization capabilities to identify and verify
that only the authorized speaker is present in audio recordings.
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import librosa


class SpeakerVerifier:
    """
    Handles speaker diarization and verification to ensure only
    the authorized user's voice is present in recordings.
    """

    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the speaker verification system.

        Args:
            auth_token: HuggingFace API token for pyannote models
                       (required for downloading pretrained models)
        """
        self.auth_token = auth_token or os.getenv("HUGGINGFACE_TOKEN")
        self.pipeline = None

    def load_diarization_pipeline(self):
        """
        Load the pyannote speaker diarization pipeline.
        Requires HuggingFace authentication token.
        """
        if not self.auth_token:
            raise ValueError(
                "HuggingFace auth token required. Set HUGGINGFACE_TOKEN env variable "
                "or pass it to the constructor. Get one at: "
                "https://huggingface.co/settings/tokens"
            )

        try:
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )

            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("✓ Using GPU for speaker diarization")
            else:
                print("✓ Using CPU for speaker diarization")

        except Exception as e:
            raise RuntimeError(f"Failed to load diarization pipeline: {e}")

    def detect_speakers(self, audio_path: str) -> Dict:
        """
        Detect and count unique speakers in an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing:
                - num_speakers: Number of unique speakers detected
                - segments: List of (start, end, speaker_id) tuples
                - speaker_times: Dict mapping speaker_id to total speaking time
                - speakers: List of unique speaker IDs
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call load_diarization_pipeline() first.")

        # Run diarization
        diarization = self.pipeline(audio_path)

        # Extract speaker information
        speakers = set()
        segments = []
        speaker_times = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append((turn.start, turn.end, speaker))

            # Calculate speaking time
            duration = turn.end - turn.start
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        return {
            'num_speakers': len(speakers),
            'segments': segments,
            'speaker_times': speaker_times,
            'speakers': list(speakers)
        }

    def verify_single_speaker(self, audio_path: str,
                             threshold: float = 0.95) -> Tuple[bool, Dict]:
        """
        Verify that only one speaker is present in the audio.

        Args:
            audio_path: Path to the audio file
            threshold: Minimum percentage of audio that should be from primary speaker
                      (default: 0.95 = 95%)

        Returns:
            Tuple of (is_valid, details) where:
                - is_valid: True if single speaker detected
                - details: Dictionary with speaker analysis
        """
        try:
            result = self.detect_speakers(audio_path)
        except Exception as e:
            return False, {'error': f'Diarization failed: {str(e)}'}

        num_speakers = result['num_speakers']

        if num_speakers == 0:
            return False, {'error': 'No speakers detected', 'result': result}

        if num_speakers == 1:
            return True, {
                'message': 'Single speaker verified',
                'result': result,
                'num_speakers': 1
            }

        # Multiple speakers detected - check if one is dominant
        speaker_times = result['speaker_times']
        total_time = sum(speaker_times.values())

        # Find primary speaker (longest speaking time)
        primary_speaker = max(speaker_times, key=speaker_times.get)
        primary_time = speaker_times[primary_speaker]
        primary_percentage = primary_time / total_time

        is_valid = primary_percentage >= threshold

        details = {
            'num_speakers': num_speakers,
            'primary_speaker': primary_speaker,
            'primary_percentage': primary_percentage,
            'threshold': threshold,
            'speaker_times': speaker_times,
            'message': (
                f"Primary speaker accounts for {primary_percentage:.1%} of audio. "
                f"{'Valid' if is_valid else 'Invalid'} (threshold: {threshold:.1%})"
            )
        }

        return is_valid, details

    def extract_primary_speaker_audio(self, audio_path: str,
                                     output_path: str) -> str:
        """
        Extract only the primary speaker's audio segments.
        Useful for cleaning up audio with background voices.

        Args:
            audio_path: Path to input audio file
            output_path: Path to save cleaned audio

        Returns:
            Path to the output file
        """
        result = self.detect_speakers(audio_path)

        if result['num_speakers'] == 0:
            raise ValueError("No speakers detected in audio")

        # Find primary speaker
        speaker_times = result['speaker_times']
        primary_speaker = max(speaker_times, key=speaker_times.get)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Extract segments with primary speaker
        primary_segments = [
            (start, end) for start, end, speaker in result['segments']
            if speaker == primary_speaker
        ]

        # Concatenate primary speaker segments
        cleaned_audio = []
        for start, end in primary_segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            cleaned_audio.append(audio[start_sample:end_sample])

        if cleaned_audio:
            cleaned_audio = np.concatenate(cleaned_audio)

            # Save cleaned audio
            import soundfile as sf
            sf.write(output_path, cleaned_audio, sr)

            return output_path
        else:
            raise ValueError("No audio segments found for primary speaker")


# Singleton instance for reuse
_verifier_instance: Optional[SpeakerVerifier] = None


def get_speaker_verifier(auth_token: Optional[str] = None) -> SpeakerVerifier:
    """
    Get or create singleton speaker verifier instance.

    Args:
        auth_token: Optional HuggingFace token (only needed on first call)

    Returns:
        SpeakerVerifier instance with loaded pipeline
    """
    global _verifier_instance

    if _verifier_instance is None:
        _verifier_instance = SpeakerVerifier(auth_token=auth_token)
        _verifier_instance.load_diarization_pipeline()

    return _verifier_instance


def validate_authorized_speaker(audio_path: str,
                                auth_token: Optional[str] = None,
                                threshold: float = 0.95) -> Tuple[bool, str]:
    """
    Quick validation function to check if audio contains single authorized speaker.

    Args:
        audio_path: Path to audio file to validate
        auth_token: HuggingFace auth token for pyannote (optional if env var set)
        threshold: Minimum percentage for primary speaker (default 0.95)

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        verifier = get_speaker_verifier(auth_token=auth_token)
        is_valid, details = verifier.verify_single_speaker(audio_path, threshold)

        if not is_valid:
            error_msg = details.get('error', details.get('message', 'Unknown error'))
            return False, f"Speaker verification failed: {error_msg}"
        else:
            return True, "Speaker verification passed"

    except Exception as e:
        return False, f"Speaker verification error: {str(e)}"
