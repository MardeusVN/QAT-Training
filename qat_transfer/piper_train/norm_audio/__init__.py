from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import pyworld
import torch

from piper_train.vits.mel_processing import spectrogram_torch

from .trim import trim_silence
from .vad import SileroVoiceActivityDetector

_DIR = Path(__file__).parent


def make_silence_detector() -> SileroVoiceActivityDetector:
    silence_model = _DIR / "models" / "silero_vad.onnx"
    return SileroVoiceActivityDetector(silence_model)


def cache_norm_audio(
    audio_path: Union[str, Path],
    cache_dir: Union[str, Path],
    detector: SileroVoiceActivityDetector,
    sample_rate: int,
    silence_threshold: float = 0.2,
    silence_samples_per_chunk: int = 480,
    silence_keep_chunks_before: int = 2,
    silence_keep_chunks_after: int = 2,
    filter_length: int = 1024,
    window_length: int = 1024,
    hop_length: int = 256,
    ignore_cache: bool = False,
) -> Tuple[Path, Path]:
    audio_path = Path(audio_path).absolute()
    cache_dir = Path(cache_dir)

    # Cache id is the SHA256 of the full audio path
    audio_cache_id = sha256(str(audio_path).encode()).hexdigest()

    audio_norm_path = cache_dir / f"{audio_cache_id}.pt"
    audio_spec_path = cache_dir / f"{audio_cache_id}.spec.pt"

    # Normalize audio
    audio_norm_tensor: Optional[torch.FloatTensor] = None
    if ignore_cache or (not audio_norm_path.exists()):
        # Trim silence first.
        #
        # The VAD model works on 16khz, so we determine the portion of audio
        # to keep and then just load that with librosa.
        vad_sample_rate = 16000
        audio_16khz, _sr = librosa.load(path=audio_path, sr=vad_sample_rate)

        offset_sec, duration_sec = trim_silence(
            audio_16khz,
            detector,
            threshold=silence_threshold,
            samples_per_chunk=silence_samples_per_chunk,
            sample_rate=vad_sample_rate,
            keep_chunks_before=silence_keep_chunks_before,
            keep_chunks_after=silence_keep_chunks_after,
        )

        # NOTE: audio is already in [-1, 1] coming from librosa
        audio_norm_array, _sr = librosa.load(
            path=audio_path,
            sr=sample_rate,
            offset=offset_sec,
            duration=duration_sec,
        )

        # Save to cache directory
        audio_norm_tensor = torch.FloatTensor(audio_norm_array).unsqueeze(0)
        torch.save(audio_norm_tensor, audio_norm_path)

    # Compute spectrogram
    if ignore_cache or (not audio_spec_path.exists()):
        if audio_norm_tensor is None:
            # Load pre-cached normalized audio
            audio_norm_tensor = torch.load(audio_norm_path)

        audio_spec_tensor = spectrogram_torch(
            y=audio_norm_tensor,
            n_fft=filter_length,
            sampling_rate=sample_rate,
            hop_size=hop_length,
            win_size=window_length,
            center=False,
        ).squeeze(0)
        torch.save(audio_spec_tensor, audio_spec_path)

    return audio_norm_path, audio_spec_path


def _interpolate_unvoiced(f0: "np.ndarray") -> "np.ndarray":
    """Fill F0=0 (unvoiced/silent) frames via linear interpolation in log-F0
    space, so the predictor isn't trained to regress towards zero in gaps."""
    voiced = f0 > 0
    if voiced.sum() in (0, len(f0)):
        return f0
    idx = np.arange(len(f0))
    log_f0 = np.log(f0 + 1e-8)
    log_f0_interp = np.interp(idx, idx[voiced], log_f0[voiced])
    f0_interp = np.exp(log_f0_interp)
    f0_interp[voiced] = f0[voiced]
    return f0_interp


def cache_f0(
    audio_norm_path: Union[str, Path],
    cache_dir: Union[str, Path],
    sample_rate: int,
    num_mel_frames: int,
    hop_length: int = 256,
    ignore_cache: bool = False,
) -> Path:
    """Extract a per-frame F0 contour (pyworld DIO + StoneMask) aligned to the
    same number of frames as the mel-spectrogram/duration ground truth.

    pyworld's frame count is consistently 1 frame longer than the
    spectrogram's for this hop_length/sample_rate combination (verified
    empirically), so the contour is truncated to num_mel_frames to align.
    """
    audio_norm_path = Path(audio_norm_path)
    cache_dir = Path(cache_dir)
    audio_cache_id = audio_norm_path.stem
    audio_f0_path = cache_dir / f"{audio_cache_id}.f0.pt"

    if (not ignore_cache) and audio_f0_path.exists():
        return audio_f0_path

    audio_norm_tensor = torch.load(audio_norm_path)
    audio64 = audio_norm_tensor.squeeze(0).numpy().astype(np.float64)

    frame_period_ms = hop_length / sample_rate * 1000.0
    f0, t = pyworld.dio(audio64, sample_rate, frame_period=frame_period_ms)
    f0 = pyworld.stonemask(audio64, f0, t, sample_rate)
    f0 = _interpolate_unvoiced(f0)

    # Align frame count with the spectrogram (off-by-one from pyworld).
    if len(f0) > num_mel_frames:
        f0 = f0[:num_mel_frames]
    elif len(f0) < num_mel_frames:
        f0 = np.pad(f0, (0, num_mel_frames - len(f0)), mode="edge")

    f0_tensor = torch.FloatTensor(f0)
    torch.save(f0_tensor, audio_f0_path)

    return audio_f0_path
