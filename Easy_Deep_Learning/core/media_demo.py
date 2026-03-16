"""Demo utilities for audio/video inputs."""

from __future__ import annotations

import io
import math
import wave
from typing import Any

import numpy as np


def generate_sine_wave(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * math.pi * freq * t)


def load_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        dtype = np.int16 if wf.getsampwidth() == 2 else np.int8
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        audio /= np.max(np.abs(audio)) + 1e-9
        return audio, sr


def audio_features(signal: np.ndarray, sr: int) -> dict[str, float]:
    if signal.size == 0:
        return {}
    rms = float(np.sqrt(np.mean(signal**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))) > 0))
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    mag = np.abs(fft)
    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))
    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "duration_sec": float(len(signal) / sr),
        "sample_rate": float(sr),
    }


def generate_synthetic_video(num_frames: int = 12, size: int = 64) -> list[np.ndarray]:
    frames = []
    for i in range(num_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        cx = int((i / max(1, num_frames - 1)) * (size - 1))
        frame[:, :, 0] = (np.arange(size) + cx) % 255
        frame[:, :, 1] = (np.arange(size).reshape(-1, 1) + i * 10) % 255
        frame[:, :, 2] = (i * 20) % 255
        frames.append(frame)
    return frames


def video_features(frames: list[np.ndarray]) -> dict[str, float]:
    if not frames:
        return {}
    means = [float(np.mean(f)) for f in frames]
    motion = 0.0
    for i in range(1, len(frames)):
        motion += float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
    motion /= max(1, len(frames) - 1)
    return {
        "num_frames": float(len(frames)),
        "mean_intensity": float(np.mean(means)),
        "motion_energy": motion,
    }


def build_audio_dataset(n: int = 200, sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic audio dataset: classify low vs high frequency."""
    rng = np.random.default_rng(42)
    X = []
    y = []
    for _ in range(n):
        label = rng.integers(0, 2)
        freq = rng.uniform(200, 400) if label == 0 else rng.uniform(600, 900)
        signal = generate_sine_wave(freq=freq, duration=1.0, sr=sr)
        feats = audio_features(signal, sr)
        X.append([feats["rms"], feats["zcr"], feats["spectral_centroid"]])
        y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def build_video_dataset(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic video dataset: classify low vs high motion."""
    rng = np.random.default_rng(42)
    X = []
    y = []
    for _ in range(n):
        label = rng.integers(0, 2)
        frames = generate_synthetic_video(num_frames=12 if label == 0 else 20)
        feats = video_features(frames)
        X.append([feats["mean_intensity"], feats["motion_energy"], feats["num_frames"]])
        y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
