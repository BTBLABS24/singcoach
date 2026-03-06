"""
Audio analysis pipeline for singing comparison.
Uses librosa for feature extraction and dtw-python for temporal alignment.
"""

import numpy as np
import librosa
from dtw import dtw
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger("singcoach.analyzer")


@dataclass
class SegmentAnalysis:
    """Analysis result for a single time window after DTW alignment."""
    window_index: int
    ref_time_start: float
    ref_time_end: float
    user_time_start: float
    user_time_end: float
    ref_pitch_median: float
    user_pitch_median: float
    pitch_delta_cents: float
    ref_energy_mean: float
    user_energy_mean: float
    energy_ratio: float
    energy_drop_mid_phrase: bool
    timing_drift_ms: float
    lyric_snippet: Optional[str] = None


class AudioAnalyzer:
    """
    Compares a reference singing recording to a user recording.

    Pipeline:
    1. Load both audio files (mono, 16kHz)
    2. Extract features: pitch (YIN), energy (RMS), onsets, chroma (CQT)
    3. DTW-align using MFCC features
    4. Segment into 500ms windows using the DTW warping path
    5. Compute per-window deltas for pitch, energy, and timing
    """

    def __init__(self, sr: int = 16000, hop_length: int = 512, window_sec: float = 0.5):
        self.sr = sr
        self.hop_length = hop_length
        self.window_sec = window_sec
        self.window_frames = int(window_sec * sr / hop_length)

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def _extract_pitch(self, y: np.ndarray) -> np.ndarray:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sr,
            hop_length=self.hop_length,
        )
        f0 = f0.astype(float)
        f0[f0 <= 0] = np.nan
        return f0

    def _extract_energy(self, y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        max_rms = np.max(rms)
        if max_rms > 0:
            rms = rms / max_rms
        return rms

    def _extract_onsets(self, y: np.ndarray) -> np.ndarray:
        return librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length
        )

    def _extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=13, hop_length=self.hop_length
        )
        return mfcc.T  # (T, 13)

    def _align_dtw(
        self,
        ref_features: np.ndarray,
        user_features: np.ndarray,
    ) -> tuple:
        """
        DTW alignment using MFCC features with no windowing constraint.
        Falls back to unconstrained if Sakoe-Chiba fails (recordings
        may differ significantly in length).
        """
        try:
            # Try with a generous Sakoe-Chiba window first
            window_size = max(len(ref_features), len(user_features))
            alignment = dtw(
                user_features,
                ref_features,
                dist_method="euclidean",
                step_pattern="symmetric2",
                window_type="sakoechiba",
                window_args={"window_size": window_size},
                keep_internals=False,
            )
        except Exception:
            # If that fails, run fully unconstrained
            logger.info("Sakoe-Chiba DTW failed, falling back to unconstrained")
            alignment = dtw(
                user_features,
                ref_features,
                dist_method="euclidean",
                step_pattern="symmetric2",
                keep_internals=False,
            )

        # index1 = query (user), index2 = reference
        return np.array(alignment.index2), np.array(alignment.index1)

    def _segment_by_windows(
        self,
        ref_indices: np.ndarray,
        user_indices: np.ndarray,
        ref_pitch: np.ndarray,
        user_pitch: np.ndarray,
        ref_energy: np.ndarray,
        user_energy: np.ndarray,
    ) -> List[SegmentAnalysis]:
        n_ref_frames = int(np.max(ref_indices)) + 1
        n_user_total = int(np.max(user_indices)) + 1
        scale = n_user_total / n_ref_frames if n_ref_frames > 0 else 1.0

        segments = []
        window_idx = 0

        for win_start in range(0, n_ref_frames, self.window_frames):
            win_end = min(win_start + self.window_frames, n_ref_frames)

            mask = (ref_indices >= win_start) & (ref_indices < win_end)
            if not np.any(mask):
                continue

            matched_user_frames = user_indices[mask]
            user_win_start = int(np.min(matched_user_frames))
            user_win_end = int(np.max(matched_user_frames)) + 1

            # Clamp to array bounds
            rws = max(0, win_start)
            rwe = min(win_end, len(ref_pitch))
            uws = max(0, user_win_start)
            uwe = min(user_win_end, len(user_pitch))

            # --- PITCH ---
            ref_pitch_win = ref_pitch[rws:rwe]
            user_pitch_win = user_pitch[uws:uwe]

            ref_p = float(np.nanmedian(ref_pitch_win)) if len(ref_pitch_win) > 0 else 0.0
            user_p = float(np.nanmedian(user_pitch_win)) if len(user_pitch_win) > 0 else 0.0

            if np.isnan(ref_p) or np.isnan(user_p) or ref_p <= 0 or user_p <= 0:
                pitch_delta_cents = 0.0
            else:
                pitch_delta_cents = 1200.0 * np.log2(user_p / ref_p)

            # --- ENERGY ---
            ref_e_win = ref_energy[rws:min(rwe, len(ref_energy))]
            user_e_win = user_energy[uws:min(uwe, len(user_energy))]

            ref_e_mean = float(np.mean(ref_e_win)) if len(ref_e_win) > 0 else 0.0
            user_e_mean = float(np.mean(user_e_win)) if len(user_e_win) > 0 else 0.0

            energy_ratio = user_e_mean / ref_e_mean if ref_e_mean > 0.01 else 1.0

            energy_drop = False
            if len(user_e_win) >= 4:
                mid = len(user_e_win) // 2
                first_half = float(np.mean(user_e_win[:mid]))
                second_half = float(np.mean(user_e_win[mid:]))
                if first_half > 0.05 and second_half < first_half * 0.6:
                    energy_drop = True

            # --- TIMING ---
            ref_center_frame = (win_start + win_end) / 2.0
            user_center_actual = float(np.mean(matched_user_frames))
            user_center_expected = ref_center_frame * scale
            drift_frames = user_center_actual - user_center_expected
            drift_sec = drift_frames * self.hop_length / self.sr
            timing_drift_ms = drift_sec * 1000.0

            # --- TIME CONVERSIONS ---
            ref_t_start = win_start * self.hop_length / self.sr
            ref_t_end = win_end * self.hop_length / self.sr
            user_t_start = user_win_start * self.hop_length / self.sr
            user_t_end = user_win_end * self.hop_length / self.sr

            ref_p_safe = ref_p if not np.isnan(ref_p) else 0.0
            user_p_safe = user_p if not np.isnan(user_p) else 0.0

            segments.append(SegmentAnalysis(
                window_index=window_idx,
                ref_time_start=round(ref_t_start, 3),
                ref_time_end=round(ref_t_end, 3),
                user_time_start=round(user_t_start, 3),
                user_time_end=round(user_t_end, 3),
                ref_pitch_median=round(ref_p_safe, 1),
                user_pitch_median=round(user_p_safe, 1),
                pitch_delta_cents=round(float(pitch_delta_cents), 1),
                ref_energy_mean=round(ref_e_mean, 4),
                user_energy_mean=round(user_e_mean, 4),
                energy_ratio=round(float(energy_ratio), 3),
                energy_drop_mid_phrase=energy_drop,
                timing_drift_ms=round(float(timing_drift_ms), 1),
            ))
            window_idx += 1

        return segments

    def compare(self, reference_path: str, user_path: str) -> List[SegmentAnalysis]:
        ref_y = self._load_audio(reference_path)
        user_y = self._load_audio(user_path)

        ref_pitch = self._extract_pitch(ref_y)
        user_pitch = self._extract_pitch(user_y)

        ref_energy = self._extract_energy(ref_y)
        user_energy = self._extract_energy(user_y)

        ref_mfcc = self._extract_mfcc(ref_y)
        user_mfcc = self._extract_mfcc(user_y)

        ref_idx, user_idx = self._align_dtw(ref_mfcc, user_mfcc)

        # Equalize feature array lengths per recording
        ref_len = min(len(ref_pitch), len(ref_energy), len(ref_mfcc))
        user_len = min(len(user_pitch), len(user_energy), len(user_mfcc))

        ref_pitch = ref_pitch[:ref_len]
        ref_energy = ref_energy[:ref_len]

        user_pitch = user_pitch[:user_len]
        user_energy = user_energy[:user_len]

        ref_idx = np.clip(ref_idx, 0, ref_len - 1)
        user_idx = np.clip(user_idx, 0, user_len - 1)

        return self._segment_by_windows(
            ref_idx, user_idx,
            ref_pitch, user_pitch,
            ref_energy, user_energy,
        )
