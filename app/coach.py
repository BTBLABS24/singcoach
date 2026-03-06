"""
Rule-based coaching engine. Deterministic rules applied to per-window
analysis results — no LLM needed.
"""

from typing import List, Dict, Any
from app.analyzer import SegmentAnalysis


# Hardcoded lyrics with timestamps matching the ~44s reference audio.
# Placeholder lyrics (not actual copyrighted lyrics).
LYRICS = [
    (0.0,   4.5,  "Walking down beneath the city lights"),
    (4.5,   9.0,  "Feeling every shadow come alive"),
    (9.0,  13.5,  "The bridge above is calling out my name"),
    (13.5, 18.0,  "Nothing here will ever be the same"),
    (18.0, 22.5,  "I don't ever wanna feel"),
    (22.5, 27.0,  "Like I did that day"),
    (27.0, 31.5,  "Take me to the place I love"),
    (31.5, 36.0,  "Take me all the way"),
    (36.0, 40.0,  "Under the bridge downtown"),
    (40.0, 44.0,  "Is where I drew some blood"),
]


class CoachingEngine:
    """
    Applies rule-based coaching logic to SegmentAnalysis results.

    Rules:
    1. Pitch flat  > 20 cents
    2. Pitch sharp > 20 cents
    3. Energy drop mid-phrase
    4. Timing drift > 100ms
    """

    PITCH_THRESHOLD_CENTS = 20.0
    TIMING_THRESHOLD_MS = 100.0
    ENERGY_RATIO_LOW = 0.5

    def generate_feedback(self, segments: List[SegmentAnalysis]) -> List[Dict[str, Any]]:
        feedback_items = []

        for seg in segments:
            issues = self._evaluate_segment(seg)
            if not issues:
                continue

            lyric = self._find_lyric(seg.ref_time_start, seg.ref_time_end)

            for issue in issues:
                feedback_items.append({
                    "timestamp_start": seg.ref_time_start,
                    "timestamp_end": seg.ref_time_end,
                    "lyric_snippet": lyric,
                    "metric": issue["metric"],
                    "delta": issue["delta"],
                    "feedback": issue["feedback"],
                })

        return feedback_items

    def _evaluate_segment(self, seg: SegmentAnalysis) -> List[Dict[str, Any]]:
        issues = []

        # Rule 1: Pitch flat
        if seg.pitch_delta_cents < -self.PITCH_THRESHOLD_CENTS:
            issues.append({
                "metric": "pitch_flat",
                "delta": f"{seg.pitch_delta_cents:+.1f} cents",
                "feedback": (
                    "You're going flat here \u2014 engage your diaphragm "
                    "for support and think the note slightly higher."
                ),
            })

        # Rule 2: Pitch sharp
        elif seg.pitch_delta_cents > self.PITCH_THRESHOLD_CENTS:
            issues.append({
                "metric": "pitch_sharp",
                "delta": f"{seg.pitch_delta_cents:+.1f} cents",
                "feedback": (
                    "You're going sharp \u2014 ease off the push "
                    "and relax your throat."
                ),
            })

        # Rule 3: Energy drop mid-phrase
        if seg.energy_drop_mid_phrase:
            issues.append({
                "metric": "energy_drop",
                "delta": f"ratio {seg.energy_ratio:.2f}",
                "feedback": (
                    "Breath support is collapsing mid-phrase \u2014 "
                    "take a deeper breath before this line and "
                    "engage your core throughout."
                ),
            })
        elif seg.energy_ratio < self.ENERGY_RATIO_LOW and seg.ref_energy_mean > 0.1:
            issues.append({
                "metric": "energy_low",
                "delta": f"ratio {seg.energy_ratio:.2f}",
                "feedback": (
                    "You're losing volume here \u2014 project more "
                    "from your diaphragm and open up."
                ),
            })

        # Rule 4: Timing drift
        if abs(seg.timing_drift_ms) > self.TIMING_THRESHOLD_MS:
            if seg.timing_drift_ms > 0:
                direction = "dragging"
                advice = "lock into the beat and stay ahead mentally"
            else:
                direction = "rushing"
                advice = "take your time and feel the groove"

            issues.append({
                "metric": "timing_drift",
                "delta": f"{seg.timing_drift_ms:+.0f}ms ({direction})",
                "feedback": f"You're {direction} here \u2014 {advice}.",
            })

        return issues

    def _find_lyric(self, t_start: float, t_end: float) -> str:
        midpoint = (t_start + t_end) / 2.0

        for lyric_start, lyric_end, text in LYRICS:
            if lyric_start <= midpoint < lyric_end:
                return text

        closest = min(LYRICS, key=lambda l: abs((l[0] + l[1]) / 2 - midpoint))
        return closest[2]
