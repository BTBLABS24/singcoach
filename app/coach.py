"""
Rule-based coaching engine with scored metrics, severity-aware feedback,
prioritized exercises, and session summary. No LLM needed.
"""

from typing import List, Dict, Any, Tuple
from app.analyzer import ComparisonResult, SegmentAnalysis


# ---------------------------------------------------------------------------
# Feedback message bank: each metric has mild / moderate / severe variants
# with feedback, why (root cause), and tip (quick fix).
# ---------------------------------------------------------------------------

FEEDBACK_BANK = {
    "pitch_flat": {
        "mild": {
            "feedback": (
                "You're dipping slightly flat here \u2014 about {cents} cents "
                "below the target pitch."
            ),
            "why": (
                "This often happens when breath pressure drops slightly or "
                "when you're approaching the bottom of your comfortable range. "
                "Mild flatness can also come from vowel shapes that darken the tone."
            ),
            "tip": (
                "Think of the note as sitting slightly above where you feel it. "
                "A small mental adjustment often corrects mild flatness. Try "
                "lifting your soft palate (like the start of a yawn) as you sing."
            ),
        },
        "moderate": {
            "feedback": (
                "You're noticeably flat in this section \u2014 {cents} cents "
                "below where you need to be. The pitch center is sagging."
            ),
            "why": (
                "This usually indicates insufficient airflow or a tendency to "
                "'reach down' for notes. Your vocal folds need more consistent "
                "air pressure to maintain pitch. It can also happen when you're "
                "not fully hearing the target note in your head before singing it."
            ),
            "tip": (
                "Practice this phrase on a 'nee' or 'nay' vowel \u2014 these "
                "bright vowels naturally lift pitch. Sing the phrase slowly first, "
                "nail each note, then gradually bring it up to speed."
            ),
        },
        "severe": {
            "feedback": (
                "You're significantly flat here \u2014 {cents} cents off target. "
                "This section needs focused pitch work."
            ),
            "why": (
                "A deviation this large usually means you're either unsure of "
                "the melody, your voice is fatiguing, or you're singing outside "
                "your comfortable range. It could also indicate tension in the "
                "throat pulling the pitch down."
            ),
            "tip": (
                "Isolate this phrase: play the reference, pause, then sing it "
                "back slowly. Use a piano or tuner app to check each note. "
                "Build accuracy at a slow tempo before adding speed. If fatigue "
                "is a factor, take a break and hydrate."
            ),
        },
    },
    "pitch_sharp": {
        "mild": {
            "feedback": (
                "You're sitting slightly sharp here \u2014 about {cents} cents "
                "above the target pitch."
            ),
            "why": (
                "Mild sharpness often comes from excess tension in the throat "
                "or jaw, or from over-supporting with too much air pressure. "
                "It's common on ascending phrases or sustained high notes."
            ),
            "tip": (
                "Relax your jaw and throat \u2014 imagine the sound floating "
                "out rather than being pushed. Drop your shoulders and let the "
                "note settle into place."
            ),
        },
        "moderate": {
            "feedback": (
                "You're noticeably sharp in this section \u2014 {cents} cents "
                "above the target. The pitch is being pushed up."
            ),
            "why": (
                "This typically comes from excess breath pressure or tension. "
                "When you push harder to be louder or more expressive, the pitch "
                "can rise. Emotional intensity can also cause sharpness."
            ),
            "tip": (
                "Try singing this section at a softer dynamic first. Once the "
                "pitch is accurate at low volume, gradually increase intensity "
                "while maintaining the same pitch center. Think 'down and back' "
                "with your support."
            ),
        },
        "severe": {
            "feedback": (
                "You're significantly sharp here \u2014 {cents} cents above "
                "target. This phrase needs careful pitch calibration."
            ),
            "why": (
                "A deviation this large usually means you're straining or the "
                "note sits in an uncomfortable part of your range. Excess "
                "tension in the larynx physically shortens the vocal folds, "
                "raising pitch involuntarily."
            ),
            "tip": (
                "Check if you're straining. If so, try the phrase in a lower "
                "key first, then slowly move it up. Practice on a gentle 'mah' "
                "to release tension. If the note is at the top of your range, "
                "work on mixed voice exercises to ease the transition."
            ),
        },
    },
    "energy_drop": {
        "mild": {
            "feedback": (
                "Your breath support dips slightly mid-phrase here. "
                "The second half loses a bit of energy."
            ),
            "why": (
                "You're likely running out of air toward the end of the phrase. "
                "This is very common and usually means you didn't take a deep "
                "enough breath, or you're releasing too much air at the start."
            ),
            "tip": (
                "Take a low belly breath before this phrase. Think of rationing "
                "your air \u2014 start the phrase at moderate volume so you have "
                "reserves for the end."
            ),
        },
        "moderate": {
            "feedback": (
                "Your breath support is dropping noticeably mid-phrase. "
                "The energy falls off significantly in the second half."
            ),
            "why": (
                "Your diaphragm support is collapsing partway through. This "
                "creates an audible fade that weakens the musical phrase. "
                "It often happens on longer lines or after a series of phrases "
                "without adequate breath recovery."
            ),
            "tip": (
                "Practice this phrase in two halves first, breathing in between. "
                "Then gradually extend into the full phrase. Engage your core "
                "muscles throughout \u2014 imagine a slow, steady stream of air "
                "like blowing through a straw."
            ),
        },
        "severe": {
            "feedback": (
                "Your breath support collapses in this phrase. The energy drops "
                "dramatically, making the second half barely audible."
            ),
            "why": (
                "This level of energy loss indicates a fundamental breath "
                "management issue. You may be using chest-only breathing, "
                "dumping air at the start of the phrase, or this phrase may "
                "simply be too long for your current breath capacity."
            ),
            "tip": (
                "Work on the 'sustained hiss' exercise: breathe in for 4 counts, "
                "hiss out on 'sss' for as long as possible (aim for 20+ seconds). "
                "This builds the core engagement you need. Also try breaking this "
                "phrase into smaller chunks and adding quick catch breaths."
            ),
        },
    },
    "energy_low": {
        "mild": {
            "feedback": (
                "You're a bit quiet in this section compared to the reference."
            ),
            "why": (
                "You may be holding back or not fully committing to the phrase. "
                "It's common when you're focused on pitch accuracy and dialing "
                "back your volume to compensate."
            ),
            "tip": (
                "Open up your mouth more and project forward. Think of singing "
                "to someone across the room rather than to yourself."
            ),
        },
        "moderate": {
            "feedback": (
                "Your volume is noticeably lower than the reference here. "
                "The phrase lacks projection."
            ),
            "why": (
                "This usually means you're either not engaging your diaphragm "
                "fully, or there's tension restricting your airflow. It can also "
                "happen when a phrase sits in an uncomfortable part of your range."
            ),
            "tip": (
                "Stand up straight, feet shoulder-width apart. Take a low breath "
                "and sing this phrase like you're trying to fill the room. "
                "Practice at an exaggerated volume first, then dial it back "
                "to a comfortable level that still projects."
            ),
        },
        "severe": {
            "feedback": (
                "You're significantly under-projecting here \u2014 the volume "
                "is much weaker than the reference needs."
            ),
            "why": (
                "A volume gap this large usually indicates either significant "
                "breath support issues, vocal fatigue, or singing in a register "
                "where your voice doesn't naturally project. There may be "
                "tension closing off the throat."
            ),
            "tip": (
                "Focus on opening your throat \u2014 imagine the space of a "
                "yawn while singing. Try the 'ng' hum (as in 'sing') to find "
                "your natural resonance, then open to the vowel. If your voice "
                "is fatiguing, take a break and come back fresh."
            ),
        },
    },
    "timing_drift": {
        "mild": {
            "feedback": (
                "You're {direction} slightly here \u2014 about {ms}ms "
                "{dir_desc} the beat."
            ),
            "why": (
                "Mild timing drift is very common and often comes from not "
                "internalizing the rhythm before singing. If {direction}, you "
                "may be overthinking the notes."
            ),
            "tip": (
                "Listen to the reference once more, tapping your foot or "
                "clapping on the beat. Then sing while maintaining that "
                "physical pulse. The body rhythm anchors your timing."
            ),
        },
        "moderate": {
            "feedback": (
                "You're {direction} noticeably here \u2014 {ms}ms {dir_desc} "
                "the beat. The phrase is drifting out of sync."
            ),
            "why": (
                "You've lost the rhythmic anchor in this section. If dragging, "
                "you may be lingering on notes or taking too long between words. "
                "If rushing, excitement or nerves may be pushing you ahead."
            ),
            "tip": (
                "Practice speaking the lyrics in rhythm before singing them. "
                "Clap the rhythm, then add pitch. Breaking it down like this "
                "helps lock in the timing before you worry about notes."
            ),
        },
        "severe": {
            "feedback": (
                "You're {direction} significantly \u2014 {ms}ms {dir_desc} "
                "the beat. This section is out of sync with the reference."
            ),
            "why": (
                "A drift this large usually means you've lost the pulse "
                "entirely for this section. This can happen when the melody "
                "is unfamiliar or when a difficult passage causes you to slow "
                "down or speed up unconsciously."
            ),
            "tip": (
                "Loop this section of the reference and listen 3-4 times, "
                "focusing only on the rhythm (ignore pitch). Clap the rhythm, "
                "then hum it, then sing with lyrics. Building timing in layers "
                "is more effective than trying to fix everything at once."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Exercise bank: selected based on weakest scoring categories.
# ---------------------------------------------------------------------------

EXERCISE_BANK = {
    "pitch": [
        {
            "name": "Siren Slides",
            "description": (
                "Slide smoothly from your lowest comfortable note to your highest "
                "on an 'oo' vowel, then back down. Do this 5 times. Focus on a "
                "continuous, connected slide with no breaks or jumps."
            ),
            "why": (
                "Builds pitch range awareness and trains your ear-to-voice "
                "connection. The sliding motion forces your vocal folds to "
                "make micro-adjustments, improving pitch accuracy over time."
            ),
        },
        {
            "name": "Scale Matching",
            "description": (
                "Play a 5-note ascending scale (do-re-mi-fa-sol) on a piano or "
                "app, then sing each note back and hold for 3 seconds. Check "
                "each note with a tuner. Repeat in 3 different keys."
            ),
            "why": (
                "Trains your pitch memory and the ability to lock onto a target "
                "note quickly. Holding the note reveals any drift that needs "
                "correction."
            ),
        },
        {
            "name": "Drone Singing",
            "description": (
                "Play a single sustained note (drone) and sing that same note, "
                "holding it for 10 seconds. Listen for 'beats' \u2014 the wobble "
                "you hear when two notes are close but not quite matched. Adjust "
                "until the beats disappear."
            ),
            "why": (
                "Trains your ear to detect fine pitch differences and your voice "
                "to make precise corrections. This is the foundation of good "
                "intonation."
            ),
        },
    ],
    "energy": [
        {
            "name": "Sustained Hiss",
            "description": (
                "Take a deep belly breath (4 counts in), then hiss out on 'sss' "
                "as steadily and quietly as possible. Time yourself \u2014 aim for "
                "20 seconds, then 30, then 40. Keep the hiss perfectly even."
            ),
            "why": (
                "Builds the breath control and core engagement needed to sustain "
                "phrases without volume drops. The 'sss' provides resistance that "
                "mirrors what happens when singing."
            ),
        },
        {
            "name": "Staccato Ha's",
            "description": (
                "Sing short, sharp 'ha' sounds on each note of a scale. Each "
                "'ha' should be a burst of energy from your diaphragm. Do 3 "
                "scales ascending and descending."
            ),
            "why": (
                "Strengthens the connection between your diaphragm and vocal "
                "onset. Each 'ha' forces a quick, strong breath engagement that "
                "carries over into sustained singing."
            ),
        },
        {
            "name": "Phrase Elongation",
            "description": (
                "Take a short phrase from the song. Sing just the first two "
                "words on one breath. Then three words. Then four. Keep adding "
                "words until you can sing the whole phrase on one breath without "
                "losing volume."
            ),
            "why": (
                "Gradually builds your phrase capacity so you learn to ration "
                "air across longer lines. This is more practical than abstract "
                "breathing exercises because it uses the actual material."
            ),
        },
    ],
    "timing": [
        {
            "name": "Metronome Clapping",
            "description": (
                "Set a metronome to the song's tempo. Clap along for 30 seconds. "
                "Then speak the lyrics in rhythm while clapping. Then sing while "
                "clapping. Each step adds a layer without losing the beat."
            ),
            "why": (
                "Builds an internal clock that you can rely on. The physical act "
                "of clapping anchors your body to the beat, which your voice can "
                "then follow."
            ),
        },
        {
            "name": "Delayed Echo",
            "description": (
                "Play one phrase of the reference. Count one beat of silence. "
                "Then sing the phrase back, matching the exact rhythm. Repeat "
                "for each phrase in the song."
            ),
            "why": (
                "Trains rhythmic memory and the ability to reproduce timing "
                "patterns accurately. The delay forces you to internalize the "
                "rhythm rather than just following along."
            ),
        },
    ],
    "similarity": [
        {
            "name": "Vowel Matching",
            "description": (
                "Listen to a phrase in the reference and identify each vowel "
                "sound. Sing the phrase focusing only on matching those exact "
                "vowel shapes with your mouth. Record yourself and compare."
            ),
            "why": (
                "Vowel shapes are the biggest factor in tone color. Matching "
                "the reference's vowels brings your timbre much closer to the "
                "target sound."
            ),
        },
        {
            "name": "Tone Mirroring",
            "description": (
                "Play the reference at low volume and sing along simultaneously, "
                "trying to blend your voice with theirs until you can barely "
                "distinguish the two. Focus on matching their tone color and "
                "resonance, not just pitch."
            ),
            "why": (
                "Singing along at low volume forces you to listen actively and "
                "match the quality of sound, not just the notes. This trains "
                "your ear for timbral similarity."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# CoachingEngine
# ---------------------------------------------------------------------------

class CoachingEngine:
    """
    Applies rule-based coaching to ComparisonResult.
    Produces scores, summary, prioritized exercises, and detailed feedback.
    """

    PITCH_THRESHOLD_CENTS = 20.0
    TIMING_THRESHOLD_MS = 100.0
    ENERGY_RATIO_LOW = 0.5

    # Severity thresholds
    PITCH_MILD = 40.0
    PITCH_MODERATE = 70.0
    ENERGY_MILD = 0.4
    ENERGY_MODERATE = 0.25
    TIMING_MILD = 200.0
    TIMING_MODERATE = 350.0

    # Score weights
    WEIGHT_SIMILARITY = 0.40
    WEIGHT_PITCH = 0.30
    WEIGHT_ENERGY = 0.20
    WEIGHT_TIMING = 0.10

    def generate_coaching(
        self,
        result: "ComparisonResult",
        lyrics: List[Tuple[float, float, str]] | None = None,
    ) -> Dict[str, Any]:
        """Main entry point. Returns full coaching response dict."""
        scores = self._compute_scores(result)
        feedback = self._generate_feedback(result.segments, lyrics)
        summary = self._generate_summary(scores, result.segments)
        exercises = self._select_exercises(scores)

        return {
            "scores": scores,
            "summary": summary,
            "exercises": exercises,
            "feedback": feedback,
        }

    # --- Scores ---

    def _compute_scores(self, result: "ComparisonResult") -> Dict[str, int]:
        sim = result.similarity_score * 100
        pitch = (sum(result.pitch_scores) / len(result.pitch_scores) * 100) if result.pitch_scores else 50
        energy = (sum(result.energy_scores) / len(result.energy_scores) * 100) if result.energy_scores else 50
        timing = (sum(result.timing_scores) / len(result.timing_scores) * 100) if result.timing_scores else 50

        overall = (
            sim * self.WEIGHT_SIMILARITY
            + pitch * self.WEIGHT_PITCH
            + energy * self.WEIGHT_ENERGY
            + timing * self.WEIGHT_TIMING
        )

        return {
            "overall": round(overall),
            "similarity": round(sim),
            "pitch": round(pitch),
            "energy": round(energy),
            "timing": round(timing),
        }

    # --- Feedback ---

    def _generate_feedback(
        self,
        segments: List[SegmentAnalysis],
        lyrics: List[Tuple[float, float, str]] | None = None,
    ) -> List[Dict[str, Any]]:
        feedback_items = []

        for seg in segments:
            issues = self._evaluate_segment(seg)
            if not issues:
                continue

            lyric = ""
            if lyrics:
                lyric = self._find_lyric(seg.ref_time_start, seg.ref_time_end, lyrics)

            for issue in issues:
                feedback_items.append({
                    "timestamp_start": seg.ref_time_start,
                    "timestamp_end": seg.ref_time_end,
                    "lyric_snippet": lyric,
                    **issue,
                })

        return feedback_items

    def _evaluate_segment(self, seg: SegmentAnalysis) -> List[Dict[str, Any]]:
        issues = []

        # Pitch
        if seg.pitch_delta_cents < -self.PITCH_THRESHOLD_CENTS:
            cents = abs(seg.pitch_delta_cents)
            severity = self._pitch_severity(cents)
            bank = FEEDBACK_BANK["pitch_flat"][severity]
            issues.append({
                "metric": "pitch_flat",
                "severity": severity,
                "delta": f"{seg.pitch_delta_cents:+.1f} cents",
                "feedback": bank["feedback"].format(cents=f"{cents:.0f}"),
                "why": bank["why"],
                "tip": bank["tip"],
            })
        elif seg.pitch_delta_cents > self.PITCH_THRESHOLD_CENTS:
            cents = abs(seg.pitch_delta_cents)
            severity = self._pitch_severity(cents)
            bank = FEEDBACK_BANK["pitch_sharp"][severity]
            issues.append({
                "metric": "pitch_sharp",
                "severity": severity,
                "delta": f"{seg.pitch_delta_cents:+.1f} cents",
                "feedback": bank["feedback"].format(cents=f"{cents:.0f}"),
                "why": bank["why"],
                "tip": bank["tip"],
            })

        # Energy
        if seg.energy_drop_mid_phrase:
            severity = self._energy_severity(seg.energy_ratio)
            bank = FEEDBACK_BANK["energy_drop"][severity]
            issues.append({
                "metric": "energy_drop",
                "severity": severity,
                "delta": f"ratio {seg.energy_ratio:.2f}",
                "feedback": bank["feedback"],
                "why": bank["why"],
                "tip": bank["tip"],
            })
        elif seg.energy_ratio < self.ENERGY_RATIO_LOW and seg.ref_energy_mean > 0.1:
            severity = self._energy_severity(seg.energy_ratio)
            bank = FEEDBACK_BANK["energy_low"][severity]
            issues.append({
                "metric": "energy_low",
                "severity": severity,
                "delta": f"ratio {seg.energy_ratio:.2f}",
                "feedback": bank["feedback"],
                "why": bank["why"],
                "tip": bank["tip"],
            })

        # Timing
        if abs(seg.timing_drift_ms) > self.TIMING_THRESHOLD_MS:
            ms = abs(seg.timing_drift_ms)
            severity = self._timing_severity(ms)
            if seg.timing_drift_ms > 0:
                direction = "dragging"
                dir_desc = "behind"
            else:
                direction = "rushing"
                dir_desc = "ahead of"
            bank = FEEDBACK_BANK["timing_drift"][severity]
            issues.append({
                "metric": "timing_drift",
                "severity": severity,
                "delta": f"{seg.timing_drift_ms:+.0f}ms ({direction})",
                "feedback": bank["feedback"].format(
                    direction=direction, ms=f"{ms:.0f}", dir_desc=dir_desc,
                ),
                "why": bank["why"].format(direction=direction),
                "tip": bank["tip"],
            })

        return issues

    def _pitch_severity(self, cents: float) -> str:
        if cents < self.PITCH_MILD:
            return "mild"
        elif cents < self.PITCH_MODERATE:
            return "moderate"
        return "severe"

    def _energy_severity(self, ratio: float) -> str:
        if ratio > self.ENERGY_MILD:
            return "mild"
        elif ratio > self.ENERGY_MODERATE:
            return "moderate"
        return "severe"

    def _timing_severity(self, ms: float) -> str:
        if ms < self.TIMING_MILD:
            return "mild"
        elif ms < self.TIMING_MODERATE:
            return "moderate"
        return "severe"

    # --- Summary ---

    def _generate_summary(
        self,
        scores: Dict[str, int],
        segments: List[SegmentAnalysis],
    ) -> Dict[str, List[str]]:
        strengths = []
        work_on = []

        categories = [
            ("similarity", "Voice Similarity", "Your voice tone closely matches the reference \u2014 great timbral match."),
            ("pitch", "Pitch Accuracy", "Your pitch accuracy is strong \u2014 notes are well-centered."),
            ("energy", "Breath & Energy", "Good breath support and energy projection throughout."),
            ("timing", "Timing", "Your timing is solid \u2014 you're locking into the rhythm well."),
        ]

        for key, label, strength_msg in categories:
            score = scores[key]
            if score >= 80:
                strengths.append(strength_msg)
            else:
                work_on.append(self._describe_weakness(key, score, segments))

        if not strengths:
            strengths.append("Keep practicing \u2014 every session builds your skill. Focus on the exercises below.")
        if not work_on:
            work_on.append("No major issues \u2014 try a harder song to keep pushing yourself!")

        return {"strengths": strengths, "work_on": work_on}

    def _describe_weakness(
        self, key: str, score: int, segments: List[SegmentAnalysis]
    ) -> str:
        if key == "pitch":
            flat_count = sum(1 for s in segments if s.pitch_delta_cents < -self.PITCH_THRESHOLD_CENTS)
            sharp_count = sum(1 for s in segments if s.pitch_delta_cents > self.PITCH_THRESHOLD_CENTS)
            total = len(segments)
            if flat_count > sharp_count:
                return (
                    f"Pitch accuracy (score: {score}) \u2014 you tend to go flat. "
                    f"{flat_count} of {total} sections were below target pitch. "
                    "Focus on breath support and lifting your soft palate."
                )
            elif sharp_count > flat_count:
                return (
                    f"Pitch accuracy (score: {score}) \u2014 you tend to go sharp. "
                    f"{sharp_count} of {total} sections were above target pitch. "
                    "Focus on relaxing your throat and easing off breath pressure."
                )
            return (
                f"Pitch accuracy (score: {score}) \u2014 pitch wanders both flat "
                "and sharp. Work on ear training and pitch centering exercises."
            )
        elif key == "energy":
            drop_count = sum(1 for s in segments if s.energy_drop_mid_phrase)
            low_count = sum(1 for s in segments if s.energy_ratio < self.ENERGY_RATIO_LOW and s.ref_energy_mean > 0.1)
            if drop_count > low_count:
                return (
                    f"Breath support (score: {score}) \u2014 energy drops mid-phrase "
                    f"in {drop_count} sections. Work on sustaining airflow throughout "
                    "each phrase."
                )
            return (
                f"Energy & projection (score: {score}) \u2014 you're under-projecting "
                f"in {low_count} sections. Open up and engage your diaphragm more."
            )
        elif key == "timing":
            drag_count = sum(1 for s in segments if s.timing_drift_ms > self.TIMING_THRESHOLD_MS)
            rush_count = sum(1 for s in segments if s.timing_drift_ms < -self.TIMING_THRESHOLD_MS)
            if drag_count > rush_count:
                return (
                    f"Timing (score: {score}) \u2014 you tend to drag behind the beat "
                    f"in {drag_count} sections. Work on internalizing the rhythm."
                )
            elif rush_count > drag_count:
                return (
                    f"Timing (score: {score}) \u2014 you tend to rush ahead "
                    f"in {rush_count} sections. Slow down and feel the groove."
                )
            return f"Timing (score: {score}) \u2014 timing is inconsistent. Practice with a metronome."
        else:  # similarity
            return (
                f"Voice similarity (score: {score}) \u2014 your tone doesn't closely "
                "match the reference yet. Work on vowel shapes and tone color."
            )

    # --- Exercises ---

    def _select_exercises(self, scores: Dict[str, int]) -> List[Dict[str, Any]]:
        # Rank categories by score (lowest first)
        category_scores = [
            ("pitch", scores["pitch"]),
            ("energy", scores["energy"]),
            ("timing", scores["timing"]),
            ("similarity", scores["similarity"]),
        ]
        category_scores.sort(key=lambda x: x[1])

        exercises = []
        priority = 1

        # Pick exercises from the 2 weakest categories (only if score < 80)
        categories_used = 0
        for cat, score in category_scores:
            if score >= 80 or categories_used >= 2:
                break
            for ex in EXERCISE_BANK.get(cat, []):
                exercises.append({
                    "priority": priority,
                    "name": ex["name"],
                    "target": cat,
                    "description": ex["description"],
                    "why": ex["why"],
                })
                priority += 1
                if priority > 4:
                    break
            categories_used += 1
            if priority > 4:
                break

        return exercises

    # --- Helpers ---

    def _find_lyric(
        self,
        t_start: float,
        t_end: float,
        lyrics: List[Tuple[float, float, str]],
    ) -> str:
        midpoint = (t_start + t_end) / 2.0
        for lyric_start, lyric_end, text in lyrics:
            if lyric_start <= midpoint < lyric_end:
                return text
        if lyrics:
            closest = min(lyrics, key=lambda l: abs((l[0] + l[1]) / 2 - midpoint))
            return closest[2]
        return ""
