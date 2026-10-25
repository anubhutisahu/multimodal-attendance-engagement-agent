
from datetime import datetime


def build_session_report(meta, attendance, audio, fusion, advisories):
    attendance_empty = (
        attendance is None or attendance.empty
    )
    audio_empty = audio is None or audio.empty
    fusion_empty = fusion is None or fusion.empty

    return {
        "report_generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "session": meta if meta is not None else {},
        "metrics": {
            "anonymous_tracks": (
                0 if attendance_empty else int(len(attendance))
            ),
            "audio_windows": (
                0 if audio_empty else int(len(audio))
            ),
            "fusion_windows": (
                0 if fusion_empty else int(len(fusion))
            ),
            "mean_estimated_engagement_index": (
                None if fusion_empty else round(
                    float(fusion["engagement_index"].mean()), 1
                )
            ),
            "peak_estimated_engagement_index": (
                None if fusion_empty else round(
                    float(fusion["engagement_index"].max()), 1
                )
            ),
        },
        "advisories": advisories if advisories is not None else [],
        "limitations": [
            "Face tracks are anonymous and may switch IDs.",
            "Still images cannot establish sustained attendance.",
            "Audio heuristics cannot reliably identify speaker roles.",
            "The engagement index is not a validated measure of attention or learning.",
            "Review results in context and follow consent and privacy policies.",
        ],
    }
