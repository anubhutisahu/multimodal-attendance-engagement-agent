import pandas as pd
import numpy as np

def fuse_signals(attendance, audio):
    # Align to audio windows when available; otherwise create coarse windows from attendance observations.
    if not audio.empty:
        out = audio[["time_seconds","estimated_class","rms"]].copy()
        activity = out["estimated_class"].apply(lambda x: 0.15 if "Silence" in x else 0.75 if "Speech-like" in x else 0.5)
        # Presence is only a coarse session-level estimate, not a measure of visual attention.
        presence = min(1.0, len(attendance)/8) if not attendance.empty else 0.5
        out["visual_presence_proxy"] = round(presence*100,1)
        out["audio_interaction_proxy"] = (activity*100).round(1)
        out["engagement_index"] = np.clip((0.55*presence + 0.45*activity)*100, 0, 100).round(1)
        out["available_modalities"] = "audio + coarse presence proxy" if not attendance.empty else "audio only"
    elif not attendance.empty:
        out = pd.DataFrame({"time_seconds":[0], "estimated_class":["Audio unavailable"], "rms":[0.0],
                            "visual_presence_proxy":[min(100, len(attendance)/8*100)],
                            "audio_interaction_proxy":[50.0],
                            "engagement_index":[round(min(100, len(attendance)/8*100)*0.55+22.5,1)],
                            "available_modalities":["presence proxy only; audio unavailable"]})
    else:
        return pd.DataFrame(columns=["time_seconds","estimated_class","rms","visual_presence_proxy","audio_interaction_proxy","engagement_index","available_modalities"])
    out["interpretation_note"] = "Exploratory heuristic only; not a validated measure of attention, learning, or emotion"
    return out
