import numpy as np
import pandas as pd

def analyze_audio(path, window_seconds=3):
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("Librosa is unavailable. Install requirements and retry.") from exc
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0: raise ValueError("Audio file is empty.")
    win = max(1, int(window_seconds*sr))
    rows = []
    for start in range(0, len(y), win):
        segment = y[start:start+win]
        if len(segment) < max(256, int(sr*0.2)): continue
        rms = float(np.sqrt(np.mean(segment.astype(float)**2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
        # Relative, file-local thresholds avoid pretending to be a trained classifier.
        rows.append({"time_seconds": round(start/sr,2), "rms": rms, "zero_crossing_rate": zcr,
                     "spectral_centroid_hz": centroid})
    if not rows: raise ValueError("No usable audio windows were found.")
    df = pd.DataFrame(rows)
    noise_floor = float(df["rms"].quantile(.25))
    active_cut = max(0.008, noise_floor*1.6)
    loud_cut = max(0.08, float(df["rms"].quantile(.85)))
    labels = []
    for _, r in df.iterrows():
        if r.rms < active_cut: label = "Silence / low energy"
        elif r.rms >= loud_cut and r.spectral_centroid_hz > 1800: label = "High ambient energy (review)"
        elif r.zero_crossing_rate > 0.08 and r.spectral_centroid_hz > 1000: label = "Speech-like activity (speaker unknown)"
        else: label = "Audible activity (source unknown)"
        labels.append(label)
    df["estimated_class"] = labels
    df["interpretation_note"] = "Heuristic signal label; not a reliable instructor/student classifier"
    return df
