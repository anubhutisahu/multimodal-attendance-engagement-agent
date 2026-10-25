import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json, tempfile, os

from modules.attendance import analyze_video, analyze_image, attendance_from_detections
from modules.audio import analyze_audio
from modules.fusion import fuse_signals
from modules.advisory import generate_advisories
from modules.reporting import build_session_report

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stMetric"] { background: white; padding: 16px; border-radius: 12px; border: 1px solid #e6eaf2; }
    .hero { padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#142b4a,#245b75); color: white; margin-bottom: 1rem; }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #e5eef7; margin: .45rem 0 0 0; }
    .note { background: #fff8e7; padding: 12px 15px; border-radius: 10px; border-left: 4px solid #e4ad38; }
</style>
""", unsafe_allow_html=True)

if "attendance" not in st.session_state: st.session_state.attendance = pd.DataFrame()
if "audio" not in st.session_state: st.session_state.audio = pd.DataFrame()
if "fusion" not in st.session_state: st.session_state.fusion = pd.DataFrame()
if "advisories" not in st.session_state: st.session_state.advisories = []
if "session_meta" not in st.session_state: st.session_state.session_meta = {}

st.markdown("""
<div class="hero">
<h1>🎓 Multimodal Attendance & Engagement Agent</h1>
<p>Privacy-first classroom signals · Local processing · Aggregate teaching support</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="note"><b>Privacy & interpretation:</b> Media is processed within this app process and is not intentionally uploaded to an external analytics service. Do not upload media without appropriate consent. Face detections are anonymous tracking estimates, not identity verification; engagement is a heuristic and must not be used to grade or discipline students.</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Session setup")
    session_name = st.text_input("Session name", "Classroom Session")
    session_minutes = st.number_input("Expected session duration (minutes)", min_value=1, value=50)
    threshold = st.slider("Presence threshold (% of observed session)", 50, 100, 70)
    st.caption("No database required. Results remain in this app session unless downloaded.")
    st.divider()
    st.markdown("**Processing status**")
    st.success("Local processing mode")
    st.caption("Internet access may be needed once to install Python packages.")

tabs = st.tabs(["1 · Presence", "2 · Audio", "3 · Fusion", "4 · Teaching Advisories", "5 · Session Report"])

with tabs[0]:
    st.subheader("Automated presence estimation")
    st.write("Upload a classroom video or image. Video sampling estimates how often anonymous face tracks appear; a single photo cannot establish sustained presence.")
    mode = st.radio("Input type", ["Classroom video", "Classroom photo"], horizontal=True)
    upload = st.file_uploader("Choose media", type=["mp4","mov","avi","mkv","jpg","jpeg","png"] if mode == "Classroom video" else ["jpg","jpeg","png"])
    sample_every = st.slider("Analyze every Nth video frame", 1, 30, 5, help="Higher values run faster but may miss brief appearances.")
    if upload and st.button("Analyze presence", type="primary"):
        suffix = Path(upload.name).suffix or (".mp4" if mode == "Classroom video" else ".jpg")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getvalue()); temp_path = tmp.name
            with st.spinner("Processing locally…"):
                if mode == "Classroom video":
                    result = analyze_video(temp_path, sample_every=sample_every)
                    detections = result["detections"]
                    st.session_state.session_meta["video_duration_seconds"] = result["duration_seconds"]
                    st.session_state.session_meta["frames_sampled"] = result["frames_sampled"]
                else:
                    result = analyze_image(temp_path)
                    detections = result["detections"]
                    st.session_state.session_meta["video_duration_seconds"] = 0
                    st.session_state.session_meta["frames_sampled"] = 1
                st.session_state.attendance = attendance_from_detections(detections, threshold_percent=threshold)
                st.session_state.session_meta["session_name"] = session_name
                st.session_state.session_meta["threshold_percent"] = threshold
                st.session_state.session_meta["created_at"] = datetime.now().isoformat(timespec="seconds")
                st.success(f"Analysis complete. Detected {result['max_faces']} face(s) in the busiest sampled frame.")
                st.caption(result.get("note", ""))
        except Exception as exc:
            st.error(f"Could not process media: {exc}")
            st.info("Try a common MP4/JPG/PNG file and ensure the video codec is supported.")
        finally:
            try: os.unlink(temp_path)
            except Exception: pass
    if not st.session_state.attendance.empty:
        df = st.session_state.attendance
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download attendance CSV", df.to_csv(index=False).encode("utf-8"), "attendance_register.csv", "text/csv")
        st.caption("Track IDs are temporary anonymous tracks, not student identities. To create a named register, an educator must map IDs manually and transparently.")

with tabs[1]:
    st.subheader("Acoustic interaction & noise dynamics")
    st.write("Analyze an audio file in short windows. The labels are signal-based estimates; audio alone cannot reliably identify who is speaking.")
    audio_upload = st.file_uploader( "Upload classroom audio", type=["wav", "mp3", "m4a", "ogg", "flac"], key="audio_upload_widget")

    window = st.slider("Window size (seconds)", 1, 10, 3)
    if audio_upload and st.button("Analyze audio", type="primary"):
        try:
            with st.spinner("Analyzing audio locally…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_upload.name).suffix) as tmp:
                    tmp.write(audio_upload.getvalue()); audio_path = tmp.name
                st.session_state.audio = analyze_audio(audio_path, window_seconds=window)
                st.success(f"Processed {len(st.session_state.audio)} audio windows.")
        except Exception as exc:
            st.error(f"Audio analysis failed: {exc}")
            st.info("WAV is the most reliable format. Some MP3/M4A codecs require additional system libraries.")
        finally:
            try: os.unlink(audio_path)
            except Exception: pass
    if not st.session_state.audio.empty:
        st.dataframe(st.session_state.audio, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(st.session_state.audio["time_seconds"], st.session_state.audio["rms"], marker="o")
        ax.set_title("Audio energy by window"); ax.set_xlabel("Time (seconds)"); ax.set_ylabel("RMS energy")
        ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
        st.download_button("Download audio analysis CSV", st.session_state.audio.to_csv(index=False).encode(), "audio_dynamics.csv", "text/csv")

with tabs[2]:
    st.subheader("Multimodal fusion & estimated engagement")
    st.write("Combines available visual-presence and audio-interaction signals into a transparent heuristic. It is not a validated psychological measure.")
    if st.button("Build engagement timeline", type="primary"):
        st.session_state.fusion = fuse_signals(st.session_state.attendance, st.session_state.audio)
        st.session_state.advisories = generate_advisories(st.session_state.fusion)
    if not st.session_state.fusion.empty:
        df = st.session_state.fusion
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean indicator", f"{df['engagement_index'].mean():.0f}/100")
        c2.metric("Peak indicator", f"{df['engagement_index'].max():.0f}/100")
        c3.metric("Time windows", len(df))
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(df["time_seconds"]/60, df["engagement_index"], marker="o")
        ax.axhline(45, linestyle="--", alpha=.7)
        ax.set_ylim(0,100); ax.set_xlabel("Minutes"); ax.set_ylabel("Estimated index (0–100)")
        ax.set_title("Session signal timeline"); ax.grid(alpha=.25)
        st.pyplot(fig); plt.close(fig)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download fusion CSV", df.to_csv(index=False).encode(), "engagement_timeline.csv", "text/csv")
    else:
        st.info("Run audio analysis and/or presence analysis, then select Build engagement timeline. Missing modalities are handled transparently.")

with tabs[3]:
    st.subheader("Teaching suggestion engine")
    st.write("Rule-based suggestions are designed to support the educator, not automatically evaluate learners.")
    if st.button("Refresh suggestions"):
        st.session_state.advisories = generate_advisories(st.session_state.fusion)
    if st.session_state.advisories:
        for item in st.session_state.advisories:
            st.info(item)
    else:
        st.write("No advisory generated yet. Build the engagement timeline in Tab 3, or review the general prompts below.")
        st.markdown("- If participation is low, try a one-minute think-pair-share or anonymous poll.")
        st.markdown("- If noise is high, pause briefly, restate the activity goal, and check whether the noise is productive discussion.")
        st.markdown("- Offer a stretch or quiet reflection break during a long lecture.")
    st.caption("Thresholds are configurable engineering heuristics, not clinical or educationally validated cut-offs.")

with tabs[4]:
    st.subheader("Session analytics & educator report")
    attendance = st.session_state.attendance
    audio = st.session_state.audio
    fusion = st.session_state.fusion
    a, b, c, d = st.columns(4)
    a.metric("Anonymous tracks", int(len(attendance)) if not attendance.empty else 0)
    b.metric("Audio windows", int(len(audio)) if not audio.empty else 0)
    c.metric("Mean engagement", f"{fusion['engagement_index'].mean():.0f}/100" if not fusion.empty else "—")
    d.metric("Advisories", len(st.session_state.advisories))
    if st.button("Generate session report", type="primary"):
        report = build_session_report(st.session_state.session_meta, attendance, audio, fusion, st.session_state.advisories)
        st.session_state["report_json"] = json.dumps(report, indent=2)
    if "report_json" in st.session_state:
        st.download_button("Download session report (JSON)", st.session_state.report_json.encode(), "session_report.json", "application/json")
        st.json(json.loads(st.session_state.report_json))
    st.markdown("**Responsible interpretation checklist**")
    st.markdown("- Review context before interpreting silence or noise.")
    st.markdown("- Do not infer identity, emotion, learning, or intent from face/audio signals.")
    st.markdown("- Explain the system to participants and follow institutional consent and retention policies.")
