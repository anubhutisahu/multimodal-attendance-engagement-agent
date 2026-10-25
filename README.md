# Multimodal Attendance & Engagement Agent

A privacy-first Streamlit capstone prototype for local classroom media analysis.

## Features
- Anonymous face detection with MediaPipe when available and OpenCV Haar fallback.
- Sampled video/photo processing and CSV attendance-style register.
- Audio window features (RMS, zero-crossing rate, spectral centroid) and heuristic signal labels.
- Exploratory multimodal timeline and rule-based teaching advisories.
- JSON session report and CSV downloads.

## Setup
Recommended: Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit (usually http://localhost:8501).

## Privacy and limitations
- Media is processed in the running app; the code does not intentionally transmit classroom media to a remote analytics API. Hosting the app on a cloud platform means the uploaded file reaches that host, so **Streamlit Community Cloud is not a fully offline deployment**. For strict offline processing, run locally on the classroom computer and disconnect the network.
- Face tracks are anonymous and are not reliable student identities. Track IDs can switch when faces overlap or leave the frame.
- A still image cannot prove sustained attendance. Video-based duration is approximate and should be manually reviewed.
- Audio labels are simple signal heuristics, not a trained VAD or speaker-role classifier. They cannot reliably tell instructor speech from student discussion or ambient noise.
- The engagement index is an illustrative heuristic, not a validated measure of attention, learning, emotion, or teaching quality. Do not use it for grading, discipline, or high-stakes decisions.
- Obtain appropriate consent, minimize retention, and follow institutional policy and applicable law.

## Implementation notes
This is a modular working prototype, not a validated production surveillance or attendance system. For production, replace the centroid tracker with a tested tracking algorithm, use a validated VAD and carefully evaluated audio classifier, implement true timestamped occupancy estimates, add unit/integration tests, and conduct privacy/security review.
