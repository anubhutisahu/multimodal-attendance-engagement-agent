import cv2
import pandas as pd
from collections import defaultdict

def _detector():
    # MediaPipe is preferred when installed; Haar cascade provides a local fallback.
    try:
        import mediapipe as mp
        return ("mediapipe", mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5))
    except Exception:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("Neither MediaPipe face detection nor OpenCV Haar cascade is available.")
        return ("haar", cascade)

def _detect(frame, detector):
    kind, model = detector
    h, w = frame.shape[:2]
    if kind == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model.process(rgb)
        boxes = []
        if result.detections:
            for det in result.detections:
                bb = det.location_data.relative_bounding_box
                x, y = int(bb.xmin*w), int(bb.ymin*h)
                boxes.append((max(0,x), max(0,y), max(1,int(bb.width*w)), max(1,int(bb.height*h))))
        return boxes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    return [(int(x),int(y),int(w),int(h)) for x,y,w,h in faces]

def analyze_video(path, sample_every=5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Video could not be opened.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = float(frame_count/fps) if frame_count else 0.0
    detector = _detector()
    detections, max_faces, sampled, frame_idx = [], 0, 0, 0
    # Lightweight centroid association: useful for demos, not robust identity tracking.
    active, next_id = {}, 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % max(1, sample_every) == 0:
                boxes = _detect(frame, detector)
                sampled += 1
                max_faces = max(max_faces, len(boxes))
                centers = [(x+w/2,y+h/2) for x,y,w,h in boxes]
                used = set()
                new_active = {}
                for center in centers:
                    best_id, best_dist = None, 90
                    for tid, old in active.items():
                        if tid in used: continue
                        dist = ((center[0]-old[0])**2+(center[1]-old[1])**2)**0.5
                        if dist < best_dist: best_id, best_dist = tid, dist
                    if best_id is None:
                        best_id = next_id; next_id += 1
                    used.add(best_id); new_active[best_id] = center
                    detections.append({"track_id": best_id, "time_seconds": frame_idx/fps})
                active = new_active
            frame_idx += 1
    finally:
        cap.release()
        if detector[0] == "mediapipe":
            try: detector[1].close()
            except Exception: pass
    return {"detections": detections, "max_faces": max_faces, "frames_sampled": sampled,
            "duration_seconds": duration,
            "note": "Anonymous centroid tracking can switch IDs when people cross, leave, or occlude one another. Validate manually before using any register."}

def analyze_image(path):
    image = cv2.imread(path)
    if image is None: raise ValueError("Image could not be read.")
    detector = _detector()
    boxes = _detect(image, detector)
    detections = [{"track_id": i+1, "time_seconds": 0.0} for i in range(len(boxes))]
    if detector[0] == "mediapipe":
        try: detector[1].close()
        except Exception: pass
    return {"detections": detections, "max_faces": len(boxes), "frames_sampled": 1,
            "note": "A still image cannot estimate sustained presence or attendance duration."}

def attendance_from_detections(detections, threshold_percent=70):
    if not detections:
        return pd.DataFrame(columns=["Student ID","First Seen","Last Seen","Total Duration (seconds)","Status"])
    # The current tracker emits observations only, not continuous occupancy. Use sampled timestamps as a conservative proxy.
    by_track = defaultdict(list)
    for d in detections: by_track[d["track_id"]].append(float(d["time_seconds"]))
    all_times = [float(d["time_seconds"]) for d in detections]
    observed_start, observed_end = min(all_times), max(all_times)
    duration = max(1.0, observed_end-observed_start)
    rows = []
    for tid, times in sorted(by_track.items()):
        first, last = min(times), max(times)
        span = max(0.0, last-first)
        ratio = (span/duration*100) if duration > 0 else 0
        rows.append({"Student ID": f"Track-{tid:03d}", "First Seen": _clock(first),
                     "Last Seen": _clock(last), "Total Duration (seconds)": round(span,1),
                     "Status": "Present" if ratio >= threshold_percent else "Review"})
    return pd.DataFrame(rows)

def _clock(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
