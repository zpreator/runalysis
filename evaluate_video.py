"""
evaluate_video.py  —  Overlay style predictions on a video using the trained model.

Usage:
    uv run python evaluate_video.py runs/20260609_181605
"""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analyze_runs import (
    CONF_THRESHOLD, SMOOTH_WINDOW, MIN_PHASE_MS, LOADING_WINDOW, SIDE, TILT_DEG,
    _apply_tilt, _joint_angle,
)

MODEL_PATH = Path("style_model.pkl")

CLASS_COLORS = {
    "normal":       (100, 220, 100),   # green
    "overstriding": (60,  60,  255),   # red
    "small_steps":  (255, 180, 30),    # blue
    "too_bouncy":   (30,  200, 255),   # yellow
}

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
POSE_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ── Feature extraction (mirrors style_features.py) ────────────────────────────

def extract_stride_predictions(csv_path: Path, model, classes: list, feature_cols: list):
    """
    Returns list of dicts:
        start_frame, end_frame, label, proba (array over classes)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    df["time_s"] = df["timestamp_ms"] / 1000
    tilt_rad = np.radians(TILT_DEG)

    ax_cor, ay_cor = _apply_tilt(
        pd.to_numeric(df[f"{SIDE}_ankle_x"], errors="coerce"),
        pd.to_numeric(df[f"{SIDE}_ankle_y"], errors="coerce"),
        tilt_rad,
    )
    hx_cor, hy_cor = _apply_tilt(
        pd.to_numeric(df["left_hip_x"], errors="coerce"),
        pd.to_numeric(df["left_hip_y"], errors="coerce"),
        tilt_rad,
    )
    kx_cor, ky_cor = _apply_tilt(
        pd.to_numeric(df["left_knee_x"], errors="coerce"),
        pd.to_numeric(df["left_knee_y"], errors="coerce"),
        tilt_rad,
    )

    conf      = pd.to_numeric(df[f"{SIDE}_ankle_conf"], errors="coerce")
    hip_conf  = pd.to_numeric(df["left_hip_conf"],      errors="coerce")
    knee_conf = pd.to_numeric(df["left_knee_conf"],     errors="coerce")

    ay_clean = ay_cor.where(conf     >= CONF_THRESHOLD).interpolate()
    ax_clean = ax_cor.where(conf     >= CONF_THRESHOLD).interpolate()
    hy_clean = hy_cor.where(hip_conf >= CONF_THRESHOLD).interpolate()

    t_s     = df["time_s"]
    xvel    = ax_clean.diff() / t_s.diff()
    xvel_sm = xvel.rolling(SMOOTH_WINDOW, center=True).mean()
    sign    = pd.Series(np.sign(xvel_sm)).fillna(0).astype(int)

    fps = 1000 / df["timestamp_ms"].diff().median()
    peaks, _ = find_peaks(ay_clean, distance=int(fps * 0.3), prominence=20)
    if len(peaks) < 2:
        return []

    contact_sign = int(pd.Series(sign.iloc[peaks]).mode()[0])

    phases, cur_sign, phase_start_ms, phase_start_idx = [], None, None, None
    for idx, (ts, s) in enumerate(zip(df["timestamp_ms"], sign)):
        if s == 0:
            continue
        if s != cur_sign:
            if cur_sign is not None:
                phases.append({
                    "label":       "contact" if cur_sign == contact_sign else "air",
                    "start_ms":    phase_start_ms,
                    "end_ms":      ts,
                    "duration_ms": ts - phase_start_ms,
                    "start_idx":   phase_start_idx,
                    "end_idx":     idx,
                })
            cur_sign, phase_start_ms, phase_start_idx = s, ts, idx

    phases_df = pd.DataFrame(phases)
    if phases_df.empty:
        return []
    phases_df = phases_df[phases_df["duration_ms"] >= MIN_PHASE_MS].reset_index(drop=True)

    ground_y = float(ay_clean.iloc[peaks].median())

    stride_preds = []

    for _, ph_c in phases_df[phases_df["label"] == "contact"].iterrows():
        following = phases_df[(phases_df["label"] == "air") & (phases_df["start_ms"] >= ph_c["end_ms"])]
        if following.empty:
            continue
        ph_a = following.iloc[0]

        contact_ms  = ph_c["duration_ms"]
        air_ms      = ph_a["duration_ms"]
        stride_ms   = contact_ms + air_ms
        c_start_idx = int(ph_c["start_idx"])
        a_start_idx = int(ph_a["start_idx"])
        a_end_idx   = int(ph_a["end_idx"])

        # Knee angle at landing
        knee_angle = np.nan
        if (c_start_idx < len(df) and
                conf.iloc[c_start_idx]      >= CONF_THRESHOLD and
                hip_conf.iloc[c_start_idx]  >= CONF_THRESHOLD and
                knee_conf.iloc[c_start_idx] >= CONF_THRESHOLD):
            p_hip   = np.array([hx_cor.iloc[c_start_idx], hy_cor.iloc[c_start_idx]])
            p_knee  = np.array([kx_cor.iloc[c_start_idx], ky_cor.iloc[c_start_idx]])
            p_ankle = np.array([ax_cor.iloc[c_start_idx], ay_cor.iloc[c_start_idx]])
            knee_angle = _joint_angle(p_hip, p_knee, p_ankle)

        # Ankle vs knee at landing
        ankle_ahead_knee = np.nan
        if (c_start_idx < len(df) and
                conf.iloc[c_start_idx]      >= CONF_THRESHOLD and
                knee_conf.iloc[c_start_idx] >= CONF_THRESHOLD):
            ankle_ahead_knee = (ax_cor.iloc[c_start_idx] - kx_cor.iloc[c_start_idx]) * contact_sign

        # Ankle vs knee at toe-off
        toeoff_ankle_knee = np.nan
        if (a_start_idx < len(df) and
                conf.iloc[a_start_idx]      >= CONF_THRESHOLD and
                knee_conf.iloc[a_start_idx] >= CONF_THRESHOLD):
            toeoff_ankle_knee = (ax_cor.iloc[a_start_idx] - kx_cor.iloc[a_start_idx]) * contact_sign

        # Loading rate
        loading_rate = np.nan
        i0 = max(c_start_idx - LOADING_WINDOW, 0)
        seg_y = ay_clean.iloc[i0:c_start_idx]
        seg_t = t_s.iloc[i0:c_start_idx] * 1000
        if seg_y.notna().sum() > 1 and (seg_t.iloc[-1] - seg_t.iloc[0]) > 0:
            loading_rate = (seg_y.iloc[-1] - seg_y.iloc[0]) / (seg_t.iloc[-1] - seg_t.iloc[0])

        # Flight arc
        flight_arc = np.nan
        seg_air = ay_clean.iloc[a_start_idx:a_end_idx]
        if seg_air.notna().sum() > 1:
            flight_arc = ground_y - seg_air.min()

        # Hip oscillation
        hip_osc = np.nan
        peak_before = peaks[peaks <= c_start_idx]
        peak_after  = peaks[peaks >= a_end_idx]
        if len(peak_before) > 0 and len(peak_after) > 0:
            seg_hip = hy_clean.iloc[peak_before[-1]:peak_after[0]]
            if seg_hip.notna().sum() > 2:
                hip_osc = seg_hip.max() - seg_hip.min()

        feat = {
            "contact_ms":           contact_ms,
            "air_ms":               air_ms,
            "ratio":                contact_ms / air_ms if air_ms > 0 else np.nan,
            "pct_contact":          contact_ms / stride_ms * 100 if stride_ms > 0 else np.nan,
            "knee_angle_deg":       knee_angle,
            "ankle_ahead_knee_px":  ankle_ahead_knee,
            "toeoff_ankle_knee_px": toeoff_ankle_knee,
            "cadence_spm":          60000.0 / stride_ms if stride_ms > 0 else np.nan,
            "loading_rate":         loading_rate,
            "flight_arc_px":        flight_arc,
            "hip_osc_px":           hip_osc,
            "view_enc":             0,   # perpendicular
        }

        row = pd.DataFrame([feat])[feature_cols]
        if row.isna().any(axis=1).iloc[0]:
            continue

        proba = model.predict_proba(row)[0]
        label = classes[proba.argmax()]

        start_frame = int(df["frame"].iloc[c_start_idx])
        end_frame   = int(df["frame"].iloc[min(a_end_idx, len(df) - 1)])

        stride_preds.append({
            "start_frame": start_frame,
            "end_frame":   end_frame,
            "label":       label,
            "proba":       proba,
        })

    return stride_preds


# ── Drawing helpers ────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_skeleton(img, row):
    kpts = {}
    for name in KEYPOINT_NAMES:
        x    = pd.to_numeric(row.get(f"{name}_x",    np.nan), errors="coerce")
        y    = pd.to_numeric(row.get(f"{name}_y",    np.nan), errors="coerce")
        conf = pd.to_numeric(row.get(f"{name}_conf", 0),      errors="coerce")
        if conf >= CONF_THRESHOLD and np.isfinite(x) and np.isfinite(y):
            kpts[name] = (int(x), int(y))

    idx_to_name = KEYPOINT_NAMES
    for a, b in POSE_CONNECTIONS:
        na, nb = idx_to_name[a], idx_to_name[b]
        if na in kpts and nb in kpts:
            cv2.line(img, kpts[na], kpts[nb], (180, 180, 180), 2, cv2.LINE_AA)
    for pt in kpts.values():
        cv2.circle(img, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_prediction(img, label: str, proba: np.ndarray, classes: list):
    h, w = img.shape[:2]
    color = CLASS_COLORS.get(label, (200, 200, 200))

    # Main label banner — tall enough for large text
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, label.upper().replace("_", " "), (24, 100),
                FONT, 4.0, color, 5, cv2.LINE_AA)

    # Probability bars (bottom-right)
    bar_x  = w - 420
    bar_y0 = h - 20 - len(classes) * 50
    for i, (cls, p) in enumerate(zip(classes, proba)):
        y = bar_y0 + i * 50
        bar_w = int(380 * p)
        c = CLASS_COLORS.get(cls, (180, 180, 180))
        cv2.rectangle(img, (bar_x, y), (bar_x + 380, y + 38), (40, 40, 40), -1)
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + 38), c, -1)
        cv2.putText(img, f"{cls.replace('_', ' ')}  {p*100:.0f}%",
                    (bar_x + 6, y + 26), FONT, 0.80, (255, 255, 255), 2, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python evaluate_video.py <run_folder>")
        sys.exit(1)

    run_dir    = Path(sys.argv[1])
    video_path = run_dir / "video.mp4"
    csv_path   = run_dir / "pose_yolo.csv"
    out_path   = Path("output") / "predicted.mp4"
    gif_path   = Path("output") / "predicted.gif"
    out_path.parent.mkdir(exist_ok=True)

    if not video_path.exists():
        sys.exit(f"Missing {video_path}")
    if not csv_path.exists():
        sys.exit(f"Missing {csv_path} — run batch_process.py first")

    pkg = joblib.load(MODEL_PATH)
    model, classes, feature_cols = pkg["model"], list(pkg["classes"]), pkg["features"]
    print(f"Model loaded — classes: {classes}")

    print("Extracting strides and classifying…")
    stride_preds = extract_stride_predictions(csv_path, model, classes, feature_cols)
    print(f"  {len(stride_preds)} strides classified")

    # Build frame → prediction lookup
    frame_pred: dict[int, tuple] = {}
    for s in stride_preds:
        for f in range(s["start_frame"], s["end_frame"] + 1):
            frame_pred[f] = (s["label"], s["proba"])

    # Print summary
    from collections import Counter
    counts = Counter(s["label"] for s in stride_preds)
    print("  Predicted distribution:", dict(counts))

    # Render annotated video
    pose_df = pd.read_csv(csv_path).set_index("frame")

    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cam_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    GIF_W, GIF_H = 480, 270
    GIF_FPS      = 8
    gif_every    = max(1, round(fps / GIF_FPS))   # subsample factor

    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"),
                          fps, (cam_w, cam_h))

    current_label = None
    current_proba = np.ones(len(classes)) / len(classes)
    frame_no      = 0
    gif_frames    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no in frame_pred:
            current_label, current_proba = frame_pred[frame_no]

        if frame_no in pose_df.index:
            draw_skeleton(frame, pose_df.loc[frame_no])

        if current_label is not None:
            draw_prediction(frame, current_label, current_proba, classes)

        out.write(frame)

        if frame_no % gif_every == 0:
            gif_frames.append(cv2.cvtColor(
                cv2.resize(frame, (GIF_W, GIF_H)), cv2.COLOR_BGR2RGB))

        frame_no += 1
        if frame_no % 100 == 0:
            print(f"  {frame_no}/{n_frames} frames", end="\r")

    cap.release()
    out.release()
    print(f"\nSaved MP4: {out_path}")

    import imageio
    imageio.mimsave(str(gif_path), gif_frames, fps=GIF_FPS,
                    loop=0, quantizer="nq", palettesize=256)
    print(f"Saved GIF: {gif_path}  ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
