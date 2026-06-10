"""
visualize_features.py  —  Generate annotated MP4s/GIFs for portfolio use.

Outputs:
  feature_annotation.mp4 / .gif  — Normal run: hip oscillation panel + ankle-vs-knee fade
  style_comparison.mp4   / .gif  — Landing frame for each style, side by side

Usage:
    uv run python visualize_features.py
"""

from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analyze_runs import (
    CONF_THRESHOLD, SMOOTH_WINDOW, MIN_PHASE_MS, SIDE, TILT_DEG,
    _apply_tilt,
)
from style_features import STYLE_RUNS

RUNS_DIR = Path("runs")
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# ── Style → run folder lookup ──────────────────────────────────────────────────

def runs_for_style(style, view="perpendicular"):
    return [f for f, m in STYLE_RUNS.items() if m["style"] == style and m["view"] == view]


# ── Core pose / phase helpers ──────────────────────────────────────────────────

def load_pose(csv_path):
    df = pd.read_csv(csv_path)
    tilt_rad = np.radians(TILT_DEG)
    ax_c, ay_c = _apply_tilt(
        pd.to_numeric(df[f"{SIDE}_ankle_x"], errors="coerce"),
        pd.to_numeric(df[f"{SIDE}_ankle_y"], errors="coerce"),
        tilt_rad,
    )
    hx_c, hy_c = _apply_tilt(
        pd.to_numeric(df["left_hip_x"],  errors="coerce"),
        pd.to_numeric(df["left_hip_y"],  errors="coerce"),
        tilt_rad,
    )
    kx_c, ky_c = _apply_tilt(
        pd.to_numeric(df["left_knee_x"], errors="coerce"),
        pd.to_numeric(df["left_knee_y"], errors="coerce"),
        tilt_rad,
    )
    conf  = pd.to_numeric(df[f"{SIDE}_ankle_conf"], errors="coerce")
    hconf = pd.to_numeric(df["left_hip_conf"],      errors="coerce")
    kconf = pd.to_numeric(df["left_knee_conf"],     errors="coerce")
    return df, ax_c, ay_c, hx_c, hy_c, kx_c, ky_c, conf, hconf, kconf


def detect_contact_events(df, ax_c, ay_c, conf):
    ay_clean = ay_c.where(conf >= CONF_THRESHOLD).interpolate()
    t_s      = df["timestamp_ms"] / 1000
    xvel     = ax_c.where(conf >= CONF_THRESHOLD).interpolate().diff() / t_s.diff()
    xvel_sm  = xvel.rolling(SMOOTH_WINDOW, center=True).mean()
    sign     = pd.Series(np.sign(xvel_sm)).fillna(0).astype(int)

    fps = 1000 / df["timestamp_ms"].diff().median()
    peaks, _ = find_peaks(ay_clean, distance=int(fps * 0.3), prominence=20)
    if len(peaks) < 2:
        return pd.DataFrame(), peaks, None, ay_clean

    contact_sign = int(pd.Series(sign.iloc[peaks]).mode()[0])
    phases, cur_sign, phase_start_ms, phase_start_idx = [], None, None, None
    for idx, (ts, s) in enumerate(zip(df["timestamp_ms"], sign)):
        if s == 0:
            continue
        if s != cur_sign:
            if cur_sign is not None:
                phases.append({
                    "label":       "contact" if cur_sign == contact_sign else "air",
                    "start_ms":    phase_start_ms, "end_ms": ts,
                    "duration_ms": ts - phase_start_ms,
                    "start_idx":   phase_start_idx, "end_idx": idx,
                })
            cur_sign, phase_start_ms, phase_start_idx = s, ts, idx

    phases_df = pd.DataFrame(phases)
    if phases_df.empty:
        return phases_df, peaks, contact_sign, ay_clean
    phases_df = phases_df[phases_df["duration_ms"] >= MIN_PHASE_MS].reset_index(drop=True)
    return phases_df, peaks, contact_sign, ay_clean


def get_kpt(pose_row, name):
    x    = float(pose_row.get(f"{name}_x",    np.nan) or np.nan)
    y    = float(pose_row.get(f"{name}_y",    np.nan) or np.nan)
    c    = float(pose_row.get(f"{name}_conf", 0)      or 0)
    return (int(x), int(y)) if c >= CONF_THRESHOLD and np.isfinite(x) and np.isfinite(y) else None


SKEL_PAIRS = [
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
]


# ── Drawing primitives ────────────────────────────────────────────────────────

def _blend(img, overlay, alpha):
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _bar(img, y0, h, alpha=0.85, color=(20, 22, 30)):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y0 + h), color, -1)
    _blend(img, ov, alpha)


def _pill(img, text, cx, cy, fg, bg, alpha=1.0,
          px=14, py=7, scale=0.60, thick=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x0, y0 = cx - tw // 2 - px, cy - th // 2 - py
    x1, y1 = cx + tw // 2 + px, cy + th // 2 + py
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), bg, -1)
    _blend(img, ov, alpha * 0.85)
    cv2.rectangle(img, (x0, y0), (x1, y1), fg, 1, cv2.LINE_AA)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                FONT, scale, fg, thick, cv2.LINE_AA)


def _glow_line(img, p1, p2, color, alpha=1.0):
    """Line with soft radial glow — three independent blended passes."""
    for thick, strength in [(14, 0.10), (7, 0.22), (2, 0.90)]:
        ov = img.copy()
        cv2.line(ov, p1, p2, color, thick, cv2.LINE_AA)
        _blend(img, ov, strength * alpha)


def _tick_ends(img, x, meas_y, color, alpha=1.0, half=10, thick=2):
    ov = img.copy()
    cv2.line(ov, (x, meas_y - half), (x, meas_y + half), color, thick, cv2.LINE_AA)
    cv2.circle(ov, (x, meas_y), 5, color, -1, cv2.LINE_AA)
    _blend(img, ov, alpha)


# ── GIF 1: Feature annotation on a normal run ─────────────────────────────────

# Output dimensions
GIF_W, GIF_H  = 900, 506
PANEL_W       = 100      # right-side oscillation panel
CROP_TOP_FRAC = 0.0      # no crop — show full frame so hip is visible

# Design tokens (BGR)
C_TEAL   = (195, 215, 0)
C_ORANGE = (0,   140, 255)  # orange — knee highlight
C_GREEN  = (90,  215, 75)
C_RED    = (55,  70,  250)
C_WHITE  = (240, 240, 240)
C_DIM    = (140, 140, 140)
DARK_BG  = (18,  20,  28)

ANKLE_HOLD = 18
ANKLE_FADE = 14
ANKLE_TOTAL = ANKLE_HOLD + ANKLE_FADE


def make_feature_gif(style="normal", n_strides=2, gif_fps=12,
                     out_path=None):
    if out_path is None:
        out_path = Path("output") / f"feature_{style}.gif"
    folder   = runs_for_style(style)[0]
    csv_path = RUNS_DIR / folder / "pose_yolo.csv"
    vid_path = RUNS_DIR / folder / "video.mp4"
    print(f"Feature annotation  ←  {folder}")

    df, ax_c, ay_c, hx_c, hy_c, kx_c, ky_c, conf, hconf, kconf = load_pose(csv_path)
    phases_df, peaks, contact_sign, _ = detect_contact_events(df, ax_c, ay_c, conf)

    contacts = phases_df[phases_df["label"] == "contact"].reset_index(drop=True)
    airs     = phases_df[phases_df["label"] == "air"].reset_index(drop=True)
    mid      = len(contacts) // 2
    selected = contacts.iloc[max(0, mid - n_strides // 2) :
                             mid + n_strides // 2 + n_strides % 2]

    first_idx = int(selected.iloc[0]["start_idx"])
    last_idx  = int(selected.iloc[-1]["end_idx"])
    for _, ph_a in airs.iterrows():
        if ph_a["end_idx"] >= last_idx:
            last_idx = int(ph_a["end_idx"])
            break

    # Per-stride hip oscillation value and range (in source-pixel Y coords)
    hy_clean   = hy_c.where(hconf >= CONF_THRESHOLD).interpolate()
    stride_osc = {}   # c_start → (min_hy_src, max_hy_src, osc_px)
    stride_ranges = []
    for _, ph_c in selected.iterrows():
        c_start  = int(ph_c["start_idx"])
        foll     = airs[airs["start_ms"] >= ph_c["end_ms"]]
        if foll.empty:
            continue
        a_end = int(foll.iloc[0]["end_idx"])
        stride_ranges.append((c_start, a_end))
        pb = peaks[peaks <= c_start]
        pa = peaks[peaks >= a_end]
        if len(pb) > 0 and len(pa) > 0:
            seg = hy_clean.iloc[pb[-1]:pa[0]]
            if seg.notna().sum() > 2:
                stride_osc[c_start] = (float(seg.min()), float(seg.max()),
                                       float(seg.max() - seg.min()))

    # Global hip Y range for stable panel scaling
    hy_global_min = float(hy_clean.iloc[first_idx:last_idx+1].min())
    hy_global_max = float(hy_clean.iloc[first_idx:last_idx+1].max())

    # Landing frames → ankle-vs-knee values (source coords stored as floats)
    landing: dict[int, dict] = {}
    for _, ph_c in selected.iterrows():
        c_start = int(ph_c["start_idx"])
        if conf.iloc[c_start] >= CONF_THRESHOLD and kconf.iloc[c_start] >= CONF_THRESHOLD:
            val = (ax_c.iloc[c_start] - kx_c.iloc[c_start]) * contact_sign
            landing[int(df["frame"].iloc[c_start])] = {
                "val": float(val),
                "ax": float(ax_c.iloc[c_start]), "ay": float(ay_c.iloc[c_start]),
                "kx": float(kx_c.iloc[c_start]), "ky": float(ky_c.iloc[c_start]),
            }

    # Coordinate mapping: source frame → output canvas (video portion only)
    cap      = cv2.VideoCapture(str(vid_path))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_y   = int(src_h * CROP_TOP_FRAC)
    vis_h    = src_h - crop_y
    vid_w    = GIF_W - PANEL_W
    sx       = vid_w / src_w
    sy       = GIF_H / vis_h

    def tc(x_src, y_src):
        """Transform source coords → canvas coords."""
        return (int(x_src * sx), int((y_src - crop_y) * sy))

    def hy_to_panel(hy_val):
        """Map hip Y source value → panel pixel Y (higher hy = lower on panel)."""
        t = (hy_val - hy_global_min) / max(1.0, hy_global_max - hy_global_min)
        return int(60 + t * (GIF_H - 110))

    frames_out  = []
    hip_trail   = []          # canvas coords
    hy_history  = []          # source Y values for panel waveform
    TRAIL_LEN   = 22
    HY_HIST     = 60

    ankle_timer = 0
    ankle_state = None
    cur_osc     = None

    for pose_idx in range(first_idx, last_idx + 1):
        frame_no = int(df["frame"].iloc[pose_idx])
        pose_row = df.iloc[pose_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, raw = cap.read()
        if not ok:
            continue

        # ── Crop + scale video portion ────────────────────────────────────────
        canvas = np.zeros((GIF_H, GIF_W, 3), dtype=np.uint8)
        canvas[:, :vid_w] = cv2.resize(raw[crop_y:, :], (vid_w, GIF_H))

        # ── Active stride ─────────────────────────────────────────────────────
        for c_s, a_e in stride_ranges:
            if c_s <= pose_idx <= a_e and c_s in stride_osc:
                cur_osc = stride_osc[c_s]
                break

        # ── Skeleton ──────────────────────────────────────────────────────────
        for na, nb in SKEL_PAIRS:
            ca = float(pose_row.get(f"{na}_conf", 0) or 0)
            cb = float(pose_row.get(f"{nb}_conf", 0) or 0)
            if ca >= CONF_THRESHOLD and cb >= CONF_THRESHOLD:
                try:
                    pa = tc(float(pose_row[f"{na}_x"]), float(pose_row[f"{na}_y"]))
                    pb = tc(float(pose_row[f"{nb}_x"]), float(pose_row[f"{nb}_y"]))
                    if 0 <= pa[0] < vid_w and 0 <= pb[0] < vid_w:
                        cv2.line(canvas, pa, pb, (170, 170, 170), 2, cv2.LINE_AA)
                except (ValueError, TypeError, KeyError):
                    pass

        for jn in ["left_hip", "right_hip", "left_knee", "right_knee",
                   "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]:
            cj = float(pose_row.get(f"{jn}_conf", 0) or 0)
            if cj >= CONF_THRESHOLD:
                try:
                    pt = tc(float(pose_row[f"{jn}_x"]), float(pose_row[f"{jn}_y"]))
                    if 0 <= pt[0] < vid_w:
                        cv2.circle(canvas, pt, 5, (215, 215, 215), -1, cv2.LINE_AA)
                except (ValueError, TypeError, KeyError):
                    pass

        # ── Highlight tracked knee and ankle ──────────────────────────────────
        for jn, color in [("left_knee", C_ORANGE), ("left_ankle", C_GREEN)]:
            cj = float(pose_row.get(f"{jn}_conf", 0) or 0)
            if cj >= CONF_THRESHOLD:
                try:
                    pt = tc(float(pose_row[f"{jn}_x"]), float(pose_row[f"{jn}_y"]))
                    if 0 <= pt[0] < vid_w:
                        cv2.circle(canvas, pt, 10, color,   -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, 10, C_WHITE,  1, cv2.LINE_AA)
                except (ValueError, TypeError, KeyError):
                    pass

        # ── Hip trail ────────────────────────────────────────────────────────
        hx_r = pose_row.get("left_hip_x", np.nan)
        hy_r = pose_row.get("left_hip_y", np.nan)
        hc_r = float(pose_row.get("left_hip_conf", 0) or 0)
        if hc_r >= CONF_THRESHOLD and np.isfinite(float(hx_r or np.nan)):
            hp  = tc(float(hx_r), float(hy_r))
            hip_trail.append(hp)
            hy_history.append(float(hy_r))
        if len(hip_trail) > TRAIL_LEN:
            hip_trail.pop(0)
        if len(hy_history) > HY_HIST:
            hy_history.pop(0)

        for j in range(1, len(hip_trail)):
            t   = j / len(hip_trail)
            col = (int(195 * t), int(215 * t), 0)   # teal gradient
            cv2.line(canvas, hip_trail[j-1], hip_trail[j], col, 3, cv2.LINE_AA)
        if hip_trail:
            cv2.circle(canvas, hip_trail[-1], 10, C_TEAL,  -1, cv2.LINE_AA)
            cv2.circle(canvas, hip_trail[-1], 10, C_WHITE,  1, cv2.LINE_AA)

        # ── Oscillation panel ─────────────────────────────────────────────────
        px0 = vid_w
        ov  = canvas.copy()
        cv2.rectangle(ov, (px0, 0), (GIF_W, GIF_H), DARK_BG, -1)
        _blend(canvas, ov, 0.90)
        cv2.line(canvas, (px0, 0), (px0, GIF_H), (48, 52, 62), 1)

        # Label
        cv2.putText(canvas, "VERTICAL", (px0 + 6, 22),
                    FONT, 0.42, C_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "OSC.",     (px0 + 6, 38),
                    FONT, 0.42, C_DIM, 1, cv2.LINE_AA)

        # Waveform
        if len(hy_history) > 1:
            pts = [
                (px0 + 6 + int(i * (PANEL_W - 14) / HY_HIST),
                 hy_to_panel(v))
                for i, v in enumerate(hy_history)
            ]
            for j in range(1, len(pts)):
                t   = j / len(pts)
                col = (int(195 * t), int(215 * t), 0)
                cv2.line(canvas, pts[j-1], pts[j], col, 2, cv2.LINE_AA)
            # Current dot
            cv2.circle(canvas, pts[-1], 5, C_TEAL, -1, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 5, C_WHITE, 1, cv2.LINE_AA)

        # Bracket + value
        if cur_osc is not None and len(hy_history) > 4:
            _, _, osc_px = cur_osc
            win  = hy_history[-min(len(hy_history), 40):]
            y_t  = hy_to_panel(min(win))
            y_b  = hy_to_panel(max(win))
            bx   = GIF_W - 12
            osc_color = C_RED if osc_px > 100 else C_TEAL
            cv2.line(canvas, (bx, y_t), (bx, y_b), osc_color, 2, cv2.LINE_AA)
            for y in (y_t, y_b):
                cv2.line(canvas, (bx - 7, y), (bx + 2, y), osc_color, 2, cv2.LINE_AA)
            label_y = (y_t + y_b) // 2
            cv2.putText(canvas, f"{osc_px:.0f}px",
                        (px0 + 5, label_y + 5), FONT, 0.52, osc_color, 1, cv2.LINE_AA)

        # ── Ankle measurement fade ────────────────────────────────────────────
        if frame_no in landing:
            ankle_timer = ANKLE_TOTAL
            ankle_state = landing[frame_no]

        if ankle_timer > 0 and ankle_state is not None:
            t     = ankle_timer / ANKLE_TOTAL
            alpha = min(1.0, t * ANKLE_TOTAL / ANKLE_HOLD)
            val   = ankle_state["val"]
            ak_c  = tc(ankle_state["ax"], ankle_state["ay"])
            kk_c  = tc(ankle_state["kx"], ankle_state["ky"])
            color = C_RED if abs(val) > 30 else C_GREEN
            my    = kk_c[1]

            # Draw at knee height; also extend line 30px beyond each endpoint
            # so the measurement is visible even when gap is small
            x_left  = min(kk_c[0], ak_c[0]) - 30
            x_right = max(kk_c[0], ak_c[0]) + 30
            _glow_line(canvas, (x_left, my), (x_right, my), color, alpha)
            _tick_ends(canvas, kk_c[0], my, color, alpha, half=14, thick=3)
            _tick_ends(canvas, ak_c[0], my, color, alpha, half=14, thick=3)

            cx = (kk_c[0] + ak_c[0]) // 2
            sign_str = "OVERSTRIDE" if val > 30 else "normal"
            _pill(canvas, f"ANKLE vs KNEE  {val:+.0f}px  {sign_str}",
                  cx, my - 30, color, DARK_BG, alpha=alpha, scale=0.58)
            ankle_timer -= 1

        # ── Footer bar ────────────────────────────────────────────────────────
        bar_y = GIF_H - 38
        _bar(canvas, bar_y, 38, alpha=1.0)
        cv2.line(canvas, (0, bar_y), (GIF_W, bar_y), C_TEAL, 1)
        cv2.putText(canvas, "GAIT FEATURE EXTRACTION", (14, bar_y + 26),
                    FONT, 0.65, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "hip oscillation  |  ankle vs knee",
                    (vid_w - 295, bar_y + 26), FONT, 0.50, C_DIM, 1, cv2.LINE_AA)

        frames_out.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    cap.release()

    # Save GIF + MP4
    imageio.mimsave(str(out_path), frames_out, fps=gif_fps,
                    loop=0, quantizer="nq", palettesize=256)
    print(f"  Saved {out_path}  ({len(frames_out)} frames @ {gif_fps}fps)")

    mp4_path = out_path.with_suffix(".mp4")
    h_, w_   = frames_out[0].shape[:2]
    wr = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter.fourcc(*"mp4v"),
                         gif_fps, (w_, h_))
    for f in frames_out:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"  Saved {mp4_path}")

    return frames_out


# ── GIF 3: All 4 styles in a 2×2 grid ────────────────────────────────────────

CELL_W, CELL_H = 640, 360

STYLE_COLORS = {
    "normal":       (100, 220, 100),
    "overstriding": (60,  60,  255),
    "small_steps":  (255, 180, 30),
    "too_bouncy":   (30,  200, 255),
}
STYLE_LABELS = {
    "normal":       "NORMAL",
    "overstriding": "OVERSTRIDING",
    "small_steps":  "SMALL STEPS",
    "too_bouncy":   "TOO BOUNCY",
}


def make_all_styles_gif(n_strides=2, gif_fps=12,
                        out_path=Path("output/all_styles.gif")):
    styles = ["normal", "overstriding", "small_steps", "too_bouncy"]

    # Render frames for each style (also saves individual files)
    all_frames = {}
    for style in styles:
        frames = make_feature_gif(style, n_strides=n_strides, gif_fps=gif_fps)
        all_frames[style] = frames

    max_frames = max(len(f) for f in all_frames.values())

    combined = []
    for i in range(max_frames):
        cells = []
        for style in styles:
            frames = all_frames[style]
            # Loop shorter clips
            frame_bgr = cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_RGB2BGR)
            cell = cv2.resize(frame_bgr, (CELL_W, CELL_H))

            # Replace bottom bar text with style name
            color   = STYLE_COLORS[style]
            bar_y_c = int((GIF_H - 38) * CELL_H / GIF_H)
            cv2.rectangle(cell, (0, bar_y_c), (CELL_W, CELL_H), DARK_BG, -1)
            cv2.line(cell, (0, bar_y_c), (CELL_W, bar_y_c), color, 1)
            cv2.putText(cell, STYLE_LABELS[style], (10, bar_y_c + 24),
                        FONT, 0.75, color, 2, cv2.LINE_AA)

            cells.append(cell)

        top    = np.concatenate([cells[0], cells[1]], axis=1)
        bottom = np.concatenate([cells[2], cells[3]], axis=1)
        grid   = np.concatenate([top, bottom], axis=0)
        combined.append(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

    imageio.mimsave(str(out_path), combined, fps=gif_fps,
                    loop=0, quantizer="nq", palettesize=256)
    print(f"  Saved {out_path}  ({len(combined)} frames @ {gif_fps}fps)")

    mp4_path = out_path.with_suffix(".mp4")
    h_, w_   = combined[0].shape[:2]
    wr = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter.fourcc(*"mp4v"),
                         gif_fps, (w_, h_))
    for f in combined:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"  Saved {mp4_path}")


# ── GIF 2: 4-style comparison at landing ─────────────────────────────────────

def make_comparison_gif(out_path=Path("output/style_comparison.gif")):
    styles      = ["normal", "overstriding", "small_steps", "too_bouncy"]
    style_label = {"normal": "NORMAL", "overstriding": "OVERSTRIDING",
                   "small_steps": "SMALL STEPS", "too_bouncy": "TOO BOUNCY"}
    style_color = {"normal": (100, 220, 100), "overstriding": (60, 60, 255),
                   "small_steps": (255, 180, 30), "too_bouncy": (30, 200, 255)}
    print("Style comparison")

    THUMB_W, THUMB_H = 480, 360
    N_FRAMES = 5

    panels: dict[str, list] = {}

    for style in styles:
        folders = runs_for_style(style)
        if not folders:
            continue
        folder   = folders[0]
        csv_path = RUNS_DIR / folder / "pose_yolo.csv"
        vid_path = RUNS_DIR / folder / "video.mp4"

        df, ax_c, ay_c, hx_c, hy_c, kx_c, ky_c, conf, hconf, kconf = load_pose(csv_path)
        phases_df, peaks, contact_sign, _ = detect_contact_events(df, ax_c, ay_c, conf)
        contacts = phases_df[phases_df["label"] == "contact"].reset_index(drop=True)

        mid      = len(contacts) // 2
        selected = contacts.iloc[max(0, mid - N_FRAMES//2) : mid + N_FRAMES//2 + 1]

        cap = cv2.VideoCapture(str(vid_path))
        style_frames = []

        for _, ph_c in selected.iterrows():
            c_start  = int(ph_c["start_idx"])
            frame_no = int(df["frame"].iloc[c_start])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, img = cap.read()
            if not ok:
                continue

            pose_row = df.iloc[c_start]

            # Skeleton
            for na, nb in SKEL_PAIRS:
                pa = get_kpt(pose_row, na)
                pb = get_kpt(pose_row, nb)
                if pa and pb:
                    cv2.line(img, pa, pb, (160, 160, 160), 2, cv2.LINE_AA)
            for jn in ["left_hip", "right_hip", "left_knee", "right_knee",
                       "left_ankle", "right_ankle"]:
                pt = get_kpt(pose_row, jn)
                if pt:
                    cv2.circle(img, pt, 5, (220, 220, 220), -1, cv2.LINE_AA)

            # Ankle-ahead measurement
            ak = get_kpt(pose_row, f"{SIDE}_ankle")
            kk = get_kpt(pose_row, "left_knee")
            if ak and kk:
                val   = (ax_c.iloc[c_start] - kx_c.iloc[c_start]) * contact_sign
                my    = kk[1]
                color = (60, 60, 255) if val > 10 else (80, 220, 80)
                _glow_line(img, (kk[0], my), (ak[0], my), color)
                for x in (kk[0], ak[0]):
                    cv2.line(img, (x, my - 10), (x, my + 10), color, 2, cv2.LINE_AA)
                    cv2.circle(img, (x, my), 6, color, -1, cv2.LINE_AA)

            # Style banner
            ov = img.copy()
            cv2.rectangle(ov, (0, 0), (img.shape[1], 66), (0, 0, 0), -1)
            _blend(img, ov, 0.65)
            cv2.putText(img, style_label[style], (16, 48),
                        FONT, 1.4, style_color[style], 2, cv2.LINE_AA)

            style_frames.append(cv2.cvtColor(
                cv2.resize(img, (THUMB_W, THUMB_H)), cv2.COLOR_BGR2RGB))

        cap.release()
        if style_frames:
            panels[style] = style_frames
            print(f"  {style}: {len(style_frames)} frames")

    if not panels:
        print("No panels generated.")
        return

    n_ticks = max(len(v) for v in panels.values())
    gif_frames = []
    for tick in range(n_ticks):
        row = [panels[s][tick % len(panels[s])] for s in styles if s in panels]
        gif_frames.append(np.concatenate(row, axis=1))

    imageio.mimsave(str(out_path), gif_frames, fps=1.5,
                    loop=0, quantizer="nq", palettesize=256)
    print(f"  Saved {out_path}  ({len(gif_frames)} frames)")

    mp4_path = out_path.with_suffix(".mp4")
    h_, w_   = gif_frames[0].shape[:2]
    wr = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter.fourcc(*"mp4v"), 1.5, (w_, h_))
    for f in gif_frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"  Saved {mp4_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_all_styles_gif()
    make_comparison_gif()
    print("\nDone.")
