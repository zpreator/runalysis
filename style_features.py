"""
style_features.py  —  Extract per-stride features for running *style* classification.

Labels: normal | overstriding | small_steps | long_steps | too_bouncy
Views:  perpendicular (side) | isometric (angled)

Outputs style_stride_features.csv with one row per stride.

Usage:
    uv run python style_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analyze_runs import (
    RUNS_DIR,
    CONF_THRESHOLD, SMOOTH_WINDOW, MIN_PHASE_MS, LOADING_WINDOW, SIDE, TILT_DEG,
    _apply_tilt, _joint_angle,
)

OUTPUT_PATH = Path("style_stride_features.csv")

# ── Labeled runs ───────────────────────────────────────────────────────────────
# Perpendicular (side view) runs 1–5, then isometric 6–10, isometric 11–15
# (shoe switch at run 11), then perpendicular 16–20.
# Fill in shoe_a / shoe_b with the actual names.

SHOE_A = "Brooks Adrenaline"   # runs 1–10  ← replace with actual shoe name
SHOE_B = "ASICS Gel-Nimbus"   # runs 11–20 ← replace with actual shoe name
SPEED  = 5.0        # mph — update if treadmill speed differed

STYLE_RUNS = {
    # ── 20260609 — perpendicular, ASICS Gel-Nimbus, 30 fps ───────────────────
    "20260609_180856": {"style": "normal",       "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 21
    "20260609_181012": {"style": "overstriding", "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 22
    "20260609_181058": {"style": "small_steps",  "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 23
    "20260609_181151": {"style": "too_bouncy",   "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 24
    "20260609_181245": {"style": "normal",       "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 25
    "20260609_181332": {"style": "overstriding", "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 26
    "20260609_181419": {"style": "small_steps",  "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 27
    "20260609_181509": {"style": "too_bouncy",   "view": "perpendicular", "shoe": SHOE_B, "speed_mph": SPEED},  # 28
    # 20260609_181605 — test video (all styles in sequence), not used for training
}


# ── Per-stride extraction ──────────────────────────────────────────────────────

def extract_strides(csv_path: Path, meta: dict, run_folder: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

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
        return pd.DataFrame()

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
        return pd.DataFrame()
    phases_df = phases_df[phases_df["duration_ms"] >= MIN_PHASE_MS].reset_index(drop=True)

    ground_y = float(ay_clean.iloc[peaks].median())

    rows = []

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

        # ── Knee angle at landing ────────────────────────────────────────────
        knee_angle = np.nan
        all_conf_ok = (
            c_start_idx < len(df) and
            conf.iloc[c_start_idx]      >= CONF_THRESHOLD and
            hip_conf.iloc[c_start_idx]  >= CONF_THRESHOLD and
            knee_conf.iloc[c_start_idx] >= CONF_THRESHOLD
        )
        if all_conf_ok:
            p_hip   = np.array([hx_cor.iloc[c_start_idx], hy_cor.iloc[c_start_idx]])
            p_knee  = np.array([kx_cor.iloc[c_start_idx], ky_cor.iloc[c_start_idx]])
            p_ankle = np.array([ax_cor.iloc[c_start_idx], ay_cor.iloc[c_start_idx]])
            knee_angle = _joint_angle(p_hip, p_knee, p_ankle)

        # ── Ankle vs knee at landing — overstriding signal ───────────────────
        # Positive = ankle is ahead of knee in direction of travel at foot strike.
        # Multiplied by contact_sign so positive always means "ahead" regardless
        # of camera orientation.
        ankle_ahead_knee = np.nan
        if (c_start_idx < len(df) and
                conf.iloc[c_start_idx]      >= CONF_THRESHOLD and
                knee_conf.iloc[c_start_idx] >= CONF_THRESHOLD):
            ankle_ahead_knee = (ax_cor.iloc[c_start_idx] - kx_cor.iloc[c_start_idx]) * contact_sign

        # ── Ankle vs knee at toe-off ─────────────────────────────────────────
        # At push-off the foot should be well behind the knee (negative = good).
        # a_start_idx is the first air frame = toe-off moment.
        toeoff_ankle_knee = np.nan
        if (a_start_idx < len(df) and
                conf.iloc[a_start_idx]      >= CONF_THRESHOLD and
                knee_conf.iloc[a_start_idx] >= CONF_THRESHOLD):
            toeoff_ankle_knee = (ax_cor.iloc[a_start_idx] - kx_cor.iloc[a_start_idx]) * contact_sign

        # ── Loading rate ─────────────────────────────────────────────────────
        loading_rate = np.nan
        i0 = max(c_start_idx - LOADING_WINDOW, 0)
        seg_y = ay_clean.iloc[i0:c_start_idx]
        seg_t = t_s.iloc[i0:c_start_idx] * 1000
        if seg_y.notna().sum() > 1 and (seg_t.iloc[-1] - seg_t.iloc[0]) > 0:
            loading_rate = (seg_y.iloc[-1] - seg_y.iloc[0]) / (seg_t.iloc[-1] - seg_t.iloc[0])

        # ── Flight arc height ────────────────────────────────────────────────
        flight_arc = np.nan
        seg_air = ay_clean.iloc[a_start_idx:a_end_idx]
        if seg_air.notna().sum() > 1:
            flight_arc = ground_y - seg_air.min()

        # ── Hip oscillation ──────────────────────────────────────────────────
        hip_osc = np.nan
        peak_before = peaks[peaks <= c_start_idx]
        peak_after  = peaks[peaks >= a_end_idx]
        if len(peak_before) > 0 and len(peak_after) > 0:
            seg_hip = hy_clean.iloc[peak_before[-1]:peak_after[0]]
            if seg_hip.notna().sum() > 2:
                hip_osc = seg_hip.max() - seg_hip.min()

        rows.append({
            "style":           meta["style"],
            "view":            meta["view"],
            "shoe":            meta["shoe"],
            "speed_mph":       meta["speed_mph"],
            "run_folder":      run_folder,
            "stride_idx":      len(rows),
            "contact_ms":      contact_ms,
            "air_ms":          air_ms,
            "stride_ms":       stride_ms,
            "ratio":           contact_ms / air_ms if air_ms > 0 else np.nan,
            "pct_contact":     contact_ms / stride_ms * 100 if stride_ms > 0 else np.nan,
            "knee_angle_deg":        knee_angle,
            "ankle_ahead_knee_px":   ankle_ahead_knee,   # ankle vs knee at landing
            "toeoff_ankle_knee_px":  toeoff_ankle_knee,  # ankle vs knee at toe-off
            "cadence_spm":           60000.0 / stride_ms if stride_ms > 0 else np.nan,
            "loading_rate":    loading_rate,
            "flight_arc_px":   flight_arc,
            "hip_osc_px":      hip_osc,
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "contact_ms", "air_ms", "ratio", "pct_contact",
    "knee_angle_deg", "ankle_ahead_knee_px", "toeoff_ankle_knee_px", "cadence_spm",
    "loading_rate", "flight_arc_px", "hip_osc_px",
]

def main():
    all_strides = []

    for folder, meta in STYLE_RUNS.items():
        csv_path = RUNS_DIR / folder / "pose_yolo.csv"
        if not csv_path.exists():
            print(f"  SKIP {folder} — pose_yolo.csv not found")
            continue

        strides = extract_strides(csv_path, meta, folder)
        if strides.empty:
            print(f"  SKIP {folder} — no strides extracted")
            continue

        all_strides.append(strides)
        print(f"  OK   {folder}  {meta['style']:14}  {meta['view']:14}  {len(strides):3d} strides")

    if not all_strides:
        print("No data extracted.")
        return

    combined = pd.concat(all_strides, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'─'*60}")
    print(f"  Total strides : {len(combined)}")
    print(f"  Runs          : {combined['run_folder'].nunique()}")
    print(f"\n  Strides per style:\n")
    print(combined.groupby("style").size().rename("strides").to_string())

    completeness = combined[FEATURE_COLS].notna().mean() * 100
    print(f"\n  Feature completeness:\n")
    for col, pct in completeness.items():
        bar = "█" * int(pct / 5)
        print(f"    {col:<22} {pct:5.1f}%  {bar}")

    print(f"\n  Saved: {OUTPUT_PATH}  ({len(combined)} rows × {len(combined.columns)} cols)")


if __name__ == "__main__":
    main()
