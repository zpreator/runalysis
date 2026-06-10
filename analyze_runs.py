import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════════════════════════════
#  FILL IN YOUR SHOE AND SPEED INFO HERE
#  Runs are listed in chronological order.
#  speed_mph: treadmill speed for that session
# ══════════════════════════════════════════════════════════════════════════════
RUNS = {
    # ── Speed 1 ───────────────────────────────────────────────────────────────
    "20260604_220717": {"shoe": "Brooks Adrenaline 23",  "speed_mph": 5},   # run  0
    "20260604_220952": {"shoe": "ASICS Gel-Nimbus",  "speed_mph": 5},   # run  1
    "20260604_221128": {"shoe": "ASICS Novablast",  "speed_mph": 5},   # run  2
    "20260604_221239": {"shoe": "Mount to Coast C1",  "speed_mph": 5},   # run  3
    "20260604_221422": {"shoe": "Hoka Mach 6",  "speed_mph": 5},   # run  4
    "20260604_221613": {"shoe": "ASICS Superblast",  "speed_mph": 5},   # run  5
    "20260604_221728": {"shoe": "Brooks Adrenaline 22",  "speed_mph": 5},   # run  6
    "20260604_221849": {"shoe": "Brooks Ghost 23",  "speed_mph": 5},   # run  7
    "20260604_222012": {"shoe": "Adidas Adistar CS 2",  "speed_mph": 5},   # run  8
    # ── Speed 2 (randomized order) ────────────────────────────────────────────
    "20260604_222628": {"shoe": "Adidas Adistar CS 2",  "speed_mph": 6},   # run  9
    "20260604_222735": {"shoe": "Brooks Adrenaline 22",  "speed_mph": 6},   # run 10
    "20260604_222842": {"shoe": "Brooks Ghost 23",  "speed_mph": 6},   # run 11
    "20260604_222957": {"shoe": "ASICS Superblast",  "speed_mph": 6},   # run 12
    "20260604_223111": {"shoe": "Hoka Mach 6",  "speed_mph": 6},   # run 13
    "20260604_223224": {"shoe": "Mount to Coast C1",  "speed_mph": 6},   # run 14
    "20260604_223341": {"shoe": "ASICS Novablast",  "speed_mph": 6},   # run 15
    "20260604_223448": {"shoe": "Brooks Adrenaline 23",  "speed_mph": 6},   # run 16
    "20260604_223600": {"shoe": "ASICS Gel-Nimbus",  "speed_mph": 6},   # run 17
}

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS SETTINGS  (derived from notebook EDA)
# ══════════════════════════════════════════════════════════════════════════════
RUNS_DIR       = Path("runs")
SIDE           = "left"
TILT_DEG       = 2.0        # counterclockwise correction
CONF_THRESHOLD = 0.5
SMOOTH_WINDOW  = 5          # frames, rolling mean on X velocity
MIN_PHASE_MS   = 80         # discard phases shorter than this
LOADING_WINDOW = 5          # frames to look back before contact for loading rate


# ── Core analysis for one CSV ──────────────────────────────────────────────────

def _apply_tilt(x: pd.Series, y: pd.Series, tilt_rad: float) -> tuple[pd.Series, pd.Series]:
    cx, cy = x.median(), y.median()
    cos_t, sin_t = np.cos(-tilt_rad), np.sin(-tilt_rad)
    dx, dy = x - cx, y - cy
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def _joint_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Angle in degrees at `vertex` formed by the p1-vertex-p2 triplet."""
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def analyze(csv_path: Path) -> dict | None:
    from scipy.signal import find_peaks

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df["time_s"] = df["timestamp_ms"] / 1000
    tilt_rad = np.radians(TILT_DEG)

    # Tilt-corrected joints
    ax_cor, ay_cor = _apply_tilt(
        pd.to_numeric(df[f"{SIDE}_ankle_x"], errors="coerce"),
        pd.to_numeric(df[f"{SIDE}_ankle_y"], errors="coerce"),
        tilt_rad,
    )
    hx_cor, hy_cor = _apply_tilt(
        pd.to_numeric(df["left_hip_x"],  errors="coerce"),
        pd.to_numeric(df["left_hip_y"],  errors="coerce"),
        tilt_rad,
    )
    kx_cor, ky_cor = _apply_tilt(
        pd.to_numeric(df["left_knee_x"],  errors="coerce"),
        pd.to_numeric(df["left_knee_y"],  errors="coerce"),
        tilt_rad,
    )

    conf       = pd.to_numeric(df[f"{SIDE}_ankle_conf"], errors="coerce")
    hip_conf   = pd.to_numeric(df["left_hip_conf"],      errors="coerce")
    knee_conf  = pd.to_numeric(df["left_knee_conf"],     errors="coerce")

    ay_clean   = ay_cor.where(conf      >= CONF_THRESHOLD).interpolate()
    ax_clean   = ax_cor.where(conf      >= CONF_THRESHOLD).interpolate()
    hy_clean   = hy_cor.where(hip_conf  >= CONF_THRESHOLD).interpolate()

    t_s     = df["time_s"]
    xvel    = ax_clean.diff() / t_s.diff()
    xvel_sm = xvel.rolling(SMOOTH_WINDOW, center=True).mean()
    sign    = pd.Series(np.sign(xvel_sm)).fillna(0).astype(int)

    # Which sign = ground contact (ankle Y peaks = midstance)
    fps = 1000 / df["timestamp_ms"].diff().median()
    peaks, _ = find_peaks(ay_clean, distance=int(fps * 0.3), prominence=20)
    if len(peaks) < 2:
        return None
    contact_sign = int(pd.Series(sign.iloc[peaks]).mode()[0])

    # Segment phases — track frame indices for per-phase analysis
    phases, cur_sign, phase_start_ms, phase_start_idx = [], None, None, None
    for idx, (ts, s) in enumerate(zip(df["timestamp_ms"], sign)):
        if s == 0:
            continue
        if s != cur_sign:
            if cur_sign is not None:
                phases.append({
                    "label":       "contact" if cur_sign == contact_sign else "air",
                    "duration_ms": ts - phase_start_ms,
                    "start_idx":   phase_start_idx,
                    "end_idx":     idx,
                    "start_ms":    phase_start_ms,
                })
            cur_sign, phase_start_ms, phase_start_idx = s, ts, idx

    phases_df = pd.DataFrame(phases)
    phases_df = phases_df[phases_df["duration_ms"] >= MIN_PHASE_MS].reset_index(drop=True)

    contact_dur = phases_df[phases_df["label"] == "contact"]["duration_ms"]
    air_dur     = phases_df[phases_df["label"] == "air"]["duration_ms"]
    if len(contact_dur) < 2 or len(air_dur) < 2:
        return None

    # ── Vertical oscillation (hip Y range per stride) ────────────────────────
    osc_per_stride = []
    for i in range(len(peaks) - 1):
        seg = hy_clean.iloc[peaks[i]:peaks[i + 1]]
        if seg.notna().sum() > 2:
            osc_per_stride.append(seg.max() - seg.min())
    osc = pd.Series(osc_per_stride)

    # ── Flight arc height (ankle rise above ground baseline during air) ───────
    # Ground baseline = median ankle Y at all midstance peaks
    ground_y = float(ay_clean.iloc[peaks].median())
    arc_heights = []
    for _, ph in phases_df[phases_df["label"] == "air"].iterrows():
        seg = ay_clean.iloc[ph["start_idx"]:ph["end_idx"]]
        if seg.notna().sum() > 1:
            # ground_y - min_y: positive = foot rose above ground level
            arc_heights.append(ground_y - seg.min())
    arc = pd.Series(arc_heights)

    # ── Loading rate (ankle descent rate in frames before contact) ────────────
    loading_rates = []
    for _, ph in phases_df[phases_df["label"] == "contact"].iterrows():
        i0 = max(ph["start_idx"] - LOADING_WINDOW, 0)
        i1 = ph["start_idx"]
        seg_y = ay_clean.iloc[i0:i1]
        seg_t = t_s.iloc[i0:i1] * 1000  # convert to ms
        if seg_y.notna().sum() > 1 and (seg_t.iloc[-1] - seg_t.iloc[0]) > 0:
            # px/ms, positive = ankle descending toward ground
            rate = (seg_y.iloc[-1] - seg_y.iloc[0]) / (seg_t.iloc[-1] - seg_t.iloc[0])
            loading_rates.append(rate)
    loading = pd.Series(loading_rates)

    # ── Knee angle at landing (angle at knee on first contact frame) ──────────
    knee_angles = []
    for _, ph in phases_df[phases_df["label"] == "contact"].iterrows():
        idx = ph["start_idx"]
        good   = (conf.iloc[idx] >= CONF_THRESHOLD and
                  hip_conf.iloc[idx] >= CONF_THRESHOLD and
                  knee_conf.iloc[idx] >= CONF_THRESHOLD)
        if not good:
            continue
        p_hip    = np.array([hx_cor.iloc[idx], hy_cor.iloc[idx]])
        p_knee   = np.array([kx_cor.iloc[idx],      ky_cor.iloc[idx]])
        p_ankle  = np.array([ax_cor.iloc[idx],       ay_cor.iloc[idx]])
        knee_angles.append(_joint_angle(p_hip, p_knee, p_ankle))
    knee_ang = pd.Series(knee_angles)

    ratio = contact_dur.mean() / air_dur.mean()
    return {
        "contact_mean_ms":    contact_dur.mean(),
        "contact_std_ms":     contact_dur.std(),
        "air_mean_ms":        air_dur.mean(),
        "air_std_ms":         air_dur.std(),
        "ratio":              ratio,
        "pct_on_ground":      ratio / (1 + ratio) * 100,
        "n_strides":          len(contact_dur),
        "cadence_spm":        len(contact_dur) / df["time_s"].max() * 60,
        "vert_osc_mean_px":   osc.mean(),
        "vert_osc_std_px":    osc.std(),
        "flight_arc_mean_px": arc.mean(),
        "flight_arc_std_px":  arc.std(),
        "loading_rate_px_ms": loading.mean(),
        "knee_angle_mean_deg": knee_ang.mean(),
        "knee_angle_std_deg":  knee_ang.std(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing_labels = [k for k, v in RUNS.items() if not v["shoe"]]
    if missing_labels:
        print("⚠  Some runs have no shoe label — fill in the RUNS dict first:")
        for k in missing_labels:
            print(f"   {k}")
        print()

    rows = []
    for folder, meta in RUNS.items():
        csv_path = RUNS_DIR / folder / "pose_yolo.csv"
        if not csv_path.exists():
            print(f"  SKIP {folder} — pose_yolo.csv not found")
            continue

        result = analyze(csv_path)
        if result is None:
            print(f"  SKIP {folder} — insufficient detections")
            continue

        rows.append({
            "run":          folder,
            "shoe":         meta["shoe"] or f"run_{folder}",
            "speed_mph":    meta["speed_mph"],
            **result,
        })
        print(f"  OK   {folder}  →  {meta['shoe'] or '(unlabeled)'}  "
              f"contact={result['contact_mean_ms']:.0f}ms  "
              f"air={result['air_mean_ms']:.0f}ms  "
              f"ratio={result['ratio']:.3f}")

    if not rows:
        sys.exit("No runs processed.")

    results = pd.DataFrame(rows)
    results_path = "run_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    _print_table(results)
    _plot(results)


def _print_table(df: pd.DataFrame):
    speeds = sorted(df["speed_mph"].dropna().unique())
    for speed in speeds:
        subset = df[df["speed_mph"] == speed].sort_values("contact_mean_ms")
        print(f"\n── Speed {speed} mph {'─'*44}")
        print(f"  {'Shoe':<28} {'Contact':>10} {'±':>6} {'Air':>10} {'±':>6} {'Ratio':>7} {'% gnd':>7} {'Stride/min':>11} {'Osc(px)':>9}")
        print(f"  {'─'*28} {'─'*10} {'─'*6} {'─'*10} {'─'*6} {'─'*7} {'─'*7} {'─'*11} {'─'*9}")
        for _, r in subset.iterrows():
            print(f"  {r['shoe']:<28} "
                  f"{r['contact_mean_ms']:>10.1f} "
                  f"{r['contact_std_ms']:>6.1f} "
                  f"{r['air_mean_ms']:>10.1f} "
                  f"{r['air_std_ms']:>6.1f} "
                  f"{r['ratio']:>7.3f} "
                  f"{r['pct_on_ground']:>6.1f}% "
                  f"{r['cadence_spm']:>11.1f} "
                  f"{r['vert_osc_mean_px']:>9.1f}")

    if df["speed_mph"].nunique() > 1:
        print(f"\n── Same shoe across speeds {'─'*38}")
        pivot = df.pivot_table(index="shoe", columns="speed_mph",
                               values=["contact_mean_ms", "air_mean_ms", "ratio"])
        print(pivot.to_string(float_format="{:.1f}".format))


def _plot(df: pd.DataFrame):
    speeds = sorted(df["speed_mph"].dropna().unique())
    # Consistent shoe order (alphabetical) across both subplots
    shoes = sorted(df["shoe"].unique())
    n_shoes = len(shoes)
    x = np.arange(n_shoes)

    # 4 bars per shoe: ratio@speed1, osc@speed1, ratio@speed2, osc@speed2
    n_bars   = len(speeds) * 2
    bar_w    = 0.8 / n_bars
    colors   = {"ratio": ["#5C6BC0", "#9FA8DA"], "osc": ["#2E7D32", "#81C784"]}
    # darker shade = lower speed, lighter = higher speed

    fig, axes = plt.subplots(2, 3, figsize=(max(18, n_shoes * 2.2), 12))
    axes = axes.flatten()
    axes[-1].set_visible(False)  # 5 metrics, 6 cells

    metrics = [
        ("ratio",              "Contact : air ratio",     axes[0], colors["ratio"]),
        ("vert_osc_mean_px",   "Vert. oscillation (px)",  axes[1], colors["osc"]),
        ("flight_arc_mean_px", "Flight arc height (px)",  axes[2], ["#E53935", "#EF9A9A"]),
        ("loading_rate_px_ms", "Loading rate (px/ms)",    axes[3], ["#F57F17", "#FFE082"]),
        ("knee_angle_mean_deg","Knee angle at landing (°)",axes[4], ["#00838F", "#80DEEA"]),
    ]

    for metric, ylabel, ax, palette in metrics:
        for i, (speed, color) in enumerate(zip(speeds, palette)):
            sub = df[df["speed_mph"] == speed].set_index("shoe").reindex(shoes)
            vals = sub[metric].values
            err  = sub["vert_osc_std_px"].values if metric == "vert_osc_mean_px" else None
            offset = (i - (len(speeds) - 1) / 2) * bar_w

            bars = ax.bar(x + offset, vals, width=bar_w, color=color, alpha=0.9,
                          yerr=err, capsize=3, label=f"{speed} mph")

            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals[np.isfinite(vals)]) * 0.01,
                            f"{val:.3f}" if metric == "ratio" else f"{val:.0f}",
                            ha="center", va="bottom", fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(shoes, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Shoe comparison — contact:air ratio & vertical oscillation\n"
        f"{TILT_DEG}° tilt correction  |  left hip oscillation  |  left ankle X-velocity phasing",
        fontsize=11
    )
    plt.tight_layout()
    out = "run_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
