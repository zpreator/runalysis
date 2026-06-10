import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2

TARGET_FPS       = 120
RECORD_SECONDS   = 30
PRE_ROLL_SECONDS = 10
CAMERA_INDEX     = 0


# ── drawing helpers ────────────────────────────────────────────────────────────

def text_centered(image, text, cy, scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (image.shape[1] - tw) // 2
    y = cy + th // 2
    cv2.putText(image, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def dim(image, alpha=0.5):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def bottom_bar(image, text):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    text_centered(image, text, h - 45, 1.1, (255, 255, 255), 2)


# ── state machine ──────────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = auto()
    STARTING  = auto()
    RECORDING = auto()
    DONE      = auto()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cam_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    print(f"Camera {CAMERA_INDEX}: {cam_w}x{cam_h} @ {cam_fps:.0f} fps")

    state        = State.IDLE
    state_start  = 0.0
    video_writer = None
    run_dir: Path | None = None
    saved_path   = ""
    frame_idx    = 0

    fps_times: list[float] = []

    def advance():
        nonlocal state, state_start
        if state == State.IDLE:
            state = State.STARTING
            state_start = time.monotonic()
        elif state == State.DONE:
            state = State.IDLE

    def on_mouse(event, _x, _y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            advance()

    cv2.namedWindow("Recorder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recorder", 1280, 720)
    cv2.setMouseCallback("Recorder", on_mouse)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        now = time.monotonic()
        fps_times.append(now)
        fps_times[:] = [t for t in fps_times if now - t <= 1.0]
        live_fps = len(fps_times)

        display = frame_bgr.copy()
        elapsed = now - state_start

        # ── IDLE ───────────────────────────────────────────────────────────────
        if state == State.IDLE:
            bottom_bar(display, "SPACE or CLICK to start")

        # ── STARTING ───────────────────────────────────────────────────────────
        elif state == State.STARTING:
            count = PRE_ROLL_SECONDS - int(elapsed)
            if count <= 0:
                run_dir    = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir.mkdir(parents=True)
                saved_path = str(run_dir)

                video_writer = cv2.VideoWriter(
                    str(run_dir / "video.mp4"),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    cam_fps,
                    (cam_w, cam_h),
                )
                frame_idx   = 0
                state       = State.RECORDING
                state_start = now
                elapsed     = 0.0
            else:
                dim(display, 0.35)
                text_centered(display, str(count), display.shape[0] // 2 - 60, 9.0, (255, 255, 255), 12)
                text_centered(display, "Get ready!", int(display.shape[0] * 0.78), 1.4, (200, 200, 200), 2)

        # ── RECORDING ──────────────────────────────────────────────────────────
        elif state == State.RECORDING:
            remaining = RECORD_SECONDS - elapsed

            if remaining <= 0:
                video_writer.release()
                video_writer = None
                state        = State.DONE
                state_start  = now
            else:
                video_writer.write(frame_bgr)
                frame_idx += 1

                secs_left = int(remaining) + 1
                color = (0, 220, 0) if remaining > 10 else (0, 100, 255)
                text_centered(display, str(secs_left), 10, 3.5, color, 5)
                cv2.circle(display, (28, 28), 10, (0, 0, 220), -1, cv2.LINE_AA)
                cv2.putText(display, "REC", (46, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 220), 2, cv2.LINE_AA)

        # ── DONE ───────────────────────────────────────────────────────────────
        elif state == State.DONE:
            dim(display, 0.55)
            text_centered(display, "DONE!", display.shape[0] // 2 - 80, 5.0, (0, 230, 0), 8)
            text_centered(display, f"Saved: {saved_path}/",
                          int(display.shape[0] * 0.65), 0.85, (230, 230, 230), 2)
            text_centered(display, f"{frame_idx} frames @ {cam_fps:.0f} fps",
                          int(display.shape[0] * 0.72), 0.9, (170, 230, 170), 2)
            text_centered(display, "SPACE or CLICK to record again",
                          int(display.shape[0] * 0.82), 1.0, (170, 170, 170), 2)
            if elapsed > 6:
                state = State.IDLE

        # Live fps in corner
        cv2.putText(display, f"{live_fps} fps", (cam_w - 110, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Recorder", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            advance()

    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
