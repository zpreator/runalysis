import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO 17-keypoint layout output by YOLOv8-pose
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

CSV_HEADER = ["frame", "timestamp_ms"] + [
    f"{name}_{axis}"
    for name in KEYPOINT_NAMES
    for axis in ("x", "y", "conf")
]


def process_video(video_path: str, model: YOLO, output_csv: str | None = None, output_video: str | None = None) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    run_dir = Path(video_path).parent
    csv_path = output_csv or str(run_dir / "pose_yolo.csv")
    annotated_path = output_video or str(run_dir / "pose_yolo_video.mp4")

    video_writer = cv2.VideoWriter(
        annotated_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    frame_idx = 0
    detected = 0

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            results = model(frame_bgr, device="mps", verbose=False)
            result = results[0]

            row = [frame_idx, timestamp_ms]

            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Pick the person with the largest bounding box
                if result.boxes is not None and len(result.boxes) > 1:
                    areas = result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]
                    best = int(areas.argmax())
                else:
                    best = 0

                kpts = result.keypoints.data[best]  # (17, 3) — x, y, conf
                row += [v.item() for kp in kpts for v in kp]
                detected += 1
            else:
                row += [""] * (len(KEYPOINT_NAMES) * 3)

            writer.writerow(row)
            video_writer.write(result.plot(labels=False, conf=False))

            frame_idx += 1

    video_writer.release()
    cap.release()
    return frame_idx, detected


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8-pose landmark extraction from video.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-m", "--model",
        default="x",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n=nano s=small m=medium l=large x=extra-large (default: x)",
    )
    parser.add_argument("--csv", help="Output CSV path (default: <run_dir>/pose_yolo.csv)")
    parser.add_argument("--video-out", help="Output video path (default: <run_dir>/pose_yolo_video.mp4)")
    args = parser.parse_args()

    model = YOLO(f"yolov8{args.model}-pose.pt")
    frames, detected = process_video(args.video, model, args.csv, args.video_out)

    run_dir = Path(args.video).parent
    print(f"Done — {frames} frames, pose detected in {detected}")
    print(f"  CSV:   {args.csv or run_dir / 'pose_yolo.csv'}")
    print(f"  Video: {args.video_out or run_dir / 'pose_yolo_video.mp4'}")


if __name__ == "__main__":
    main()
