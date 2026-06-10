import argparse
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from yolo_process import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YOLOv8-pose processing for all runs.")
    parser.add_argument(
        "--runs-dir", default="runs", help="Root folder containing run subdirectories (default: runs/)"
    )
    parser.add_argument(
        "-m", "--model",
        default="x",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (default: x)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process runs that already have pose_yolo.csv"
    )
    args = parser.parse_args()

    videos = sorted(Path(args.runs_dir).glob("*/video.mp4"))
    if not videos:
        print(f"No video.mp4 files found under {args.runs_dir}/")
        return

    if not args.force:
        pending = [v for v in videos if not (v.parent / "pose_yolo.csv").exists()]
        skipped = len(videos) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-processed run(s). Use --force to reprocess.")
    else:
        pending = videos

    if not pending:
        print("All runs already processed.")
        return

    print(f"Loading yolov8{args.model}-pose model...")
    model = YOLO(f"yolov8{args.model}-pose.pt")

    total_frames = 0
    total_detected = 0

    for video_path in tqdm(pending, desc="Runs", unit="run"):
        tqdm.write(f"  → {video_path.parent.name}")
        frames, detected = process_video(str(video_path), model)
        total_frames += frames
        total_detected += detected

    print(f"\nAll done — {len(pending)} runs, {total_frames} frames, pose detected in {total_detected}")


if __name__ == "__main__":
    main()
