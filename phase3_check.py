import sys
from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIDENCE, FRAME_HEIGHT, FRAME_WIDTH, TARGET_FPS
from detection.video_processor import VideoProcessor
from models.detection import PersonDetector


def main() -> None:
    print("=== Phase 3 YOLO Person Detection Test ===")
    # Uses yolo11n.pt from project root if present; falls back to auto-download
    root_weights = ROOT_DIR / "yolo11n.pt"
    model_source = root_weights if root_weights.exists() else "yolo11n.pt"

    detector = PersonDetector(
        model_path=model_source,
        confidence_threshold=DEFAULT_CONFIDENCE,
        imgsz=640,
    )

    processor = VideoProcessor(
        source=0,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        target_fps=TARGET_FPS,
    )

    if not processor.open():
        print("Webcam could not be opened.")
        return

    print("Running person detection. Press 'q' in video window to quit.")
    try:
        while True:
            ok, frame = processor.read_frame()
            if not ok or frame is None:
                break

            detections = detector.detect(frame)
            annotated = detector.draw_detections(frame, detections)
            annotated = processor.draw_fps(annotated)

            cv2.imshow("Phase 3 - YOLO Person Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        processor.release()
        cv2.destroyAllWindows()
        print("Phase 3 test finished cleanly.")


if __name__ == "__main__":
    main()