import sys
from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIDENCE, PPE_MODEL_PATH
from models.ppe_detection import PPEDetector


def main() -> None:
    if not PPE_MODEL_PATH.exists():
        print(f"Missing model file: {PPE_MODEL_PATH}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera (index 0) could not be opened. Check if another app is using it.")
        return

    detector = PPEDetector(
        model_path=PPE_MODEL_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE,
        imgsz=640,
    )

    print("Live camera started. Press 'q' on the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame from camera.")
            break

        summary = detector.detect(frame)
        rendered = detector.draw_detections(frame, summary)

        cv2.imshow("Live PPE Camera Check (Press q to quit)", rendered)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()