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
        print("Place your trained PPE weights at weights\\ppe_best.pt first.")
        return

    # Put a test image named helmet_test.jpg inside uploads/
    img_path = ROOT_DIR / "uploads" / "helmet_test.jpg"
    if not img_path.exists():
        print(f"Put a helmet test image at: {img_path}")
        return

    frame = cv2.imread(str(img_path))
    if frame is None:
        print("Could not read image.")
        return

    detector = PPEDetector(model_path=PPE_MODEL_PATH, confidence_threshold=DEFAULT_CONFIDENCE, imgsz=640)
    summary = detector.detect(frame)
    rendered = detector.draw_detections(frame, summary)

    print(f"Helmets: {summary.helmet_count} | No-Helmet: {summary.no_helmet_count}")
    cv2.imshow("Helmet Detection Check (Press q to close)", rendered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()