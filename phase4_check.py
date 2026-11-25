import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIDENCE, PPE_MODEL_PATH
from models.ppe_detection import PPEDetector


def main() -> None:
    print("=== Phase 4 PPE Detection Module Test ===")
    detector = PPEDetector(
        model_path=PPE_MODEL_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE,
        imgsz=640,
    )

    # Dummy test frame (360x640x3) to verify module initialization & drawing without crashing
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    summary = detector.detect(dummy_frame)
    rendered = detector.draw_detections(dummy_frame, summary)

    print(f"PPE weights found ({PPE_MODEL_PATH}): {summary.model_loaded}")
    print(f"Detections count: {len(summary.items)} | Rendered shape: {rendered.shape}")
    print("Phase 4 module test passed cleanly.")


if __name__ == "__main__":
    main()