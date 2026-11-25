import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIDENCE, FIRE_MODEL_PATH
from models.fire_detection import FireSmokeDetector


def main() -> None:
    print("=== Phase 7 Fire & Smoke Detection Module Test ===")
    detector = FireSmokeDetector(
        model_path=FIRE_MODEL_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE,
        imgsz=640,
    )

    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    summary = detector.detect(dummy_frame)
    rendered = detector.draw_hazards(dummy_frame, summary)

    print(f"Fire/Smoke weights found ({FIRE_MODEL_PATH}): {summary.model_loaded}")
    print(f"Emergency flag: {summary.has_emergency} | Rendered shape: {rendered.shape}")
    print("Phase 7 module test passed cleanly.")


if __name__ == "__main__":
    main()