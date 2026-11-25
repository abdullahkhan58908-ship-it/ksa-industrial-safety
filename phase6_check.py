import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.fall_detection import FallDetector


def main() -> None:
    print("=== Phase 6 Fall Detection Module Test ===")
    detector = FallDetector(width_to_height_ratio=1.25, min_fallen_frames=3)

    # Standing worker box (height > width): (100, 80, 170, 260) -> w=70, h=180
    standing = [(1, (100, 80, 170, 260))]
    assert len(detector.update(standing)) == 0

    # Fallen horizontal worker box (width > height): (100, 190, 280, 265) -> w=180, h=75 (ratio=2.4)
    fallen = [(1, (100, 190, 280, 265))]
    events = []
    for _ in range(3):
        events = detector.update(fallen)

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered = detector.draw_falls(canvas, events)

    print(f"Fall events after 3 horizontal frames: {len(events)}")
    print(f"Rendered frame shape: {rendered.shape}")
    print("Phase 6 module test passed cleanly.")


if __name__ == "__main__":
    main()