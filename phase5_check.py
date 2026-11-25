import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.worker_tracking import WorkerTracker
from detection.zone_detection import RestrictedZone, RestrictedZoneDetector


def main() -> None:
    print("=== Phase 5 Worker Tracking & Restricted Zone Test ===")
    tracker = WorkerTracker()

    # Define a sample restricted zone polygon on a 640x360 frame
    zone = RestrictedZone(
        name="High-Voltage Rig Zone",
        points=[(350, 100), (610, 100), (610, 340), (350, 340)],
    )
    zone_detector = RestrictedZoneDetector(zones=[zone])

    # Simulated detections moving from outside (x=120..220) into the restricted zone (x=420)
    frame1_dets = [((100, 120, 170, 300), 0.91)]
    frame2_dets = [((400, 120, 470, 300), 0.93)]

    w1 = tracker.update(frame1_dets)
    v1 = zone_detector.check_violations([(w.track_id, w.foot_point, w.center) for w in w1])

    w2 = tracker.update(frame2_dets)
    v2 = zone_detector.check_violations([(w.track_id, w.foot_point, w.center) for w in w2])

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas = zone_detector.draw_zones(canvas, v2)
    canvas = tracker.draw_tracks(canvas, w2)

    print(f"Frame 1 tracked IDs: {[w.track_id for w in w1]} | Violations: {len(v1)}")
    print(f"Frame 2 tracked IDs: {[w.track_id for w in w2]} | Violations: {len(v2)}")
    print(f"Rendered frame shape: {canvas.shape}")
    print("Phase 5 module test passed cleanly.")


if __name__ == "__main__":
    main()