from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_CONFIDENCE,
    FIRE_MODEL_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PPE_MODEL_PATH,
    REPORTS_DIR,
)
from models.detection import PersonDetector
from models.fall_detection import FallDetector
from models.fire_detection import FireSmokeDetector
from models.ppe_detection import PPEDetector


def benchmark_person_detector(model_path: Path, runs: int = 20) -> dict:
    """Measures real CPU inference latency (ms) and FPS on dummy frames."""
    detector = PersonDetector(model_path=model_path, confidence_threshold=DEFAULT_CONFIDENCE, imgsz=640)
    sample = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Warm-up run
    detector.detect(sample)

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detector.detect(sample)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        "runs": runs,
        "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "avg_latency_ms": round(avg_ms, 2),
        "estimated_cpu_fps": round(fps, 2),
    }


def evaluate_yolo_dataset(weights_path: Path, data_yaml: Path) -> dict:
    """Runs official Ultralytics validation only when both weights and data.yaml exist."""
    if not weights_path.exists() or not data_yaml.exists():
        return {
            "status": "skipped",
            "reason": "Weights or dataset data.yaml not provided yet (no fabricated metrics).",
        }

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=640, verbose=False)
    return {
        "status": "evaluated",
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
    }


def main() -> None:
    print("=== Phase 11 Testing & Evaluation Runner ===")
    root_yolo = ROOT_DIR / "yolo11n.pt"
    person_weights = root_yolo if root_yolo.exists() else Path("yolo11n.pt")

    perf = benchmark_person_detector(person_weights, runs=15)
    ppe_eval = evaluate_yolo_dataset(PPE_MODEL_PATH, ROOT_DIR / "datasets" / "ppe" / "data.yaml")
    fire_eval = evaluate_yolo_dataset(FIRE_MODEL_PATH, ROOT_DIR / "datasets" / "fire" / "data.yaml")

    # Verify fall & PPE & fire module instantiation cleanly
    _ = PPEDetector(model_path=PPE_MODEL_PATH)
    _ = FallDetector()
    _ = FireSmokeDetector(model_path=FIRE_MODEL_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_summary.json"
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_cpu_benchmark": perf,
        "ppe_dataset_evaluation": ppe_eval,
        "fire_smoke_dataset_evaluation": fire_eval,
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"CPU Benchmark: {perf}")
    print(f"PPE Evaluation: {ppe_eval}")
    print(f"Fire/Smoke Evaluation: {fire_eval}")
    print(f"Saved report to: {report_path}")
    print("Phase 11 evaluation run completed cleanly.")


if __name__ == "__main__":
    main()