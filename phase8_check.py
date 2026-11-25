import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ALERT_COOLDOWN_SECONDS, DB_PATH, INCIDENTS_IMG_DIR
from database.db import SafetyDatabase


def main() -> None:
    print("=== Phase 8 Database, Auth & Incident Cooldown Test ===")
    db = SafetyDatabase(
        db_path=DB_PATH,
        incidents_img_dir=INCIDENTS_IMG_DIR,
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
    )

    # 1. Verify authentication & role-based access
    admin_user = db.authenticate_user("admin", "Admin@123")
    wrong_user = db.authenticate_user("admin", "WrongPassword")
    assert admin_user is not None and admin_user["role"] == "admin"
    assert wrong_user is None

    # 2. Verify incident logging + cooldown suppression on immediate repeat
    sample_img = np.zeros((360, 640, 3), dtype=np.uint8)
    first_id = db.log_incident(
        camera_id="CAM-SA-01",
        violation_type="Missing Helmet",
        severity="Warning",
        confidence=0.88,
        frame=sample_img,
        notes="Phase 8 test violation",
    )
    second_id = db.log_incident(
        camera_id="CAM-SA-01",
        violation_type="Missing Helmet",
        severity="Warning",
        confidence=0.89,
        frame=sample_img,
        notes="Duplicate within cooldown",
    )

    assert first_id is not None
    assert second_id is None  # Suppressed by cooldown

    db.record_safety_stat("CAM-SA-01", workers_detected=4, compliance_pct=75.0)
    rows = db.get_incidents(camera_id="CAM-SA-01", limit=5)

    print(f"Auth role verified: {admin_user['role']} | Logged incident ID: {first_id}")
    print(f"Cooldown suppressed duplicate: {second_id is None} | Total rows fetched: {len(rows)}")
    print("Phase 8 module test passed cleanly.")


if __name__ == "__main__":
    main()