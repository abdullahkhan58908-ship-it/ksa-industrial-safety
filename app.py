from datetime import datetime
from pathlib import Path
import importlib
import sqlite3
import sys
import tempfile
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

try:
    import winsound
except Exception:
    winsound = None

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIDENCE, PPE_MODEL_PATH
from models.ppe_detection import PPEDetector

DB_DIR = ROOT_DIR / "incidents_db"
SNAPSHOTS_DIR = DB_DIR / "snapshots"
DB_PATH = DB_DIR / "safety_incidents.db"

st.set_page_config(
    page_title="KSA Smart Industrial Safety | Operations Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_modern_saas_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f6f9;
            color: #111827;
            font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #0b1320;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] * {
            color: #cbd5e1 !important;
        }
        .brand-box {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.35rem 0.25rem 0.9rem 0.25rem;
            margin-bottom: 0.65rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        }
        .brand-badge {
            width: 36px;
            height: 36px;
            border-radius: 9px;
            background: #0d9488;
            color: #ffffff !important;
            font-weight: 800;
            font-size: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .brand-name {
            font-size: 0.96rem;
            font-weight: 700;
            color: #f8fafc !important;
            line-height: 1.15;
        }
        .brand-sub {
            font-size: 0.7rem;
            color: #94a3b8 !important;
        }
        .side-group-title {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b !important;
            margin: 0.85rem 0 0.32rem 0.25rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.18rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            width: 100%;
            margin: 0 !important;
            padding: 0.5rem 0.7rem !important;
            border-radius: 7px !important;
            background: transparent !important;
            border: none !important;
            transition: all 0.15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.06) !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: #0d9488 !important;
            color: #ffffff !important;
            font-weight: 600;
        }
        .side-alert-card {
            margin-top: 1rem;
            padding: 0.7rem 0.8rem;
            border-radius: 8px;
            background: rgba(13, 148, 136, 0.14);
            border: 1px solid rgba(20, 184, 166, 0.35);
            font-size: 0.75rem;
            color: #99f6e4 !important;
        }
        .top-greeting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 1.05rem;
        }
        .greeting-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            line-height: 1.25;
        }
        .greeting-meta {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.18rem;
        }
        .top-right-pills {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .status-chip {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 0.32rem 0.72rem;
            font-size: 0.76rem;
            font-weight: 600;
            color: #1e293b;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .avatar-chip {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: #0d9488;
            color: #ffffff;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .saas-kpi {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            height: 100%;
        }
        .kpi-top-teal { border-top: 3px solid #0d9488; }
        .kpi-top-blue { border-top: 3px solid #2563eb; }
        .kpi-top-amber { border-top: 3px solid #f59e0b; }
        .kpi-smart-alert {
            background: #ffffff;
            border: 1.5px dashed #14b8a6;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            height: 100%;
        }
        .saas-kpi-label {
            font-size: 0.76rem;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 0.3rem;
        }
        .saas-kpi-val {
            font-size: 1.65rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
            margin-bottom: 0.3rem;
        }
        .saas-kpi-trend-up {
            font-size: 0.73rem;
            font-weight: 600;
            color: #059669;
        }
        .saas-kpi-trend-warn {
            font-size: 0.73rem;
            font-weight: 600;
            color: #d97706;
        }
        .saas-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.9rem;
        }
        .saas-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .saas-card-sub {
            font-size: 0.74rem;
            color: #64748b;
            margin-bottom: 0.75rem;
        }
        .metric-row {
            margin-bottom: 0.72rem;
        }
        .metric-row-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.25rem;
        }
        .metric-track {
            width: 100%;
            height: 7px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
        }
        .metric-fill {
            height: 100%;
            background: #0d9488;
            border-radius: 999px;
        }
        .standby-box {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 1.3rem 0.9rem;
            text-align: center;
            color: #475569;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_modern_saas_theme()


def get_current_shift() -> str:
    hour = datetime.now().hour
    return "Day Shift (06:00-18:00)" if 6 <= hour < 18 else "Night Shift (18:00-06:00)"


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'HIGH',
                shift_info TEXT NOT NULL DEFAULT 'Day Shift',
                ptw_status TEXT NOT NULL DEFAULT 'Active PTW',
                status TEXT NOT NULL DEFAULT 'OPEN',
                details TEXT NOT NULL,
                snapshot_path TEXT NOT NULL
            )
            """
        )
        # Safe schema migration for existing DB files
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(incidents)").fetchall()
        }
        migrations = {
            "severity": "ALTER TABLE incidents ADD COLUMN severity TEXT NOT NULL DEFAULT 'HIGH'",
            "shift_info": "ALTER TABLE incidents ADD COLUMN shift_info TEXT NOT NULL DEFAULT 'Day Shift'",
            "ptw_status": "ALTER TABLE incidents ADD COLUMN ptw_status TEXT NOT NULL DEFAULT 'Active PTW'",
            "status": "ALTER TABLE incidents ADD COLUMN status TEXT NOT NULL DEFAULT 'OPEN'",
        }
        for col_name, ddl in migrations.items():
            if col_name not in existing_cols:
                conn.execute(ddl)
        conn.commit()


def fetch_incidents_from_db(limit: int = 200) -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, source, type, severity, shift_info, ptw_status, status, details, snapshot_path
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_incident_status(incident_id: int, new_status: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE incidents SET status = ? WHERE id = ?",
            (new_status, int(incident_id)),
        )
        conn.commit()


def save_incident_with_red_circle(
    bgr_frame: np.ndarray,
    source: str,
    incident_type: str,
    severity: str,
    shift_info: str,
    ptw_status: str,
    details: str,
    alert_boxes: list[tuple[int, int, int, int]],
) -> str:
    init_db()
    snap_img = bgr_frame.copy()
    h_img, w_img = snap_img.shape[:2]

    if alert_boxes:
        for (x1, y1, x2, y2) in alert_boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = max(24, int(max(x2 - x1, y2 - y1) * 0.62))
            cv2.circle(snap_img, (cx, cy), radius, (0, 0, 255), 4)
            cv2.circle(snap_img, (cx, cy), max(8, radius // 4), (0, 0, 255), 2)
    else:
        cx, cy = w_img // 2, h_img // 2
        radius = max(35, min(w_img, h_img) // 6)
        cv2.circle(snap_img, (cx, cy), radius, (0, 0, 255), 4)

    cv2.putText(
        snap_img,
        f"[{severity}] {incident_type}",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255),
        2,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_type = "".join(c if c.isalnum() else "_" for c in incident_type)
    file_name = f"{stamp}_{safe_type}.jpg"
    full_snap_path = SNAPSHOTS_DIR / file_name
    cv2.imwrite(str(full_snap_path), snap_img)
    rel_path = f"incidents_db/snapshots/{file_name}"

    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO incidents
                (timestamp, source, type, severity, shift_info, ptw_status, status, details, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """,
            (ts_human, source, incident_type, severity, shift_info, ptw_status, details, rel_path),
        )
        conn.commit()

    return rel_path


init_db()

if "last_alarm_beep_ts" not in st.session_state:
    st.session_state.last_alarm_beep_ts = 0.0
if "fall_state" not in st.session_state:
    st.session_state.fall_state = {"prev_centers": [], "streak": 0}
if "prev_smoke_gray" not in st.session_state:
    st.session_state.prev_smoke_gray = None
if "show_incident_details" not in st.session_state:
    st.session_state.show_incident_details = False


def reset_stream_trackers() -> None:
    st.session_state.fall_state = {"prev_centers": [], "streak": 0}
    st.session_state.prev_smoke_gray = None


def play_alert_alarm(freq: int = 1550) -> None:
    now = time.time()
    if now - float(st.session_state.last_alarm_beep_ts) < 1.2:
        return
    st.session_state.last_alarm_beep_ts = now
    if winsound is not None:
        try:
            winsound.Beep(freq, 350)
        except Exception:
            pass


@st.cache_resource
def load_ppe_detector(conf: float):
    if not PPE_MODEL_PATH.exists():
        return None
    return PPEDetector(
        model_path=PPE_MODEL_PATH,
        confidence_threshold=conf,
        imgsz=640,
    )


@st.cache_resource
def load_general_yolo():
    general_weights = ROOT_DIR / "yolo11n.pt"
    if not general_weights.exists():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(general_weights))
    except Exception:
        return None


def check_module_availability(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def run_advanced_fall_check(
    bgr_frame: np.ndarray,
    summary,
    conf_thresh: float,
    sensitivity: str = "High (Video & CCTV)",
) -> tuple[bool, list[tuple[int, int, int, int]], str]:
    frame_h, frame_w = bgr_frame.shape[:2]
    fall_boxes: list[tuple[int, int, int, int]] = []
    reasons: list[str] = []
    candidate_boxes: list[tuple[int, int, int, int]] = []

    ratio_thresh = 1.02 if "High" in sensitivity else 1.25
    min_box_w = 35 if "High" in sensitivity else 55
    required_streak = 1 if "High" in sensitivity else 2

    try:
        fall_mod = importlib.import_module("models.fall_detection")
        for attr_name in ("FallDetector", "detect_fall", "check_fall"):
            obj = getattr(fall_mod, attr_name, None)
            if obj is not None and callable(obj):
                if isinstance(obj, type):
                    inst = obj()
                    for method_name in ("detect", "predict", "process_frame", "check"):
                        fn = getattr(inst, method_name, None)
                        if callable(fn):
                            res = fn(bgr_frame)
                            if isinstance(res, bool) and res:
                                reasons.append("FALL-MOD")
                            elif isinstance(res, dict) and (
                                res.get("fall_detected") or res.get("fall")
                            ):
                                reasons.append("FALL-MOD")
                            break
                else:
                    res = obj(bgr_frame)
                    if isinstance(res, bool) and res:
                        reasons.append("FALL-MOD")
                    elif isinstance(res, dict) and (
                        res.get("fall_detected") or res.get("fall")
                    ):
                        reasons.append("FALL-MOD")
                break
    except Exception:
        pass

    dets = getattr(summary, "detections", None) or [] if summary is not None else []
    for det in dets:
        label = str(getattr(det, "label", getattr(det, "class_name", ""))).lower().strip()
        bbox = getattr(det, "bbox", None)
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            if "fall" in label:
                candidate_boxes.append((x1, y1, x2, y2))
                reasons.append("CLASS:FALL")
            elif ("person" in label or "worker" in label or "human" in label) and w >= min_box_w:
                if w >= h * ratio_thresh:
                    candidate_boxes.append((x1, y1, x2, y2))
                    reasons.append("LYING-POSTURE")

    person_centers: list[tuple[float, float, float]] = []
    gen_model = load_general_yolo()
    if gen_model is not None:
        try:
            results = gen_model(bgr_frame, conf=max(0.15, conf_thresh * 0.8), verbose=False)
            for r in results:
                names = getattr(r, "names", {}) or {}
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = str(names.get(cls_id, "")).lower()
                    if cls_name == "person":
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        person_centers.append((cx, cy, float(h)))

                        if w >= min_box_w and w >= h * ratio_thresh:
                            candidate_boxes.append((x1, y1, x2, y2))
                            reasons.append("HORIZONTAL-BODY")

                        prev_list = st.session_state.fall_state.get("prev_centers", [])
                        for (pcx, pcy, ph) in prev_list:
                            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                            dy = cy - pcy
                            if dist < max(120.0, frame_w * 0.25) and dy > frame_h * 0.12 and w >= h * 0.85:
                                candidate_boxes.append((x1, y1, x2, y2))
                                reasons.append("RAPID-DROP")
                                break
        except Exception:
            pass

    st.session_state.fall_state["prev_centers"] = person_centers

    if candidate_boxes or reasons:
        st.session_state.fall_state["streak"] += 1
    else:
        st.session_state.fall_state["streak"] = max(0, st.session_state.fall_state["streak"] - 1)

    fall_confirmed = st.session_state.fall_state["streak"] >= required_streak and (
        len(candidate_boxes) > 0 or len(reasons) > 0
    )

    seen = set()
    for box in candidate_boxes:
        if box not in seen:
            seen.add(box)
            fall_boxes.append(box)

    reason_text = "+".join(sorted(set(reasons))) if reasons else ""
    return fall_confirmed, fall_boxes, reason_text


def run_fire_smoke_check(bgr_frame: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int, str]]]:
    boxes_out: list[tuple[int, int, int, int, str]] = []
    detected = False

    try:
        fire_mod = importlib.import_module("models.fire_detection")
        for attr_name in ("FireDetector", "detect_fire", "check_fire_smoke", "detect_fire_smoke"):
            obj = getattr(fire_mod, attr_name, None)
            if obj is not None and callable(obj):
                if isinstance(obj, type):
                    inst = obj()
                    for method_name in ("detect", "predict", "process_frame", "check"):
                        fn = getattr(inst, method_name, None)
                        if callable(fn):
                            res = fn(bgr_frame)
                            if isinstance(res, bool) and res:
                                return True, []
                            if isinstance(res, dict) and (
                                res.get("fire")
                                or res.get("smoke")
                                or res.get("fire_detected")
                                or res.get("smoke_detected")
                            ):
                                return True, []
                            break
                else:
                    res = obj(bgr_frame)
                    if isinstance(res, bool) and res:
                        return True, []
                    if isinstance(res, dict) and (
                        res.get("fire")
                        or res.get("smoke")
                        or res.get("fire_detected")
                        or res.get("smoke_detected")
                    ):
                        return True, []
                break
    except Exception:
        pass

    h_img, w_img = bgr_frame.shape[:2]
    frame_area = float(max(1, h_img * w_img))

    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    flame_mask = (
        ((h_ch <= 24) | (h_ch >= 175))
        & (s_ch >= 145)
        & (v_ch >= 200)
    ).astype(np.uint8) * 255
    flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    flame_mask = cv2.dilate(flame_mask, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(flame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= max(900.0, frame_area * 0.012):
            x, y, w, h = cv2.boundingRect(cnt)
            detected = True
            boxes_out.append((x, y, x + w, y + h, "FIRE ALERT!"))

    prev_gray = st.session_state.get("prev_smoke_gray")
    curr_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    curr_gray_blur = cv2.GaussianBlur(curr_gray, (9, 9), 0)
    st.session_state.prev_smoke_gray = curr_gray_blur

    if prev_gray is not None and prev_gray.shape == curr_gray_blur.shape:
        motion_mask = (cv2.absdiff(curr_gray_blur, prev_gray) >= 12).astype(np.uint8) * 255
        smoke_color_mask = ((s_ch <= 30) & (v_ch >= 140) & (v_ch <= 220)).astype(np.uint8) * 255
        smoke_mask = cv2.bitwise_and(smoke_color_mask, motion_mask)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        smoke_mask = cv2.dilate(smoke_mask, np.ones((13, 13), np.uint8), iterations=1)

        s_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in s_contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if y > int(h_img * 0.18) and area >= max(6500.0, frame_area * 0.045):
                detected = True
                boxes_out.append((x, y, x + w, y + h, "SMOKE ALERT!"))

    return detected, boxes_out


def run_restricted_zone_check(
    bgr_frame: np.ndarray,
    summary,
    conf_thresh: float,
) -> tuple[bool, tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    h_img, w_img = bgr_frame.shape[:2]
    zx1 = int(w_img * 0.45)
    zy1 = int(h_img * 0.25)
    zx2 = int(w_img * 0.92)
    zy2 = int(h_img * 0.88)
    zone_rect = (zx1, zy1, zx2, zy2)

    intruding_boxes: list[tuple[int, int, int, int]] = []
    candidate_boxes: list[tuple[int, int, int, int]] = []

    dets = getattr(summary, "detections", None) or [] if summary is not None else []
    for det in dets:
        bbox = getattr(det, "bbox", None)
        if bbox and len(bbox) == 4:
            candidate_boxes.append(tuple(int(v) for v in bbox))

    gen_model = load_general_yolo()
    if gen_model is not None:
        try:
            results = gen_model(bgr_frame, conf=max(0.15, conf_thresh * 0.8), verbose=False)
            for r in results:
                names = getattr(r, "names", {}) or {}
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = str(names.get(cls_id, "")).lower()
                    if cls_name == "person":
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                        candidate_boxes.append((x1, y1, x2, y2))
        except Exception:
            pass

    for (x1, y1, x2, y2) in candidate_boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        overlaps = not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)
        center_inside = (zx1 <= cx <= zx2) and (zy1 <= cy <= zy2)
        if center_inside or overlaps:
            intruding_boxes.append((x1, y1, x2, y2))

    return len(intruding_boxes) > 0, zone_rect, intruding_boxes


def process_frame(
    bgr_frame: np.ndarray,
    detector: PPEDetector | None,
    conf_thresh: float,
    use_ppe: bool,
    use_fall: bool,
    fall_sens: str,
    use_fire: bool,
    use_zone: bool,
):
    rendered = bgr_frame.copy()
    counts = {"helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "person": 0}
    extra_alerts = []
    ppe_viol_boxes: list[tuple[int, int, int, int]] = []
    fall_boxes_out: list[tuple[int, int, int, int]] = []
    fire_boxes_out: list[tuple[int, int, int, int]] = []
    zone_boxes_out: list[tuple[int, int, int, int]] = []
    summary = None

    if use_ppe and detector is not None:
        summary = detector.detect(bgr_frame)
        rendered = detector.draw_detections(bgr_frame, summary)

        counts["helmet"] = int(getattr(summary, "helmet_count", 0) or 0)
        counts["no_helmet"] = int(getattr(summary, "no_helmet_count", 0) or 0)

        dets = getattr(summary, "detections", None) or []
        for det in dets:
            label = str(getattr(det, "label", getattr(det, "class_name", ""))).lower().strip()
            conf_score = float(getattr(det, "confidence", 0.0))
            bbox = getattr(det, "bbox", None)

            if "person" in label:
                counts["person"] += 1
            if "no_vest" in label or "without_vest" in label or "no-vest" in label:
                counts["no_vest"] += 1
            elif "vest" in label:
                counts["vest"] += 1

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                is_violation = (
                    "no_helmet" in label
                    or "no_vest" in label
                    or "without" in label
                    or "violation" in label
                )
                color = (0, 0, 255) if is_violation else (13, 148, 136)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    rendered,
                    f"{label} {conf_score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                )
                if is_violation:
                    ppe_viol_boxes.append((x1, y1, x2, y2))

    if use_fall:
        fall_detected, f_boxes, reason_txt = run_advanced_fall_check(
            bgr_frame, summary, conf_thresh, fall_sens
        )
        fall_boxes_out.extend(f_boxes)
        for (fx1, fy1, fx2, fy2) in f_boxes:
            cv2.rectangle(rendered, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
            tag_label = f"FALL ALERT! [{reason_txt}]" if reason_txt else "FALL ALERT!"
            cv2.putText(
                rendered,
                tag_label,
                (fx1, max(22, fy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
        if fall_detected:
            extra_alerts.append(f"FALL DETECTED ({reason_txt or 'ALERT'})!")

    if use_fire:
        fire_smoke_detected, fs_boxes = run_fire_smoke_check(bgr_frame)
        for (x1, y1, x2, y2, tag) in fs_boxes:
            fire_boxes_out.append((x1, y1, x2, y2))
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 69, 255), 3)
            cv2.putText(
                rendered,
                tag,
                (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 69, 255),
                2,
            )
        if fire_smoke_detected:
            extra_alerts.append("FIRE / SMOKE DETECTED!")
        elif check_module_availability("models.fire_detection"):
            extra_alerts.append("FIRE-MOD: ACTIVE")

    if use_zone:
        zone_intrusion, z_rect, intr_boxes = run_restricted_zone_check(bgr_frame, summary, conf_thresh)
        zx1, zy1, zx2, zy2 = z_rect
        z_color = (0, 0, 255) if zone_intrusion else (13, 148, 136)
        cv2.rectangle(rendered, (zx1, zy1), (zx2, zy2), z_color, 2)
        cv2.putText(
            rendered,
            "RESTRICTED ZONE [INTRUSION!]" if zone_intrusion else "RESTRICTED ZONE",
            (zx1 + 6, max(22, zy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            z_color,
            2,
        )
        zone_boxes_out.extend(intr_boxes)
        for (ix1, iy1, ix2, iy2) in intr_boxes:
            cv2.rectangle(rendered, (ix1, iy1), (ix2, iy2), (0, 0, 255), 3)
            cv2.putText(
                rendered,
                "ZONE INTRUSION!",
                (ix1, max(22, iy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 0, 255),
                2,
            )
        if zone_intrusion:
            extra_alerts.append("ZONE INTRUSION DETECTED!")
        elif check_module_availability("detection.zone_detection"):
            extra_alerts.append("ZONE-MOD: ACTIVE")

    boxes_map = {
        "ppe": ppe_viol_boxes,
        "fall": fall_boxes_out,
        "fire": fire_boxes_out,
        "zone": zone_boxes_out,
    }
    return rendered, counts, extra_alerts, boxes_map


def handle_alerts_and_db(
    raw_bgr_frame: np.ndarray,
    source_name: str,
    cnt: dict,
    extras: list[str],
    boxes_map: dict,
    alarm_on: bool,
    shift_info: str,
    ptw_status: str,
    allow_ppe: bool = True,
    allow_fall: bool = True,
    allow_fire: bool = True,
    allow_zone: bool = True,
) -> None:
    has_ppe_viol = (cnt["no_helmet"] > 0 or cnt["no_vest"] > 0) and allow_ppe
    has_fall = any("FALL DETECTED" in e for e in extras) and allow_fall
    has_fire = any("FIRE / SMOKE DETECTED" in e for e in extras) and allow_fire
    has_zone = any("ZONE INTRUSION DETECTED" in e for e in extras) and allow_zone

    if (has_ppe_viol or has_fall or has_fire or has_zone) and alarm_on:
        play_alert_alarm(1650 if (has_fire or has_zone) else 1500)

    if has_ppe_viol:
        sev = "HIGH" if cnt["no_helmet"] > 0 else "MEDIUM"
        save_incident_with_red_circle(
            raw_bgr_frame,
            source_name,
            "PPE VIOLATION (NO HELMET/VEST)",
            sev,
            shift_info,
            ptw_status,
            f"no_helmet={cnt['no_helmet']}, no_vest={cnt['no_vest']}",
            boxes_map.get("ppe", []),
        )
    if has_fall:
        save_incident_with_red_circle(
            raw_bgr_frame,
            source_name,
            "FALL EMERGENCY",
            "CRITICAL",
            shift_info,
            ptw_status,
            "Worker fall detected",
            boxes_map.get("fall", []),
        )
    if has_fire:
        save_incident_with_red_circle(
            raw_bgr_frame,
            source_name,
            "FIRE / SMOKE ALERT",
            "CRITICAL",
            shift_info,
            ptw_status,
            "Fire or smoke detected",
            boxes_map.get("fire", []),
        )
    if has_zone:
        save_incident_with_red_circle(
            raw_bgr_frame,
            source_name,
            "RESTRICTED ZONE ALERT",
            "HIGH",
            shift_info,
            ptw_status,
            "Object/person detected inside restricted zone",
            boxes_map.get("zone", []),
        )


with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-badge">S</div>
            <div>
                <div class="brand-name">SafeSite Cloud</div>
                <div class="brand-sub">Industrial HSE Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-group-title">Menu</div>', unsafe_allow_html=True)
    page = st.radio(
        "MENU",
        ["Dashboard", "Live Monitor", "Incidents & Snapshots", "System Status"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-group-title">Input Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "INPUT MODE",
        ["Upload Image (Pic)", "Upload Video", "Live Webcam"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-group-title">Site &amp; PTW Controls</div>', unsafe_allow_html=True)
    branch_name = st.selectbox(
        "BRANCH",
        ["Main Branch — Jubail", "North Yard — Yanbu", "East Hub — Dammam"],
        index=0,
    )
    ptw_status = st.selectbox(
        "PERMIT-TO-WORK (PTW)",
        ["Active PTW Verified", "Hot Work Permit", "Confined Space PTW", "No PTW / Unverified"],
        index=0,
    )
    conf_val = st.slider("Confidence Threshold", 0.10, 0.90, float(DEFAULT_CONFIDENCE), 0.05)

    st.markdown('<div class="side-group-title">Detection Modules</div>', unsafe_allow_html=True)
    mod_ppe = st.checkbox("PPE (Helmet / Vest)", value=True)
    mod_fall = st.checkbox("Fall Detection", value=True)
    mod_fire = st.checkbox("Fire & Smoke", value=True)
    mod_zone = st.checkbox("Restricted Zone", value=True)
    alarm_sound_on = st.checkbox("Alert Alarm Sound", value=True)
    fall_sens = st.selectbox(
        "Fall Sensitivity",
        ["High (Video & CCTV)", "Standard (Low False-Alarm)"],
        index=0,
    )

    cam_index = 0
    live_cam_active = False
    if input_mode == "Live Webcam":
        cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        live_cam_active = st.toggle("Live Camera On / Off", value=False, key="live_cam_toggle")

db_incidents = fetch_incidents_from_db(limit=200)
total_incidents = len(db_incidents)
open_incidents = sum(1 for x in db_incidents if x.get("status", "OPEN") == "OPEN")
critical_incidents = sum(1 for x in db_incidents if x.get("severity") == "CRITICAL")
ppe_violations_logged = sum(1 for x in db_incidents if "PPE" in x["type"])
fall_incidents_logged = sum(1 for x in db_incidents if "FALL" in x["type"])
hazard_incidents_logged = sum(
    1 for x in db_incidents if any(k in x["type"] for k in ("FIRE", "SMOKE", "ZONE"))
)
compliance_rate = max(0.0, round(100.0 - (total_incidents * 2.5), 1))
active_shift = get_current_shift()

with st.sidebar:
    latest_alert_txt = db_incidents[0]["type"] if db_incidents else "No active alert"
    st.markdown(
        f"""
        <div class="side-alert-card">
            <b>KSA HSE Shift Tracker</b><br/>
            Shift: <b>{active_shift.split(" ")[0]}</b><br/>
            Open Alerts: <b>{open_incidents}</b> • Critical: <b>{critical_incidents}</b><br/>
            Latest: {latest_alert_txt}
        </div>
        """,
        unsafe_allow_html=True,
    )

now_str = datetime.now().strftime("%A %d %b %Y • %I:%M %p")
st.markdown(
    f"""
    <div class="top-greeting-row">
        <div>
            <h1 class="greeting-title">Good afternoon, Safety Lead — {branch_name}</h1>
            <div class="greeting-meta">{now_str} • {active_shift} • {ptw_status}</div>
        </div>
        <div class="top-right-pills">
            <span class="status-chip">Open: {open_incidents}</span>
            <span class="status-chip">Critical: {critical_incidents}</span>
            <span class="status-chip">Compliance: {compliance_rate:.1f}%</span>
            <span class="avatar-chip">A</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="saas-kpi kpi-top-teal">
            <div class="saas-kpi-label">Safety Compliance</div>
            <div class="saas-kpi-val">{compliance_rate:.1f}%</div>
            <div class="saas-kpi-trend-up">&#9650; Target benchmark &ge; 95.0%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="saas-kpi kpi-top-blue">
            <div class="saas-kpi-label">PPE Violations</div>
            <div class="saas-kpi-val">{ppe_violations_logged}</div>
            <div class="saas-kpi-trend-up">Helmet &amp; vest checks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="saas-kpi kpi-top-amber">
            <div class="saas-kpi-label">Fall &amp; Hazard Alerts</div>
            <div class="saas-kpi-val">{fall_incidents_logged + hazard_incidents_logged}</div>
            <div class="saas-kpi-trend-warn">Critical: {critical_incidents} • Open: {open_incidents}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    smart_msg = (
        f"<b>[{db_incidents[0].get('severity', 'HIGH')}] {db_incidents[0]['type']}</b> ({db_incidents[0].get('status', 'OPEN')})"
        if db_incidents
        else "All monitored zones &amp; workers nominal."
    )
    st.markdown(
        f"""
        <div class="kpi-smart-alert">
            <div class="saas-kpi-label" style="color:#0f766e;">&#129302; AI Smart Alert</div>
            <div style="font-size:0.86rem; font-weight:700; color:#0f172a; margin-bottom:0.25rem;">
                {smart_msg}
            </div>
            <div class="saas-kpi-trend-up">&#8594; PTW &amp; red-circle snapshot logged</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

left_analytics_col, right_monitor_col = st.columns([1.45, 1.05], gap="medium")

with left_analytics_col:
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-title">Safety &amp; Incident Overview</div>
            <div class="saas-card-sub">Logged incident distribution across active modules</div>
        """,
        unsafe_allow_html=True,
    )
    chart_data = {
        "PPE": ppe_violations_logged,
        "Fall": fall_incidents_logged,
        "Fire/Zone": hazard_incidents_logged,
        "Total": total_incidents,
    }
    st.bar_chart(chart_data, height=195)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-title">Module Readiness &amp; Compliance</div>
            <div class="saas-card-sub">Current operational indicators</div>
        """,
        unsafe_allow_html=True,
    )
    for label_txt, pct in [
        ("PPE Compliance Rate", int(compliance_rate)),
        ("Fall Detection Module", 100 if mod_fall else 25),
        ("Fire & Smoke Module", 100 if mod_fire else 25),
        ("Restricted Zone Guard", 100 if mod_zone else 25),
    ]:
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-row-head"><span>{label_txt}</span><span>{pct}%</span></div>
                <div class="metric-track"><div class="metric-fill" style="width:{max(0, min(100, pct))}%;"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_monitor_col:
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="saas-card-title">Live Feed &amp; Media Inspector</div>
            <div class="saas-card-sub">Mode: {input_mode} • {ptw_status}</div>
        """,
        unsafe_allow_html=True,
    )

    detector = load_ppe_detector(conf_val) if mod_ppe else None
    if mod_ppe and detector is None:
        st.error(f"Missing weights file: {PPE_MODEL_PATH}")

    elif input_mode == "Live Webcam":
        cam_frame_slot = st.empty()
        cam_stats_slot = st.empty()
        if not live_cam_active:
            reset_stream_trackers()
            cam_frame_slot.markdown(
                '<div class="standby-box">Camera Standby<br/>Sidebar se <b>Live Camera On / Off</b> toggle on karein.</div>',
                unsafe_allow_html=True,
            )
        else:
            reset_stream_trackers()
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                cam_frame_slot.warning(f"Camera {int(cam_index)} open nahi ho saka.")
            else:
                tick = 0
                while st.session_state.get("live_cam_toggle", False):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    rendered, cnt, extras, boxes_map = process_frame(
                        frame, detector, conf_val, mod_ppe, mod_fall, fall_sens, mod_fire, mod_zone
                    )
                    rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                    cam_frame_slot.image(rgb, channels="RGB", width="stretch")

                    extra_str = f" | {' | '.join(extras)}" if extras else ""
                    cam_stats_slot.markdown(
                        f"`H: {cnt['helmet']}` | `NO-H: {cnt['no_helmet']}` | `V: {cnt['vest']}` | `NO-V: {cnt['no_vest']}`{extra_str}"
                    )
                    handle_alerts_and_db(
                        frame,
                        f"{branch_name} CAM-{int(cam_index)}",
                        cnt,
                        extras,
                        boxes_map,
                        alarm_sound_on,
                        active_shift,
                        ptw_status,
                        allow_ppe=(tick % 25 == 0),
                        allow_fall=(tick % 20 == 0),
                        allow_fire=(tick % 20 == 0),
                        allow_zone=(tick % 20 == 0),
                    )
                    tick += 1
                cap.release()

    elif input_mode == "Upload Image (Pic)":
        uploaded_img = st.file_uploader(
            "Select Image File",
            type=["jpg", "jpeg", "png", "webp"],
            key="saas_img_uploader",
        )
        if uploaded_img is None:
            st.markdown(
                '<div class="standby-box">Image Standby<br/>Browse files se image upload karein.</div>',
                unsafe_allow_html=True,
            )
        else:
            reset_stream_trackers()
            pil_img = Image.open(uploaded_img).convert("RGB")
            if max(pil_img.size) > 1280:
                pil_img.thumbnail((1280, 1280))
            bgr_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            rendered, cnt, extras, boxes_map = process_frame(
                bgr_frame, detector, conf_val, mod_ppe, mod_fall, fall_sens, mod_fire, mod_zone
            )
            rgb_out = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            st.image(rgb_out, channels="RGB", width="stretch")

            extra_str = f" | {' | '.join(extras)}" if extras else ""
            st.markdown(
                f"`H: {cnt['helmet']}` | `NO-H: {cnt['no_helmet']}` | `V: {cnt['vest']}` | `NO-V: {cnt['no_vest']}`{extra_str}"
            )
            handle_alerts_and_db(
                bgr_frame,
                uploaded_img.name,
                cnt,
                extras,
                boxes_map,
                alarm_sound_on,
                active_shift,
                ptw_status,
            )

    elif input_mode == "Upload Video":
        uploaded_vid = st.file_uploader(
            "Select Video File",
            type=["mp4", "avi", "mov", "mkv"],
            key="saas_vid_uploader",
        )
        if uploaded_vid is None:
            st.markdown(
                '<div class="standby-box">Video Standby<br/>Browse files se video upload karein.</div>',
                unsafe_allow_html=True,
            )
        else:
            run_vid = st.button("Run Safety Video Scan", use_container_width=True)
            vid_frame_slot = st.empty()
            vid_stats_slot = st.empty()
            if run_vid:
                reset_stream_trackers()
                suffix = Path(uploaded_vid.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_vid.read())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    st.warning("Uploaded video open nahi ho saki.")
                else:
                    frame_idx = 0
                    saved_types: set[str] = set()
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        frame_idx += 1
                        if frame_idx % 2 != 0:
                            continue
                        rendered, cnt, extras, boxes_map = process_frame(
                            frame, detector, conf_val, mod_ppe, mod_fall, fall_sens, mod_fire, mod_zone
                        )
                        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                        vid_frame_slot.image(rgb, channels="RGB", width="stretch")

                        extra_str = f" | {' | '.join(extras)}" if extras else ""
                        vid_stats_slot.markdown(
                            f"`F: {frame_idx}` | `H: {cnt['helmet']}` | `NO-H: {cnt['no_helmet']}` | `V: {cnt['vest']}` | `NO-V: {cnt['no_vest']}`{extra_str}"
                        )

                        has_ppe_viol = (cnt["no_helmet"] > 0 or cnt["no_vest"] > 0) and ("ppe" not in saved_types)
                        has_fall = any("FALL DETECTED" in e for e in extras) and ("fall" not in saved_types)
                        has_fire = any("FIRE / SMOKE DETECTED" in e for e in extras) and ("fire" not in saved_types)
                        has_zone = any("ZONE INTRUSION DETECTED" in e for e in extras) and ("zone" not in saved_types)

                        if has_ppe_viol:
                            saved_types.add("ppe")
                        if has_fall:
                            saved_types.add("fall")
                        if has_fire:
                            saved_types.add("fire")
                        if has_zone:
                            saved_types.add("zone")

                        handle_alerts_and_db(
                            frame,
                            f"{uploaded_vid.name} (F:{frame_idx})",
                            cnt,
                            extras,
                            boxes_map,
                            alarm_sound_on,
                            active_shift,
                            ptw_status,
                            allow_ppe=has_ppe_viol,
                            allow_fall=has_fall,
                            allow_fire=has_fire,
                            allow_zone=has_zone,
                        )
                    cap.release()
                    st.success("Video scan complete.")

    st.markdown("</div>", unsafe_allow_html=True)

is_details_open = (page == "Incidents & Snapshots") or st.session_state.show_incident_details

st.markdown(
    """
    <div class="saas-card">
        <div class="saas-card-title">Safety Incident Archive &amp; HSE Audit</div>
        <div class="saas-card-sub">Click option below to open incident table, resolve alerts, export CSV/report &amp; view red-circle snapshots</div>
    """,
    unsafe_allow_html=True,
)

btn_label = (
    "Hide Incident Details"
    if is_details_open
    else f"Show Incident Details ({total_incidents})"
)
if st.button(btn_label, key="toggle_incident_details_btn"):
    st.session_state.show_incident_details = not st.session_state.show_incident_details
    st.rerun()

if is_details_open:
    if db_incidents:
        filter_sev = st.selectbox(
            "Filter by Severity",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM"],
            index=0,
            key="sev_filter_box",
        )
        filtered_rows = (
            db_incidents
            if filter_sev == "ALL"
            else [r for r in db_incidents if r.get("severity") == filter_sev]
        )

        table_rows = [
            {
                "ID": f"#{r['id']}",
                "SEVERITY": r.get("severity", "HIGH"),
                "STATUS": r.get("status", "OPEN"),
                "ALERT TYPE": r["type"],
                "SOURCE": r["source"],
                "SHIFT": r.get("shift_info", active_shift),
                "PTW": r.get("ptw_status", ptw_status),
                "DETAILS": r["details"],
                "TIME": r["timestamp"],
            }
            for r in filtered_rows
        ]
        st.dataframe(table_rows, use_container_width=True)

        # Quick Resolve action for OPEN incidents
        open_options = {
            f"#{r['id']} • {r['type']} ({r['timestamp']})": r["id"]
            for r in db_incidents
            if r.get("status", "OPEN") == "OPEN"
        }
        if open_options:
            res_c1, res_c2 = st.columns([2.2, 1])
            with res_c1:
                selected_inc_label = st.selectbox(
                    "Select Open Alert to Mark Resolved",
                    list(open_options.keys()),
                    key="resolve_inc_select",
                )
            with res_c2:
                st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
                if st.button("Mark Resolved", key="mark_resolved_btn", use_container_width=True):
                    update_incident_status(open_options[selected_inc_label], "RESOLVED")
                    st.rerun()

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            csv_lines = ["id,timestamp,source,type,severity,shift_info,ptw_status,status,details,snapshot_path"]
            for r in db_incidents:
                csv_lines.append(
                    f"{r['id']},\"{r['timestamp']}\",\"{r['source']}\",\"{r['type']}\",\"{r.get('severity','HIGH')}\",\"{r.get('shift_info','')}\",\"{r.get('ptw_status','')}\",\"{r.get('status','OPEN')}\",\"{r['details']}\",\"{r['snapshot_path']}\""
                )
            st.download_button(
                "Download Incidents CSV",
                data="\n".join(csv_lines),
                file_name="safesite_ksa_incidents.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_c2:
            summary_report = (
                f"KSA HSE SHIFT SUMMARY REPORT\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Branch: {branch_name}\n"
                f"Shift: {active_shift}\n"
                f"Compliance Rate: {compliance_rate:.1f}%\n"
                f"Total Incidents: {total_incidents} (Open: {open_incidents}, Critical: {critical_incidents})\n"
                f"PPE Violations: {ppe_violations_logged} | Fall Alerts: {fall_incidents_logged} | Fire/Zone: {hazard_incidents_logged}\n"
            )
            st.download_button(
                "Download HSE Shift Report",
                data=summary_report,
                file_name="ksa_hse_shift_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("**Saved Red-Circle Snapshots**")
        snap_cols = st.columns(3)
        shown = 0
        for r in filtered_rows[:6]:
            p = ROOT_DIR / r["snapshot_path"]
            if p.exists():
                with snap_cols[shown % 3]:
                    st.image(
                        str(p),
                        caption=f"#{r['id']} • [{r.get('severity','HIGH')}] {r['type']} ({r.get('status','OPEN')})",
                        use_container_width=True,
                    )
                shown += 1
    else:
        st.info("Abhi koi incident record save nahi hua. Violation, fall, fire/smoke, ya zone alert par snapshot + DB log save hoga.")

st.markdown("</div>", unsafe_allow_html=True)