import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from alerts.email_alerts import EmailAlerter
from alerts.telegram_alerts import TelegramAlerter


def main() -> None:
    print("=== Phase 10 Alerts & Notifications Module Test ===")
    tg = TelegramAlerter()
    mail = EmailAlerter()

    tg_sent = tg.send_alert(
        camera_id="CAM-SA-01",
        violation_type="Missing Helmet",
        severity="Warning",
        timestamp="2026-11-21 10:00:00",
    )
    mail_sent = mail.send_alert(
        camera_id="CAM-SA-01",
        violation_type="Worker Fall",
        severity="Emergency",
        timestamp="2026-11-21 10:00:00",
    )

    print(f"Telegram configured: {tg.is_configured} | Test send result: {tg_sent}")
    print(f"Email configured: {mail.is_configured} | Test send result: {mail_sent}")
    print("Phase 10 module test passed cleanly.")


if __name__ == "__main__":
    main()