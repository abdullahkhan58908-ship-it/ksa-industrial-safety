import sys
from pathlib import Path
import cv2

# Ensure project root is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FRAME_HEIGHT, FRAME_WIDTH, TARGET_FPS
from detection.video_processor import VideoProcessor


def main() -> None:
    print("=== Phase 2 Video Processor Test ===")
    print("Opening webcam (source=0). Press 'q' inside the video window to quit.")

    processor = VideoProcessor(
        source=0,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        target_fps=TARGET_FPS,
    )

    if not processor.open():
        print("Webcam not found or could not be opened. If testing with a video file, pass its path as source.")
        return

    try:
        while True:
            ok, frame = processor.read_frame()
            if not ok or frame is None:
                break

            annotated = processor.draw_fps(frame)
            cv2.imshow("Phase 2 - OpenCV Video Test", annotated)

            # Press 'q' to exit cleanly
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        processor.release()
        cv2.destroyAllWindows()
        print("Video stream closed cleanly.")


if __name__ == "__main__":
    main()