import platform
import cv2
import torch
from ultralytics import YOLO


def main() -> None:
    print("=== Phase 1 Environment Check ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

    # Downloads lightweight nano model on first run into current working directory
    model = YOLO("yolo11n.pt")
    print("YOLO11n model loaded successfully.")


if __name__ == "__main__":
    main()