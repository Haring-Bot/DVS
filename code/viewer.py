import cv2
from pathlib import Path


def fit_with_padding(frame, target_w, target_h):
    h, w = frame.shape[:2]

    # Scale to fit inside target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on a black canvas so output is exactly target size
    canvas = cv2.copyMakeBorder(
        resized,
        (target_h - new_h) // 2,
        target_h - new_h - (target_h - new_h) // 2,
        (target_w - new_w) // 2,
        target_w - new_w - (target_w - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas


def stack_videos_vertically(video1_path, video2_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Common canvas: both videos fit fully inside this size
    target_w = max(w1, w2)
    target_h = max(h1, h2)

    print("Playing videos... Press 'q' to exit.")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1_fit = fit_with_padding(frame1, target_w, target_h)
        frame2_fit = fit_with_padding(frame2, target_w, target_h)

        stacked_frame = cv2.vconcat([frame1_fit, frame2_fit])
        cv2.imshow("Stacked Videos (Top / Bottom)", stacked_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stamp = input("Enter time stamp: ")
    recDir = Path(__file__).resolve().parent.parent / "recordings" / stamp
    rgbPath = recDir / f"{stamp}_RGB.mp4"
    dvsPath = recDir / f"{stamp}_DVS.avi"

    stack_videos_vertically(rgbPath, dvsPath)