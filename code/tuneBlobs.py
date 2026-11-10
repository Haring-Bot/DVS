import cv2
import numpy as np
import os

def nothing(x):
    pass

# Pick one sample frame to tune on
image_path = "/home/julian/rpg_e2vid/results/Screenshot from events_preview.mp4.png"

if not os.path.exists(image_path):
    debug_dir = "/home/julian/rpg_e2vid/results/debug_event_frames/"
    if os.path.exists(debug_dir):
        files = sorted(f for f in os.listdir(debug_dir) if f.endswith(".jpg"))
        if files:
            image_path = os.path.join(debug_dir, files[0])

print(f"Loading frame for single-frame tuning: {image_path}")
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image from {image_path}.")
    raise SystemExit(1)

window_name = "Single-Frame Blob Tuner"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Trackbars
cv2.createTrackbar("minArea", window_name, 100, 5000, nothing)
cv2.createTrackbar("maxArea", window_name, 3000, 15000, nothing)
cv2.createTrackbar("minCirc (x100)", window_name, 40, 100, nothing)
cv2.createTrackbar("minConv (x100)", window_name, 75, 100, nothing)
cv2.createTrackbar("minInertia (x100)", window_name, 30, 100, nothing)

print("\nInstructions:")
print("1. Adjust the sliders.")
print("2. Watch the detected blobs update live.")
print("3. Press 'q' or ESC to save the final values.\n")

while True:
    min_area = cv2.getTrackbarPos("minArea", window_name)
    max_area = cv2.getTrackbarPos("maxArea", window_name)
    min_circ = cv2.getTrackbarPos("minCirc (x100)", window_name) / 100.0
    min_conv = cv2.getTrackbarPos("minConv (x100)", window_name) / 100.0
    min_inertia = max(0.01, cv2.getTrackbarPos("minInertia (x100)", window_name) / 100.0)

    if min_area >= max_area:
        max_area = min_area + 1

    # Same preprocessing path as the real finder
    gray = img[:, :, 0]
    kernel = np.ones((19, 19), np.uint8)
    processed = cv2.dilate(gray, kernel, iterations=1)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255

    params.minArea = float(min_area)
    params.maxArea = float(max_area)

    params.filterByCircularity = True
    params.minCircularity = min_circ

    params.filterByConvexity = True
    params.minConvexity = min_conv

    params.filterByInertia = True
    params.minInertiaRatio = min_inertia
    params.maxInertiaRatio = 1.0

    params.minDistBetweenBlobs = 20.0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(processed)

    pattern_size = (2, 5)

    ret_grid, corners = cv2.findCirclesGrid(
        processed,
        pattern_size,
        flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
        blobDetector=detector,
    )

    display = img.copy()

    if ret_grid:
        cv2.drawChessboardCorners(display, pattern_size, corners, ret_grid)
        cv2.putText(
            display,
            f"Asymmetric Grid: FOUND ({len(corners)})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            display,
            "Asymmetric Grid: NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # Draw keypoints on the display image
    cv2.drawKeypoints(
        display,
        keypoints,
        display,
        (0, 255, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    h, w, _ = display.shape
    panel = np.zeros((h, 360, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "SINGLE-FRAME TUNER", (15, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, f"Blobs Found: {len(keypoints)}", (15, 65), font, 0.6, (0, 255, 0), 2)
    cv2.putText(panel, f"minArea: {min_area}", (15, 120), font, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"maxArea: {max_area}", (15, 160), font, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"minCircularity: {min_circ:.2f}", (15, 200), font, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"minConvexity: {min_conv:.2f}", (15, 240), font, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"minInertiaRatio: {min_inertia:.2f}", (15, 280), font, 0.55, (200, 200, 200), 1)

    grid_status = "FOUND" if ret_grid else "NOT FOUND"
    grid_color = (0, 255, 0) if ret_grid else (0, 0, 255)
    cv2.putText(panel, f"Grid: {grid_status}", (15, 320), font, 0.55, grid_color, 1)
    cv2.putText(panel, f"Grid corners: {len(corners) if ret_grid else 0}", (15, 360), font, 0.55, (255,255,255), 1)

    cv2.putText(panel, "[Q] Quit and save", (15, h - 30), font, 0.5, (0, 165, 255), 1)

    combined = np.hstack((display, panel))
    cv2.imshow(window_name, combined)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()

print("\n=== Final Tuned Parameters ===")
print(f"params.minArea = {min_area}")
print(f"params.maxArea = {max_area}")
print(f"params.minCircularity = {min_circ}")
print(f"params.minConvexity = {min_conv}")
print(f"params.minInertiaRatio = {min_inertia}")