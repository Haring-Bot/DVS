import cv2
import numpy as np

# --- Configuration ---
stamp = 202611101716
output_dir = f"/home/julian/Downloads/{stamp}/"
transform_file = f"{output_dir}/{stamp}_rgb_to_dvs_transformation.npz"
rgb_video_path = f"{output_dir}/{stamp}_RGB_cut.mp4"
dvs_video_path = f"{output_dir}/{stamp}_DVS_cut.avi"
overlay_output_path = f"{output_dir}/events_over_rgb.mp4"

# Load calibration & transformation data
with np.load(transform_file) as data:
    R = data["R"]
    T = data["T"]
    rgb_mtx = data["rgb_camera_matrix"]
    dvs_mtx = data["dvs_camera_matrix"]
    dist_rgb = data["rgb_dist_coeffs"] if "rgb_dist_coeffs" in data.files else np.zeros(5)
    dist_dvs = data["dvs_dist_coeffs"] if "dvs_dist_coeffs" in data.files else np.zeros(5)

rgb_cap = cv2.VideoCapture(rgb_video_path)
dvs_cap = cv2.VideoCapture(dvs_video_path)

if not rgb_cap.isOpened() or not dvs_cap.isOpened():
    raise FileNotFoundError("Could not open RGB and/or DVS video files.")

rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)

# We want DVS events projected into the RGB camera view.
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    rgb_mtx, dist_rgb,
    dvs_mtx, dist_dvs,
    (rgb_width, rgb_height),
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
)

# DVS -> RGB mapping
map1, map2 = cv2.initUndistortRectifyMap(
    dvs_mtx, dist_dvs, R2, P2, (rgb_width, rgb_height), cv2.CV_16SC2
)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(overlay_output_path, fourcc, rgb_fps, (rgb_width, rgb_height))

print("Overlaying DVS events onto RGB video...")
frame_idx = 0

while True:
    ret_rgb, rgb_frame = rgb_cap.read()
    ret_dvs, dvs_frame = dvs_cap.read()

    if not ret_rgb or not ret_dvs:
        break

    dvs_gray = cv2.cvtColor(dvs_frame, cv2.COLOR_BGR2GRAY) if len(dvs_frame.shape) == 3 else dvs_frame

    # Warp the DVS event image into the RGB frame
    warped_events = cv2.remap(dvs_gray, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Keep only actual event pixels
    _, event_mask = cv2.threshold(warped_events, 10, 255, cv2.THRESH_BINARY)

    # Create an overlay in red on the RGB image
    overlay = rgb_frame.copy()
    overlay[event_mask > 0] = [0, 0, 255]

    # Blend so the scene stays visible
    blended = cv2.addWeighted(rgb_frame, 0.75, overlay, 0.55, 0)

    out_writer.write(blended)
    frame_idx += 1

rgb_cap.release()
dvs_cap.release()
out_writer.release()

print(f"Overlay video saved to: {overlay_output_path}")