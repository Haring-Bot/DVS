import os
import cv2
import numpy as np

# --- Configuration ---
stamp = 202611101716
dvs_video_path = "/home/julian/rpg_e2vid/results/events_preview.mp4"
rgb_video_path = f"/home/julian/Downloads/{stamp}/{stamp}_RGB_cut.mp4"

# Output paths for the debug annotated videos
output_dir = f"/home/julian/Downloads/{stamp}/"
os.makedirs(output_dir, exist_ok=True)
dvs_output_video_path = os.path.join(output_dir, "annotated_dvs_reconstruction.mp4")
rgb_output_video_path = os.path.join(output_dir, "annotated_rgb_video.mp4")

# Pattern specifications (Asymmetric grid)
pattern_cols = 2
pattern_rows = 5
pattern_size = (pattern_cols, pattern_rows)
square_size = 45.0  # Diagonal spacing in mm

# Prepare 3D object points for the asymmetric circle grid
objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
for i in range(pattern_rows):
  for j in range(pattern_cols):
    objp[i * pattern_cols + j] = np.array(
        [(2 * j + i % 2) * (square_size / 2.0), i * (square_size / 2.0), 0],
        np.float32,
    )


def process_video_for_calibration_and_video(video_path, output_video_path, is_rgb=False):
  print(f"\n--- Processing & Exporting Video: {'RGB' if is_rgb else 'DVS'} ---")
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    return None

  fps = cap.get(cv2.CAP_PROP_FPS)
  if fps == 0 or np.isnan(fps):
    fps = 30.0
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

  # Configure Blob Detector based on mode
  params = cv2.SimpleBlobDetector_Params()
  params.filterByColor = True
  if is_rgb:
    params.blobColor = 0  # Dark blobs on bright background for RGB
    params.minArea = 1500
    params.maxArea = 15000
    params.minCircularity = 0.30
    params.minConvexity = 0.40
    params.minInertiaRatio = 0.40
  else:
    params.blobColor = 255  # Bright blobs on black background for DVS
    params.minArea = 800
    params.maxArea = 3000
    params.minCircularity = 0.10
    params.minConvexity = 0.40
    params.minInertiaRatio = 0.30

  blob_detector = cv2.SimpleBlobDetector_create(params)

  objpoints = []
  imgpoints = []
  frame_count = 0
  success_count = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    if not is_rgb:
      # DVS: Isolate Blue channel & dilate
      gray = frame[:, :, 0]
      kernel = np.ones((19, 19), np.uint8)
      processed_img = cv2.dilate(gray, kernel, iterations=1)
    else:
      # RGB: Convert to grayscale
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      processed_img = gray

    ret_grid, corners = cv2.findCirclesGrid(
        processed_img,
        pattern_size,
        flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
        blobDetector=blob_detector,
    )

    debug_frame = frame.copy()
    if ret_grid:
      objpoints.append(objp)
      imgpoints.append(corners)
      success_count += 1
      
      # Draw detected grid corners in green
      cv2.drawChessboardCorners(debug_frame, pattern_size, corners, ret_grid)
      cv2.putText(
          debug_frame, 
          f"Frame {frame_count}: SUCCESS", 
          (30, 40), 
          cv2.FONT_HERSHEY_SIMPLEX, 
          1, 
          (0, 255, 0), 
          2
      )
    else:
      # Draw keypoints in yellow when detection fails
      keypoints = blob_detector.detect(processed_img)
      cv2.drawKeypoints(
          debug_frame,
          keypoints,
          debug_frame,
          (0, 255, 255),
          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
      )
      cv2.putText(
          debug_frame, 
          f"Frame {frame_count}: FAILED", 
          (30, 40), 
          cv2.FONT_HERSHEY_SIMPLEX, 
          1, 
          (0, 0, 255), 
          2
      )

    out_writer.write(debug_frame)
    frame_count += 1

  cap.release()
  out_writer.release()
  print(f"Successfully detected grid in {success_count} / {frame_count} frames.")
  print(f"Annotated debug video saved to: {output_video_path}")

  if success_count == 0:
    return None

  # Calibrate individual camera
  ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
      objpoints, imgpoints, (width, height), None, None
  )
  print(f"Calibration RMS Error ({'RGB' if is_rgb else 'DVS'}): {ret_calib:.4f}")

  return {
      "objpoints": objpoints,
      "imgpoints": imgpoints,
      "camera_matrix": mtx,
      "dist_coeffs": dist,
      "image_size": (width, height),
  }


# 1. Process and Calibrate DVS Camera
dvs_data = process_video_for_calibration_and_video(dvs_video_path, dvs_output_video_path, is_rgb=False)

# 2. Process and Calibrate RGB Camera
rgb_data = process_video_for_calibration_and_video(rgb_video_path, rgb_output_video_path, is_rgb=True)

if dvs_data is not None and rgb_data is not None:
  print("\n--- Computing Stereo Calibration / Transformation Matrix ---")
  min_frames = min(
      len(dvs_data["objpoints"]), len(rgb_data["objpoints"])
  )

  if min_frames > 0:
    obj_points_stereo = dvs_data["objpoints"][:min_frames]
    img_points_dvs = dvs_data["imgpoints"][:min_frames]
    img_points_rgb = rgb_data["imgpoints"][:min_frames]

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )

    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points_stereo,
        img_points_rgb,
        img_points_dvs,
        rgb_data["camera_matrix"],
        rgb_data["dist_coeffs"],
        dvs_data["camera_matrix"],
        dvs_data["dist_coeffs"],
        dvs_data["image_size"],
        criteria=criteria_stereo,
        flags=flags,
    )

    print(f"Stereo Calibration RMS Error: {ret_stereo:.4f}")
    print("\nRotation Matrix (R) from RGB to DVS:\n", R)
    print("\nTranslation Vector (T) from RGB to DVS:\n", T)

    # Save Transformation Matrix / Stereo Results
    output_transform_file = os.path.join(output_dir, f"{stamp}_rgb_to_dvs_transformation.npz")
    np.savez(
        output_transform_file,
        R=R,
        T=T,
        rgb_camera_matrix=rgb_data["camera_matrix"],
        dvs_camera_matrix=dvs_data["camera_matrix"],
        rgb_dist_coeffs=rgb_data["dist_coeffs"],
        dvs_dist_coeffs=dvs_data["dist_coeffs"],
    )
    print(f"\nTransformation parameters successfully saved to '{output_transform_file}'.")
  else:
    print(
        "[!] Error: Could not find overlapping frames with valid grid detections in both videos for stereo calibration."
    )
else:
  print(
      "[!] Calibration failed for one or both video streams. Check your blob detector parameters and video paths."
  )