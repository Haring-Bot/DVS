import os
import cv2
import numpy as np

# Configuration for your DVS event recording
video_path = "/home/julian/Downloads/202611041605/202611041605_DVS_cut.avi"
debug_dir = "/home/julian/Downloads/202611041605/debug_event_frames/"
os.makedirs(debug_dir, exist_ok=True)

# Pattern specifications (5 columns, 4 rows asymmetric grid)
pattern_cols = 5
pattern_rows = 4
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

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
  print(f"Error: Could not open video at {video_path}")
  exit()

# Configure SimpleBlobDetector parameters optimized for event-accumulated blobs
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0  # Dark circles

params.minArea = 50
params.maxArea = 50000
params.filterByCircularity = True
params.minCircularity = 0.35
params.filterByConvexity = True
params.minConvexity = 0.7
params.filterByInertia = True
params.minInertiaRatio = 0.1

blob_detector = cv2.SimpleBlobDetector_create(params)

frame_count = 0
success_count = 0

print("Processing DVS event-accumulated frames for grid calibration...")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect the 5x4 asymmetric grid on the event frame
  ret_grid, corners = cv2.findCirclesGrid(
      gray,
      pattern_size,
      flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
      blobDetector=blob_detector,
  )

  debug_frame = frame.copy()
  if ret_grid:
    objpoints.append(objp)
    imgpoints.append(corners)
    success_count += 1
    print(f"[+] Asymmetric grid successfully detected in event frame {frame_count}!")
    
    # Draw detected grid corners in green
    cv2.drawChessboardCorners(debug_frame, pattern_size, corners, ret_grid)
    cv2.imwrite(
        os.path.join(debug_dir, f"event_success_{frame_count}.jpg"),
        debug_frame,
    )
  else:
    # Save debug view periodically for troubleshooting sparse event windows
    if frame_count % 30 == 0:
      keypoints = blob_detector.detect(gray)
      cv2.drawKeypoints(
          debug_frame,
          keypoints,
          debug_frame,
          (0, 0, 255),
          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
      )
      cv2.imwrite(
          os.path.join(debug_dir, f"event_failed_{frame_count}.jpg"),
          debug_frame,
      )

  frame_count += 1

cap.release()
print(f"\nTotal successful event frames captured: {success_count}")

if success_count > 0:
  ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
      objpoints, imgpoints, gray.shape[::-1], None, None
  )
  print("\n--- DVS Camera Calibration Results ---")
  print("RMS Re-projection Error:", ret_calib)
  print("\nCamera Matrix (mtx):\n", mtx)
  print("\nDistortion Coefficients (dist):\n", dist)

  output_file = "/home/julian/Downloads/202611041605/dvs_camera_calibration_data.npz"
  np.savez(output_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
  print(f"\nCalibration parameters saved successfully to '{output_file}'.")
else:
  print(
      "\n[!] 0 event frames detected. Ensure the event accumulation window "
      "captures enough contrast for the 5x4 pattern."
  )