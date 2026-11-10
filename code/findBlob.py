import os
import cv2
import numpy as np

# Configuration for your DVS event recording
video_path = "/home/julian/rpg_e2vid/results/events_preview.mp4"              #DVS
#video_path = "/home/julian/Downloads/202611091132/202611091132_RGB_cut.mp4"  #RGB
debug_dir = "/home/julian/rpg_e2vid/results/debug_event_frames/"
os.makedirs(debug_dir, exist_ok=True)

# Output video path for the annotated visualization
output_video_path = "/home/julian/rpg_e2vid/results/annotated_reconstruction.mp4"

# Pattern specifications (5 columns, 4 rows asymmetric grid)
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

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
  print(f"Error: Could not open video at {video_path}")
  exit()

# Get video properties for writing the output video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
  fps = 30.0  # Fallback FPS if metadata is missing
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter (mp4v codec works well with .mp4 files)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Configure SimpleBlobDetector parameters with your tuned values from the live tuner
rgb = False
if rgb:
  params = cv2.SimpleBlobDetector_Params()
  params.filterByColor = True
  params.blobColor = 0  # Bright blobs on black background

  params.minArea = 1500
  params.maxArea = 15000

  params.filterByCircularity = True
  params.minCircularity = 0.40

  params.filterByConvexity = True
  params.minConvexity = 0.80

  params.filterByInertia = True
  params.minInertiaRatio = 0.40

else:
  params = cv2.SimpleBlobDetector_Params()
  params.filterByColor = True
  params.blobColor = 255  # Bright blobs on black background

  params.minArea = 500
  params.maxArea = 3000

  params.filterByCircularity = True
  params.minCircularity = 0.10

  params.filterByConvexity = True
  params.minConvexity = 0.40

  params.filterByInertia = True
  params.minInertiaRatio = 0.30

blob_detector = cv2.SimpleBlobDetector_create(params)

frame_count = 0
success_count = 0

print(f"Processing DVS video, saving individual annotated frames to: {debug_dir}")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 1. Isolate the Blue channel
  gray = frame[:, :, 0]

  # 2. Apply a slight dilation to fatten up sparse/dim event clusters
  kernel = np.ones((19, 19), np.uint8)
  processed_img = cv2.dilate(gray, kernel, iterations=1)

  # Detect the 5x4 asymmetric grid using the dilated image
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
    print(f"[+] Asymmetric grid successfully detected in event frame {frame_count}!")
    
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
    # Draw keypoints in yellow when detection fails so you can inspect tracking quality
    keypoints = blob_detector.detect(gray)
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

  # Save every annotated frame individually as a numbered image
  frame_filename = os.path.join(debug_dir, f"annotated_frame_{frame_count:05d}.jpg")
  cv2.imwrite(frame_filename, debug_frame)

  # Write the annotated frame to the output video file as well
  out_writer.write(debug_frame)
  frame_count += 1

cap.release()
out_writer.release()
print(f"\nProcessing complete!")
print(f"Annotated video saved to: {output_video_path}")
print(f"Individual annotated frames saved to: {debug_dir}")
print(f"Total successful event frames captured: {success_count}")

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