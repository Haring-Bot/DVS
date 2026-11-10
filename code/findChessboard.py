import cv2
import numpy as np

# --- Configuration ---
# Define the dimensions of the chessboard (number of inner corners per a row and column)
CHECKBOARD_DIMENSIONS = (5, 3)
SQUARE_SIZE_METERS = 0.034  # The size of a square in meters

# Path to your input video file and desired output debug video
VIDEO_PATH = "/home/julian/rpg_e2vid/results/reconstruction.mp4"
OUTPUT_VIDEO_PATH = "calibration_debug_output.mp4"

# Frame skipping: Process only 1 out of every N frames to avoid redundant data
SKIP_FRAMES = 1

# Termination criteria for corner sub-pixel optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros(
    (CHECKBOARD_DIMENSIONS[0] * CHECKBOARD_DIMENSIONS[1], 3), np.float32
)
objp[:, :2] = np.mgrid[
    0 : CHECKBOARD_DIMENSIONS[0], 0 : CHECKBOARD_DIMENSIONS[1]
].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS

# Arrays to store object points and image points
objpoints = []
imgpoints = []

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties for writing the output video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0  # Fallback if FPS cannot be read
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(
    f"Processing video: {VIDEO_PATH} and saving debug output to {OUTPUT_VIDEO_PATH}..."
)

frame_id = 0

while cap.isOpened():
    ret_val, img = cap.read()
    if not ret_val:
        break  # End of video

    # Process only every Nth frame
    if frame_id % SKIP_FRAMES == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKBOARD_DIMENSIONS, None)

        if ret:
            objpoints.append(objp)
            # Refine corner locations for higher accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw corners and success status text (Green)
            cv2.drawChessboardCorners(img, CHECKBOARD_DIMENSIONS, corners2, ret)
            cv2.putText(
                img,
                f"Frame {frame_id}: FOUND",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            # Draw failure status text (Red)
            cv2.putText(
                img,
                f"Frame {frame_id}: NOT FOUND",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        # For skipped frames
        cv2.putText(
            img,
            f"Frame {frame_id}: SKIPPED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Write the annotated frame to the output video
    out.write(img)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDebug video successfully saved to: {OUTPUT_VIDEO_PATH}")

if len(objpoints) == 0:
    print(
        "Error: Could not find chessboard corners in any frame of the video."
    )
    exit()

print(
    f"Successfully collected points from {len(objpoints)} frames. Performing camera calibration..."
)

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (width, height), None, None
)

# --- Results ---
print("\n--- Calibration Results ---")
print(f"RMS Re-projection Error: {ret:.4f}")
print("\nCamera Matrix (Intrinsic Parameters):\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs.ravel())

# Save calibration parameters using NumPy
np.savez(
    "camera_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
)
print("\nCalibration parameters saved successfully to 'camera_calibration.npz'.")