import cv2
import numpy as np
import os

# --- Configuration ---
recordingNumber = 202611022116
GRID_SIZE = (11,3)        # Asymmetric circles grid dimensions (rows, cols)
SPACING = 0.01             # Distance between circles in meters (3cm)

OAK_VIDEO_PATH = f"/home/julian/Downloads/{recordingNumber}/{recordingNumber}_RGB_cut.mp4"
DVS_VIDEO_PATH = f"/home/julian/Downloads/{recordingNumber}/{recordingNumber}_DVS_cut.avi"

def run_calibration():
    debug_dir = "debug_matches"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    cap_oak = cv2.VideoCapture(OAK_VIDEO_PATH)
    cap_dvs = cv2.VideoCapture(DVS_VIDEO_PATH)

    if not cap_oak.isOpened() or not cap_dvs.isOpened():
        print("Error: Could not open one or both video files.")
        return

    # Prepare 3D object points for asymmetric circles grid
    objp = np.zeros((GRID_SIZE[0] * GRID_SIZE[1], 3), np.float32)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            objp[i * GRID_SIZE[1] + j] = np.array([j * SPACING, (i * 0.5 + (j % 2) * 0.5) * SPACING, 0.0], np.float32)

    objpoints = []     
    imgpoints_oak = [] 
    imgpoints_dvs = [] 

    oak_success_count = 0
    dvs_success_count = 0
    paired_success_count = 0
    
    frame_idx = 0
    print("\nProcessing videos, extracting circle grid points, and saving matches...")

    while True:
        ret_oak, frame_oak = cap_oak.read()
        ret_dvs, frame_dvs = cap_dvs.read()
        
        if not ret_oak or not ret_dvs:
            break
            
        gray_oak = cv2.cvtColor(frame_oak, cv2.COLOR_BGR2GRAY)
        gray_dvs = cv2.cvtColor(frame_dvs, cv2.COLOR_BGR2GRAY)
        
        # 1. OAK-D: Detect asymmetric circles grid
        found_oak, corners_oak = cv2.findCirclesGrid(
            gray_oak, GRID_SIZE, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )
        if found_oak:
            oak_success_count += 1

        # 2. DVS: Threshold event frame to isolate dot clusters
        _, thresh_dvs = cv2.threshold(gray_dvs, 40, 255, cv2.THRESH_BINARY)
        
        # DVS: Detect asymmetric circles grid natively on event clusters
        found_dvs, corners_dvs = cv2.findCirclesGrid(
            thresh_dvs, GRID_SIZE, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )
        if found_dvs:
            dvs_success_count += 1

        # 3. Synchronized Pairing & Visual Export
        if found_oak and found_dvs:
            objpoints.append(objp)
            imgpoints_oak.append(corners_oak)
            imgpoints_dvs.append(corners_dvs)
            paired_success_count += 1

            # --- SAVE DEBUG MATCH VISUALIZATION ---
            vis_oak = frame_oak.copy()
            cv2.drawChessboardCorners(vis_oak, GRID_SIZE, corners_oak, found_oak)

            vis_dvs = cv2.cvtColor(thresh_dvs, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis_dvs, GRID_SIZE, corners_dvs, found_dvs)

            if vis_oak.shape[0] != vis_dvs.shape[0]:
                vis_dvs = cv2.resize(vis_dvs, (vis_oak.shape[1], vis_oak.shape[0]))

            combined_match = np.hstack((vis_oak, vis_dvs))
            cv2.imwrite(f"{debug_dir}/match_circle_{frame_idx:04d}.png", combined_match)

        frame_idx += 1

    cap_oak.release()
    cap_dvs.release()

    # --- Summary Report ---
    print("\n==============================")
    print("      TEST RESULTS SUMMARY    ")
    print("==============================")
    print(f"Total Frames Processed:       {frame_idx}")
    print(f"OAK-D Detections:             {oak_success_count} / {frame_idx}")
    print(f"DVS (Event) Grid Detections:  {dvs_success_count} / {frame_idx}")
    print(f"Synchronized Dual Detections: {paired_success_count} / {frame_idx}")
    print(f"Match images saved to:        {os.path.abspath(debug_dir)}/")
    print("==============================\n")

    # --- Stereo Calibration Execution ---
    if paired_success_count > 10:
        print("Running stereo calibration...")
        image_size = gray_oak.shape[::-1]
        
        ret, K_oak, d_oak, K_dvs, d_dvs, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_oak, imgpoints_dvs,
            None, None, None, None, image_size,
            criteria=(3, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        print(f"Calibration Complete! RMS Re-projection Error: {ret:.4f} pixels")
        np.savez("oak_dvs_extrinsics.npz", R=R, T=T, K_oak=K_oak, d_oak=d_oak, K_dvs=K_dvs, d_dvs=d_dvs)
        print("Saved extrinsics to oak_dvs_extrinsics.npz")
    else:
        print("[!] Not enough paired frames found for calibration (< 10). Adjust motion or check pattern dimensions.")

if __name__ == "__main__":
    run_calibration()