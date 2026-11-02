import cv2
import numpy as np
import os

def test_videos(oak_video_path, dvs_video_path, chessboard_size=(10, 7), save_debug_frames=True):
    """
    Tests both the OAK-D and DVS videos frame-by-frame, applies pre-processing 
    specifically for event-based frames, and reports chessboard detection counts.
    """
    if save_debug_frames and not os.path.exists("debug_output"):
        os.makedirs("debug_output")

    # Open both video captures
    cap_oak = cv2.VideoCapture(oak_video_path)
    cap_dvs = cv2.VideoCapture(dvs_video_path)

    if not cap_oak.isOpened():
        print(f"Error: Could not open OAK video at {oak_video_path}")
        return
    if not cap_dvs.isOpened():
        print(f"Error: Could not open DVS video at {dvs_video_path}")
        return

    oak_total_frames = int(cap_oak.get(cv2.CAP_PROP_FRAME_COUNT))
    dvs_total_frames = int(cap_dvs.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n--- Video Properties ---")
    print(f"OAK Video Total Frames: {oak_total_frames}")
    print(f"DVS Video Total Frames: {dvs_total_frames}")
    print(f"Target Chessboard Inner Corners: {chessboard_size}\n")

    print("Testing OAK video...")
    oak_success_count = 0
    dvs_success_count = 0
    paired_success_count = 0
    
    frame_idx = 0
    while True:
        ret_oak, frame_oak = cap_oak.read()
        ret_dvs, frame_dvs = cap_dvs.read()
        
        # Stop if either video stream ends
        if not ret_oak or not ret_dvs:
            break
            
        # 1. Process OAK frame
        gray_oak = cv2.cvtColor(frame_oak, cv2.COLOR_BGR2GRAY)
        found_oak, corners_oak = cv2.findChessboardCorners(gray_oak, chessboard_size, None)
        if found_oak:
            oak_success_count += 1

        # 2. Process DVS frame (Event-based frame requires thickening/dilation)
        gray_dvs = cv2.cvtColor(frame_dvs, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create clean binary lines from event accumulation
        _, thresh_dvs = cv2.threshold(gray_dvs, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate/Thicken the thin event lines into solid blocks for OpenCV
        kernel = np.ones((3, 3), np.uint8)
        processed_dvs = cv2.dilate(thresh_dvs, kernel, iterations=1)
        
        # Optional: Invert if your event stream has a white background with black lines
        # processed_dvs = cv2.bitwise_not(processed_dvs)

        found_dvs, corners_dvs = cv2.findChessboardCorners(processed_dvs, chessboard_size, None)
        if found_dvs:
            dvs_success_count += 1

        # Check if both succeeded on this frame index
        if found_oak and found_dvs:
            paired_success_count += 1

        # Save a few processed DVS debug images for inspection if counts are still 0
        if save_debug_frames and frame_idx < 5:
            cv2.imwrite(f"debug_output/dvs_raw_{frame_idx}.png", gray_dvs)
            cv2.imwrite(f"debug_output/dvs_processed_{frame_idx}.png", processed_dvs)

        frame_idx += 1

    cap_oak.release()
    cap_dvs.release()

    # --- Summary Report ---
    print("\n==============================")
    print("      TEST RESULTS SUMMARY    ")
    print("==============================")
    print(f"Total Frames Processed Pairs: {frame_idx}")
    print(f"OAK-D Detections:             {oak_success_count} / {frame_idx}")
    print(f"DVS (Event) Detections:       {dvs_success_count} / {frame_idx}")
    print(f"Synchronized Dual Detections: {paired_success_count} / {frame_idx}")
    print("==============================\n")
    
    if dvs_success_count == 0:
        print("[!] Warning: DVS detections are still 0.")
        print("    Check the './debug_output/' folder. If the processed DVS frames look blank")
        print("    or missing lines, your recording lacked enough motion or contrast to trigger events.")
    else:
        print("[✓] Success! DVS corners are now being detected. You can proceed with stereo calibration.")

if __name__ == "__main__":
    OAK_PATH = "/home/julian/Downloads/202611020950/202611020950_RGB_cut.mp4"
    DVS_PATH = "/home/julian/Downloads/202611020950/202611020950_DVS_cut.avi"
    
    test_videos(OAK_PATH, DVS_PATH, chessboard_size=(10, 7))