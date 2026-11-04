import cv2
import numpy as np
from datetime import datetime
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm

def main():
    print("Initializing GenX320 sensor...")
    
    # 1. Connect to the live camera hardware layer
    device = initiate_device("")
    if not device:
        print("Error: Could not connect to the GenX320 camera hardware.")
        return

    # 2. Gain access to low-level biases and the raw event stream recorder
    i_ll_biases = device.get_i_ll_biases()
    events_stream = device.get_i_events_stream()
    
    if not i_ll_biases or not events_stream:
        print("Error: Could not access essential hardware facilities (biases/stream).")
        return

    # 3. Initialize the EventsIterator (feeding chunks every 10ms to the loop)
    mv_iterator = EventsIterator.from_device(device=device, delta_t=10000)
    width, height = mv_iterator.get_size()
    
    # 4. Set up a tracking dictionary to share frames across the callback
    runtime_data = {
        "frame": np.zeros((height, width, 3), np.uint8),
        "is_ready": False
    }

    # 5. Define the callback that the SDK calls when a frame is fully baked
    def frame_generation_callback(ts, generated_frame):
        runtime_data["frame"] = generated_frame.copy()
        runtime_data["is_ready"] = True

    # 6. Initialize frame generator (30ms event accumulation at 30 FPS)
    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, 
        sensor_height=height, 
        accumulation_time_us=30000, 
        fps=30.0
    )
    frame_gen.set_output_callback(frame_generation_callback)
    
    # --- Fullscreen Window Setup ---
    window_name = "GenX320 Custom Live Viewport"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\n--- Live Streaming Active ---")
    print("Controls (Focus on the Video Window):")
    print("  'w' / 's' : Increase / Decrease bias_diff_on  (ON contrast sensitivity)")
    print("  'e' / 'd' : Increase / Decrease bias_diff_off (OFF contrast sensitivity)")
    print("  'r'        : Start / Stop RAW Recording Toggle")
    print("  'f'        : Toggle Fullscreen On / Off")
    print("  'q'        : Quit Program\n")

    offset_on = 0
    offset_off = 0
    is_recording = False
    is_fullscreen = True

    # 7. Main streaming loop
    for evs in mv_iterator:
        # Pass the events to the generator—this automatically triggers the callback when ready
        frame_gen.process_events(evs)
        
        # If the callback captured a new frame, display it
        if runtime_data["is_ready"]:
            display_frame = runtime_data["frame"].copy()
            
            # On-screen Recording HUD Overlay
            if is_recording:
                cv2.circle(display_frame, (15, 20), 5, (0, 0, 255), -1) # Red dot
                cv2.putText(display_frame, "REC", (26, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.imshow(window_name, display_frame)
            runtime_data["is_ready"] = False # Reset flag
        
        # Intercept keystrokes to configure parameters in real-time
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if is_recording:
                events_stream.stop_log_raw_data()
            break
            
        elif key == ord('w'):
            offset_on += 5
            i_ll_biases.set("bias_diff_on", offset_on)
            print(f"Live Update -> bias_diff_on offset set to: {offset_on}")
            
        elif key == ord('s'):
            offset_on -= 5
            i_ll_biases.set("bias_diff_on", offset_on)
            print(f"Live Update -> bias_diff_on offset set to: {offset_on}")
            
        elif key == ord('e'):
            offset_off += 5
            i_ll_biases.set("bias_diff_off", offset_off)
            print(f"Live Update -> bias_diff_off offset set to: {offset_off}")
            
        elif key == ord('d'):
            offset_off -= 5
            i_ll_biases.set("bias_diff_off", offset_off)
            print(f"Live Update -> bias_diff_off offset set to: {offset_off}")
            
        elif key == ord('r'):
            if not is_recording:
                # Generate unique filename containing timestamp and active parameters
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"calibrationVid/rec_{timestamp}_on_{offset_on}_off_{offset_off}.raw"
                
                # Start logging the raw stream
                events_stream.log_raw_data(filename)
                is_recording = True
                print(f"▶️ RECORDING STARTED: Target file -> {filename}")
            else:
                # Stop logging the raw stream safely
                events_stream.stop_log_raw_data()
                is_recording = False
                print("⏹️ RECORDING STOPPED. File saved successfully.")
                
        elif key == ord('f'):
            # Toggle fullscreen mode live using the 'f' key
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_fullscreen = False
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_fullscreen = True

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
