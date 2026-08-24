#import depthai as dai
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from metavision_hal import DeviceDiscovery


def recordRGB(stamp):
    folderPath = Path(__file__).resolve().parent.parent / "recordings" / stamp
    output_file = folderPath / f"{stamp}_RGB.mp4"

    with dai.Pipeline() as pipeline:
        # Graceful stop on Ctrl+C
        def signal_handler(sig, frame):
            print("\nStopping recording...")
            pipeline.stop()
        signal.signal(signal.SIGINT, signal_handler)

        # 1. Initialize Camera (CAM_A / RGB)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # 2. Create Video Encoder
        videoEncoder = pipeline.create(dai.node.VideoEncoder).build(
            cam.requestOutput((1280, 720), dai.ImgFrame.Type.NV12)
        )
        videoEncoder.setProfile(dai.VideoEncoderProperties.Profile.H264_MAIN)

        # 3. Create RecordVideo host node
        record = pipeline.create(dai.node.RecordVideo)
        record.setRecordVideoFile(str(output_file))

        # 4. Link Encoder output directly to RecordVideo
        videoEncoder.out.link(record.input)

        # 5. Start recording pipeline
        pipeline.start()
        print("Recording started. Saving to '" + str(output_file) + "'. Press Ctrl+C to stop.")
        
        while pipeline.isRunning():
            time.sleep(1)
def record_genx320(output_file: Path):
    # 1. Discover and open the Prophesee GenX320 camera
    if DeviceDiscovery is None:
        print("Error: Metavision SDK not available.")
        sys.exit(1)
    
    device = DeviceDiscovery.open("")
    if not device:
        print("Error: Could not connect to Prophesee GenX320 camera.")
        sys.exit(1)

    # 2. Get access to the event stream
    raw_facility = device.get_i_events_stream()
    if not raw_facility:
        print("Error: Could not access event stream facility.")
        sys.exit(1)

    # 3. Start logging raw data to file
    print(f"Starting GenX320 recording to: {output_file}")
    raw_facility.start()
    raw_facility.log_raw_data(str(output_file))

    # Handle Ctrl+C gracefully
    recording = True
    def signal_handler(sig, frame):
        nonlocal recording
        print("\nStopping GenX320 recording...")
        recording = False

    signal.signal(signal.SIGINT, signal_handler)

    # 4. Main recording loop
    try:
        while recording:
            if raw_facility.poll_buffer():
                raw_facility.get_latest_raw_data()
            time.sleep(0.001)
    finally:
        # 5. Safely stop recording
        raw_facility.stop_log_raw_data()
        raw_facility.stop()
        print("GenX320 recording saved successfully.")


def main():
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    
    # Save to DVS/recordings/YYYYMMDDHHMM/genx320.raw
    target_dir = Path(__file__).resolve().parent.parent / "recordings" / stamp
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_raw = target_dir / "genx320.raw"
    record_genx320(output_raw)

if __name__ == "__main__":
    main()