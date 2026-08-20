import depthai as dai
import time
import signal
from pathlib import Path

output_file = Path("test_video.mp4")

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
    record.setRecordVideoFile(output_file)

    # 4. Link Encoder output directly to RecordVideo
    videoEncoder.out.link(record.input)

    # 5. Start recording pipeline
    pipeline.start()
    print("Recording started. Saving to 'test_video.mp4'. Press Ctrl+C to stop.")
    
    while pipeline.isRunning():
        time.sleep(1)