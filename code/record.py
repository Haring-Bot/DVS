import depthai as dai
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Event, get_context

from metavision_hal import DeviceDiscovery


def recordRGB(stamp, stop_event):
    folder_path = Path(__file__).resolve().parent.parent / "recordings" / stamp
    output_file = folder_path / f"{stamp}_RGB.mp4"

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )

        encoder = pipeline.create(dai.node.VideoEncoder).build(
            cam.requestOutput((1280, 720), dai.ImgFrame.Type.NV12)
        )
        encoder.setProfile(dai.VideoEncoderProperties.Profile.H264_MAIN)

        record = pipeline.create(dai.node.RecordVideo)
        record.setRecordVideoFile(str(output_file))
        encoder.out.link(record.input)

        pipeline.start()
        print(f"RGB recording to: {output_file}")

        try:
            while not stop_event.is_set() and pipeline.isRunning():
                time.sleep(0.1)
        finally:
            if pipeline.isRunning():
                pipeline.stop()


def recordDVS(output_file, stop_event):
    device = DeviceDiscovery.open("")
    if not device:
        raise RuntimeError("Could not connect to DVS camera")

    raw_facility = device.get_i_events_stream()
    if not raw_facility:
        raise RuntimeError("Could not access DVS event stream")

    print(f"DVS recording: {output_file}")
    raw_facility.start()
    raw_facility.log_raw_data(str(output_file))

    try:
        while not stop_event.is_set():
            if raw_facility.poll_buffer():
                raw_facility.get_latest_raw_data()
            time.sleep(0.001)
    finally:
        raw_facility.stop_log_raw_data()
        raw_facility.stop()


def main():
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    target_dir = Path(__file__).resolve().parent.parent / "recordings" / stamp
    target_dir.mkdir(parents=True, exist_ok=True)

    context = get_context("spawn")
    stop_event = context.Event()

    def handle_stop(sig, frame):
        print("\nStopping both recordings...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)

    dvs_process = context.Process(
        target=recordDVS,
        args=(target_dir / f"DVS_{stamp}.raw", stop_event),
        name="DVS recorder",
    )
    rgb_process = context.Process(
        target=recordRGB,
        args=(stamp, stop_event),
        name="RGB recorder",
    )

    dvs_process.start()
    rgb_process.start()

    print(f"Started DVS process: {dvs_process.pid}")
    print(f"Started RGB process: {rgb_process.pid}")

    try:
        while dvs_process.is_alive() or rgb_process.is_alive():
            dvs_process.join(timeout=0.5)
            rgb_process.join(timeout=0.5)

            if rgb_process.exitcode not in (None, 0):
                print(f"RGB recorder exited with code {rgb_process.exitcode}")
                stop_event.set()

            if dvs_process.exitcode not in (None, 0):
                print(f"DVS recorder exited with code {dvs_process.exitcode}")
                stop_event.set()

    except KeyboardInterrupt:
        handle_stop(None, None)

    finally:
        stop_event.set()

        for process in (dvs_process, rgb_process):
            process.join(timeout=3)

        for process in (dvs_process, rgb_process):
            if process.is_alive():
                print(f"Force-stopping {process.name}")
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()