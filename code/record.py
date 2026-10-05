import depthai as dai
import signal
import sys
import time
import docker
import subprocess
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import Event, get_context

from metavision_hal import DeviceDiscovery


def recordRGB(stamp, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    folder_path = Path(__file__).resolve().parent.parent / "recordings" / stamp
    output_file = folder_path / f"{stamp}_RGB.mp4"

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A,
            sensorResolution=(3840, 2160),
            sensorFps=15,
        )

        encoder = pipeline.create(dai.node.VideoEncoder).build(
            cam.requestOutput(
                (3840, 2160),
                dai.ImgFrame.Type.NV12,
                fps=15,
            )
        )
        encoder.setProfile(dai.VideoEncoderProperties.Profile.H264_MAIN)
        encoder.setBitrateKbps(12000)

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
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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


def recordLIDAR(output_folder, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    output_folder = Path(output_folder)

    image_name = "ros2humble-livox:latest"
    container_name = f"livox_recorder_{output_folder.name}"

    dockerClient = docker.from_env()

    try:
        old_container = dockerClient.containers.get(container_name)
        old_container.remove(force=True)
    except docker.errors.NotFound:
        pass

    print(f"Starting LIDAR Docker container: {container_name}")

    host_ros2_ws = os.path.expanduser("~/ros2_ws")
    host_target_dir = str(output_folder.parent.resolve())
    bag_folder_name = output_folder.name
    container_bag_path = f"/recordings/{bag_folder_name}"

    metadata_path = output_folder / "metadata.yaml"

    container = dockerClient.containers.run(
        image=image_name,
        name=container_name,
        network_mode="host",
        volumes={
        host_ros2_ws: {"bind": "/root/ros2_ws", "mode": "rw"},
        host_target_dir: {"bind": "/recordings", "mode": "rw"}
    },
        detach=True,
        tty=True,
        command="bash",
    )

    try:
        driver_cmd = (
            "bash -c 'source /root/ros2_ws/install/setup.bash && "
            "ros2 launch livox_ros_driver2 msg_MID360_launch.py'"
        )
        container.exec_run(driver_cmd, detach=True)

        print("LIDAR driver initializing...")
        time.sleep(4)

        record_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            f"source /root/ros2_ws/install/setup.bash && "
            f"ros2 bag record --storage sqlite3 "
            f"/livox/lidar /livox/imu -o {container_bag_path}",
        ]

        print(f"LIDAR recording to: {output_folder}")
        proc = subprocess.Popen(record_cmd)

        while not stop_event.is_set():
            if proc.poll() is not None:
                raise RuntimeError(
                    f"ROS2 bag recording exited unexpectedly with code {proc.returncode}"
                )
            time.sleep(0.2)

        print("Stopping LIDAR bag recording gracefully...")
        container.exec_run(
            ["bash", "-lc", "pkill -INT -f '[r]os2 bag record'"]
        )

        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("Recorder did not stop cleanly; forcing termination.")
            container.exec_run(
                ["bash", "-lc", "pkill -TERM -f '[r]os2 bag record'"]
            )
            proc.wait(timeout=5)

        print(f"ROS 2 bag finalized: {metadata_path}")

    finally:
        print("Stopping and removing LIDAR container...")
        try:
            container.stop(timeout=10)
            container.remove()
        except Exception as e:
            print(f"Error during LIDAR container cleanup: {e}")


def main():
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    target_dir = Path(__file__).resolve().parent.parent / "recordings" / stamp
    target_dir.mkdir(parents=True, exist_ok=True)

    context = get_context("spawn")
    stop_event = context.Event()

    def handle_stop(sig, frame):
        print("\nStopping all recordings (RGB, DVS, LIDAR)...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)

    dvs_process = context.Process(
        target=recordDVS,
        args=(target_dir / f"{stamp}_DVS.raw", stop_event),
        name="DVS recorder",
    )
    rgb_process = context.Process(
        target=recordRGB,
        args=(stamp, stop_event),
        name="RGB recorder",
    )

    lidar_process = context.Process(
        target=recordLIDAR,
        args=(target_dir / f"{stamp}_LIDAR", stop_event),
        name="LIDAR recorder",
    )

    lidar_process.start()
    dvs_process.start()
    rgb_process.start()

    print(f"Started LIDAR process: {lidar_process.pid}")
    print(f"Started DVS process:   {dvs_process.pid}")
    print(f"Started RGB process:   {rgb_process.pid}")

    try:
        while (
            dvs_process.is_alive()
            or rgb_process.is_alive()
            or lidar_process.is_alive()
        ):
            lidar_process.join(timeout=0.5)
            dvs_process.join(timeout=0.5)
            rgb_process.join(timeout=0.5)

            if lidar_process.exitcode not in (None, 0):
                print(
                    f"LIDAR recorder exited with code {lidar_process.exitcode}"
                )
                stop_event.set()

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

        all_processes = (dvs_process, rgb_process, lidar_process)

        for process in all_processes:
            process.join(timeout=3)

        for process in all_processes:
            if process.is_alive():
                print(f"Force-stopping {process.name}")
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()