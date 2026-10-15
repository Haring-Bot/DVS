import depthai as dai
import signal
import sys
import time
import docker
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Event, get_context

from metavision_hal import DeviceDiscovery

def modifyJson(stamp, sensor, key, value):
    path = Path(__file__).resolve().parent.parent / "recordings" / f"{stamp}/{stamp}.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        print(f".json couldn't be found at {path}")
        return
    
    if sensor == None:
        data["experiment"][key]=value
    elif sensor in ["RGB","DVS","LIDAR"]:
        data["sensors"][sensor][key]=value
    else:
        print("sensor not found in list")
        return

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    

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
        startPerf = time.perf_counter()

        modifyJson(stamp, "RGB", "timestampStart", startPerf)
        modifyJson(stamp, "RGB", "status", "started")
        
        print(f"RGB recording to: {output_file}")

        try:
            while not stop_event.is_set() and pipeline.isRunning():
                time.sleep(0.1)
        finally:
            if pipeline.isRunning():
                pipeline.stop()
                endPerf = time.perf_counter()
                modifyJson(stamp, "RGB", "timestampStop", endPerf)
                modifyJson(stamp, "RGB", "status", "stopped")

               
def recordDVS(stamp, output_file, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    device = DeviceDiscovery.open("")
    if not device:
        raise RuntimeError("Could not connect to DVS camera")

    raw_facility = device.get_i_events_stream()
    if not raw_facility:
        raise RuntimeError("Could not access DVS event stream")

    raw_facility.start()
    raw_facility.log_raw_data(str(output_file))
    
    print(f"DVS recording: {output_file}")

    startPerf = time.perf_counter()
    modifyJson(stamp, "DVS", "timestampStart", startPerf)
    modifyJson(stamp, "DVS", "status", "started")


    try:
        while not stop_event.is_set():
            if raw_facility.poll_buffer():
                raw_facility.get_latest_raw_data()
            time.sleep(0.001)
    finally:
        raw_facility.stop_log_raw_data()
        raw_facility.stop()
        endPerf = time.perf_counter()
        modifyJson(stamp, "DVS", "timestampStop", endPerf)
        modifyJson(stamp, "DVS", "status", "stopped")

def recordLIDAR(stamp, output_folder, stop_event):
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

        startPerf = time.perf_counter()
        modifyJson(stamp, "LIDAR", "timestampStart", startPerf)
        modifyJson(stamp, "LIDAR", "status", "started")

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
        endPerf = time.perf_counter()
        modifyJson(stamp, "LIDAR", "timestampStop", endPerf)
        modifyJson(stamp, "LIDAR", "status", "stopped")


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

    jsonFile = {
        "experiment":{
            "ID" : 0,
            "status": "empty"
        },
        "sensors":{
            "RGB":{
                "status": "pending",
                "timestampStart": None,
                "timestampStop": None
            },
            "DVS":{
                "status": "pending",
                "timestampStart": None,
                "timestampStop": None
            },
            "LIDAR":{
                "status": "pending",
                "timestampStart": None,
                "timestampStop": None
            }
        }
    }

    jsonPath = f"{target_dir}/{stamp}.json"

    with open(jsonPath, "w") as f:
        json.dump(jsonFile, f, indent=4)

    modifyJson(stamp, None, "ID", stamp)
    modifyJson(stamp, None, "status", "started")

    dvs_process = context.Process(
        target=recordDVS,
        args=(stamp, target_dir / f"{stamp}_DVS.raw", stop_event),
        name="DVS recorder",
    )
    rgb_process = context.Process(
        target=recordRGB,
        args=(stamp, stop_event),
        name="RGB recorder",
    )

    lidar_process = context.Process(
        target=recordLIDAR,
        args=(stamp, target_dir / f"{stamp}_LIDAR", stop_event),
        name="LIDAR recorder",
    )

    lidar_process.start()
    dvs_process.start()
    rgb_process.start()

    print(f"Started LIDAR process: {lidar_process.pid}")
    print(f"Started DVS process:   {dvs_process.pid}")
    print(f"Started RGB process:   {rgb_process.pid}")

    modifyJson(stamp, None, "status", "started")

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

    modifyJson(stamp, None, "status", "finished")


if __name__ == "__main__":
    main()