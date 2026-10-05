import os
import sys
import time
import subprocess
import docker
from pathlib import Path


def generate_map(recording_folder: Path):
    """
    Runs FAST-LIO2 inside the Docker container on a recorded LIDAR bag directory
    and saves the resulting 3D PCD map directly into the recording folder.
    """
    recording_folder = Path(recording_folder).resolve()

    # Locate the LIDAR bag folder within the timestamped recording folder
    bag_dirs = list(recording_folder.glob("*_LIDAR"))
    if not bag_dirs:
        print(f"[!] Error: Could not find *_LIDAR bag directory in {recording_folder}")
        sys.exit(1)

    lidar_bag_path = bag_dirs[0]
    bag_folder_name = lidar_bag_path.name
    container_bag_path = (f"/recordings/{recording_folder.name}/{bag_folder_name}")
    target_pcd_path = lidar_bag_path / "test.pcd"

    print(f"[+] Found LIDAR bag: {bag_folder_name}")

    metadata_path = lidar_bag_path / "metadata.yaml"
    db_files = list(lidar_bag_path.glob("*.db3"))

    if not metadata_path.exists():
        raise RuntimeError(
            f"Invalid ROS 2 bag: missing {metadata_path}"
        )

    if not db_files:
        raise RuntimeError(
            f"Invalid ROS 2 bag: no .db3 database in {lidar_bag_path}"
        )

    image_name = "ros2humble-livox:latest"
    container_name = "fast_lio_processor"

    # Define host paths
    host_ros2_ws = os.path.expanduser("~/ros2_ws")
    host_recordings_dir = str(recording_folder.parent)

    client = docker.from_env()

    # Clean up stale container if present
    try:
        client.containers.get(container_name).remove(force=True)
    except docker.errors.NotFound:
        pass

    print("[+] Starting FAST-LIO Docker workspace...")
    container = client.containers.run(
        image=image_name,
        name=container_name,
        network_mode="host",
        volumes={
            host_ros2_ws: {"bind": "/root/ros2_ws", "mode": "rw"},
            host_recordings_dir: {"bind": "/recordings", "mode": "rw"},
        },
        detach=True,
        tty=True,
        command="bash",
    )

    try:
        # 1. Start FAST-LIO mapping launch file in background inside container
        print("[+] Launching FAST-LIO2 node...")


        fast_lio_launch_cmd = (
            "bash -lc 'source /opt/ros/humble/setup.bash && "
            "source /root/ros2_ws/install/setup.bash && "
            "ros2 launch fast_lio mapping.launch.py "
            "config_file:=mid360.yaml rviz:=false "
            "> /tmp/fast_lio.log 2>&1'"
        )

        container.exec_run(
            fast_lio_launch_cmd,
            detach=True,
            workdir=container_bag_path,
        )

        # Allow node initialization
        time.sleep(4)

        # 2. Replay ROS 2 Bag inside container
        print(f"[+] Playing ROS 2 bag from: {container_bag_path}...")
        
        play_bag_cmd = [
            "docker", "exec", container_name,
            "bash", "-c",
            f"source /root/ros2_ws/install/setup.bash && ros2 bag play --storage sqlite3 {container_bag_path}"
        ]
        
        # Block until bag playback finishes
        subprocess.run(play_bag_cmd, check=True)
        print("[+] Bag playback complete!")

        # 3. Trigger map save service
        print("[+] Triggering /map_save service...")
        save_result = container.exec_run(
            [
                "bash",
                "-lc",
                "source /root/ros2_ws/install/setup.bash && "
                "ros2 service call /map_save std_srvs/srv/Trigger '{}'",
            ],
            demux=True,
        )

        stdout, stderr = save_result.output
        print((stdout or b"").decode(), end="")
        print((stderr or b"").decode(), end="")

        if save_result.exit_code != 0:
            raise RuntimeError(
                f"/map_save failed with exit code {save_result.exit_code}"
            )

        # Wait for FAST-LIO to finish writing the map.
        deadline = time.monotonic() + 60
        candidates = []

        while time.monotonic() < deadline:
            if target_pcd_path.exists() and target_pcd_path.stat().st_size > 0:
                break
            time.sleep(1)
        else:
            log_result = container.exec_run(
                ["bash", "-lc", "cat /tmp/fast_lio.log"]
            )
            print(log_result.output.decode())
            raise RuntimeError(
                f"FAST-LIO reported map saving, but no PCD was found at "
                f"{target_pcd_path}"
            )

        print(f"[+] 3D PCD map saved to: {target_pcd_path}")

    finally:
        print("[+] Stopping processing container...")
        container.stop(timeout=10)
        container.remove()


if __name__ == "__main__":
    # Example usage: Pass timestamped folder path or default to newest recording
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        # Grab latest timestamp folder in recordings directory
        recordings_root = Path(__file__).resolve().parent.parent / "recordings"
        target_dir = sorted(list(recordings_root.glob("*")))[-1]

    generate_map(target_dir)