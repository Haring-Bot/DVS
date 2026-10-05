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
    
    print(f"[+] Found LIDAR bag: {bag_folder_name}")

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
            host_recordings_dir: {"bind": "/recordings", "mode": "rw"}
        },
        detach=True,
        tty=True,
        command="bash"
    )

    try:
        # 1. Start FAST-LIO mapping launch file in background inside container
        print("[+] Launching FAST-LIO2 node...")
        fast_lio_launch_cmd = (
            "bash -c 'source /root/ros2_ws/install/setup.bash && "
            "ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml'"
        )
        container.exec_run(fast_lio_launch_cmd, detach=True)

        # Allow node initialization
        time.sleep(4)

        # 2. Replay ROS 2 Bag inside container
        container_bag_path = f"/recordings/{recording_folder.name}/{bag_folder_name}"
        print(f"[+] Playing ROS 2 bag from: {container_bag_path}...")
        
        play_bag_cmd = [
            "docker", "exec", container_name,
            "bash", "-c",
            f"source /root/ros2_ws/install/setup.bash && ros2 bag play {container_bag_path}"
        ]
        
        # Block until bag playback finishes
        subprocess.run(play_bag_cmd, check=True)
        print("[+] Bag playback complete!")

        # 3. Trigger map save service
        print("[+] Triggering /map_save service...")
        save_srv_cmd = (
            "bash -c 'source /root/ros2_ws/install/setup.bash && "
            "ros2 service call /map_save std_srvs/srv/Trigger {}'"
        )
        container.exec_run(save_srv_cmd)
        time.sleep(2)

        # 4. Copy generated PCD file from default workspace output to host recording folder
        pcd_source_path = Path(host_ros2_ws) / "src" / "FAST_LIO_ROS2" / "PCD" / "scans.pcd"
        target_pcd_path = recording_folder / f"{recording_folder.name}_map.pcd"

        if pcd_source_path.exists():
            # Rename/move PCD file into target recording directory
            pcd_source_path.rename(target_pcd_path)
            print(f"[✔] 3D PCD Map successfully generated and saved to:\n    {target_pcd_path}")
        else:
            print(f"[!] Warning: Could not locate map at {pcd_source_path}. Ensure pcd_save_en: true in mid360.yaml")

    finally:
        print("[+] Stopping processing container...")
        container.stop()
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