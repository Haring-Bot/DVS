import argparse
import json
import os
import subprocess

def cutVideos(rgbPath, dvsPath, startRGB, startDVS, stopRGB, stopDVS):
    return None

def cutDVS(dvsPath, startStamp, endStamp):
    return False

def main(path):
    recordingNum = os.path.basename(os.path.normpath(path))

    with open(os.path.join(path, f"{recordingNum}.json"), "r") as f:
        data = json.load(f)

    if data["experiment"]["status"] == "finished":
        if data["sensors"]["RGB"]["status"] == "stopped" and data["sensors"]["DVS"]["status"] == "stopped":
            rgbStart = data["sensors"]["RGB"]["timestampStart"]
            dvsStart = data["sensors"]["DVS"]["timestampStart"]
            rgbStop = data["sensors"]["RGB"]["timestampStop"]
            dvsStop = data["sensors"]["DVS"]["timestampStop"]
            stamp = data["experiment"]["ID"]

            rgbFile = os.path.join(path, f"{stamp}_RGB.mp4")
            dvsFile = os.path.join(path, f"{stamp}_DVS.raw")

            cutVideos(rgbFile, dvsFile, rgbStart, dvsStart, rgbStop, dvsStop)

            if data["sensors"]["LIDAR"]["status"] == "stopped":
                print("LIDAR data detected. Starting to create map")
        
        else:
            print("DVS or RGB data incomplete. Exiting program...")
            return False

    else:
        print("Recording not complete. Exiting program...")
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RGB and DVS videos to be spatialy synched")
    parser.add_argument("path", type=str, help="Path to recordings directory")
    args = parser.parse_args()

    main(args.path)