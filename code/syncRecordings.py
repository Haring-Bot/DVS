import argparse
import json
import os
import subprocess

def cutVideos(rgbPath, dvsPath, startRGB, startDVS, stopRGB, stopDVS):
    """ print(f"startRGB:{startRGB}")
    print(f"stopRGB:{stopRGB}")
    print(f"startDVS:{startDVS}")
    print(f"stopDVS:{stopDVS}") """
    globalStart = max(startRGB, startDVS)
    globalStop = min(stopRGB, stopDVS)
    globalDuration = globalStop - globalStart

    startCutRGB = globalStart - startRGB
    startCutDVS = globalStart - startDVS

    stopCutRGB = startCutRGB + globalDuration
    stopCutDVS = startCutDVS + globalDuration

    
    cutDVS(dvsPath, startCutDVS, stopCutDVS)
    cutRGB(rgbPath, startCutRGB, stopCutRGB)
    
    return None

def cutRGB(rgbPath, startStamp, stopStamp):
    root, ext = os.path.splitext(rgbPath)
    outputPath = f"{root}_cut{ext}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", rgbPath,
        "-ss", str(startStamp),
        "-to", str(stopStamp),
        "-c", "copy",
        outputPath
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully saved trimmed RGB file to {outputPath}")
    except subprocess.CalledProcessError as e:
        print(f"RGB trimming failed: {e}")
        raise

    return outputPath

def cutDVS(dvsPath, startStamp, stopStamp):
    root, ext = os.path.splitext(dvsPath)
    outputPath = f"{root}_cut{ext}"

    """ print(f"start: {startStamp}")
    print(f"stop: {stopStamp}") """
    
    cmd = [
        "metavision_file_cutter",
        "-i", dvsPath,
        "-o", outputPath,
        "-s", str(startStamp),
        "-e", str(stopStamp)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully saved trimmed DVS file to {outputPath}")
    except subprocess.CalledProcessError as e:
        print(f"DVS trimming failed: {e}")
        raise

    return outputPath

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