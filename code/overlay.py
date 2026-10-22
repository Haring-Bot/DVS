import numpy as np
import subprocess
import os
import argparse
from pathlib import Path

def raw2CSV(path):
    root, ext = os.path.splitext(path)
    outputPath = f"{root}.csv"

    if Path(outputPath).is_file():
        print(f"{outputPath} already exists")
        return outputPath

    cmd = [
        "metavision_file_to_csv",
        "-i", path,
        "-o", outputPath
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Successfully transformed .raw to .csv file")
    except subprocess.CalledProcessError as e:
        print(f".raw to .csv transformation failed: {e}")
        raise

    return outputPath

def CSV2NP(csvFile):
    npArray = np.loadtxt(
        csvFile,
        delimiter=",",
        skiprows=0,
        dtype={"names":("x","y","p","t"),
               "formats":("u2","u2","i1","u8")}
    )
    return npArray

def events2buckets(events):
    buckets = ()
    return buckets

def main(path):
    recordingNum = os.path.basename(os.path.normpath(path))
    rawFilePath = f"{path}/{recordingNum}_DVS_cut.raw"

    csvFilePath = raw2CSV(rawFilePath)
    events = CSV2NP(csvFilePath)

    print(events[1]["t"])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay events from .raw file over RGB video")
    parser.add_argument("path", type=str, help="Path to recordings directory")
    args = parser.parse_args()

    main(args.path)