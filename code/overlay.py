import numpy as np
import subprocess
import os
import argparse
from pathlib import Path
import cv2

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

def load_dvs_to_rgb(transform_file):
    data = np.load(transform_file)
    R = data["R"].astype(np.float64)
    T = data["T"].astype(np.float64).reshape(3, 1)
    K_dvs = data["dvs_camera_matrix"].astype(np.float64)
    K_rgb = data["rgb_camera_matrix"].astype(np.float64)
    dist_dvs = data["dvs_dist_coeffs"].astype(np.float64) if "dvs_dist_coeffs" in data.files else np.zeros(5)
    dist_rgb = data["rgb_dist_coeffs"].astype(np.float64) if "rgb_dist_coeffs" in data.files else np.zeros(5)

    if "rgb_to_dvs" in os.path.basename(transform_file):
        R, T = R.T, -R.T @ T

    return K_dvs, dist_dvs, K_rgb, dist_rgb, R, T

def createLookupTable():
    K_dvs, dist_dvs, K_rgb, dist_rgb, R, T = load_dvs_to_rgb("/home/julian/Downloads/dvs_to_rgb_transformation.npz")
    w, h = 320, 320
    depth = 2000    #mm

    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))  #create one table for each coordinate
    pix = np.stack([x, y], -1).reshape(-1, 1, 2)   #reshapes into list of matching coordinates to fullfill nps reqs
    norm = cv2.undistortPoints(pix, K_dvs, dist_dvs).reshape(-1 ,2)     #removes intrinsics and inverts distortion
    X = np.hstack([norm, np.ones((norm.shape[0], 1))])  #adds dimension for depth
    X = X * depth
    tvec = T.ravel()

    #so far we have created a lookup table that transforms a dvs event into an undistorted variant of itself at distance depth in an 3d space

    rvec, _ = cv2.Rodrigues(R)  #transforms rotation from 3x3 jacobian into 3 vector axis-angle form
    proj, _ = cv2.projectPoints(X, rvec, tvec, K_rgb, dist_rgb)     #adds distortion of RGB onto dvs pixels

    lookuptableDVS = proj.reshape(h, w, 2).astype(np.float32)   #reshape back

    return lookuptableDVS

def transformEventstream(eventsNp, lut, chunkSize = 20_000_000):
    x, y = eventsNp["x"], eventsNp["y"]
    H, W, _ = lut.shape
    flat = lut.reshape(-1, 2)
    out = np.empty(len(x), dtype=[("x", "f4"), ("y", "f4"), ("p", "i1"), ("t", "u8")])
    out["p"] = eventsNp["p"]
    out["t"] = eventsNp["t"]

    size = len(x)
    for s in range(0, len(x), chunkSize):
        xs = np.asarray(x[s:s+chunkSize], dtype=np.int64)
        ys = np.asarray(y[s:s+chunkSize], dtype=np.int64)
        mapped = flat[ys*W + xs]
        out["x"][s:s+chunkSize] = mapped[:, 0]
        out["y"][s:s+chunkSize] = mapped[:, 1]
        done = min(s + chunkSize, size)
        print(f"Transforming eventstream. Progress {done / size * 100:.1f}%")
    
    return out

def events2buckets(events, fps):
    t = events["t"]
    bucketsize = 1_000_000 / fps
    nBuckets = int(np.ceil((t[-1] - t[0]) / bucketsize))

    # integer edges; ceil matches the old float comparison exactly
    edges = t[0] + np.ceil(np.arange(nBuckets + 1) * bucketsize).astype(np.uint64)
    idx = np.searchsorted(t, edges, side="left")

    buckets = [events[a:b] for a, b in zip(idx[:-1], idx[1:])]

    print(f"Split eventStream into {len(buckets)} frames with size {bucketsize/1000:.2f}ms at {fps}fps")
    return buckets

def buckets2npImages(buckets, xRes=None, yRes=None):
    if xRes is None or yRes is None:
        nonEmpty = [b for b in buckets if len(b) > 0]
        xRes = int(max(b["x"].max() for b in nonEmpty)) + 1
        yRes = int(max(b["y"].max() for b in nonEmpty)) + 1
        print(f"Detected resolution: {xRes}x{yRes}")

    images = np.zeros((len(buckets), yRes, xRes), dtype=np.int8)
    
    for imageN, bucket in enumerate(buckets):
        xi = np.rint(bucket["x"]).astype(np.int64)
        yi = np.rint(bucket["y"]).astype(np.int64)
        valid = (xi >= 0) & (xi < xRes) & (yi >= 0) & (yi < yRes)
        images[imageN, yi[valid], xi[valid]] = np.where(bucket["p"][valid] == 1, 1, -1)

    print(f"Successfully created {len(images)} images!")

    return images

def npImages2png(path, npImages):
    N, H, W = npImages.shape
    bgr = np.zeros((N, H, W, 3), dtype=np.uint8)

    pos = npImages == 1
    neg = npImages == -1

    bgr[pos] = (0, 0, 255)      #red
    bgr[neg] = (255, 0, 0)      #blue

    os.makedirs("dvs_frames", exist_ok=True)
    for i, image in enumerate(bgr):
        cv2.imwrite(f"dvs_frames/frame_{i:05d}.png", image)

def overlayOnRgb(npImages, rgbVideoPath, outPath, alpha=1.0, kernelSize=3):
    cap = cv2.VideoCapture(rgbVideoPath)
    if not cap.isOpened():
        raise IOError(f"Could not open {rgbVideoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(outPath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    #3x3 kernel to mitigate difference in resolution
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    print("Starting to create video from buckets")

    for i, dvs in enumerate(npImages):
        ok, frame = cap.read()
        if not ok:
            break

        pos = cv2.dilate((dvs == 1).astype(np.uint8), kernel).astype(bool)
        neg = cv2.dilate((dvs == -1).astype(np.uint8), kernel).astype(bool)

        colored = frame.copy()
        colored[pos] = (0, 0, 255)      #red
        colored[neg] = (255, 0, 0)      #blue

        frame = cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0)    #alpha = transparency

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {i} overlay frames to {outPath}")

def npImages2Video(npImages, outPath, fps=30, kernelSize=3):
    N, H, W = npImages.shape
    writer = cv2.VideoWriter(outPath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise IOError(f"Could not open {outPath} for writing")

    kernel = np.ones((kernelSize, kernelSize), np.uint8) if kernelSize > 1 else None

    for dvs in npImages:
        pos = dvs == 1
        neg = dvs == -1
        if kernel is not None:
            pos = cv2.dilate(pos.astype(np.uint8), kernel).astype(bool)
            neg = cv2.dilate(neg.astype(np.uint8), kernel).astype(bool)

        frame = np.zeros((H, W, 3), dtype=np.uint8)   #black background
        frame[pos] = (0, 0, 255)      #red
        frame[neg] = (255, 0, 0)      #blue

        writer.write(frame)

    writer.release()
    print(f"Wrote {N} DVS frames to {outPath}")

def main(path, saveChekpoints, saveImages, saveVid):
    recordingNum = os.path.basename(os.path.normpath(path))
    rawFilePath = f"{path}/{recordingNum}_DVS_cut.raw"
    dvsNpPath = f"{path}/{recordingNum}_DVS_eventstream.npy"
    dvsImagesPath = f"{path}/{recordingNum}_DVS_images.npy"
    rgbVideoPath = f"{path}/{recordingNum}_RGB_cut.mp4"
    outputVideoPath = f"{path}/{recordingNum}_overlayed_video.mp4"
    outputVideoPathDVS = f"{path}/{recordingNum}_events.mp4"

    if Path(dvsNpPath).is_file():
        print("Saved np array found. Trying to load file...")
        events_cor = np.load(dvsNpPath)
        print("Np array successfully loaded. Continuing...")
    else:
        print("Saved np array missing. Starting creation")
        csvFilePath = raw2CSV(rawFilePath)
        events = CSV2NP(csvFilePath)
        lut = createLookupTable()
        events_cor = transformEventstream(events, lut, chunkSize=20_000_000)
        print("New np array created.")
        if saveChekpoints:
            np.save(dvsNpPath, events_cor)
            print("Np array successfully saved. Continuing...")

    

    if Path(dvsImagesPath).is_file():
        print("Saved images found. Continuing with overlay...")
        images = np.load(dvsImagesPath, allow_pickle=False)
    else:
        print(f"No saved images found. Creating from np array.")
        buckets = events2buckets(events_cor, 30)
        images = buckets2npImages(buckets, 1280, 960)
        print("Images successfully created.")
        if saveChekpoints:
            np.save(dvsImagesPath, images)
            print(f"Images saved to {dvsImagesPath}")

    if saveImages:
        print("Starting to save images to png.")
        npImages2png(path, images)
        print("Images successfully saved.")

    if saveVid:
        overlayOnRgb(images, rgbVideoPath, outputVideoPath, alpha=1.0)
        npImages2Video(images, outputVideoPathDVS, 30, 3)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay events from .raw file over RGB video")
    parser.add_argument("path", type=str, help="Path to recordings directory")
    parser.add_argument("--saveCps", action="store_true", help="Should checkpoints be saved? (increases compile time and file size but decreases risk)")
    parser.add_argument("--saveImg", action="store_true", help="Save images as png?")
    parser.add_argument("--saveVid", action="store_true", help="Save videos?")
    args = parser.parse_args()

    main(args.path, args.saveCps, args.saveImg, args.saveVid)