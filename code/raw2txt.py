import argparse
import os
from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator

def raw2CustomTxt(path):
    root, ext = os.path.splitext(path)
    outputPath = f"{root}.txt"

    if Path(outputPath).is_file():
        print(f"{outputPath} already exists")
        return outputPath

    try:
        print(f"Opening {path}...")
        mv_iterator = EventsIterator(path, mode="n_events", n_events=1_000_000)
        height, width = mv_iterator.get_size()

        print(f"Writing to {outputPath} in custom format...")
        with open(outputPath, "w") as f:
            f.write(f"{width} {height}\n")
            
            for evs in mv_iterator:
                if evs.size == 0:
                    continue
                
                t = evs['t'].astype(np.float64) / 1e6
                x = evs['x']
                y = evs['y']
                p = evs['p']
                
                chunk = np.column_stack((t, x, y, p))
                np.savetxt(f, chunk, fmt=['%.12f', '%d', '%d', '%d'], delimiter=' ')
                
        print("Successfully transformed .raw to custom .txt file")
    except Exception as e:
        print(f".raw to .txt transformation failed: {e}")
        raise

    return outputPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Metavision .raw file to custom txt format.")
    parser.add_argument("path", type=str, help="Path to the input .raw file")
    args = parser.parse_args()
    
    raw2CustomTxt(args.path)