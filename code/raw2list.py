import numpy as np
from metavision_core.event_io import EventsIterator

def load_raw_as_array(file_path):
    # Initialize the iterator with your recorded file path
    mv_iterator = EventsIterator(input_path=file_path)
    
    event_chunks = []
    
    print("Reading DVS file...")
    for evs in mv_iterator:
        # 'evs' is a NumPy structured array for the current time slice
        if evs.size > 0:
            event_chunks.append(evs)
            
    # Combine all individual chunks into one giant continuous array
    all_events = np.concatenate(event_chunks)
    
    print(f"Successfully loaded {len(all_events)} total events.")
    return all_events

def main(path, save):
    print("still waiting for saving implementation")

if __name__ == "main":
    main()
