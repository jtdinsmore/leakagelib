import numpy as np

WIDTH=36
HEIGHT=int(36/0.866)

def process_track(track):
    track = track.astype(float) - np.nanpercentile(track, 25) # Remove background so that the image can be padded with zeros later
    track = np.maximum(track, 0)

    # Pad the track if it is misaligned
    if track.shape[0] % 4 == 0:
        track = np.vstack((
            track,
            np.zeros((1,track.shape[1])),
            np.zeros((1,track.shape[1]))
        ))
    
    if track.shape[0] > HEIGHT: # Trim
        start_x = (track.shape[0]-HEIGHT)//2
        stop_x = (track.shape[0]+HEIGHT)//2
        if start_x % 2 == 0:
            start_x += 1
            stop_x += 1
        track = track[start_x:stop_x, :]
    elif track.shape[0] < HEIGHT: # Pad
        initial_x = (HEIGHT-track.shape[0])//2
        odd = (HEIGHT - track.shape[0]) % 2
        track = np.vstack((np.zeros((initial_x, track.shape[1])), track, np.zeros((initial_x+odd, track.shape[1]))))
    if track.shape[1] > WIDTH: # Trim
        start_y = (track.shape[1]-WIDTH)//2
        stop_y = (track.shape[1]+WIDTH)//2
        track = track[:,start_y:stop_y]
    elif track.shape[1] < WIDTH: # Pad
        initial_y = (WIDTH - track.shape[1]) // 2
        odd = (WIDTH - track.shape[1]) % 2
        track = np.hstack((np.zeros((track.shape[0], initial_y)), track, np.zeros((track.shape[0], initial_y+odd))))
    return track

def load_tracks(hdul, indices):
    tracks = []
    # Load the event
    for event_index in indices:
        min_x = hdul[1].data["MIN_CHIPX"][event_index]
        min_y = hdul[1].data["MIN_CHIPY"][event_index]
        max_x = hdul[1].data["MAX_CHIPX"][event_index]
        max_y = hdul[1].data["MAX_CHIPY"][event_index]
        track = hdul[1].data["PIX_PHAS"][event_index].reshape(max_y-min_y+1, max_x-min_x+1)
        tracks.append(process_track(track))

    tracks = np.array(tracks)
    shape = tuple(np.concatenate([tracks.shape, [1]]))
    return tracks.reshape(shape)

def associate_events(l1_times, l2_times):
    # Figure out which events are associated with which
    if len(l1_times) == len(l2_times):
        l1_indices = np.arange(len(l1_times))
    else:
        _, indices, counts = np.unique(l1_times, return_counts=True, return_index=True)
        duplicate_mask = np.zeros(len(l1_times), bool)
        if np.max(counts) > 1:
            print("WARNING: The level 1 file contained multiple events with the same time. Since the l1 and l2 files do not have the same number of rows, flag_background uses the time of each event to associate events between the l1 and l2 files. It cannot do this if there are multiple events with the same time. The first event listed in the file is used.")
            duplicate_mask = np.ones(len(l1_times), bool)
            duplicate_mask[np.sort(indices)] = False
        l1_mask = np.isin(l1_times, l2_times) & ~duplicate_mask
        l1_indices = np.where(l1_mask)[0]
        n_unassociated_events = np.sum(~np.isin(l2_times, l1_times))
        if n_unassociated_events > 0:
            raise Exception(f"{n_unassociated_events} events in the l2 file had times which are not in the l1 file.")

    return l1_indices