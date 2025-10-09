import numpy as np
from ..io.read_h5 import read_tracks_from_h5, get_nsamples_from_h5
from ..track import Track

import concurrent.futures


def load_tracks(filename, dt, rep_rate, run=0, min_length=3):
    nsamples = get_nsamples_from_h5(filename)
    frame_range = [range(i * 4, (i + 1) * 4) for i in range(nsamples)]
    samples = []

    def process_sample(args):
        sample_idx, sample_frames = args
        X, Y, Z, Slice, Count = read_tracks_from_h5(filename, sample_frames)
        mask = Count != 0
        X, Y, Z, Count = X[mask], Y[mask], Z[mask], Count[mask]

        unique_counts = np.unique(Count)
        tracks = []
        for i, uc in enumerate(unique_counts):
            idx = Count == uc  # group all points with this count value
            track = Track(
                X[idx], Y[idx], Z[idx],
                dt,
                uc,
                1 / rep_rate * sample_idx,
                run
            )
            track.compute_velocity()
            track.compute_acceleration()
            if track.track_length < min_length:
                continue
            tracks.append(track)
        return tracks

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sample, enumerate(frame_range)))
        for tracks in results:
            samples.extend(tracks)

    return samples
