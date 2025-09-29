import numpy as np
from ..io.read_h5 import read_tracks_from_h5, get_nsamples_from_h5
from ..track import Track


def load_tracks(filename, dt, rep_rate, run=0, min_length=3):
    nsamples = get_nsamples_from_h5(filename)
    frame_range = [range(i * 4, (i + 1) * 4) for i in range(nsamples)]
    samples = []
    for sample_idx, sample_frames in enumerate(frame_range):
        X, Y, Z, Slice, Count = read_tracks_from_h5(
            filename, sample_frames)
        mask = Count != 0
        X, Y, Z, Count = X[mask], Y[mask], Z[mask], Count[mask]

        unique_counts = np.unique(Count)
        tracks = np.empty(unique_counts.shape[0], dtype=object)
        for i, uc in enumerate(unique_counts):
            idx = Count == uc  # group all points with this count value
            track = Track(
                X[idx], Y[idx], Z[idx],
                dt,
                idx,
                1 / rep_rate * sample_idx,
                run
            )
            track.compute_velocity()
            track.compute_acceleration()
            if track.track_length < min_length:
                continue
            samples.append(track)

    return samples
