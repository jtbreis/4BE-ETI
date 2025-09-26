import h5py
import numpy as np


def get_nsamples_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        return len(f.keys())


def read_tracks_from_h5(filename, frame_range):
    with h5py.File(filename, 'r') as f:
        group_name = f"frames{min(frame_range)}-{max(frame_range)}"
        if group_name not in f:
            X = np.zeros([1, 1])
            Y = np.zeros([1, 1])
            Z = np.zeros([1, 1])
            Slice = np.zeros([1, 1])
            Count = np.zeros([1, 1])
            return X, Y, Z, Slice, Count
        group = f[group_name]
        X = np.array(group['X'])
        Y = np.array(group['Y'])
        Z = np.array(group['Z'])
        Slice = np.array(group['Slice'])
        Count = np.array(group['Count'])

        return X, Y, Z, Slice, Count
