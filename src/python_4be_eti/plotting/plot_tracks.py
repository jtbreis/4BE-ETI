import matplotlib.pyplot as plt

from ..io.read_h5 import read_tracks_from_h5
import numpy as np


def plot_tracks(filename, frame_range):
    X, Y, Z, Slice, Count = read_tracks_from_h5(
        filename=filename, frame_range=frame_range)

    mask = Count != 0
    X, Y, Z, Count = np.array(X)[mask], np.array(
        Y)[mask], np.array(Z)[mask], np.array(Count)[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=Count, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Count')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
