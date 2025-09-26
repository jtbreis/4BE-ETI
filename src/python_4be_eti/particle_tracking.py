import os
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .utils.basic_utils import load_data_h5, write_data_h5
from .particle_tracking_code import no_previous_tracks_3d, no_previous_tracks, previous_tracks, previous_tracks_3d


def process_frame_range(frame_range, folder, filename, dimension,
                        box_size, box_size_initial_x, box_size_initial_y,
                        box_size_initial_z, start_time):

    data_range = load_data_h5(
        os.path.join(folder, filename), frame_range)

    for jj in range(min(frame_range), max(frame_range)):
        if jj % 500 == 0:
            logging.info(str(jj))
            logging.info(str(time.time() - start_time) + ' seconds')

        imInit = np.where(data_range.Slice == jj)[0]

        for ii in range(len(imInit)):
            if data_range.Count[imInit[ii]] == 0:
                if dimension == '3d':
                    data_range = no_previous_tracks_3d(
                        data_range, jj, ii, box_size,
                        box_size_initial_x, box_size_initial_y,
                        box_size_initial_z
                    )
                else:
                    data_range = no_previous_tracks(
                        data_range, jj, ii, box_size,
                        box_size_initial_x, box_size_initial_y
                    )
            else:
                if dimension == '3d':
                    data_range = previous_tracks_3d(
                        data_range, jj, ii, box_size)
                else:
                    data_range = previous_tracks(data_range, jj, ii, box_size)

        if len(data_range.Count[data_range.Count == -1]) > 0:
            data_range.Count[data_range.Count == -1] = 0

    write_data_h5(data_range, folder, frame_range)
    return f"[PID {os.getpid()}] Finished frame range {frame_range} in {time.time() - start_time:.1f}s)"


def process_batch(batch, folder, filename, dimension,
                  box_size, box_size_initial_x, box_size_initial_y,
                  box_size_initial_z, start_time):
    results = []
    for frame_range in batch:
        progress = process_frame_range(frame_range, folder, filename, dimension,
                                       box_size, box_size_initial_x, box_size_initial_y,
                                       box_size_initial_z, start_time)
        print(progress)
    return results
