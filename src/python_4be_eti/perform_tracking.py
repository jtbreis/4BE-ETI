from concurrent.futures import ProcessPoolExecutor

import os
import time
import logging

from .utils.basic_utils import get_nframes, chunk_list, create_h5_file
from .particle_tracking import process_batch


class FourFrameTracking():
    def __init__(self, path, filename, box_size_x, box_size_y, box_size_z, box_size_track):
        self.path = path
        self.filename = filename
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.dimension = '3d'
        self.box_size_track = box_size_track
        self.num_frames = get_nframes(
            os.path.join(path, filename))
        self.track_length = 4
        self.frame_ranges = [list(range(i, min(i + self.track_length, self.num_frames)))
                             for i in range(0, self.num_frames, self.track_length)]

        create_h5_file(folder=self.path)

    def run_tracking(self, workers=8):
        start_time = time.time()
        batches = list(chunk_list(self.frame_ranges, workers))
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_batch, batch, self.path, self.filename,
                    self.dimension, self.box_size_track, self.box_size_x,
                    self.box_size_y, self.box_size_z, start_time
                )
                for batch in batches
            ]

        logging.info('Particle tracking program took ' +
                     str(time.time() - start_time) + ' seconds to run.')
