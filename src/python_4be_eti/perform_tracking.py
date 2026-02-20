from concurrent.futures import ProcessPoolExecutor

import os
import time
import logging

from .utils.basic_utils import get_nframes, chunk_list, create_h5_file
from .particle_tracking import process_batch


class FourFrameTracking():
    def __init__(
        self,
        path,
        filename,
        box_size_x,
        box_size_y,
        box_size_z,
        box_size_track,
        dt=None,
        rep_rate=None,
        run=0,
        min_track_length=2,
        write_paraview=False,
    ):
        self.path = path
        self.filename = filename
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.dimension = '3d'
        self.box_size_track = box_size_track
        self.dt = dt
        self.rep_rate = rep_rate
        self.run = run
        self.min_track_length = min_track_length
        self.write_paraview = write_paraview
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

        if self.write_paraview and self.dt is not None and self.rep_rate is not None:
            self._export_paraview()

    def _export_paraview(self):
        """Write HDF5 + XDMF for ParaView from tracks.h5part in self.path."""
        from .io.read_h5 import load_tracks_from_h5part
        from .io.write_paraview import write_tracks_paraview

        tracks = load_tracks_from_h5part(
            self.path,
            self.dt,
            self.rep_rate,
            run=self.run,
            min_length=self.min_track_length,
        )
        if not tracks:
            logging.warning(
                "No tracks loaded from %s; skipping ParaView export.", self.path)
            return
        out_base = os.path.join(self.path, "tracks_paraview")
        write_tracks_paraview(
            tracks, out_base,
            include_velocity_magnitude=True,
            per_snapshot=True,
        )
        logging.info("Wrote ParaView files: %s.h5 and %s.xmf",
                     out_base, out_base)
