import os
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .utils.basic_utils import load_data_h5, write_data_h5
from .particle_tracking_code import no_previous_tracks_3d, no_previous_tracks, previous_tracks, previous_tracks_3d


def process_frame_range(frame_range, folder, filename, dimension,
                        box_size,
                        box_size_initial_x_lo, box_size_initial_x_hi,
                        box_size_initial_y_lo, box_size_initial_y_hi,
                        box_size_initial_z_lo, box_size_initial_z_hi,
                        start_time, show_progress=True,
                        output_h5part=None, dt=1.0, write_failed_tracks=False):
    print("  Loading frames {}-{}...".format(frame_range[0], frame_range[-1]), flush=True)
    data_range = load_data_h5(
        os.path.join(folder, filename), frame_range)

    frame_steps = list(range(min(frame_range), max(frame_range)))
    iterator = tqdm(
        frame_steps,
        desc=f"Track {frame_range[0]}-{frame_range[-1]}",
        unit="frame",
        leave=False,
        disable=not show_progress or tqdm is None,
    )
    for jj in iterator:
        if jj % 500 == 0:
            logging.info(str(jj))
            logging.info(str(time.time() - start_time) + ' seconds')

        imInit = np.where(data_range.Slice == jj)[0]
        if tqdm is not None and show_progress:
            iterator.set_postfix(particles=len(imInit))

        # Precompute frame indices once per frame step (same for all particles)
        if dimension == '3d':
            idx_0 = np.where(data_range.Slice == jj)[0]
            idx_1 = np.where(data_range.Slice == jj + 1)[0]
            idx_2 = np.where(data_range.Slice == jj + 2)[0]
            idx_3 = np.where(data_range.Slice == jj + 3)[0]
            idx_m1 = np.where(data_range.Slice == jj - 1)[0] if jj > min(frame_range) else None
        else:
            idx_0 = idx_1 = idx_2 = idx_3 = idx_m1 = None

        n_particles = len(imInit)
        for ii in range(n_particles):
            # Heartbeat every 1000 particles so you can see progress (avoids "stuck" impression)
            if (ii + 1) % 1000 == 0 or ii == 0:
                print("  frame {}: particle {}/{}".format(jj, ii + 1, n_particles), flush=True)
            if data_range.Count[imInit[ii]] == 0:
                if dimension == '3d':
                    data_range = no_previous_tracks_3d(
                        data_range, jj, ii, box_size,
                        box_size_initial_x_lo, box_size_initial_x_hi,
                        box_size_initial_y_lo, box_size_initial_y_hi,
                        box_size_initial_z_lo, box_size_initial_z_hi,
                        _im0=idx_0, _im1=idx_1, _im2=idx_2, _im3=idx_3,
                    )
                else:
                    data_range = no_previous_tracks(
                        data_range, jj, ii, box_size,
                        box_size_initial_x_lo, box_size_initial_x_hi,
                        box_size_initial_y_lo, box_size_initial_y_hi,
                    )
            else:
                if dimension == '3d':
                    data_range = previous_tracks_3d(
                        data_range, jj, ii, box_size,
                        _im0=idx_m1, _im1=idx_0, _im2=idx_1, _im3=idx_2,
                    )
                else:
                    data_range = previous_tracks(data_range, jj, ii, box_size)

        if len(data_range.Count[data_range.Count == -1]) > 0:
            data_range.Count[data_range.Count == -1] = 0

    write_data_h5(data_range, folder, frame_range, output_h5part=output_h5part, dt=dt, include_failed_tracks=write_failed_tracks)
    return f"[PID {os.getpid()}] Finished frame range {frame_range} in {time.time() - start_time:.1f}s)"


def process_batch(batch, folder, filename, dimension,
                  box_size,
                  box_size_initial_x_lo, box_size_initial_x_hi,
                  box_size_initial_y_lo, box_size_initial_y_hi,
                  box_size_initial_z_lo, box_size_initial_z_hi,
                  start_time, show_progress=True,
                  output_h5part=None, dt=1.0, write_failed_tracks=False):
    print("[Worker] Processing batch ({} frame ranges)...".format(len(batch)), flush=True)
    results = []
    for frame_range in batch:
        progress = process_frame_range(
            frame_range, folder, filename, dimension,
            box_size,
            box_size_initial_x_lo, box_size_initial_x_hi,
            box_size_initial_y_lo, box_size_initial_y_hi,
            box_size_initial_z_lo, box_size_initial_z_hi,
            start_time, show_progress=show_progress,
            output_h5part=output_h5part,
            dt=dt,
            write_failed_tracks=write_failed_tracks,
        )
        print(progress)
    return results
