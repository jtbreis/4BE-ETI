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
from .particle_tracking_code import (
    no_previous_tracks,
    previous_tracks,
    set_mesh_debug_match_prints,
)
from .particle_tracking_numba import no_previous_tracks_3d, previous_tracks_3d


def process_frame_range(frame_range, folder, filename, dimension,
                        box_size,
                        box_size_initial_x_lo, box_size_initial_x_hi,
                        box_size_initial_y_lo, box_size_initial_y_hi,
                        box_size_initial_z_lo, box_size_initial_z_hi,
                        start_time, show_progress=True,
                        output_h5part=None, dt=1.0, write_failed_tracks=False,
                        use_bspline=False, export_bspline_paraview=False,
                        rep_rate=None, run=0,
                        track_length=4,
                        position_scale_to_m=1e-3,
                        position_units_str="mm",
                        particle_progress_interval=1000,
                        max_candidates_mesh=None,
                        max_targets_mesh=None,
                        debug_mesh_match_prints=0):
    set_mesh_debug_match_prints(debug_mesh_match_prints)
    print(
        "  Loading frames {}-{} from HDF5...".format(frame_range[0], frame_range[-1]), flush=True)
    data_range = load_data_h5(
        os.path.join(folder, filename), frame_range)
    print("  Loaded {} particles total, starting tracking...".format(len(data_range.x)), flush=True)

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
            idx_m1 = np.where(data_range.Slice == jj -
                              1)[0] if jj > min(frame_range) else None
        else:
            idx_0 = idx_1 = idx_2 = idx_3 = idx_m1 = None

        n_particles = len(imInit)
        for ii in range(n_particles):
            # Heartbeat: print every ``particle_progress_interval`` particles (and first); 0 = off
            if particle_progress_interval and (
                (ii + 1) % particle_progress_interval == 0
                or ii == 0
            ):
                print("  frame {}: particle {}/{}".format(jj,
                      ii + 1, n_particles), flush=True)
            if data_range.Count[imInit[ii]] == 0:
                if dimension == '3d':
                    data_range = no_previous_tracks_3d(
                        data_range, jj, ii, box_size,
                        box_size_initial_x_lo, box_size_initial_x_hi,
                        box_size_initial_y_lo, box_size_initial_y_hi,
                        box_size_initial_z_lo, box_size_initial_z_hi,
                        max_candidates_mesh=max_candidates_mesh,
                        max_targets_mesh=max_targets_mesh,
                        debug_mesh_match_prints=debug_mesh_match_prints,
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
                        max_candidates_mesh=max_candidates_mesh,
                        max_targets_mesh=max_targets_mesh,
                        _im0=idx_m1, _im1=idx_0, _im2=idx_1, _im3=idx_2,
                    )
                else:
                    data_range = previous_tracks(data_range, jj, ii, box_size)

        if len(data_range.Count[data_range.Count == -1]) > 0:
            data_range.Count[data_range.Count == -1] = 0

    write_data_h5(
        data_range, folder, frame_range, output_h5part=output_h5part,
        dt=dt, include_failed_tracks=write_failed_tracks, use_bspline=use_bspline,
        rep_rate_hz=rep_rate,
        track_length=track_length,
        position_scale_to_m=position_scale_to_m,
        position_units_str=position_units_str,
    )
    # Export B-spline curves after each frame range so partial results are saved if process is killed
    if export_bspline_paraview and use_bspline and dt is not None and rep_rate is not None:
        try:
            from .io.read_h5 import load_tracks_from_h5part
            from .io.write_paraview import write_bspline_curves_vtk
            h5_path = output_h5part if output_h5part else os.path.join(folder, "tracks.h5part")
            tracks = load_tracks_from_h5part(
                folder, dt, rep_rate, run=run, min_length=2,
                use_bspline=True, filepath=h5_path,
            )
            if tracks:
                out_path = os.path.join(folder, "bspline_curves.vtk")
                write_bspline_curves_vtk(tracks, out_path)
                print(f"  Wrote B-spline curves to {out_path}", flush=True)
        except Exception as e:
            logging.warning("Could not export B-spline curves after frame range %s: %s", frame_range, e)
    return f"[PID {os.getpid()}] Finished frame range {frame_range} in {time.time() - start_time:.1f}s)"


def process_batch(batch, folder, filename, dimension,
                  box_size,
                  box_size_initial_x_lo, box_size_initial_x_hi,
                  box_size_initial_y_lo, box_size_initial_y_hi,
                  box_size_initial_z_lo, box_size_initial_z_hi,
                  start_time, show_progress=True,
                  output_h5part=None, dt=1.0, write_failed_tracks=False,
                  use_bspline=False, export_bspline_paraview=False,
                  rep_rate=None, run=0,
                  track_length=4,
                  position_scale_to_m=1e-3,
                  position_units_str="mm",
                  particle_progress_interval=1000,
                  max_candidates_mesh=None,
                  max_targets_mesh=None,
                  debug_mesh_match_prints=0):
    print("[Worker] Processing batch ({} frame ranges)...".format(
        len(batch)), flush=True)
    results = []
    for idx, frame_range in enumerate(batch):
        print(f"  Batch: starting frame range {idx + 1}/{len(batch)} {frame_range}", flush=True)
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
            use_bspline=use_bspline,
            export_bspline_paraview=export_bspline_paraview,
            rep_rate=rep_rate,
            run=run,
            track_length=track_length,
            position_scale_to_m=position_scale_to_m,
            position_units_str=position_units_str,
            particle_progress_interval=particle_progress_interval,
            max_candidates_mesh=max_candidates_mesh,
            max_targets_mesh=max_targets_mesh,
            debug_mesh_match_prints=debug_mesh_match_prints,
        )
        print(progress)
    return results
