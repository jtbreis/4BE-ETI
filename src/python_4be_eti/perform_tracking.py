from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import time
import logging

from .utils.basic_utils import (
    get_nframes,
    balance_frame_ranges,
    format_tracking_batch_plan,
    merge_tracks_h5part_parts,
    remove_tracks_h5part_parts,
    remove_staging_h5part,
    staging_h5part_path,
    canonical_h5part_path,
    write_h5_tracking_metadata,
    TRACKS_H5PART_CANONICAL,
    TRACKS_H5PART_STAGING,
    position_unit_to_meters_per_unit,
)
from .particle_tracking import process_batch
from .particle_tracking_code import MAX_CANDIDATES_MESH, MAX_TARGETS_MESH


def _tracking_job_kwargs(tracker, start_time, ppi, show_progress, verbose):
    return dict(
        folder=tracker.path,
        filename=tracker.filename,
        dimension=tracker.dimension,
        box_size=tracker.box_size_track,
        box_size_initial_x_lo=tracker.box_size_initial_x_lo,
        box_size_initial_x_hi=tracker.box_size_initial_x_hi,
        box_size_initial_y_lo=tracker.box_size_initial_y_lo,
        box_size_initial_y_hi=tracker.box_size_initial_y_hi,
        box_size_initial_z_lo=tracker.box_size_initial_z_lo,
        box_size_initial_z_hi=tracker.box_size_initial_z_hi,
        start_time=start_time,
        show_progress=show_progress,
        dt=tracker.dt,
        write_failed_tracks=tracker.write_failed_tracks,
        use_bspline=tracker.use_bspline,
        rep_rate=tracker.rep_rate,
        run=tracker.run,
        track_length=tracker.track_length,
        position_scale_to_m=tracker.position_scale_to_m,
        position_units_str=tracker.position_unit,
        particle_progress_interval=ppi,
        max_candidates_mesh=tracker.max_candidates_mesh,
        max_targets_mesh=tracker.max_targets_mesh,
        debug_mesh_match_prints=tracker.debug_mesh_match_prints,
        verbose=verbose,
    )


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
        write_failed_tracks=False,
        use_bspline=False,
        *,
        box_size_initial_x_lo=None,
        box_size_initial_x_hi=None,
        box_size_initial_y_lo=None,
        box_size_initial_y_hi=None,
        box_size_initial_z_lo=None,
        box_size_initial_z_hi=None,
        position_unit="mm",
        particle_progress_interval=1000,
        max_candidates_mesh=None,
        max_targets_mesh=None,
        debug_mesh_match_prints=0,
        verbose=True,
        rays_input_path=None,
    ):
        self.path = path
        self.rays_input_path = rays_input_path if rays_input_path is not None else path
        self.filename = filename
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.box_size_initial_x_lo = box_size_initial_x_lo if box_size_initial_x_lo is not None else -box_size_x
        self.box_size_initial_x_hi = box_size_initial_x_hi if box_size_initial_x_hi is not None else box_size_x
        self.box_size_initial_y_lo = box_size_initial_y_lo if box_size_initial_y_lo is not None else -box_size_y
        self.box_size_initial_y_hi = box_size_initial_y_hi if box_size_initial_y_hi is not None else box_size_y
        self.box_size_initial_z_lo = box_size_initial_z_lo if box_size_initial_z_lo is not None else -box_size_z
        self.box_size_initial_z_hi = box_size_initial_z_hi if box_size_initial_z_hi is not None else box_size_z
        self.dimension = '3d'
        self.box_size_track = box_size_track
        self.dt = dt
        self.rep_rate = rep_rate
        self.run = run
        self.min_track_length = min_track_length
        self.write_failed_tracks = write_failed_tracks
        self.use_bspline = use_bspline
        self.position_unit = position_unit
        self.position_scale_to_m = position_unit_to_meters_per_unit(
            position_unit)
        self.num_frames = get_nframes(
            os.path.join(self.rays_input_path, filename))
        self.track_length = 4
        self.frame_ranges = [list(range(i, min(i + self.track_length, self.num_frames)))
                             for i in range(0, self.num_frames, self.track_length)]
        ppi = int(particle_progress_interval)
        if ppi < 0:
            raise ValueError("particle_progress_interval must be >= 0 (0 = disable)")
        self.particle_progress_interval = ppi

        self.max_candidates_mesh = (
            int(max_candidates_mesh)
            if max_candidates_mesh is not None
            else MAX_CANDIDATES_MESH
        )
        self.max_targets_mesh = (
            int(max_targets_mesh)
            if max_targets_mesh is not None
            else MAX_TARGETS_MESH
        )
        if self.max_candidates_mesh < 1 or self.max_targets_mesh < 1:
            raise ValueError(
                "max_candidates_mesh and max_targets_mesh must be integers >= 1"
            )

        dmp = int(debug_mesh_match_prints)
        if dmp < 0:
            raise ValueError("debug_mesh_match_prints must be >= 0")
        self.debug_mesh_match_prints = dmp
        self.verbose = bool(verbose)

    def tracking_parameters_dict(self, *, workers=None, max_frame_ranges=None):
        """Serializable tracking parameters for metadata / JSON sidecar."""
        return {
            "job_id": "tracking",
            "tracker": "four_frame",
            "box_size_x": self.box_size_x,
            "box_size_y": self.box_size_y,
            "box_size_z": self.box_size_z,
            "box_size_initial_x_lo": self.box_size_initial_x_lo,
            "box_size_initial_x_hi": self.box_size_initial_x_hi,
            "box_size_initial_y_lo": self.box_size_initial_y_lo,
            "box_size_initial_y_hi": self.box_size_initial_y_hi,
            "box_size_initial_z_lo": self.box_size_initial_z_lo,
            "box_size_initial_z_hi": self.box_size_initial_z_hi,
            "box_size_track": self.box_size_track,
            "dt": self.dt,
            "rep_rate": self.rep_rate,
            "position_unit": self.position_unit,
            "write_failed_tracks": self.write_failed_tracks,
            "use_bspline": self.use_bspline,
            "rays_filename": self.filename,
            "rays_input_path": self.rays_input_path,
            "output_path": self.path,
            "num_frames": self.num_frames,
            "track_length": self.track_length,
            "workers": workers,
            "max_frame_ranges": max_frame_ranges,
            "max_candidates_mesh": self.max_candidates_mesh,
            "max_targets_mesh": self.max_targets_mesh,
            "debug_mesh_match_prints": self.debug_mesh_match_prints,
            "particle_progress_interval": self.particle_progress_interval,
        }

    def run_tracking(
        self,
        workers=8,
        show_progress=True,
        max_frame_ranges=None,
        particle_progress_interval=None,
        verbose=None,
        run_metadata=None,
    ):
        start_time = time.time()
        if particle_progress_interval is not None:
            ppi = int(particle_progress_interval)
            if ppi < 0:
                raise ValueError("particle_progress_interval must be >= 0 (0 = disable)")
        else:
            ppi = self.particle_progress_interval
        if verbose is None:
            verbose = self.verbose
        frame_ranges_to_run = self.frame_ranges
        if max_frame_ranges is not None:
            max_frame_ranges = int(max_frame_ranges)
            if max_frame_ranges <= 0:
                raise ValueError("max_frame_ranges must be a positive integer or None")
            frame_ranges_to_run = self.frame_ranges[:max_frame_ranges]

        input_h5 = os.path.join(self.rays_input_path, self.filename)
        batches, batch_costs = balance_frame_ranges(
            frame_ranges_to_run, workers, filename=input_h5,
        )
        n_ranges = len(frame_ranges_to_run)
        batch_plan_lines = format_tracking_batch_plan(batches, batch_costs)
        for line in batch_plan_lines:
            logging.info(line)
            print(line, flush=True)

        staging_path = staging_h5part_path(self.path)
        canonical_path = canonical_h5part_path(self.path)
        if os.path.isfile(canonical_path):
            print(
                f"Note: existing {TRACKS_H5PART_CANONICAL} will not be overwritten "
                f"(writing to {TRACKS_H5PART_STAGING}; rename manually when satisfied).",
                flush=True,
            )

        use_part_files = workers > 1
        remove_staging_h5part(self.path)
        remove_tracks_h5part_parts(self.path)
        n_batches = len(batches)

        logging.info(
            "Starting tracking: %d frame range(s) in %d load-balanced batch(es), "
            "%d parallel worker(s)",
            n_ranges, n_batches, workers,
        )
        print(
            "Tracking started ({} ranges, {} batch(es), {} worker(s))...".format(
                n_ranges, n_batches, workers),
            flush=True,
        )

        job_kw = _tracking_job_kwargs(
            self, start_time, ppi, show_progress and workers == 1, verbose,
        )

        if workers <= 1:
            process_batch(
                frame_ranges_to_run,
                output_h5part=staging_path,
                **job_kw,
            )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for batch_idx, batch in enumerate(batches):
                    output_h5part = os.path.join(
                        self.path,
                        "tracks_part_{:d}.h5part".format(batch_idx),
                    )
                    fut = executor.submit(
                        process_batch,
                        batch,
                        output_h5part=output_h5part,
                        **job_kw,
                    )
                    futures.append((batch_idx, batch, fut))

                if verbose:
                    for batch_idx, batch, fut in futures:
                        fut.result()
                else:
                    fut_to_batch = {fut: (idx, b) for idx, b, fut in futures}
                    done = 0
                    for fut in as_completed(fut_to_batch):
                        batch_idx, batch = fut_to_batch[fut]
                        fut.result()
                        done += 1
                        print(
                            "Tracking batch {}/{} complete ({} ranges, frames {}-{})".format(
                                done,
                                n_batches,
                                len(batch),
                                min(min(fr) for fr in batch),
                                max(max(fr) for fr in batch),
                            ),
                            flush=True,
                        )

        if use_part_files:
            merge_tracks_h5part_parts(self.path, out_path=staging_path)

        meta = dict(run_metadata or {})
        meta.update(self.tracking_parameters_dict(
            workers=workers, max_frame_ranges=max_frame_ranges,
        ))
        meta["duration_seconds"] = time.time() - start_time
        meta["output_h5part"] = TRACKS_H5PART_STAGING
        write_h5_tracking_metadata(staging_path, meta)

        elapsed = time.time() - start_time
        logging.info('Particle tracking program took %s seconds to run.', elapsed)
        print(
            f"Tracking output: {staging_path} "
            f"(rename to {TRACKS_H5PART_CANONICAL} when ready)",
            flush=True,
        )
        return staging_path, meta
