from concurrent.futures import ProcessPoolExecutor

import os
import time
import logging

from .utils.basic_utils import (
    get_nframes,
    chunk_list,
    create_h5_file,
    merge_tracks_h5part_parts,
    position_unit_to_meters_per_unit,
)
from .particle_tracking import process_batch
from .particle_tracking_code import MAX_CANDIDATES_MESH, MAX_TARGETS_MESH


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
        write_failed_tracks=False,
        use_bspline=False,
        export_bspline_paraview=False,
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
    ):
        self.path = path
        self.filename = filename
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        # Asymmetric initial box: lo/hi are signed *offsets* added to the seed on frame im.
        # Physical bounds: [seed + min(lo, hi), seed + max(lo, hi)] per axis (same as Numba).
        # Examples in x: lo=-1, hi=3 => [x0-1, x0+3]; lo=0.5, hi=4 => [x0+0.5, x0+4];
        # lo=+1, hi=+3 => [x0+1, x0+3] (both edges on the +x side of the seed), NOT [x0-1, x0+3].
        # If None, use symmetric defaults lo=-box_size_*, hi=+box_size_*.
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
        self.write_paraview = write_paraview
        self.write_failed_tracks = write_failed_tracks
        self.use_bspline = use_bspline
        self.export_bspline_paraview = export_bspline_paraview
        self.position_unit = position_unit
        self.position_scale_to_m = position_unit_to_meters_per_unit(
            position_unit)
        self.num_frames = get_nframes(
            os.path.join(path, filename))
        self.track_length = 4
        self.frame_ranges = [list(range(i, min(i + self.track_length, self.num_frames)))
                             for i in range(0, self.num_frames, self.track_length)]
        # Print ``frame: particle i/n`` every N particles; 0 = no per-particle progress lines
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

        create_h5_file(folder=self.path)

    def run_tracking(
        self,
        workers=8,
        show_progress=True,
        max_frame_ranges=None,
        particle_progress_interval=None,
    ):
        start_time = time.time()
        if particle_progress_interval is not None:
            ppi = int(particle_progress_interval)
            if ppi < 0:
                raise ValueError("particle_progress_interval must be >= 0 (0 = disable)")
        else:
            ppi = self.particle_progress_interval
        frame_ranges_to_run = self.frame_ranges
        if max_frame_ranges is not None:
            max_frame_ranges = int(max_frame_ranges)
            if max_frame_ranges <= 0:
                raise ValueError("max_frame_ranges must be a positive integer or None")
            frame_ranges_to_run = self.frame_ranges[:max_frame_ranges]
        batches = list(chunk_list(frame_ranges_to_run, workers))
        n_ranges = sum(len(b) for b in batches)
        logging.info(
            "Starting tracking: %d frame ranges in %d batch(es), workers=%d",
            n_ranges, len(batches), workers,
        )
        print("Tracking started ({} frame ranges, {} worker(s))...".format(n_ranges, workers), flush=True)
        # Only show per-track progress bar when running sequentially (one worker)
        use_progress = show_progress and (workers == 1)
        # When using multiple workers, each batch writes to its own file to avoid HDF5 concurrent-write issues
        use_part_files = workers > 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for batch_idx, batch in enumerate(batches):
                output_h5part = None
                if use_part_files:
                    output_h5part = os.path.join(
                        self.path, "tracks_part_{:d}.h5part".format(batch_idx)
                    )
                fut = executor.submit(
                    process_batch, batch, self.path, self.filename,
                    self.dimension, self.box_size_track,
                    self.box_size_initial_x_lo, self.box_size_initial_x_hi,
                    self.box_size_initial_y_lo, self.box_size_initial_y_hi,
                    self.box_size_initial_z_lo, self.box_size_initial_z_hi,
                    start_time,
                    show_progress=use_progress,
                    output_h5part=output_h5part,
                    dt=self.dt,
                    write_failed_tracks=self.write_failed_tracks,
                    use_bspline=self.use_bspline,
                    export_bspline_paraview=self.export_bspline_paraview,
                    rep_rate=self.rep_rate,
                    run=self.run,
                    track_length=self.track_length,
                    position_scale_to_m=self.position_scale_to_m,
                    position_units_str=self.position_unit,
                    particle_progress_interval=ppi,
                    max_candidates_mesh=self.max_candidates_mesh,
                    max_targets_mesh=self.max_targets_mesh,
                    debug_mesh_match_prints=self.debug_mesh_match_prints,
                )
                futures.append(fut)
            for fut in futures:
                fut.result()

        if use_part_files:
            merge_tracks_h5part_parts(self.path)

        logging.info('Particle tracking program took ' +
                     str(time.time() - start_time) + ' seconds to run.')

        if self.write_paraview and self.dt is not None and self.rep_rate is not None:
            self._export_paraview()

        if self.export_bspline_paraview and self.use_bspline and self.dt is not None and self.rep_rate is not None:
            self._export_bspline_curves_vtk()

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
                "No tracks loaded from %s (tracks.h5part missing, empty, or all tracks shorter than min_length=%s); skipping ParaView export.",
                self.path,
                self.min_track_length,
            )
            return
        out_base = os.path.join(self.path, "tracks_paraview")
        try:
            write_tracks_paraview(
                tracks,
                out_base,
                include_velocity_magnitude=True,
                per_snapshot=True,
            )
        except MemoryError:
            logging.warning(
                "ParaView per-snapshot export ran out of memory for %s; retrying with compact single-grid export.",
                out_base,
            )
            write_tracks_paraview(
                tracks,
                out_base,
                include_velocity_magnitude=True,
                per_snapshot=False,
            )
        logging.info("Wrote ParaView files: %s.h5 and %s.xmf",
                     out_base, out_base)

    def _export_bspline_curves_vtk(self):
        """Write B-spline curves for 4-frame tracks to a VTK file (ParaView)."""
        from .io.read_h5 import load_tracks_from_h5part
        from .io.write_paraview import write_bspline_curves_vtk

        tracks = load_tracks_from_h5part(
            self.path,
            self.dt,
            self.rep_rate,
            run=self.run,
            min_length=2,
        )
        if not tracks:
            logging.warning("No tracks to export B-spline curves; skipping.")
            return
        out_path = os.path.join(self.path, "bspline_curves.vtk")
        write_bspline_curves_vtk(tracks, out_path)
