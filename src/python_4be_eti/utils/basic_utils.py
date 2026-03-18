import logging
import math
import os
import re

import numpy as np
import pandas as pd
import h5py

# This file introduces functions that are needed


def initialize_logfile(main_dir, filename):
    """
    Rounds a number up to the nearest multiple of the value defined by increment
    Input: main_dir - directory that contains the results (tracking, plotting, etc.)
           filename - specified name for logfile
    Output: initialized logfile
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(main_dir + filename)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)-15s %(levelname)-8s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def roundup(x, increment):
    """
    Rounds a number up to the nearest multiple of the value defined by increment
    Input: x - number to round up to the nearest multiple value
           increment - number to use to round up to (ie. if x is 225 and increment
                       is 50, then the output of the function would be 250)
    Output: Rounded value of x
    """
    return int(math.ceil(x / increment)) * increment


# Orders files sequentially
# (ie. 1, 2, 3, 4)
def natural_key(string_):
    '''See http://www.codinghorror.com/blog/archives/001018.html'''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)',
            string_)]


class a:
    """
    Defines a class named a to label the data more easily. For example, this
    class allows data['X'] to be redefined as data.x
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.Area = []
        self.Slice = []
        self.Count = []
        self.CountTemp = []
        self.Cost = []


def convert_class(data_df):
    """
    Changes the way that data is stored and also adds the columns for count,
    tempcount, and cost
    Input: data_df - pandas df containing the particle data
    Output: data - class data
    """
    data = a()
    data.x = data_df['X'].values
    data.y = data_df['Y'].values
    if 'Z' in data_df:
        data.z = data_df['Z'].values
    data.Area = data_df['Area'].values
    data.Slice = data_df['Slice'].values.astype('int')
    data.Count = data_df['Count'].values.astype('int')
    data.CountTemp = data_df['CountTemp'].values.astype('int')
    data.Cost = data_df['Cost'].values.astype('float')

    return data


def load_data(main_dir, trial_name, file_type):
    """
    Loads data from a file into a numpy array to use with the tracking program
    Input: main_dir - directory that contains the particle data
    Output: data - data class containing the results
    """

    if file_type == '.csv':
        data_df = pd.read_csv(main_dir + trial_name + file_type)
    elif file_type == '.txt':
        data_df = pd.read_csv(main_dir + trial_name + file_type,
                              sep=None, engine='python')
    elif file_type == '.pkl':
        data_df = pd.read_pickle(main_dir + trial_name + file_type)
    else:
        raise ValueError(
            'Particle file must be either a csv, txt, or pkl file')

    if 'Area' not in data_df:
        data_df['Area'] = -np.ones(len(data_df))

    data_df['Count'] = np.zeros(len(data_df))
    data_df['CountTemp'] = np.zeros(len(data_df))
    data_df['Cost'] = -np.ones(len(data_df))

    data = convert_class(data_df)

    return data


def get_nframes(filename):
    with h5py.File(filename, 'r') as f:
        return len(f.values())


def load_data_h5(filename, frame_range):
    data = a()
    with h5py.File(filename, 'r') as f:
        nframes = len(frame_range)
        slices = np.empty(nframes, dtype=object)
        x = np.empty(nframes, dtype=object)
        y = np.empty(nframes, dtype=object)
        z = np.empty(nframes, dtype=object)
        has_props = False
        diameter_list = np.empty(nframes, dtype=object)
        intensity_list = np.empty(nframes, dtype=object)
        mass_list = np.empty(nframes, dtype=object)

        for idx, frame_idx in enumerate(frame_range):
            key = f"frame{frame_idx:05d}"
            frame = f[key]
            xyze = np.array(frame['xyze'])
            n_pts = xyze[0].shape[0]
            x[idx] = xyze[0].reshape(-1, 1)
            y[idx] = xyze[1].reshape(-1, 1)
            z[idx] = xyze[2].reshape(-1, 1)
            slices[idx] = np.ones([n_pts, 1], dtype=int) * frame_idx

            if 'diameter' in frame and 'intensity' in frame and 'mass' in frame:
                # (n_pts,) or (n_pts, maxcams)
                d = np.array(frame['diameter'])
                i = np.array(frame['intensity'])
                m = np.array(frame['mass'])
                if d.ndim == 2:
                    d = np.nanmean(d, axis=1)
                    i = np.nanmean(i, axis=1)
                    m = np.nanmean(m, axis=1)
                diameter_list[idx] = d.reshape(-1, 1)
                intensity_list[idx] = i.reshape(-1, 1)
                mass_list[idx] = m.reshape(-1, 1)
                has_props = True
            else:
                diameter_list[idx] = np.full(
                    (n_pts, 1), np.nan, dtype=np.float64)
                intensity_list[idx] = np.full(
                    (n_pts, 1), np.nan, dtype=np.float64)
                mass_list[idx] = np.full((n_pts, 1), np.nan, dtype=np.float64)

        data.x = np.concatenate(np.vstack(x))
        data.y = np.concatenate(np.vstack(y))
        data.z = np.concatenate(np.vstack(z))
        data.Slice = np.concatenate(np.vstack(slices))
        data.Count = np.zeros_like(data.x)
        data.CountTemp = np.zeros_like(data.x)
        data.Cost = -np.ones_like(data.x)
        data.Area = -np.ones_like(data.x)
        if has_props:
            data.diameter = np.concatenate(np.vstack(diameter_list))
            data.intensity = np.concatenate(np.vstack(intensity_list))
            data.mass = np.concatenate(np.vstack(mass_list))

    return data


def chunk_list(seq, workers):
    size = math.ceil(len(seq) / workers)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def create_h5_file(folder):
    filename = os.path.join(folder, 'tracks.h5part')
    with h5py.File(filename, 'w') as f:
        # Create an empty HDF5 file with no datasets or groups
        pass


def merge_tracks_h5part_parts(folder):
    """
    Merge tracks_part_*.h5part files into tracks.h5part.
    Used after parallel tracking so a single file is available for ParaView export.
    """
    import glob
    pattern = os.path.join(folder, 'tracks_part_*.h5part')
    part_files = sorted(glob.glob(pattern))
    if not part_files:
        return
    out_path = os.path.join(folder, 'tracks.h5part')
    with h5py.File(out_path, 'w') as out:
        for part_path in part_files:
            with h5py.File(part_path, 'r') as inc:
                for name in inc.keys():
                    inc.copy(name, out)
    for part_path in part_files:
        try:
            os.remove(part_path)
        except OSError:
            pass
    logging.info("Merged %d part file(s) into %s", len(part_files), out_path)


def _build_tracks_from_data(data: a, frame_range, include_failed_tracks):
    """Build list of (track_id, frames, x, y, z) for each track present in frame_range."""
    frame_range_set = set(frame_range)
    if include_failed_tracks:
        mask = np.isin(data.Slice, frame_range)
    else:
        mask = np.isin(data.Slice, frame_range) & (data.Count != 0)
    ids = data.Count[mask]
    frames = data.Slice[mask]
    xx = data.x[mask]
    yy = data.y[mask]
    zz = data.z[mask]
    tracks = {}
    for i in range(len(ids)):
        tid = int(ids[i])
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append((int(frames[i]), float(
            xx[i]), float(yy[i]), float(zz[i])))
    out = []
    for tid, points in tracks.items():
        points.sort(key=lambda p: p[0])
        frames_arr = np.array([p[0] for p in points])
        x_arr = np.array([p[1] for p in points])
        y_arr = np.array([p[2] for p in points])
        z_arr = np.array([p[3] for p in points])
        out.append((tid, frames_arr, x_arr, y_arr, z_arr))
    return out


def _bspline_velocity_acceleration_lookup(data: a, frame_range, dt, include_failed_tracks):
    """
    For each track with at least 4 points, compute velocity and acceleration at each frame
    using a cubic B-spline fit. Returns dict (track_id, frame) -> (vx, vy, vz, ax, ay, az).
    """
    try:
        from ..track import velocity_acceleration_from_bspline
    except ImportError:
        return {}
    tracks = _build_tracks_from_data(data, frame_range, include_failed_tracks)
    lookup = {}
    for tid, frames_arr, x_arr, y_arr, z_arr in tracks:
        if len(frames_arr) < 4:
            continue
        t = frames_arr.astype(np.float64) * dt
        try:
            vx, vy, vz, ax, ay, az = velocity_acceleration_from_bspline(
                t, x_arr, y_arr, z_arr)
        except Exception:
            continue
        for i, frame in enumerate(frames_arr):
            lookup[(tid, int(frame))] = (
                vx[i], vy[i], vz[i], ax[i], ay[i], az[i])
    return lookup


def _write_bspline_curves_paraview(data: a, frame_range, dt, filepath, include_failed_tracks=False, num_samples=50):
    """
    Write B-spline curves for every 4-frame track to a VTK file viewable in ParaView.
    Each track is rendered as a polyline sampled at num_samples points along the curve.
    """
    try:
        from ..track import sample_bspline_curve
    except ImportError:
        logging.warning(
            "Cannot write B-spline curves: track.sample_bspline_curve not available")
        return
    tracks = _build_tracks_from_data(data, frame_range, include_failed_tracks)
    polylines = []
    for tid, frames_arr, x_arr, y_arr, z_arr in tracks:
        if len(frames_arr) != 4:
            continue
        t = frames_arr.astype(np.float64) * dt
        try:
            _, x_curve, y_curve, z_curve = sample_bspline_curve(
                t, x_arr, y_arr, z_arr, num_samples=num_samples)
        except Exception:
            continue
        polylines.append(np.column_stack([x_curve, y_curve, z_curve]))
    if not polylines:
        return
    points_list = []
    lines_list = []
    pt_offset = 0
    for pts in polylines:
        n_pts = len(pts)
        points_list.append(pts)
        lines_list.append((n_pts, list(range(pt_offset, pt_offset + n_pts))))
        pt_offset += n_pts
    all_points = np.vstack(points_list).astype(np.float64)
    n_points = all_points.shape[0]
    n_lines = len(polylines)
    # VTK legacy format: LINES n_lines (total_ints) then per line: num_pts i0 i1 ...
    line_data = []
    for n_pts, indices in lines_list:
        line_data.append(n_pts)
        line_data.extend(indices)
    with open(filepath, 'w') as vtk:
        vtk.write("# vtk DataFile Version 3.0\n")
        vtk.write("B-spline curves (4-frame tracks)\n")
        vtk.write("ASCII\n")
        vtk.write("DATASET POLYDATA\n")
        vtk.write("POINTS %d float\n" % n_points)
        for i in range(n_points):
            vtk.write("%.6g %.6g %.6g\n" %
                      (all_points[i, 0], all_points[i, 1], all_points[i, 2]))
        vtk.write("LINES %d %d\n" % (n_lines, len(line_data)))
        idx = 0
        while idx < len(line_data):
            n_pts = line_data[idx]
            vtk.write(str(n_pts))
            for j in range(1, n_pts + 1):
                vtk.write(" %d" % line_data[idx + j])
            vtk.write("\n")
            idx += 1 + n_pts
    logging.info("Wrote B-spline curves to %s (%d polylines)",
                 filepath, n_lines)


def write_data_h5(data: a, folder, frame_range, output_h5part=None, dt=1.0, include_failed_tracks=False,
                  use_bspline=False):
    """Write track data to HDF5. If output_h5part is set (parallel mode), write to that file instead of folder/tracks.h5part.
    Velocity (vx, vy, vz) and acceleration (ax, ay, az) are computed from position history and written when dt is provided.
    If use_bspline is True, velocity and acceleration are computed from a cubic B-spline fit (tracks with at least 4 points).
    If include_failed_tracks is True, particles with Count==0 (tracking failed) are also written; their id is 0 and a 'tracked' dataset (bool) is written per step.
    Only particles whose track appears in all 4 frames of the block are given a tracked index (treated as tracked).
    """
    filename = output_h5part if output_h5part else os.path.join(
        folder, 'tracks.h5part')
    frame_range_set = set(frame_range)
    dt = float(dt) if dt is not None else 1.0

    # Only treat a track as tracked (non-zero id counts) if it appears in a full 4 frames
    max_id = int(np.max(data.Count)) if data.Count.size else 0
    id_has_4_frames = np.zeros(max(1, max_id + 1), dtype=bool)
    if max_id > 0:
        in_range = np.isin(data.Slice, frame_range)
        for tid in np.unique(data.Count[data.Count != 0]):
            tid = int(tid)
            n_frames = len(
                np.unique(data.Slice[(data.Count == tid) & in_range]))
            id_has_4_frames[tid] = (n_frames >= 4)

    bspline_lookup = {}
    if use_bspline:
        bspline_lookup = _bspline_velocity_acceleration_lookup(
            data, frame_range, dt, include_failed_tracks
        )
    # B-spline curves VTK is written from perform_tracking after merge so one file contains all 4-frame tracks

    with h5py.File(filename, 'a') as f:
        print(frame_range)
        for frame in frame_range:
            step_name = f"Step#{frame}"
            if step_name in f:
                del f[step_name]
            grp = f.create_group(step_name)

            if include_failed_tracks:
                mask = (data.Slice == frame)
            else:
                # Only include particles whose track appears in 4 frames
                mask = (data.Slice == frame) & (data.Count != 0)
                if max_id > 0:
                    mask = mask & id_has_4_frames[np.asarray(
                        data.Count, dtype=int)]
            n = int(np.sum(mask))
            ids_cur = data.Count[mask]
            x_cur = data.x[mask]
            y_cur = data.y[mask]
            z_cur = data.z[mask]

            grp.create_dataset('x', data=x_cur)
            grp.create_dataset('y', data=y_cur)
            grp.create_dataset('z', data=z_cur)

            # Build prev-frame index once per frame (for finite-difference fallback)
            id_to_idx_prev = None
            x_prev_arr = y_prev_arr = z_prev_arr = None
            id_to_idx_prev_prev = None
            x_prev_prev_arr = y_prev_prev_arr = z_prev_prev_arr = None
            if (frame - 1) in frame_range_set:
                mask_prev = (data.Slice == frame - 1) & (data.Count != 0)
                ids_prev = data.Count[mask_prev]
                id_to_idx_prev = {int(t): idx for idx,
                                  t in enumerate(ids_prev)}
                x_prev_arr = data.x[mask_prev]
                y_prev_arr = data.y[mask_prev]
                z_prev_arr = data.z[mask_prev]
            if (frame - 2) in frame_range_set:
                mask_prev_prev = (data.Slice == frame - 2) & (data.Count != 0)
                ids_prev_prev = data.Count[mask_prev_prev]
                id_to_idx_prev_prev = {
                    int(t): idx for idx, t in enumerate(ids_prev_prev)}
                x_prev_prev_arr = data.x[mask_prev_prev]
                y_prev_prev_arr = data.y[mask_prev_prev]
                z_prev_prev_arr = data.z[mask_prev_prev]

            # Velocity: from B-spline lookup if available, else finite difference
            vx = np.full(n, np.nan, dtype=np.float64)
            vy = np.full(n, np.nan, dtype=np.float64)
            vz = np.full(n, np.nan, dtype=np.float64)
            for i in range(n):
                tid = int(ids_cur[i])
                key = (tid, frame)
                vals = bspline_lookup.get(key, None)
                if vals is not None:
                    vx[i], vy[i], vz[i] = vals[0], vals[1], vals[2]
                elif id_to_idx_prev is not None and tid in id_to_idx_prev:
                    j = id_to_idx_prev[tid]
                    vx[i] = (x_cur[i] - x_prev_arr[j]) / dt
                    vy[i] = (y_cur[i] - y_prev_arr[j]) / dt
                    vz[i] = (z_cur[i] - z_prev_arr[j]) / dt
            grp.create_dataset('vx', data=vx)
            grp.create_dataset('vy', data=vy)
            grp.create_dataset('vz', data=vz)

            # Acceleration: from B-spline lookup if available, else finite difference
            ax = np.full(n, np.nan, dtype=np.float64)
            ay = np.full(n, np.nan, dtype=np.float64)
            az = np.full(n, np.nan, dtype=np.float64)
            dt2 = dt * dt
            for i in range(n):
                tid = int(ids_cur[i])
                key = (tid, frame)
                vals = bspline_lookup.get(key, None)
                if vals is not None:
                    ax[i], ay[i], az[i] = vals[3], vals[4], vals[5]
                elif id_to_idx_prev is not None and id_to_idx_prev_prev is not None and tid in id_to_idx_prev and tid in id_to_idx_prev_prev:
                    j = id_to_idx_prev[tid]
                    k = id_to_idx_prev_prev[tid]
                    ax[i] = (x_cur[i] - 2 * x_prev_arr[j] +
                             x_prev_prev_arr[k]) / dt2
                    ay[i] = (y_cur[i] - 2 * y_prev_arr[j] +
                             y_prev_prev_arr[k]) / dt2
                    az[i] = (z_cur[i] - 2 * z_prev_arr[j] +
                             z_prev_prev_arr[k]) / dt2
            grp.create_dataset('ax', data=ax)
            grp.create_dataset('ay', data=ay)
            grp.create_dataset('az', data=az)

            grp.create_dataset("id", data=ids_cur)
            if include_failed_tracks:
                # 1 = tracked for full 4 frames, 0 = failed or not yet 4-frame
                tid_arr = np.asarray(ids_cur, dtype=int)
                in_range = (tid_arr > 0) & (tid_arr < len(id_has_4_frames))
                tracked = np.where(
                    in_range & id_has_4_frames[tid_arr], 1, 0).astype(np.uint8)
                grp.create_dataset('tracked', data=tracked)
            if hasattr(data, 'diameter'):
                grp.create_dataset('diameter', data=data.diameter[mask])
                grp.create_dataset('intensity', data=data.intensity[mask])
                grp.create_dataset('mass', data=data.mass[mask])
