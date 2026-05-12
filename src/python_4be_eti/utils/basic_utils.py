import logging
import math
import os
import re

import numpy as np
import pandas as pd
import h5py

# This file introduces functions that are needed


def position_unit_to_meters_per_unit(unit):
    """Meters per one unit of position (e.g. mm -> 1e-3). Used to convert v/a to SI."""
    if unit is None:
        return 1.0
    u = str(unit).strip().lower()
    mapping = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "μm": 1e-6,
        "micron": 1e-6,
        "micrometer": 1e-6,
        "nm": 1e-9,
    }
    if u not in mapping:
        raise ValueError(
            "Unknown position unit %r; use one of: %s"
            % (unit, sorted(set(mapping)))
        )
    return mapping[u]


def physical_time_seconds(frame, dt, rep_rate_hz=None, track_length=4):
    """
    Physical time in seconds for global stereo frame index ``frame``.

    If ``rep_rate_hz`` is set and positive, each 4-frame block (length ``track_length``)
    is assumed to start every 1/rep_rate seconds, with subframes spaced by ``dt``:

        t = (frame // L) / rep_rate + (frame % L) * dt

    Otherwise ``t = frame * dt`` (uniform frame clock).
    """
    dt = float(dt)
    frame = int(frame)
    L = int(track_length)
    if L <= 0:
        L = 4
    if rep_rate_hz is None or rep_rate_hz <= 0:
        return float(frame) * dt
    return float(frame // L) / float(rep_rate_hz) + float(frame % L) * dt


def _acceleration_three_point(x0, x1, x2, t0, t1, t2):
    """Approximate d^2 x / dt^2 at t1 for non-uniform times t0 < t1 < t2."""
    dt01 = t1 - t0
    dt12 = t2 - t1
    if dt01 <= 0 or dt12 <= 0:
        return np.nan
    v01 = (x1 - x0) / dt01
    v12 = (x2 - x1) / dt12
    t02 = t2 - t0
    if t02 <= 0:
        return np.nan
    return 2.0 * (v12 - v01) / t02


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
                # (n_pts,) or (n_pts, maxcams) per-ray values from stereomatching
                d = np.asarray(frame['diameter'], dtype=np.float64)
                i = np.asarray(frame['intensity'], dtype=np.float64)
                m = np.asarray(frame['mass'], dtype=np.float64)
                if d.ndim == 1:
                    d = d.reshape(-1, 1)
                    i = i.reshape(-1, 1)
                    m = m.reshape(-1, 1)
                diameter_list[idx] = d
                intensity_list[idx] = i
                mass_list[idx] = m
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
            data.diameter = np.vstack([diameter_list[i] for i in range(nframes)])
            data.intensity = np.vstack([intensity_list[i] for i in range(nframes)])
            data.mass = np.vstack([mass_list[i] for i in range(nframes)])

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


def _bspline_velocity_acceleration_lookup(
    data: a, frame_range, dt, include_failed_tracks,
    rep_rate_hz=None, track_length=4,
):
    """
    For each track with at least 4 points, compute velocity and acceleration at each frame
    using a cubic B-spline fit. Returns dict (track_id, frame) -> (vx, vy, vz, ax, ay, az).
    Derivatives are w.r.t. physical time (see physical_time_seconds).
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
        t = np.array(
            [
                physical_time_seconds(
                    int(f), dt, rep_rate_hz, track_length)
                for f in frames_arr
            ],
            dtype=np.float64,
        )
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
        t = np.array(
            [
                physical_time_seconds(int(f), dt, None, 4)
                for f in frames_arr
            ],
            dtype=np.float64,
        )
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


def write_data_h5(
    data: a,
    folder,
    frame_range,
    output_h5part=None,
    dt=1.0,
    include_failed_tracks=False,
    use_bspline=False,
    rep_rate_hz=None,
    track_length=4,
    position_scale_to_m=1e-3,
    position_units_str="mm",
):
    """Write track data to HDF5. If output_h5part is set (parallel mode), write to that file instead of folder/tracks.h5part.

    Physical time (seconds) for each step is stored as group attribute ``Time``:
    ``(frame // track_length) / rep_rate_hz + (frame % track_length) * dt`` when
    ``rep_rate_hz`` is set and positive; otherwise ``frame * dt``.

    Positions (x, y, z) are written in the user's length unit (``position_units_str``).
    Velocities and accelerations are converted to SI (m/s and m/s^2) using
    ``position_scale_to_m`` (meters per unit of position, e.g. 1e-3 for mm).

    If use_bspline is True, velocity and acceleration use a cubic B-spline (tracks with at least 4 points).
    If include_failed_tracks is True, particles with Count==0 are also written with id 0 and optional 'tracked' flag.
    """
    filename = output_h5part if output_h5part else os.path.join(
        folder, 'tracks.h5part')
    frame_range_set = set(frame_range)
    dt = float(dt) if dt is not None else 1.0
    rep_rate_hz = float(rep_rate_hz) if rep_rate_hz is not None else None
    track_length = int(track_length) if track_length is not None else 4
    if track_length <= 0:
        track_length = 4
    position_scale_to_m = float(position_scale_to_m)

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
            data, frame_range, dt, include_failed_tracks,
            rep_rate_hz=rep_rate_hz, track_length=track_length,
        )

    with h5py.File(filename, 'a') as f:
        f.attrs["dt_s"] = dt
        if rep_rate_hz is not None and rep_rate_hz > 0:
            f.attrs["rep_rate_hz"] = rep_rate_hz
        f.attrs["track_length"] = track_length
        f.attrs["position_units"] = str(position_units_str)
        f.attrs["position_scale_to_m"] = position_scale_to_m
        f.attrs["time_units"] = "s"
        f.attrs["velocity_units"] = "m/s"
        f.attrs["acceleration_units"] = "m/s^2"

        print(frame_range)
        for frame in frame_range:
            step_name = f"Step#{frame}"
            if step_name in f:
                del f[step_name]
            grp = f.create_group(step_name)
            t_phys = physical_time_seconds(
                frame, dt, rep_rate_hz, track_length)
            grp.attrs["Time"] = float(t_phys)
            grp.attrs["time_units"] = "s"
            grp.attrs["position_units"] = str(position_units_str)

            if include_failed_tracks:
                mask = (data.Slice == frame)
            else:
                mask = (data.Slice == frame) & (data.Count != 0)
                if max_id > 0:
                    mask = mask & id_has_4_frames[np.asarray(
                        data.Count, dtype=int)]
            row_mask = np.asarray(mask).ravel()
            n = int(np.sum(row_mask))
            ids_cur = np.asarray(data.Count).ravel()[row_mask]
            x_cur = np.asarray(data.x).ravel()[row_mask]
            y_cur = np.asarray(data.y).ravel()[row_mask]
            z_cur = np.asarray(data.z).ravel()[row_mask]

            dsx = grp.create_dataset('x', data=x_cur)
            dsy = grp.create_dataset('y', data=y_cur)
            dsz = grp.create_dataset('z', data=z_cur)
            dsx.attrs["units"] = str(position_units_str)
            dsy.attrs["units"] = str(position_units_str)
            dsz.attrs["units"] = str(position_units_str)

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

            t_cur = physical_time_seconds(
                frame, dt, rep_rate_hz, track_length)
            t_m1 = physical_time_seconds(
                frame - 1, dt, rep_rate_hz, track_length)
            t_m2 = physical_time_seconds(
                frame - 2, dt, rep_rate_hz, track_length)

            vx = np.full(n, np.nan, dtype=np.float64)
            vy = np.full(n, np.nan, dtype=np.float64)
            vz = np.full(n, np.nan, dtype=np.float64)
            for i in range(n):
                tid = int(ids_cur[i])
                key = (tid, frame)
                vals = bspline_lookup.get(key, None)
                if vals is not None:
                    vx[i] = vals[0] * position_scale_to_m
                    vy[i] = vals[1] * position_scale_to_m
                    vz[i] = vals[2] * position_scale_to_m
                elif id_to_idx_prev is not None and tid in id_to_idx_prev:
                    j = id_to_idx_prev[tid]
                    dt_step = t_cur - t_m1
                    if dt_step > 0:
                        vx[i] = (
                            x_cur[i] - x_prev_arr[j]) / dt_step * position_scale_to_m
                        vy[i] = (
                            y_cur[i] - y_prev_arr[j]) / dt_step * position_scale_to_m
                        vz[i] = (
                            z_cur[i] - z_prev_arr[j]) / dt_step * position_scale_to_m
            dvx = grp.create_dataset('vx', data=vx)
            dvy = grp.create_dataset('vy', data=vy)
            dvz = grp.create_dataset('vz', data=vz)
            dvx.attrs["units"] = "m/s"
            dvy.attrs["units"] = "m/s"
            dvz.attrs["units"] = "m/s"

            ax = np.full(n, np.nan, dtype=np.float64)
            ay = np.full(n, np.nan, dtype=np.float64)
            az = np.full(n, np.nan, dtype=np.float64)
            for i in range(n):
                tid = int(ids_cur[i])
                key = (tid, frame)
                vals = bspline_lookup.get(key, None)
                if vals is not None:
                    ax[i] = vals[3] * position_scale_to_m
                    ay[i] = vals[4] * position_scale_to_m
                    az[i] = vals[5] * position_scale_to_m
                elif id_to_idx_prev is not None and id_to_idx_prev_prev is not None and tid in id_to_idx_prev and tid in id_to_idx_prev_prev:
                    j = id_to_idx_prev[tid]
                    k = id_to_idx_prev_prev[tid]
                    ax[i] = _acceleration_three_point(
                        x_prev_prev_arr[k], x_prev_arr[j], x_cur[i],
                        t_m2, t_m1, t_cur) * position_scale_to_m
                    ay[i] = _acceleration_three_point(
                        y_prev_prev_arr[k], y_prev_arr[j], y_cur[i],
                        t_m2, t_m1, t_cur) * position_scale_to_m
                    az[i] = _acceleration_three_point(
                        z_prev_prev_arr[k], z_prev_arr[j], z_cur[i],
                        t_m2, t_m1, t_cur) * position_scale_to_m
            dax = grp.create_dataset('ax', data=ax)
            day = grp.create_dataset('ay', data=ay)
            daz = grp.create_dataset('az', data=az)
            dax.attrs["units"] = "m/s^2"
            day.attrs["units"] = "m/s^2"
            daz.attrs["units"] = "m/s^2"

            grp.create_dataset("id", data=ids_cur)
            if include_failed_tracks:
                tid_arr = np.asarray(ids_cur, dtype=int)
                in_range = (tid_arr > 0) & (tid_arr < len(id_has_4_frames))
                tracked = np.where(
                    in_range & id_has_4_frames[tid_arr], 1, 0).astype(np.uint8)
                grp.create_dataset('tracked', data=tracked)
            if hasattr(data, 'diameter'):
                d_all = np.asarray(data.diameter, dtype=np.float64)
                i_all = np.asarray(data.intensity, dtype=np.float64)
                m_all = np.asarray(data.mass, dtype=np.float64)
                d = d_all[row_mask] if d_all.ndim == 2 else d_all.ravel()[row_mask]
                i = i_all[row_mask] if i_all.ndim == 2 else i_all.ravel()[row_mask]
                m = m_all[row_mask] if m_all.ndim == 2 else m_all.ravel()[row_mask]
                grp.create_dataset('diameter', data=d)
                grp.create_dataset('intensity', data=i)
                grp.create_dataset('mass', data=m)
                # Also expose per-ray components as separate datasets in h5part.
                if i.ndim == 1:
                    grp.create_dataset('intensity_0', data=i)
                elif i.ndim == 2:
                    for comp in range(i.shape[1]):
                        grp.create_dataset(f'intensity_{comp}', data=i[:, comp])
                if m.ndim == 1:
                    grp.create_dataset('mass_0', data=m)
                elif m.ndim == 2:
                    for comp in range(m.shape[1]):
                        grp.create_dataset(f'mass_{comp}', data=m[:, comp])
