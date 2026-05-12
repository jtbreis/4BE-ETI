import os
import re
import h5py
import numpy as np

from ..utils.basic_utils import physical_time_seconds, position_unit_to_meters_per_unit


def load_tracks_from_h5part(folder, dt, rep_rate, run=0, min_length=2, use_bspline=False, filepath=None):
    """
    Load Track objects from 4BE-ETI output (tracks.h5part with Step#frame groups).

    Used to export tracks to ParaView format from the same folder.
    If use_bspline is True, velocity and acceleration are computed from a cubic B-spline
    fit (for tracks with at least 4 points); otherwise finite differences are used.
    If filepath is given, load from that file; otherwise use folder/tracks.h5part.
    """
    import logging
    from ..track import Track

    filename = filepath if filepath is not None else os.path.join(folder, "tracks.h5part")
    if not os.path.isfile(filename):
        logging.warning("No tracks loaded: %s does not exist.", filename)
        return []
    with h5py.File(filename, "r") as f:
        step_keys = [k for k in f.keys() if k.startswith("Step#")]
        if not step_keys:
            logging.warning(
                "No tracks loaded from %s: file has no Step# groups (empty or corrupted). "
                "If you ran with workers>1, ensure merge of tracks_part_*.h5part ran.",
                filename,
            )
            return []
        # Sort by frame number
        def frame_num(k):
            m = re.match(r"Step#(\d+)", k)
            return int(m.group(1)) if m else -1

        step_keys.sort(key=frame_num)
        has_props = "diameter" in f[step_keys[0]]
        has_tracked_flag = "tracked" in f[step_keys[0]]
        dt_file = float(f.attrs.get("dt_s", dt))
        rep_file = f.attrs.get("rep_rate_hz", None)
        if rep_file is not None:
            rep_file = float(rep_file)
            if not np.isfinite(rep_file) or rep_file <= 0:
                rep_file = None
        tl_file = int(f.attrs.get("track_length", 4))
        scale_to_m = f.attrs.get("position_scale_to_m", None)
        if scale_to_m is None:
            try:
                scale_to_m = position_unit_to_meters_per_unit(
                    str(f.attrs.get("position_units", "mm")))
            except ValueError:
                scale_to_m = 1.0
        else:
            scale_to_m = float(scale_to_m)
        frame_to_time = {}
        # Build per-track: track_id -> list of (frame, x, y, z, ...) sorted by frame
        tracks_raw = {}
        for key in step_keys:
            frame = frame_num(key)
            grp = f[key]
            x = np.asarray(grp["x"]).ravel()
            y = np.asarray(grp["y"]).ravel()
            z = np.asarray(grp["z"]).ravel()
            id_ = np.asarray(grp["id"]).ravel().astype(int)
            tracked = None
            if has_tracked_flag:
                tracked = np.asarray(grp["tracked"]).ravel().astype(np.uint8)
            if "Time" in grp.attrs:
                frame_to_time[frame] = float(grp.attrs["Time"])
            else:
                frame_to_time[frame] = physical_time_seconds(
                    frame, dt_file, rep_file, tl_file)
            if has_props:
                d = np.asarray(grp["diameter"])
                intensity = np.asarray(grp["intensity"])
                m = np.asarray(grp["mass"])
                props_2d = d.ndim == 2
            for i in range(len(id_)):
                tid = id_[i]
                # Failed/untracked detections may use id=0 and can dominate memory if grouped.
                # For track export, only keep valid tracked IDs.
                if tid <= 0:
                    continue
                if tracked is not None and tracked[i] == 0:
                    continue
                if tid not in tracks_raw:
                    tracks_raw[tid] = []
                if has_props:
                    if props_2d:
                        tracks_raw[tid].append((
                            frame, x[i], y[i], z[i],
                            np.asarray(d[i], dtype=np.float64).ravel(),
                            np.asarray(intensity[i], dtype=np.float64).ravel(),
                            np.asarray(m[i], dtype=np.float64).ravel(),
                        ))
                    else:
                        tracks_raw[tid].append((
                            frame, x[i], y[i], z[i],
                            float(d.ravel()[i]),
                            float(intensity.ravel()[i]),
                            float(m.ravel()[i]),
                        ))
                else:
                    tracks_raw[tid].append((frame, x[i], y[i], z[i], np.nan, np.nan, np.nan))
        tracks = []
        for tid, points in tracks_raw.items():
            points.sort(key=lambda p: p[0])
            if has_props:
                frames, X, Y, Z, D, I, M = zip(*points)
                D = np.stack([np.atleast_1d(x) for x in D], axis=0)
                I = np.stack([np.atleast_1d(x) for x in I], axis=0)
                M = np.stack([np.atleast_1d(x) for x in M], axis=0)
            else:
                frames, X, Y, Z = zip(*[(p[0], p[1], p[2], p[3]) for p in points])
                D = I = M = np.full(len(X), np.nan)
            X, Y, Z = np.array(X), np.array(Y), np.array(Z)
            if len(X) < min_length:
                continue
            frames = list(frames)
            time_arr = np.array(
                [frame_to_time[int(fr)] for fr in frames], dtype=np.float64)
            t0 = float(time_arr[0])
            tr = Track(X, Y, Z, dt, tid, t0, run, physical_times=time_arr)
            tr.diameter = D
            tr.intensity = I
            tr.mass = M
            tr.compute_velocity(use_bspline=use_bspline)
            tr.compute_acceleration(use_bspline=use_bspline)
            # Positions are in position_units; derivatives are in those units per second^2.
            # Scale to m/s and m/s^2 for consistency with datasets written by write_data_h5.
            if scale_to_m != 1.0 and np.isfinite(scale_to_m):
                for name in ("vx", "vy", "vz", "ax", "ay", "az"):
                    if hasattr(tr, name):
                        arr = getattr(tr, name)
                        if arr is not None:
                            setattr(tr, name, np.asarray(arr, dtype=np.float64) * scale_to_m)
                if getattr(tr, "v", None) is not None:
                    tr.v = np.asarray(tr.v, dtype=np.float64) * scale_to_m
                    tr.vmag = np.linalg.norm(tr.v, axis=1)
                if getattr(tr, "a", None) is not None:
                    tr.a = np.asarray(tr.a, dtype=np.float64) * scale_to_m
                    tr.amag = np.linalg.norm(tr.a, axis=1)
            tracks.append(tr)
    return tracks


def get_nsamples_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        return len(f.keys())


def read_tracks_from_h5(filename, frame_range):
    with h5py.File(filename, 'r') as f:
        group_name = f"frames{min(frame_range)}-{max(frame_range)}"
        if group_name not in f:
            X = np.zeros([1, 1])
            Y = np.zeros([1, 1])
            Z = np.zeros([1, 1])
            Slice = np.zeros([1, 1])
            Count = np.zeros([1, 1])
            return X, Y, Z, Slice, Count
        group = f[group_name]
        X = np.array(group['X'])
        Y = np.array(group['Y'])
        Z = np.array(group['Z'])
        Slice = np.array(group['Slice'])
        Count = np.array(group['Count'])

        return X, Y, Z, Slice, Count
