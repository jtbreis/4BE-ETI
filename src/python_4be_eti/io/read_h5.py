import os
import re
import h5py
import numpy as np


def load_tracks_from_h5part(folder, dt, rep_rate, run=0, min_length=2):
    """
    Load Track objects from 4BE-ETI output (tracks.h5part with Step#frame groups).

    Used to export tracks to ParaView format from the same folder.
    """
    import logging
    from ..track import Track

    filename = os.path.join(folder, "tracks.h5part")
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
        # Build per-track: track_id -> list of (frame, x, y, z, ...) sorted by frame
        tracks_raw = {}
        for key in step_keys:
            frame = frame_num(key)
            grp = f[key]
            x = np.asarray(grp["x"]).ravel()
            y = np.asarray(grp["y"]).ravel()
            z = np.asarray(grp["z"]).ravel()
            id_ = np.asarray(grp["id"]).ravel().astype(int)
            if has_props:
                d = np.asarray(grp["diameter"]).ravel()
                intensity = np.asarray(grp["intensity"]).ravel()
                m = np.asarray(grp["mass"]).ravel()
            for i in range(len(id_)):
                tid = id_[i]
                if tid not in tracks_raw:
                    tracks_raw[tid] = []
                if has_props:
                    tracks_raw[tid].append((frame, x[i], y[i], z[i], d[i], intensity[i], m[i]))
                else:
                    tracks_raw[tid].append((frame, x[i], y[i], z[i], np.nan, np.nan, np.nan))
        tracks = []
        for tid, points in tracks_raw.items():
            points.sort(key=lambda p: p[0])
            if has_props:
                frames, X, Y, Z, D, I, M = zip(*points)
                D, I, M = np.array(D), np.array(I), np.array(M)
            else:
                frames, X, Y, Z = zip(*[(p[0], p[1], p[2], p[3]) for p in points])
                D = I = M = np.full(len(X), np.nan)
            X, Y, Z = np.array(X), np.array(Y), np.array(Z)
            if len(X) < min_length:
                continue
            t0 = frames[0] * dt  # or (frames[0] // 4) / rep_rate for sample-based time
            tr = Track(X, Y, Z, dt, tid, t0, run)
            tr.diameter = D
            tr.intensity = I
            tr.mass = M
            tr.compute_velocity()
            tr.compute_acceleration()
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
