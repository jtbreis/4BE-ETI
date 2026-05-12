"""
Numba-accelerated 3D track initialization: ``no_previous_tracks_3d``.

``previous_tracks_3d`` is unchanged and remains the NumPy implementation in
``particle_tracking_code`` (complex else-branch; kept for bit-for-bit safety).

Disable Numba with: 4BE_ETI_NUMBA=0

Falls back to the pure NumPy path if ``numba`` is unavailable.
"""
from __future__ import annotations

import os

import numpy as np

from .particle_tracking_code import (
    MAX_CANDIDATES_MESH,
    MAX_TARGETS_MESH,
    no_previous_tracks_3d as _no_previous_tracks_3d_numpy,
    previous_tracks_3d as _previous_tracks_3d_numpy,
)

# Re-export for callers that import 3D symbols from this module
previous_tracks_3d = _previous_tracks_3d_numpy

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

NUMBA_AVAILABLE = njit is not None
USE_NUMBA = NUMBA_AVAILABLE and os.environ.get("4BE_ETI_NUMBA", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

if njit is None:  # pragma: no cover
    no_previous_tracks_3d = _no_previous_tracks_3d_numpy
else:

    @njit(cache=True)
    def _count_max(Count):
        m = 0
        for i in range(Count.shape[0]):
            v = Count[i]
            if v > m:
                m = v
        return m

    @njit(cache=True)
    def _filter_aabb(
        im, x, y, z, x_pred, y_pred, z_pred, box_size, n_pred, out_glob, max_targets
    ):
        if n_pred <= 0:
            return 0
        xlo, xhi = x_pred[0], x_pred[0]
        ylo, yhi = y_pred[0], y_pred[0]
        zlo, zhi = z_pred[0], z_pred[0]
        for k in range(1, n_pred):
            v = x_pred[k]
            if v < xlo:
                xlo = v
            if v > xhi:
                xhi = v
            v = y_pred[k]
            if v < ylo:
                ylo = v
            if v > yhi:
                yhi = v
            v = z_pred[k]
            if v < zlo:
                zlo = v
            if v > zhi:
                zhi = v
        xlo -= box_size
        xhi += box_size
        ylo -= box_size
        yhi += box_size
        zlo -= box_size
        zhi += box_size
        n_out = 0
        for t in range(im.shape[0]):
            idx = im[t]
            xv, yv, zv = x[idx], y[idx], z[idx]
            if (
                xv >= xlo
                and xv <= xhi
                and yv >= ylo
                and yv <= yhi
                and zv >= zlo
                and zv <= zhi
            ):
                if n_out < max_targets:
                    out_glob[n_out] = idx
                    n_out += 1
                else:
                    break
        return n_out

    @njit(cache=True)
    def _no_prev_3d_jit(
        x,
        y,
        z,
        Count,
        CountTemp,
        Cost,
        im0,
        im1,
        im2,
        im3,
        ii,
        box_size,
        bx_lo,
        bx_hi,
        by_lo,
        by_hi,
        bz_lo,
        bz_hi,
        w_ind1,
        w_x2,
        w_y2,
        w_z2,
        w_im2sub,
        w_im3sub,
        w_p2t,
        w_p2k,
        w_x3,
        w_y3,
        w_z3,
        max_candidates,
        max_targets,
    ):
        im0i = im0[ii]
        x0, y0, z0 = x[im0i], y[im0i], z[im0i]
        x_lo = x0 + bx_lo if bx_lo < bx_hi else x0 + bx_hi
        x_hi2 = x0 + bx_hi if bx_lo < bx_hi else x0 + bx_lo
        y_lo = y0 + by_lo if by_lo < by_hi else y0 + by_hi
        y_hi2 = y0 + by_hi if by_lo < by_hi else y0 + by_lo
        z_lo = z0 + bz_lo if bz_lo < bz_hi else z0 + bz_hi
        z_hi2 = z0 + bz_hi if bz_lo < bz_hi else z0 + bz_lo

        n1 = 0
        for j in range(im1.shape[0]):
            g = im1[j]
            xv, yv, zv = x[g], y[g], z[g]
            if (
                xv >= x_lo
                and xv <= x_hi2
                and yv >= y_lo
                and yv <= y_hi2
                and zv >= z_lo
                and zv <= z_hi2
            ):
                if n1 < max_candidates:
                    w_ind1[n1] = j
                    n1 += 1
        if n1 == 0:
            return
        for k in range(n1):
            g1 = im1[w_ind1[k]]
            w_x2[k] = 2.0 * x[g1] - x0
            w_y2[k] = 2.0 * y[g1] - y0
            w_z2[k] = 2.0 * z[g1] - z0

        n2s = _filter_aabb(
            im2, x, y, z, w_x2, w_y2, w_z2, box_size, n1, w_im2sub, max_targets
        )
        n_pair2 = 0
        for t in range(n2s):
            g2 = w_im2sub[t]
            xa, yb, zc = x[g2], y[g2], z[g2]
            for k in range(n1):
                px, py, pz = w_x2[k], w_y2[k], w_z2[k]
                if (
                    xa >= px - box_size
                    and xa <= px + box_size
                    and yb >= py - box_size
                    and yb <= py + box_size
                    and zc >= pz - box_size
                    and zc <= pz + box_size
                ):
                    if n_pair2 < max_candidates:
                        w_p2t[n_pair2] = t
                        w_p2k[n_pair2] = k
                        n_pair2 += 1
        if n_pair2 == 0:
            return
        for p in range(n_pair2):
            g2b = w_im2sub[w_p2t[p]]
            k = w_p2k[p]
            g1b = im1[w_ind1[k]]
            w_x3[p] = 3.0 * x[g2b] - 3.0 * x[g1b] + x0
            w_y3[p] = 3.0 * y[g2b] - 3.0 * y[g1b] + y0
            w_z3[p] = 3.0 * z[g2b] - 3.0 * z[g1b] + z0
        n3s = _filter_aabb(
            im3, x, y, z, w_x3, w_y3, w_z3, box_size, n_pair2, w_im3sub, max_targets
        )
        if n3s == 0:
            return
        min1 = 1.0e300
        for t3 in range(n3s):
            g3 = w_im3sub[t3]
            xa, yb, zc = x[g3], y[g3], z[g3]
            for p in range(n_pair2):
                px, py, pz = w_x3[p], w_y3[p], w_z3[p]
                if (
                    xa >= px - box_size
                    and xa <= px + box_size
                    and yb >= py - box_size
                    and yb <= py + box_size
                    and zc >= pz - box_size
                    and zc <= pz + box_size
                ):
                    c = (
                        (xa - px) * (xa - px)
                        + (yb - py) * (yb - py)
                        + (zc - pz) * (zc - pz)
                    )
                    if c < min1:
                        min1 = c
        n_eq = 0
        best_t3 = 0
        best_p = 0
        for t3 in range(n3s):
            g3 = w_im3sub[t3]
            xa, yb, zc = x[g3], y[g3], z[g3]
            for p in range(n_pair2):
                px, py, pz = w_x3[p], w_y3[p], w_z3[p]
                if (
                    xa >= px - box_size
                    and xa <= px + box_size
                    and yb >= py - box_size
                    and yb <= py + box_size
                    and zc >= pz - box_size
                    and zc <= pz + box_size
                ):
                    c = (
                        (xa - px) * (xa - px)
                        + (yb - py) * (yb - py)
                        + (zc - pz) * (zc - pz)
                    )
                    if c == min1:
                        n_eq += 1
                        best_t3 = t3
                        best_p = p
        if n_eq != 1:
            return
        cst = min1**0.5
        k_star = w_p2k[best_p]
        t2_star = w_p2t[best_p]
        g1_fin = im1[w_ind1[k_star]]
        g2_fin = w_im2sub[t2_star]
        g3_fin = w_im3sub[best_t3]

        if Count[g1_fin] == 0:
            cnew = _count_max(Count) + 1
            Count[im0i] = cnew
            Count[g1_fin] = cnew
            Cost[g1_fin] = cst
            CountTemp[g2_fin] = cnew
        else:
            c_ref = 1.0e300
            found = False
            n_im1 = im1.shape[0]
            for j in range(n_im1):
                if Count[im1[j]] == Count[g1_fin]:
                    found = True
                    v = Cost[im1[j]]
                    if v < c_ref:
                        c_ref = v
            if found and cst < c_ref:
                cnew = _count_max(Count) + 1
                Count[im0i] = cnew
                Count[g1_fin] = cnew
                Cost[g1_fin] = cst
                CountTemp[g2_fin] = cnew

    class _Work3D:
        __slots__ = (
            "w_ind1",
            "w_x2",
            "w_y2",
            "w_z2",
            "w_im2sub",
            "w_im3sub",
            "w_p2t",
            "w_p2k",
            "w_x3",
            "w_y3",
            "w_z3",
        )

        def __init__(self, max_candidates: int, max_targets: int):
            self.w_ind1 = np.zeros(max_candidates, dtype=np.int64)
            self.w_x2 = np.empty(max_candidates, dtype=np.float64)
            self.w_y2 = np.empty(max_candidates, dtype=np.float64)
            self.w_z2 = np.empty(max_candidates, dtype=np.float64)
            self.w_im2sub = np.empty(max_targets, dtype=np.int64)
            self.w_im3sub = np.empty(max_targets, dtype=np.int64)
            self.w_p2t = np.empty(max_candidates, dtype=np.int64)
            self.w_p2k = np.empty(max_candidates, dtype=np.int64)
            self.w_x3 = np.empty(max_candidates, dtype=np.float64)
            self.w_y3 = np.empty(max_candidates, dtype=np.float64)
            self.w_z3 = np.empty(max_candidates, dtype=np.float64)

    _work3d_state = {"mc": 0, "mt": 0, "w": None}

    def _ensure_work3d(max_candidates: int, max_targets: int) -> _Work3D:
        st = _work3d_state
        w = st["w"]
        if w is not None and st["mc"] >= max_candidates and st["mt"] >= max_targets:
            return w
        st["mc"] = max_candidates
        st["mt"] = max_targets
        st["w"] = _Work3D(max_candidates, max_targets)
        return st["w"]

    def _ensure_c_contiguous_arrays(data):
        return (
            np.ascontiguousarray(data.x, dtype=np.float64),
            np.ascontiguousarray(data.y, dtype=np.float64),
            np.ascontiguousarray(data.z, dtype=np.float64),
            np.ascontiguousarray(data.Count, dtype=np.int64),
            np.ascontiguousarray(data.CountTemp, dtype=np.int64),
            np.ascontiguousarray(data.Cost, dtype=np.float64),
        )

    def no_previous_tracks_3d(
        data,
        im,
        ii,
        box_size,
        box_size_initial_x_lo,
        box_size_initial_x_hi,
        box_size_initial_y_lo,
        box_size_initial_y_hi,
        box_size_initial_z_lo,
        box_size_initial_z_hi,
        max_candidates_mesh=None,
        max_targets_mesh=None,
        debug_mesh_match_prints=0,
        _im0=None,
        _im1=None,
        _im2=None,
        _im3=None,
    ):
        _mc = MAX_CANDIDATES_MESH if max_candidates_mesh is None else int(
            max_candidates_mesh)
        _mt = MAX_TARGETS_MESH if max_targets_mesh is None else int(
            max_targets_mesh)
        _use_numpy_for_debug = int(debug_mesh_match_prints or 0) > 0
        if not (
            _im0 is not None
            and _im1 is not None
            and _im2 is not None
            and _im3 is not None
        ):
            return _no_previous_tracks_3d_numpy(
                data,
                im,
                ii,
                box_size,
                box_size_initial_x_lo,
                box_size_initial_x_hi,
                box_size_initial_y_lo,
                box_size_initial_y_hi,
                box_size_initial_z_lo,
                box_size_initial_z_hi,
                max_candidates_mesh,
                max_targets_mesh,
                debug_mesh_match_prints,
                _im0,
                _im1,
                _im2,
                _im3,
            )
        if not USE_NUMBA or _use_numpy_for_debug:
            return _no_previous_tracks_3d_numpy(
                data,
                im,
                ii,
                box_size,
                box_size_initial_x_lo,
                box_size_initial_x_hi,
                box_size_initial_y_lo,
                box_size_initial_y_hi,
                box_size_initial_z_lo,
                box_size_initial_z_hi,
                max_candidates_mesh,
                max_targets_mesh,
                debug_mesh_match_prints,
                _im0,
                _im1,
                _im2,
                _im3,
            )
        w = _ensure_work3d(_mc, _mt)
        im0 = np.ascontiguousarray(_im0, dtype=np.int64)
        im1 = np.ascontiguousarray(_im1, dtype=np.int64)
        im2 = np.ascontiguousarray(_im2, dtype=np.int64)
        im3 = np.ascontiguousarray(_im3, dtype=np.int64)
        x, y, z, Count, CountTemp, Cost = _ensure_c_contiguous_arrays(data)
        _no_prev_3d_jit(
            x,
            y,
            z,
            Count,
            CountTemp,
            Cost,
            im0,
            im1,
            im2,
            im3,
            np.int64(ii),
            float(box_size),
            float(box_size_initial_x_lo),
            float(box_size_initial_x_hi),
            float(box_size_initial_y_lo),
            float(box_size_initial_y_hi),
            float(box_size_initial_z_lo),
            float(box_size_initial_z_hi),
            w.w_ind1,
            w.w_x2,
            w.w_y2,
            w.w_z2,
            w.w_im2sub,
            w.w_im3sub,
            w.w_p2t,
            w.w_p2k,
            w.w_x3,
            w.w_y3,
            w.w_z3,
            np.int64(_mc),
            np.int64(_mt),
        )
        data.x = x
        data.y = y
        data.z = z
        data.Count = Count
        data.CountTemp = CountTemp
        data.Cost = Cost
        return data
