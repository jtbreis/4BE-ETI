"""
Export completed particle tracks to HDF5 + XDMF for visualization in ParaView.

Produces:
- <basename>.h5  : HDF5 file with Points, Connectivity, and optional point/cell attributes
- <basename>.xmf : XDMF descriptor so ParaView can open the HDF5 as polylines

When per_snapshot=True, one grid per 4-frame track is written in a Temporal collection,
so you can view each track individually in ParaView by stepping through the time slider.
"""

import logging
import os
import numpy as np
import h5py


# XDMF Mixed topology: 2 = POLYLINE
_POLYLINE_TYPE = 2


def write_bspline_curves_vtk(tracks, filepath, num_samples=50):
    """
    Write B-spline curves for 4-frame tracks to a VTK file viewable in ParaView.

    Parameters
    ----------
    tracks : sequence of Track
        Track objects; only those with track_length == 4 are used.
    filepath : str
        Output path (e.g. 'folder/bspline_curves.vtk').
    num_samples : int
        Number of points to sample along each B-spline curve.
    """
    from ..track import sample_bspline_curve

    tracks_4 = [t for t in tracks if len(t.X) == 4]
    if not tracks_4:
        logging.warning("No 4-frame tracks for B-spline VTK export")
        return

    polylines = []
    for tr in tracks_4:
        t = _track_time_array(tr)
        try:
            _, x_curve, y_curve, z_curve = sample_bspline_curve(
                t, np.asarray(tr.X), np.asarray(tr.Y), np.asarray(tr.Z),
                num_samples=num_samples,
            )
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
            vtk.write("%.6g %.6g %.6g\n" % (all_points[i, 0], all_points[i, 1], all_points[i, 2]))
        vtk.write("LINES %d %d\n" % (n_lines, len(line_data)))
        idx = 0
        while idx < len(line_data):
            n_pts = line_data[idx]
            vtk.write(str(n_pts))
            for j in range(1, n_pts + 1):
                vtk.write(" %d" % line_data[idx + j])
            vtk.write("\n")
            idx += 1 + n_pts
    logging.info("Wrote B-spline curves to %s (%d polylines)", filepath, n_lines)


def _track_time_array(tr):
    """Physical time (s) at each trajectory point; uses ``physical_times`` if set."""
    n = len(tr.X)
    pt = getattr(tr, "physical_times", None)
    if pt is not None and len(pt) == n:
        return np.asarray(pt, dtype=np.float64)
    t0 = getattr(tr, "time", 0.0)
    dt = getattr(tr, "dt", 0.0)
    return (t0 + np.arange(n) * dt).astype(np.float64)


def _track_attr_scalar_per_point(tr, name, n):
    """One scalar per trajectory point for ParaView: mean over cameras if stored as (n, n_cams)."""
    a = getattr(tr, name, None)
    if a is None:
        return np.full(n, np.nan, dtype=np.float64)
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == n:
        return np.nanmean(arr, axis=1)
    arr = arr.ravel()
    if arr.size == n:
        return arr
    return np.full(n, np.nan, dtype=np.float64)


def _track_attr_components_per_point(tr, name, n):
    """Per-ray components per trajectory point as shape (n, n_components)."""
    a = getattr(tr, name, None)
    if a is None:
        return np.full((n, 0), np.nan, dtype=np.float64)
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == n:
        return arr
    arr = arr.ravel()
    if arr.size == n:
        return arr.reshape(n, 1)
    return np.full((n, 0), np.nan, dtype=np.float64)


def _single_track_geometry(tr):
    """Points, connectivity, and point attributes for one track (one snapshot)."""
    n = len(tr.X)
    if n < 2:
        return None
    points = np.column_stack(
        [np.asarray(tr.X), np.asarray(tr.Y), np.asarray(tr.Z)]
    ).astype(np.float64)
    conn = np.array(
        [_POLYLINE_TYPE, n] + list(range(n)),
        dtype=np.int32,
    )
    track_id = np.full(n, getattr(tr, "idx", 0), dtype=np.int32)
    time_arr = _track_time_array(tr)
    if hasattr(tr, "vmag") and tr.vmag is not None:
        if len(tr.vmag) == n:
            v_at_pts = np.asarray(tr.vmag, dtype=np.float64)  # B-spline: one value per point
        elif len(tr.vmag) == n - 1:
            v_at_pts = np.empty(n, dtype=np.float64)
            v_at_pts[0] = tr.vmag[0]
            v_at_pts[-1] = tr.vmag[-1]
            if n > 2:
                v_at_pts[1:-1] = 0.5 * (tr.vmag[:-1] + tr.vmag[1:])
        else:
            v_at_pts = np.full(n, np.nan, dtype=np.float64)
    else:
        v_at_pts = np.full(n, np.nan, dtype=np.float64)
    diameter_pts = _track_attr_scalar_per_point(tr, "diameter", n)
    intensity_pts = _track_attr_scalar_per_point(tr, "intensity", n)
    mass_pts = _track_attr_scalar_per_point(tr, "mass", n)
    return points, conn, track_id, time_arr, v_at_pts, diameter_pts, intensity_pts, mass_pts


def _tracks_to_geometry(tracks):
    """Build global points array and mixed polyline connectivity from Track list."""
    points_list = []
    connectivity_list = []
    track_id_list = []
    time_list = []
    vmag_at_points_list = []
    diameter_list = []
    intensity_list = []
    mass_list = []

    pt_offset = 0
    for tr in tracks:
        n = len(tr.X)
        if n < 2:
            continue
        # Points for this track
        pts = np.column_stack(
            [np.asarray(tr.X), np.asarray(tr.Y), np.asarray(tr.Z)])
        points_list.append(pts)
        # Mixed polyline: type=2, num_nodes, node0, node1, ...
        conn = [_POLYLINE_TYPE, n] + list(range(pt_offset, pt_offset + n))
        connectivity_list.append(conn)
        pt_offset += n
        # Point attributes
        track_id_list.append(np.full(n, getattr(tr, 'idx', 0)))
        time_list.append(_track_time_array(tr))
        # Velocity magnitude at points (vmag length n = B-spline, n-1 = finite diff)
        if hasattr(tr, 'vmag') and tr.vmag is not None:
            if len(tr.vmag) == n:
                vmag_at_points_list.append(np.asarray(tr.vmag, dtype=np.float64))
            elif len(tr.vmag) == n - 1:
                v_at_pts = np.empty(n)
                v_at_pts[0] = tr.vmag[0]
                v_at_pts[-1] = tr.vmag[-1]
                if n > 2:
                    v_at_pts[1:-1] = 0.5 * (tr.vmag[:-1] + tr.vmag[1:])
                vmag_at_points_list.append(v_at_pts)
            else:
                vmag_at_points_list.append(np.full(n, np.nan))
        else:
            vmag_at_points_list.append(np.full(n, np.nan))
        diameter_list.append(_track_attr_scalar_per_point(tr, 'diameter', n))
        intensity_list.append(_track_attr_scalar_per_point(tr, 'intensity', n))
        mass_list.append(_track_attr_scalar_per_point(tr, 'mass', n))

    if not points_list:
        return None, None, None, None, None, None, None, None

    points = np.vstack(points_list).astype(np.float64)
    track_id = np.concatenate(track_id_list).astype(np.int32)
    time_arr = np.concatenate(time_list).astype(np.float64)
    vmag_at_points = np.concatenate(vmag_at_points_list).astype(np.float64)
    diameter_pts = np.concatenate(diameter_list).astype(np.float64)
    intensity_pts = np.concatenate(intensity_list).astype(np.float64)
    mass_pts = np.concatenate(mass_list).astype(np.float64)
    # Mixed connectivity: flat list of ints
    connectivity = np.array(
        [x for c in connectivity_list for x in c], dtype=np.int32)

    return points, connectivity, track_id, time_arr, vmag_at_points, diameter_pts, intensity_pts, mass_pts


def _write_paraview_per_snapshot(
    tracks, base, h5_path, xmf_path, h5_name, include_velocity_magnitude
):
    """
    Write a single UnstructuredGrid with time-varying geometry (one snapshot per timestep).
    Produces vtkUnstructuredGrid so ParaView does not show MultiBlock warnings.
    Use the time slider to view each 4-frame track individually.
    """
    valid = [
        (i, tr, _single_track_geometry(tr))
        for i, tr in enumerate(tracks)
        if len(tr.X) >= 2
    ]
    if not valid:
        raise ValueError(
            "No valid tracks (each track must have at least 2 points)")

    n_snapshots = len(valid)
    n_pts_max = max(geom[0].shape[0] for _, _, geom in valid)
    max_intensity_components = max(
        _track_attr_components_per_point(tr, "intensity", len(tr.X)).shape[1]
        for _, tr, _ in valid
    )
    max_mass_components = max(
        _track_attr_components_per_point(tr, "mass", len(tr.X)).shape[1]
        for _, tr, _ in valid
    )

    # Pad all to same number of points (4 for 4-frame tracks)
    points_all = np.full((n_snapshots, n_pts_max, 3), np.nan, dtype=np.float64)
    track_id_all = np.zeros((n_snapshots, n_pts_max), dtype=np.int32)
    time_all = np.zeros((n_snapshots, n_pts_max), dtype=np.float64)
    vmag_all = np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
    diameter_all = np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
    intensity_all = np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
    mass_all = np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
    intensity_components_all = [
        np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
        for _ in range(max_intensity_components)
    ]
    mass_components_all = [
        np.full((n_snapshots, n_pts_max), np.nan, dtype=np.float64)
        for _ in range(max_mass_components)
    ]
    time_values = []

    for snap_idx, (tr_idx, tr, geom) in enumerate(valid):
        points, conn, track_id, time_arr, v_at_pts, d_pts, i_pts, m_pts = geom
        n = points.shape[0]
        points_all[snap_idx, :n, :] = points
        track_id_all[snap_idx, :n] = track_id
        time_all[snap_idx, :n] = time_arr
        if include_velocity_magnitude:
            vmag_all[snap_idx, :n] = v_at_pts
        diameter_all[snap_idx, :n] = d_pts
        intensity_all[snap_idx, :n] = i_pts
        mass_all[snap_idx, :n] = m_pts
        i_comp = _track_attr_components_per_point(tr, "intensity", n)
        for c in range(max_intensity_components):
            if c < i_comp.shape[1]:
                intensity_components_all[c][snap_idx, :n] = i_comp[:, c]
        m_comp = _track_attr_components_per_point(tr, "mass", n)
        for c in range(max_mass_components):
            if c < m_comp.shape[1]:
                mass_components_all[c][snap_idx, :n] = m_comp[:, c]
        time_values.append(float(time_arr[0]))

    # Static connectivity: one polyline, n_pts_max nodes (same for every timestep)
    conn = np.array(
        [_POLYLINE_TYPE, n_pts_max] + list(range(n_pts_max)),
        dtype=np.int32,
    )

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Points", data=points_all)
        f.create_dataset("Connectivity", data=conn)
        f.create_dataset("TimeValues", data=np.array(time_values, dtype=np.float64))
        f.create_dataset("TrackId", data=track_id_all)
        f.create_dataset("Time", data=time_all)
        f.create_dataset("Diameter", data=diameter_all)
        f.create_dataset("Intensity", data=intensity_all)
        f.create_dataset("Mass", data=mass_all)
        for c in range(max_intensity_components):
            f.create_dataset(
                f"Intensity_{c}", data=intensity_components_all[c]
            )
        for c in range(max_mass_components):
            f.create_dataset(
                f"Mass_{c}", data=mass_components_all[c]
            )
        if include_velocity_magnitude:
            f.create_dataset("VelocityMagnitude", data=vmag_all)

    h5_ref = h5_name
    # Single Grid with time-varying Geometry -> vtkUnstructuredGrid
    xdmf_lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        '  <Domain>',
        '    <Grid Name="Tracks" GridType="Uniform">',
        '      <Time TimeType="List">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">%s:/TimeValues</DataItem>' % (n_snapshots, h5_ref),
        '      </Time>',
        '      <Topology TopologyType="Mixed" NumberOfElements="1">',
        '        <DataItem Format="HDF" DataType="Int" Dimensions="%d">%s:/Connectivity</DataItem>' % (conn.size, h5_ref),
        '      </Topology>',
        '      <Geometry GeometryType="XYZ">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d 3">%s:/Points</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Geometry>',
        '      <Attribute Name="TrackId" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" DataType="Int" Dimensions="%d %d">%s:/TrackId</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Attribute>',
        '      <Attribute Name="Time" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Time</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Attribute>',
        '      <Attribute Name="Diameter" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Diameter</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Attribute>',
        '      <Attribute Name="Intensity" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Intensity</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Attribute>',
        '      <Attribute Name="Mass" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Mass</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
        '      </Attribute>',
    ]
    for c in range(max_intensity_components):
        xdmf_lines.extend([
            '      <Attribute Name="Intensity_%d" Type="Scalar" Center="Node">' % c,
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Intensity_%d</DataItem>' % (n_snapshots, n_pts_max, h5_ref, c),
            '      </Attribute>',
        ])
    for c in range(max_mass_components):
        xdmf_lines.extend([
            '      <Attribute Name="Mass_%d" Type="Scalar" Center="Node">' % c,
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/Mass_%d</DataItem>' % (n_snapshots, n_pts_max, h5_ref, c),
            '      </Attribute>',
        ])
    if include_velocity_magnitude:
        xdmf_lines.extend([
            '      <Attribute Name="VelocityMagnitude" Type="Scalar" Center="Node">',
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d %d">%s:/VelocityMagnitude</DataItem>' % (n_snapshots, n_pts_max, h5_ref),
            '      </Attribute>',
        ])
    xdmf_lines.extend([
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>',
    ])

    with open(xmf_path, "w") as f:
        f.write("\n".join(xdmf_lines))


def write_tracks_paraview(
    tracks,
    filepath,
    include_velocity_magnitude=True,
    per_snapshot=False,
):
    """
    Write a list of Track objects to HDF5 + XDMF for ParaView.

    Parameters
    ----------
    tracks : sequence of Track
        Completed particle tracks (e.g. from load_tracks).
    filepath : str
        Output path. Use a base path without extension (e.g. 'output/tracks').
        Creates filepath.h5 and filepath.xmf. If filepath ends with .h5 or .xmf,
        the extension is stripped for the base name.
    include_velocity_magnitude : bool
        If True, write point data 'VelocityMagnitude' (from track.vmag).
    per_snapshot : bool
        If True, write one grid per track in a Temporal collection so each
        4-frame track can be viewed individually in ParaView (use time slider).
        If False, all tracks are in a single grid.
    """
    base = filepath
    for ext in (".h5", ".xmf", ".hdf5"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    h5_path = base + ".h5"
    xmf_path = base + ".xmf"
    h5_name = os.path.basename(h5_path)

    if per_snapshot:
        _write_paraview_per_snapshot(
            tracks, base, h5_path, xmf_path, h5_name, include_velocity_magnitude
        )
        return h5_path, xmf_path

    geom = _tracks_to_geometry(tracks)
    if geom[0] is None:
        raise ValueError(
            "No valid tracks (each track must have at least 2 points)")

    points, connectivity, track_id, time_arr, vmag_at_points, diameter_pts, intensity_pts, mass_pts = geom
    n_points = points.shape[0]
    valid_tracks = [t for t in tracks if len(t.X) >= 2]
    n_cells = len(valid_tracks)
    max_intensity_components = max(
        (_track_attr_components_per_point(t, "intensity", len(t.X)).shape[1]
         for t in valid_tracks),
        default=0,
    )
    max_mass_components = max(
        (_track_attr_components_per_point(t, "mass", len(t.X)).shape[1]
         for t in valid_tracks),
        default=0,
    )
    intensity_component_pts = []
    for c in range(max_intensity_components):
        parts = []
        for tr in valid_tracks:
            n = len(tr.X)
            arr = _track_attr_components_per_point(tr, "intensity", n)
            if c < arr.shape[1]:
                parts.append(arr[:, c])
            else:
                parts.append(np.full(n, np.nan, dtype=np.float64))
        intensity_component_pts.append(np.concatenate(parts).astype(np.float64))
    mass_component_pts = []
    for c in range(max_mass_components):
        parts = []
        for tr in valid_tracks:
            n = len(tr.X)
            arr = _track_attr_components_per_point(tr, "mass", n)
            if c < arr.shape[1]:
                parts.append(arr[:, c])
            else:
                parts.append(np.full(n, np.nan, dtype=np.float64))
        mass_component_pts.append(np.concatenate(parts).astype(np.float64))

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Points", data=points)
        f.create_dataset("Connectivity", data=connectivity)
        f.create_dataset("TrackId", data=track_id)
        f.create_dataset("Time", data=time_arr)
        f.create_dataset("Diameter", data=diameter_pts)
        f.create_dataset("Intensity", data=intensity_pts)
        f.create_dataset("Mass", data=mass_pts)
        for c in range(max_intensity_components):
            f.create_dataset(
                f"Intensity_{c}", data=intensity_component_pts[c]
            )
        for c in range(max_mass_components):
            f.create_dataset(
                f"Mass_{c}", data=mass_component_pts[c]
            )
        if include_velocity_magnitude and np.any(np.isfinite(vmag_at_points)):
            f.create_dataset("VelocityMagnitude", data=vmag_at_points)

    h5_ref = h5_name

    xdmf_lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        '  <Domain>',
        '    <Grid Name="ParticleTracks" GridType="Uniform">',
        '      <Topology TopologyType="Mixed" NumberOfElements="%d">' % n_cells,
        '        <DataItem Format="HDF" DataType="Int" Dimensions="%d">' % connectivity.size,
        '          %s:/Connectivity' % h5_ref,
        '        </DataItem>',
        '      </Topology>',
        '      <Geometry GeometryType="XYZ">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d 3">' % n_points,
        '          %s:/Points' % h5_ref,
        '        </DataItem>',
        '      </Geometry>',
        '      <Attribute Name="TrackId" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" DataType="Int" Dimensions="%d">' % n_points,
        '          %s:/TrackId' % h5_ref,
        '        </DataItem>',
        '      </Attribute>',
        '      <Attribute Name="Time" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
        '          %s:/Time' % h5_ref,
        '        </DataItem>',
        '      </Attribute>',
        '      <Attribute Name="Diameter" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
        '          %s:/Diameter' % h5_ref,
        '        </DataItem>',
        '      </Attribute>',
        '      <Attribute Name="Intensity" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
        '          %s:/Intensity' % h5_ref,
        '        </DataItem>',
        '      </Attribute>',
        '      <Attribute Name="Mass" Type="Scalar" Center="Node">',
        '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
        '          %s:/Mass' % h5_ref,
        '        </DataItem>',
        '      </Attribute>',
    ]
    for c in range(max_intensity_components):
        xdmf_lines.extend([
            '      <Attribute Name="Intensity_%d" Type="Scalar" Center="Node">' % c,
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
            '          %s:/Intensity_%d' % (h5_ref, c),
            '        </DataItem>',
            '      </Attribute>',
        ])
    for c in range(max_mass_components):
        xdmf_lines.extend([
            '      <Attribute Name="Mass_%d" Type="Scalar" Center="Node">' % c,
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
            '          %s:/Mass_%d' % (h5_ref, c),
            '        </DataItem>',
            '      </Attribute>',
        ])
    if include_velocity_magnitude and np.any(np.isfinite(vmag_at_points)):
        xdmf_lines.extend([
            '      <Attribute Name="VelocityMagnitude" Type="Scalar" Center="Node">',
            '        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d">' % n_points,
            '          %s:/VelocityMagnitude' % h5_ref,
            '        </DataItem>',
            '      </Attribute>',
        ])
    xdmf_lines.extend([
        "    </Grid>",
        "  </Domain>",
        "</Xdmf>",
    ])

    with open(xmf_path, "w") as f:
        f.write("\n".join(xdmf_lines))

    return h5_path, xmf_path
