import numpy as np

try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None


def velocity_acceleration_from_bspline(t, X, Y, Z):
    """
    Compute velocity and acceleration at each time point using a cubic B-spline fit.

    Parameters
    ----------
    t : array-like, shape (n,)
        Time at each position.
    X, Y, Z : array-like, shape (n,)
        Position components.

    Returns
    -------
    vx, vy, vz : ndarray, shape (n,)
        Velocity components at each time.
    ax, ay, az : ndarray, shape (n,)
        Acceleration components at each time.

    Only fits when 4 or more points are tracked (cubic B-spline, k=3).
    """
    if make_interp_spline is None:
        raise RuntimeError(
            "scipy is required for B-spline velocity/acceleration")
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = len(t)
    if n < 4:
        raise ValueError("B-spline fit requires at least 4 tracked points, got %d" % n)

    spl_x = make_interp_spline(t, X, k=3)
    spl_y = make_interp_spline(t, Y, k=3)
    spl_z = make_interp_spline(t, Z, k=3)

    vx = spl_x.derivative(1)(t)
    vy = spl_y.derivative(1)(t)
    vz = spl_z.derivative(1)(t)
    ax = spl_x.derivative(2)(t)
    ay = spl_y.derivative(2)(t)
    az = spl_z.derivative(2)(t)

    return vx, vy, vz, ax, ay, az


def sample_bspline_curve(t, X, Y, Z, num_samples=50):
    """
    Sample a cubic B-spline curve through (t, X, Y, Z) at num_samples points.

    Returns (t_sample, x_sample, y_sample, z_sample) for visualization.
    Only fits when 4 or more points are tracked (cubic B-spline, k=3).
    """
    if make_interp_spline is None:
        raise RuntimeError("scipy is required for B-spline sampling")
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = len(t)
    if n < 4:
        raise ValueError("B-spline curve requires at least 4 tracked points, got %d" % n)
    t_flat = np.linspace(t[0], t[-1], num_samples)
    spl_x = make_interp_spline(t, X, k=3)
    spl_y = make_interp_spline(t, Y, k=3)
    spl_z = make_interp_spline(t, Z, k=3)
    x_curve = spl_x(t_flat)
    y_curve = spl_y(t_flat)
    z_curve = spl_z(t_flat)
    return t_flat, x_curve, y_curve, z_curve


class Track():
    def __init__(self, X, Y, Z, dt, idx, time, run, physical_times=None):
        self.time = time
        self.dt = dt
        self.X = X
        self.Y = Y
        self.Z = Z
        self.idx = idx
        self.track_length = X.shape[0]
        self.run = run
        self.physical_times = (
            None if physical_times is None
            else np.asarray(physical_times, dtype=np.float64)
        )
        self.v = None
        self.a = None

    def compute_velocity(self, use_bspline=False):
        """Compute velocity. If use_bspline=True and track has at least 4 points, use cubic B-spline (also sets acceleration)."""
        if use_bspline and self.track_length >= 4 and make_interp_spline is not None:
            self._compute_velocity_acceleration_bspline()
            return
        if self.physical_times is not None and len(self.physical_times) == self.track_length:
            dt_seg = np.diff(self.physical_times)
            self.vx = np.diff(self.X) / dt_seg
            self.vy = np.diff(self.Y) / dt_seg
            self.vz = np.diff(self.Z) / dt_seg
        else:
            self.vx = np.diff(self.X) / self.dt
            self.vy = np.diff(self.Y) / self.dt
            self.vz = np.diff(self.Z) / self.dt
        self.v = np.stack([self.vx, self.vy, self.vz], axis=-1)
        self.vmag = np.linalg.norm(self.v, axis=1)

    def compute_acceleration(self, use_bspline=False):
        """Compute acceleration. If use_bspline=True and track has at least 4 points, use cubic B-spline (also sets velocity if not already)."""
        if use_bspline and self.track_length >= 4 and make_interp_spline is not None:
            if self.v is None or len(self.v) != self.track_length:
                self._compute_velocity_acceleration_bspline()
            return
        if self.v is None:
            print('Calculate Velocity first!')
            return
        if self.physical_times is not None and len(self.physical_times) == self.track_length:
            dt_seg = np.diff(self.physical_times)
            dt_mid = 0.5 * (dt_seg[1:] + dt_seg[:-1])
            self.ax = np.diff(self.vx) / dt_mid
            self.ay = np.diff(self.vy) / dt_mid
            self.az = np.diff(self.vz) / dt_mid
        else:
            self.ax = np.diff(self.vx) / self.dt
            self.ay = np.diff(self.vy) / self.dt
            self.az = np.diff(self.vz) / self.dt
        self.a = np.stack([self.ax, self.ay, self.az], axis=-1)
        self.amag = np.linalg.norm(self.a, axis=1)

    def _compute_velocity_acceleration_bspline(self):
        """Set velocity and acceleration at each particle position from a cubic B-spline fit. Requires at least 4 points."""
        if self.physical_times is not None and len(self.physical_times) == self.track_length:
            t = self.physical_times
        else:
            t = self.time + np.arange(self.track_length) * self.dt
        vx, vy, vz, ax, ay, az = velocity_acceleration_from_bspline(
            t, np.asarray(self.X), np.asarray(self.Y), np.asarray(self.Z)
        )
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.v = np.stack([self.vx, self.vy, self.vz], axis=-1)
        self.vmag = np.linalg.norm(self.v, axis=1)
        self.ax = ax
        self.ay = ay
        self.az = az
        self.a = np.stack([self.ax, self.ay, self.az], axis=-1)
        self.amag = np.linalg.norm(self.a, axis=1)
