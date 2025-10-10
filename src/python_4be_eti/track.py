import numpy as np


class Track():
    def __init__(self, X, Y, Z, dt, idx, time, run):
        self.time = time
        self.dt = dt
        self.X = X
        self.Y = Y
        self.Z = Z
        self.idx = idx
        self.track_length = X.shape[0]
        self.run = run

    def compute_velocity(self):
        # if self.track_length < 2:
        #     print('You need at least 2 entries to calculate velocity')
        self.vx = np.diff(self.X) / self.dt
        self.vy = np.diff(self.Y) / self.dt
        self.vz = np.diff(self.Z) / self.dt
        self.v = np.stack([self.vx, self.vy, self.vz], axis=-1)
        self.vmag = np.linalg.norm(self.v, axis=1)
        self.vmean = np.mean(self.v, axis=0)
        self.vstd = np.std(self.v, axis=0)

    def compute_acceleration(self):
        if self.v is None:
            print('Calculate Velocity first!')
            return
        # if self.track_length < 3:
        #     print('You need at least 3 entries to calculate acceleration!')
        self.ax = np.diff(self.vx) / self.dt
        self.ay = np.diff(self.vy) / self.dt
        self.az = np.diff(self.vz) / self.dt
        self.a = np.stack([self.ax, self.ay, self.az], axis=-1)
        self.amag = np.linalg.norm(self.a, axis=1)
        self.amean = np.mean(self.a, axis=0)
        self.astd = np.std(self.a, axis=0)

    def mean_velocity(self):
        return np.mean(self.v)

    def std_dev_velocity(self):
        return np.std(self.v)

    def mean_acceleration(self):
        return np.mean(self.a)

    def std_dev_acceleration(self):
        return np.std(self.a)
