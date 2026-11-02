"""
8-State Kalman Filter for Multi-Object Vehicle Tracking.
State representation: [x, y, a, h, vx, vy, va, vh]^T
- (x, y): Bounding box center coordinates
- a: Aspect ratio (w / h)
- h: Box height
- (vx, vy, va, vh): Corresponding temporal velocities
Constant velocity linear dynamical model with discrete time step.
"""

import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0, class_id: int = 0):
        """
        bbox: [x1, y1, x2, y2]
        """
        # Linear Kalman Filter: 8 state variables, 4 measurement variables
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix F
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z)
        self.R[0:2, 0:2] *= 1.0
        self.R[2:4, 2:4] *= 10.0

        # State covariance matrix P
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 1000.0  # High initial uncertainty on velocities

        # Initial state x
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4, 0] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = score
        self.class_id = class_id
        self.state = 'tracked'  # 'tracked', 'lost', 'removed'

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Converts [x1, y1, x2, y2] to [x_center, y_center, aspect_ratio, height]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, r, h])

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> np.ndarray:
        """Converts state [x_center, y_center, aspect_ratio, height] to [x1, y1, x2, y2]."""
        c_x = x[0]
        c_y = x[1]
        r = x[2]
        h = x[3]
        w = r * h
        x1 = c_x - w / 2.0
        y1 = c_y - h / 2.0
        x2 = c_x + w / 2.0
        y2 = c_y + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self) -> np.ndarray:
        """Advances state vector by 1 time step and returns predicted bbox."""
        # Check for negative height
        if self.x[3] + self.x[7] <= 0:
            self.x[7] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.x[:, 0]))
        return self.history[-1]

    def update(self, bbox: np.ndarray, score: float = 1.0, class_id: int = None):
        """Updates state with observed bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        if class_id is not None:
            self.class_id = class_id

        z = self._bbox_to_z(bbox).reshape((self.dim_z, 1))

        # Innovation (residual)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        self.state = 'tracked'

    def get_state(self) -> np.ndarray:
        """Returns current bounding box estimate [x1, y1, x2, y2]."""
        return self._x_to_bbox(self.x[:, 0])

    def get_velocity(self) -> np.ndarray:
        """Returns pixel velocity vector (vx, vy) in pixels/frame."""
        return np.array([self.x[4, 0], self.x[5, 0]])
