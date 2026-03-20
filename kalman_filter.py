import numpy as np
from typing import Tuple


class KalmanFilter2D:
    """
    2D Kalman filter with constant-velocity motion model.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]

    This filter tracks target position and estimates velocity, which can
    be used to predict the next-frame position for stereo matching constraints.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        """
        Args:
            process_noise: Covariance of process noise (controls velocity change uncertainty).
                           Smaller values = more trust in constant-velocity assumption.
            measurement_noise: Covariance of measurement noise (reflects detection uncertainty).
                               Larger values = more trust in the prediction vs. observation.
        """
        self.dim_x = 4  # state: [x, y, vx, vy]
        self.dim_z = 2  # measurement: [x, y]

        # Measurement matrix: observe position only
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y

        # Process noise covariance
        self.Q = np.eye(self.dim_x, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self.R = np.eye(self.dim_z, dtype=np.float64) * measurement_noise

        # State estimate vector [x, y, vx, vy]
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        # State covariance (start with high uncertainty)
        self.P = np.eye(self.dim_x, dtype=np.float64) * 100.0

        self.initialized = False

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Initialize filter with known position and optional velocity."""
        self.x = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64) * 100.0
        self.initialized = True

    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        """
        Advance the state by one time step using constant-velocity model.

        Args:
            dt: Time step (in frames; keep at 1.0 for frame-by-frame processing).

        Returns:
            Predicted (x, y) position.
        """
        if not self.initialized:
            return float(self.x[0, 0]), float(self.x[1, 0])

        # State transition matrix: x_new = x + vx*dt, y_new = y + vy*dt
        F = np.eye(self.dim_x, dtype=np.float64)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, x: float, y: float) -> None:
        """
        Correct the state estimate with a new position measurement.

        Args:
            x: Observed x position.
            y: Observed y position.
        """
        if not self.initialized:
            self.initialize(x, y)
            return

        z = np.array([[x], [y]], dtype=np.float64)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_position(self) -> Tuple[float, float]:
        """Return current position estimate (x, y)."""
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        """Return current velocity estimate (vx, vy) in pixels per frame."""
        return float(self.x[2, 0]), float(self.x[3, 0])
