from collections import deque


class SpeedEstimator:
    NOISE_THRESHOLD = 1.0   # m/s — anything below this is treated as zero

    def __init__(self, fps=30, smoothing=15):
        self.fps         = fps
        self._smoothing  = smoothing
        self._prev_dist  = {}
        self._speed_hist = {}

    def estimate_speed(self, obj_id, current_distance):
        if obj_id not in self._prev_dist:
            self._prev_dist[obj_id]  = current_distance
            self._speed_hist[obj_id] = deque(maxlen=self._smoothing)
            return 0.0

        delta_dist = self._prev_dist[obj_id] - current_distance
        delta_time = 1.0 / self.fps
        raw_speed  = delta_dist / delta_time

        self._prev_dist[obj_id] = current_distance

        if obj_id not in self._speed_hist:
            self._speed_hist[obj_id] = deque(maxlen=self._smoothing)
        self._speed_hist[obj_id].append(raw_speed)

        smoothed = sum(self._speed_hist[obj_id]) / len(self._speed_hist[obj_id])

        if abs(smoothed) < self.NOISE_THRESHOLD:
            smoothed = 0.0

        return round(smoothed, 2)

    def is_estimate_reliable(self, obj_id):
        """Only trust speed once smoothing buffer is filled."""
        if obj_id not in self._speed_hist:
            return False
        return len(self._speed_hist[obj_id]) >= 10