from collections import deque


class DistanceEstimator:
    FOCAL_LENGTH    = 600
    REAL_CAR_HEIGHT = 1.5

    def __init__(self, smoothing=15):
        self._history   = {}
        self._smoothing = smoothing

    def estimate(self, bbox, obj_id=None):
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        if bbox_height <= 0:
            return 999.0

        raw = (self.REAL_CAR_HEIGHT * self.FOCAL_LENGTH) / bbox_height

        if obj_id is None:
            return round(raw, 2)

        if obj_id not in self._history:
            self._history[obj_id] = deque(maxlen=self._smoothing)
        self._history[obj_id].append(raw)

        smoothed = sum(self._history[obj_id]) / len(self._history[obj_id])
        return round(smoothed, 2)
    
    def is_estimate_reliable(self, obj_id):
        """Only trust distance once we have enough history frames."""
        if obj_id not in self._history:
            return False
        return len(self._history[obj_id]) >= 8