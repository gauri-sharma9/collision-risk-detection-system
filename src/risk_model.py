import cv2
import numpy as np


class RiskModel:
    # TTC thresholds (seconds)
    TTC_HIGH   = 3.0
    TTC_MEDIUM = 6.0

    # Distance-only thresholds (when speed data is reliable)
    DIST_HIGH   = 6.0    # metres
    DIST_MEDIUM = 12.0   # metres

    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

    def _hog_confidence(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.8   # too small to compute, return neutral

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 0.8

        try:
            region_resized = cv2.resize(region, (64, 64))
            gray           = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
            features       = self.hog.compute(gray)
            if features is None or len(features) == 0:
                return 0.8
            magnitude  = float(np.mean(np.abs(features)))
            confidence = float(np.clip(0.5 + magnitude * 2, 0.5, 1.0))
            return round(confidence, 3)
        except Exception:
            return 0.8

    def compute_ttc(self, distance, speed):
        # Only valid if object is approaching (speed > 0)
        if speed <= 0:
            return 999.0
        return round(distance / speed, 2)

    def classify_risk(self, ttc, distance, speed):
        """
        Risk logic:
        - If object is NOT approaching (speed <= 0), max risk is LOW
          regardless of distance (parked cars should not be HIGH)
        - If object IS approaching, use TTC as primary signal
        - Distance alone can raise to MEDIUM only if very close
        """
        if speed <= 0:
            # Object moving away or stationary — never HIGH from speed
            if distance < self.DIST_HIGH:
                return "MEDIUM"   # close but not approaching
            return "LOW"

        # Object is approaching — use TTC
        if ttc < self.TTC_HIGH:
            return "HIGH"
        elif ttc < self.TTC_MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"

    def assess(self, frame, bbox, distance, speed):
        ttc      = self.compute_ttc(distance, speed)
        risk     = self.classify_risk(ttc, distance, speed)
        hog_conf = self._hog_confidence(frame, bbox)

        # HOG confidence: weak visual evidence downgrades HIGH → MEDIUM
        if hog_conf < 0.6 and risk == "HIGH":
            risk = "MEDIUM"

        return {
            "ttc":            ttc,
            "risk":           risk,
            "hog_confidence": hog_conf
        }
        magnitude = float(np.mean(np.abs(features)))
        confidence = float(np.clip(0.5 + magnitude * 2, 0.5, 1.0))
        return round(confidence, 3)

    def compute_ttc(self, distance, speed):
        # Only compute TTC if object is genuinely approaching
        # Speed must be positive AND above noise threshold (0.3 m/s)
        if speed < 0.3:
            return 999.0
        return round(distance / speed, 2)

    def classify_risk(self, ttc, distance):
        # Also use raw distance as a safety net —
        # anything closer than 4m is always at least MEDIUM
        if ttc < self.TTC_HIGH or distance < 4.0:
            return "HIGH"
        elif ttc < self.TTC_MEDIUM or distance < 8.0:
            return "MEDIUM"
        else:
            return "LOW"

    def assess(self, frame, bbox, distance, speed):
        ttc       = self.compute_ttc(distance, speed)
        risk      = self.classify_risk(ttc, distance)
        hog_conf  = self._hog_confidence(frame, bbox)

        # HOG confidence check — weak feature downgrades HIGH → MEDIUM
        if hog_conf < 0.6 and risk == "HIGH":
            risk = "MEDIUM"

        return {"ttc": ttc, "risk": risk, "hog_confidence": hog_conf}