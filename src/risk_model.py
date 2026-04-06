import cv2
import numpy as np


class RiskModel:
    """
    Risk assessment with distance-zone gating.
    
    Zone logic:
    - Beyond 20m : our monocular distance estimate is unreliable.
                   Never show HIGH regardless of TTC.
    - 10m - 20m  : MEDIUM max unless TTC < 2.5s AND speed > 3 m/s
    - Under 10m  : full TTC-based assessment
    - Under 5m   : always at least MEDIUM (very close proximity)
    
    Speed must be positive (approaching) to ever show HIGH.
    """

    TTC_HIGH         = 3.0    # seconds
    TTC_MEDIUM       = 7.0    # seconds
    SPEED_MIN        = 1.0    # m/s minimum to consider approaching
    RELIABLE_DIST    = 20.0   # metres beyond which estimate is unreliable
    CLOSE_ZONE       = 10.0   # metres — full risk assessment zone
    VERY_CLOSE       = 5.0    # metres — always at least MEDIUM

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

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return 0.8

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 0.8

        try:
            resized   = cv2.resize(region, (64, 64))
            gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            features  = self.hog.compute(gray)
            if features is None or len(features) == 0:
                return 0.8
            magnitude  = float(np.mean(np.abs(features)))
            confidence = float(np.clip(0.5 + magnitude * 2, 0.5, 1.0))
            return round(confidence, 3)
        except Exception:
            return 0.8

    def compute_ttc(self, distance, speed):
        if speed < self.SPEED_MIN:
            return 999.0
        return round(distance / speed, 2)

    def classify_risk(self, ttc, distance, speed, reliable):
        """
        Zone-gated risk classification.
        reliable = True only after enough smoothing frames collected.
        """
        # Not enough data yet — always LOW
        if not reliable:
            return "LOW"

        # Object moving away or stationary
        if speed < self.SPEED_MIN:
            if distance < self.VERY_CLOSE:
                return "MEDIUM"
            return "LOW"

        # --- Object IS approaching ---

        # Beyond reliable range — cap at MEDIUM
        if distance > self.RELIABLE_DIST:
            return "LOW"

        # Medium zone (10–20m): only HIGH if TTC very tight and speed significant
        if distance > self.CLOSE_ZONE:
            if ttc < 2.5 and speed > 3.0:
                return "MEDIUM"   # still cap at MEDIUM in this zone
            return "LOW"

        # Close zone (under 10m): full TTC assessment
        if ttc < self.TTC_HIGH:
            return "HIGH"
        elif ttc < self.TTC_MEDIUM or distance < self.VERY_CLOSE:
            return "MEDIUM"
        else:
            return "LOW"

    def assess(self, frame, bbox, distance, speed, reliable):
        ttc      = self.compute_ttc(distance, speed)
        risk     = self.classify_risk(ttc, distance, speed, reliable)
        hog_conf = self._hog_confidence(frame, bbox)

        # HOG check: weak visual evidence downgrades HIGH → MEDIUM
        if hog_conf < 0.6 and risk == "HIGH":
            risk = "MEDIUM"

        return {
            "ttc":            ttc,
            "risk":           risk,
            "hog_confidence": hog_conf
        }