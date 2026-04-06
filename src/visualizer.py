import cv2


class Visualizer:
    COLORS = {
        "LOW":    (0, 200, 0),
        "MEDIUM": (0, 165, 255),
        "HIGH":   (0, 0, 255),
    }

    def draw(self, frame, det, distance, speed, risk_result):
        x1, y1, x2, y2 = det["bbox"]
        obj_id   = det["id"]
        label    = det["class"]
        risk     = risk_result["risk"]
        ttc      = risk_result["ttc"]
        hog_conf = risk_result["hog_confidence"]
        color    = self.COLORS[risk]

        # Bounding box — thicker for HIGH risk
        thickness = 3 if risk == "HIGH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        ttc_str = f"{ttc:.1f}s" if ttc < 99 else "--"
        spd_str = f"{speed:.1f}" if speed > 0 else "0.0"

        # Single compact label inside the box at bottom
        info_line = f"{label}#{obj_id} {distance:.1f}m {spd_str}m/s TTC:{ttc_str}"
        risk_line = f"RISK:{risk} HOG:{hog_conf:.2f}"

        # Draw background strip for readability
        label_y = min(y2 + 14, frame.shape[0] - 20)
        cv2.rectangle(frame,
                      (x1, y2),
                      (x1 + max(len(info_line), len(risk_line)) * 7, y2 + 30),
                      (0, 0, 0), -1)

        cv2.putText(frame, info_line,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, risk_line,
                    (x1 + 2, label_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def draw_hud(self, frame, fps, total_objects, any_high_risk):
        h, w = frame.shape[:2]

        if any_high_risk:
            cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 180), -1)
            cv2.putText(frame, "!! COLLISION RISK DETECTED !!",
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

        cv2.putText(frame,
                    f"FPS:{fps:.1f}  Objects:{total_objects}",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        return frame