import cv2
import time

from src.video_loader       import VideoLoader
from src.detector           import Detector
from src.tracker            import Tracker
from src.distance_estimator import DistanceEstimator
from src.speed_estimator    import SpeedEstimator
from src.risk_model         import RiskModel
from src.visualizer         import Visualizer


def main():
    video_path = "data/raw/drive.mov"

    video      = VideoLoader(video_path)
    detector   = Detector()
    tracker    = Tracker()
    estimator  = DistanceEstimator(smoothing=5)
    speeder    = SpeedEstimator(fps=video.fps if video.fps > 0 else 30,
                                smoothing=7)
    risk_model = RiskModel()
    visualizer = Visualizer()

    prev_time = time.time()

    while True:
        ret, frame = video.read_frame()
        if not ret:
            print("End of video reached.")
            break

        detections = detector.detect(frame)

        h, w, _ = frame.shape
        filtered = [d for d in detections if d["bbox"][3] <= h * 0.85]

        tracked = tracker.update(filtered)

        any_high_risk = False

        for det in tracked:
            bbox   = det["bbox"]
            obj_id = det["id"]

            distance = estimator.estimate(bbox, obj_id)
            speed    = speeder.estimate_speed(obj_id, distance)
            result   = risk_model.assess(frame, bbox, distance, speed)

            if result["risk"] == "HIGH":
                any_high_risk = True

            frame = visualizer.draw(frame, det, distance, speed, result)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        frame = visualizer.draw_hud(frame, fps, len(tracked), any_high_risk)

        cv2.imshow("Collision Risk Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()


if __name__ == "__main__":
    main()