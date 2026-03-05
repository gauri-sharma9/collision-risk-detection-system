import cv2
from src.video_loader import VideoLoader
from src.detector import Detector


def main():
    video_path = "data/raw/drive.mov"

    video = VideoLoader(video_path)
    detector = Detector()

    while True:
        ret, frame = video.read_frame()

        if not ret:
            print("End of video reached.")
            break

        detections = detector.detect(frame)

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            conf = det["confidence"]

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Phase 2 - Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()


if __name__ == "__main__":
    main()