import cv2
import os

def test_video_playback(video_path):
    if not os.path.exists(video_path):
        print("Error: Video file not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Video opened successfully.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        cv2.imshow("Phase 0 - Video Test", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "data/raw/drive.mov"
    test_video_playback(video_path)