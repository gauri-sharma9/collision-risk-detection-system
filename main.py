import cv2
from src.video_loader import VideoLoader


def main():
    video_path = "data/raw/drive.mov"
    video = VideoLoader(video_path)

    print(f"FPS: {video.fps}")
    print(f"Total Frames: {video.frame_count}")

    while True:
        ret, frame = video.read_frame()

        if not ret:
            print("End of video reached.")
            break

        cv2.imshow("Phase 1 - Video Pipeline", frame)

        # Delay adjusted based on FPS for smooth playback
        delay = int(1000 / video.fps)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Playback stopped by user.")
            break

    video.release()


if __name__ == "__main__":
    main()