import imageio
import numpy as np

TARGET_COLOR = np.array([249, 249, 249], dtype=np.uint8)   # #f9f9f9
TOL = 3  # allow small compression noise


def is_frame_flat(frame, target_color=TARGET_COLOR, tol=TOL):
    """
    Check if a frame is uniformly the target color within tolerance.
    """
    # |frame - target| <= tol everywhere?
    diff = np.abs(frame.astype(np.int16) - target_color)
    return np.all(diff <= tol)


def is_video_flat_color(video_path, num_samples=12):
    """
    Sample N evenly spaced frames from the video and check if they are flat.
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    total_frames = reader.count_frames()

    if total_frames == 0:
        return False

    # sample evenly spaced frames
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    for idx in indices:
        frame = reader.get_data(idx)
        if not is_frame_flat(frame):
            return False

    return True


if __name__ == "__main__":
    path = "src/evaluation/data/BASE/video/normal-fwd-Action Camera.mp4"
    if is_video_flat_color(path):
        print("Video is a flat #f9f9f9 background.")
    else:
        print("Video contains content / not flat.")
