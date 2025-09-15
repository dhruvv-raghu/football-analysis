import cv2
import numpy as np
from typing import List, Tuple


def read_video(path_video: str) -> List[np.ndarray]:
    """
    Reads a video file and returns its frames as a list of numpy arrays.

    Args:
        path_video (str): The path to the video file.

    Returns:
        List[np.ndarray]: A list of frames, where each frame is represented as a numpy array.

    Raises:
        FileNotFoundError: If the video file cannot be found or opened.
    """
    cap = cv2.VideoCapture(path_video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path_video}")

    frames: List[np.ndarray] = []

    while cap.isOpened():
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames


def get_video_info(path_video: str) -> Tuple[int, int, float, int]:
    """
    Get basic information about a video file.

    Args:
        path_video (str): The path to the video file.

    Returns:
        Tuple[int, int, float, int]: A tuple containing (width, height, fps, frame_count).

    Raises:
        FileNotFoundError: If the video file cannot be found or opened.
    """
    cap = cv2.VideoCapture(path_video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return width, height, fps, frame_count


def save_video(output_video_frames, output_video_path):
    """
    Save a list of frames as a video file.
    Args:
        output_video_frames (List[np.ndarray]): List of frames to be saved as a video.
        output_video_path (str): The path where the output video will be saved.
    """
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],
        output_video_frames[0].shape[0]))
   
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_video_path}")
    
    

