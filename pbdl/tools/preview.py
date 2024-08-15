"""
This module provides functions for creating videos from datasets.
"""

# non-local package imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# local class imports
from pbdl.loader import Dataset


# TODO what if there are more that 2 spatial dim?
def create_preview_video(
    dataset_name,
    path="preview.mp4",
    channels=(0, 1),
    fps=30,
    sec=5,
    res_width=512,  # maintain width-height ratio
    cmap=plt.get_cmap("twilight"),
):

    # TODO what if fps * sec too large?
    dataset = Dataset(
        dataset_name,
        fps * sec,
        sel_sims=[0],
        intermediate_time_steps=True,
        normalize=False,
    )

    if dataset.num_spatial_dims() < 2:
        raise ValueError("Error: Dataset must have at least 2 spatial dimensions.")

    first_frame, _, rem_frames = dataset[0]
    frames = np.append([first_frame], rem_frames, axis=0)

    # take vector norm
    if len(channels) > 1:
        frames = np.sqrt(np.sum(frames[:, channels, :, :] ** 2, axis=1))
    else:
        frames = frames[:, channels[0], :, :]

    height = frames.shape[1]
    width = frames.shape[2]

    high_res_size = (res_width, int((height / width) * res_width))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, float(fps), high_res_size)

    # normalize
    min, max = frames.min(), frames.max()
    frames = (frames - min) / (max - min)
    frames = cmap(frames)
    frames = (frames[:, :, :, :3] * 255).astype(np.uint8)

    for frame_count in range(len(frames)):
        high_res_frame = cv2.resize(
            frames[frame_count],
            high_res_size,
            interpolation=cv2.INTER_NEAREST,
        )

        video.write(high_res_frame)

    video.release()

    # TODO quick fix to convert to browser compatible video codec
    os.system(f"ffmpeg -i {path} -vcodec libx264 -f mp4 {path}")
