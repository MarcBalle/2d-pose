"""Script for creating a video with a given 2D skeleton"""

import argparse
import os

import numpy as np
import cv2
from tqdm import tqdm

from utils import show2Dpose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="video file path")
    parser.add_argument("--kpts", type=str, required=True, help="keypoints file path")
    parser.add_argument("--output", type=str, default=".", help="output directory")

    args = parser.parse_args()

    keypoints = np.load(args.kpts)["keypoints"]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    assert (
        keypoints.shape[0] == video_length
    ), f"Keypoints frames ({keypoints.shape[0]}) and video frame ({video_length}) do not match."

    output_dir = os.path.dirname(args.video)
    video_name, video_ext = os.path.splitext(os.path.basename(args.video))
    out = cv2.VideoWriter(
        filename=os.path.join(args.output, video_name + ".2dkeypoints_1euro" + video_ext),
        fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
        fps=fps,
        frameSize=(int(width), int(height)),
    )

    for i in tqdm(range(video_length)):
        _, img = cap.read()
        keypoints_frame = keypoints[i]
        skeleton_image = show2Dpose(img, keypoints_frame)
        out.write(skeleton_image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
