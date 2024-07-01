"""
This script performs pose estimation on a video using the MMPose library. 
The script processes each frame of the video, generates pose heatmaps, and saves them as images in an output directory. 
The indices of the frames to be processed can be specified using an optional indices file. 
The script utilizes the MMPoseInferencer class from the MMPose library for pose estimation.
"""

import argparse
import os
import pickle

import cv2
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from mmpose.apis import MMPoseInferencer


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", "-v", type=str, required=True, help="Path to the input video")
    parser.add_argument("--config_pose", "-cp", type=str, required=True, help="mmpose model config file")
    parser.add_argument(
        "--config_det",
        "-cd",
        type=str,
        required=True,
        help="mmdetection model config file",
    )
    parser.add_argument("--checkpoint", "-ckpt", type=str, required=False, help="pth checkpoint file")
    parser.add_argument("--output", "-o", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--indices",
        type=str,
        required=False,
        help="Path to the indices pkl file. This file stores the indices of the frames to be processed.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    assert os.path.isfile(args.video), "Video file not found"
    assert os.path.isdir(args.output), "Output directory not found"
    if args.indices:
        assert os.path.isfile(args.indices), "Indices file not found"

    video_name, _ = os.path.splitext(os.path.basename(args.video))
    output_dir = os.path.join(args.output, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the DetInferencer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inferencer = MMPoseInferencer(
        pose2d=args.config_pose,
        det_model=args.config_det,
        det_cat_ids=None if args.config_det == "whole_image" else [0],
        device=device,
    )

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Performing inference...")

    frames = pickle.load(open(args.indices, "rb")) if args.indices else range(video_length)

    for i in tqdm(frames):
        # Image read in BGR order (required by mmdetections)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, img = cap.read()

        # Perform inference
        result_generator = inferencer(img, return_vis=True, draw_heatmap=True)
        results = next(result_generator)
        heatmap = results["visualization"][0]

        fig = plt.figure(figsize=(32, 16))
        plt.imshow(heatmap)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"heatmap_frame{i}.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    cap.release()
    cv2.destroyAllWindows()
