import argparse
import os

import cv2
from tqdm import tqdm
import torch
import numpy as np

from mmpose.apis import MMPoseInferencer

from utils import skeleton_coco_to_h36m, show2Dpose


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", "-v", type=str, required=True, help="Path to the input video")
    parser.add_argument("--frames", "-f", type=str, default=None, help="Number of frames to analyze")
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
    parser.add_argument("--output_filename", type=str, required=False, default="keypoints.npz")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    assert os.path.isfile(args.video), "Video file not found"
    assert os.path.isdir(args.output), "Output directory not found"

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

    video_name, video_ext = os.path.splitext(os.path.split(args.video)[1])
    out = cv2.VideoWriter(
        os.path.join(args.output, video_name + ".2dkeypoints" + video_ext),
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    print("Performing inference...")

    video_length = int(args.frames) if args.frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints = np.zeros((video_length, 17, 2))

    for i in tqdm(range(video_length)):
        # Image read in BGR order (required by mmdetections)
        _, img = cap.read()

        # Perform inference
        result_generator = inferencer(img, return_vis=True)
        results = next(result_generator)
        coco_skeleton_image = results["visualization"][0]
        keypoints_coco = results["predictions"][0][0]["keypoints"]
        scores_keypoints = results["predictions"][0][0]["keypoint_scores"]
        keypoints_h36m = skeleton_coco_to_h36m(keypoints_coco, scores_keypoints)
        h36m_skeleton_image = show2Dpose(keypoints_h36m, img)
        out.write(h36m_skeleton_image)
        if keypoints_h36m.size > 0:
            keypoints[i] = keypoints_h36m

    np.savez(os.path.join(args.output, args.output_filename), keypoints=keypoints)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
