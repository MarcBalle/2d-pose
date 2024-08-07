import argparse
import math
import os

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts", type=str, required=True, help="keypoints file path")
    # Default value for angle and length found experimentally
    parser.add_argument("--angle", type=float, required=False, default=0.15, help="angle wrt knee joint in radians")
    parser.add_argument(
        "--length", type=float, required=False, default=162.656, help="lenght between knee and ankle in pixels"
    )

    args = parser.parse_args()

    keypoints = np.load(args.kpts)["keypoints"]

    for frame_idx in tqdm(range(keypoints.shape[0])):
        left_knee, right_knee = keypoints[frame_idx][5], keypoints[frame_idx][2]

        # If left or right knee joints are not available
        if not left_knee.any() or not right_knee.any():
            continue

        # Right ankle
        keypoints[frame_idx][3] = np.array(
            [right_knee[0] + args.length * math.cos(args.angle), right_knee[1] + args.length * math.sin(args.angle)]
        )

        # Left ankle
        keypoints[frame_idx][6] = np.array(
            [left_knee[0] + args.length * math.cos(args.angle), left_knee[1] + args.length * math.sin(args.angle)]
        )

    parent_dir, filename = os.path.split(args.kpts)
    output_dir = os.path.join(parent_dir, "ankle_fixed")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    np.savez(os.path.join(output_dir, filename), keypoints=keypoints)
