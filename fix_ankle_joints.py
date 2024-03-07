import argparse
import math

import numpy as np 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts", type=str, required=True, help="keypoints file path")
    # Default value for angle and length found experimentally
    parser.add_argument("--angle", type=float, required=False, default=0.09212, help="angle wrt knee joint in radians")
    parser.add_argument("--length", type=float, required=False, default=162.656, help="lenght between knee and ankle in pixels")

    args = parser.parse_args()

    keypoints = np.load(args.kpts)["keypoints"]

    # Take frames with at least one missing joint
    frames_with_missing_joints = np.unique(np.where(~np.any(keypoints, axis=2))[0])

    for frame_idx in frames_with_missing_joints:
        left_knee, right_knee = keypoints[frame_idx][5], keypoints[frame_idx][2]

        # If left or right knee joints are not available 
        if not left_knee.any() or not right_knee.any():
            continue
        
        # Right ankle
        keypoints[frame_idx][3] = np.array([right_knee[0] + args.length * math.cos(-args.angle), right_knee[1] + args.length * math.sin(-args.angle)])

        # Left ankle
        keypoints[frame_idx][6] = np.array([left_knee[0] + args.length * math.cos(-args.angle), left_knee[1] + args.length * math.sin(-args.angle)])

    np.savez("C:\\Users\\marcw\\master_thesis\\ankle-fixed_run1_2018-05-03-14-08-31.kinect_color.npz", keypoints=keypoints)
