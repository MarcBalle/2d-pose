""" Filter 2D poses using 1€ filter """


import os
import argparse
import numpy as np
from tqdm import tqdm
from one_euro_filter import OneEuroFilter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts", type=str, required=True, help="keypoints file path")

    args = parser.parse_args()

    keypoints = np.load(args.kpts)["keypoints"]

    # Filter parameters found by Iiulia 
    min_cutoff = 0.1
    beta = 0.00001

    n_frames = keypoints.shape[0]
    n_joints = keypoints.shape[1]
    
    for i in tqdm(range(n_joints)):
        joint_x = keypoints[:, i, 0]
        joint_y = keypoints[:, i, 1]

        # Filter x coordinate 
        one_euro_filter = OneEuroFilter(
                    t0=0, x0=joint_x[0],
                    min_cutoff=min_cutoff,
                    beta=beta
                )

        joint_x_hat = [one_euro_filter(j, joint_x[j]) if j > 0 else joint_x[0] for j in range(n_frames)]
        joint_x_hat = np.array(joint_x_hat)[..., np.newaxis]

        # Filter y coordinate 
        one_euro_filter = OneEuroFilter(
                    t0=0, x0=joint_y[0],
                    min_cutoff=min_cutoff,
                    beta=beta
                )

        joint_y_hat = [one_euro_filter(j, joint_y[j]) if j > 0 else joint_y[0] for j in range(n_frames)]
        joint_y_hat = np.array(joint_y_hat)[..., np.newaxis]

        joint_hat = np.concatenate((joint_x_hat, joint_y_hat), axis=-1)
        keypoints[:, i] = joint_hat
    
    parent_dir, filename = os.path.split(args.kpts)
    output_dir = os.path.join(parent_dir, "1euro_filtered")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    np.savez(os.path.join(output_dir, filename), keypoints=keypoints)
            
    

