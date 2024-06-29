""" This script creates a video comparing the 2D pose predictions of the different models. 
    If needed, the predictions can be changed to filtered prediction, therefore the script compares the original and filtered predictions.
"""

import argparse
import os

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from utils import show2Dpose

matplotlib.use("Agg")

if __name__ == "__main__":

    k0 = np.load("C:\\Users\\marcw\\master_thesis\\intro_presentation\\3d_poses_poseformer.npz")["arr_0"]
    # k0_filt = np.load("presentation\\run1b_2018-05-29-14-02-47.kinect_color.1euro.npz")["keypoints"]

    k1 = np.load("C:\\Users\\marcw\\master_thesis\\intro_presentation\\3d_poses_poseformerV2.npz")["arr_0"]
    # k1_filt = np.load("presentation\\run1_2018-05-23-10-21-45.kinect_color.1euro.npz")["keypoints"]

    k2 = np.load("C:\\Users\\marcw\\master_thesis\\intro_presentation\\3d_poses_mhformer.npz")["arr_0"]
    # k2_filt = np.load("presentation\\run1_2018-05-24-13-44-01.kinect_color.1euro.npz")["keypoints"]

    cap0 = cv2.VideoCapture(
        "C:\\Users\\marcw\\master_thesis\\intro_presentation\\CUT_run1_2018-05-23-13-16-52.kinect_color.2dkeypoints_1euro.mp4"
    )
    fps = cap0.get(cv2.CAP_PROP_FPS)
    video_length = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cap1 = cv2.VideoCapture("presentation\\run1_2018-05-23-10-21-45.kinect_color.mp4")

    # cap2 = cv2.VideoCapture("presentation\\run1_2018-05-24-13-44-01.kinect_color.mp4")

    out = cv2.VideoWriter(
        filename="presentation\\predictions.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
        fps=fps,
        frameSize=(int(width), int(height)),
    )

    for i in tqdm(range(2300, 2600)):
        cap0.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
        # cap1.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
        # cap2.set(cv2.CAP_PROP_POS_FRAMES, i - 1)

        _, img0 = cap0.read()
        # _, img1 = cap1.read()
        # _, img2 = cap2.read()

        keypoints_frame0 = k0[i]
        # keypoints_frame0_filt = k0_filt[i]
        keypoints_frame1 = k1[i]
        # keypoints_frame1_filt = k1_filt[i]
        keypoints_frame2 = k2[i]
        # keypoints_frame2_filt = k2_filt[i]

        # skeleton_frame0 = show2Dpose(keypoints_frame0, np.copy(img0))
        # skeleton_frame0_filt = show2Dpose(keypoints_frame0_filt, np.copy(img0))
        # skeleton_frame1 = show2Dpose(keypoints_frame1, img1)
        # skeleton_frame1_filt = show2Dpose(keypoints_frame1_filt, img1)
        # skeleton_frame2 = show2Dpose(keypoints_frame2, img2)
        # skeleton_frame2_filt = show2Dpose(keypoints_frame2_filt, img2)

        # plt.ioff
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(width / 100, height / 100),
            dpi=100,
        )
        ax[0].imshow(cv2.cvtColor(skeleton_frame0, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[0].set_title("Original")
        ax[1].imshow(cv2.cvtColor(skeleton_frame0_filt, cv2.COLOR_BGR2RGB))
        ax[1].axis("off")
        ax[1].set_title("1 Euro filtered")
        # ax[0, 1].imshow(cv2.cvtColor(skeleton_frame1, cv2.COLOR_BGR2RGB))
        # ax[0, 1].axis("off")
        # ax[0, 1].set_title("Original")
        # ax[1, 1].imshow(cv2.cvtColor(skeleton_frame1_filt, cv2.COLOR_BGR2RGB))
        # ax[1, 1].axis("off")
        # ax[1, 1].set_title("Filtered")
        # ax[0, 2].imshow(cv2.cvtColor(skeleton_frame2, cv2.COLOR_BGR2RGB))
        # ax[0, 2].axis("off")
        # ax[0, 2].set_title("Original")
        # ax[1, 2].imshow(cv2.cvtColor(skeleton_frame2_filt, cv2.COLOR_BGR2RGB))
        # ax[1, 2].axis("off")
        # ax[1, 2].set_title("Filtered")

        canvas = FigureCanvas(fig)
        canvas.draw()

        frame = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape((height, width, -1))

        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        del fig, ax, canvas, frame

    cap0.release()
    # cap1.release()
    # cap2.release()

    out.release()

    cv2.destroyAllWindows()
