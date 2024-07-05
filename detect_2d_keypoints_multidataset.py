import os
import argparse

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import cv2

from utils import show2Dpose
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--video", "-v", type=str, required=True, help="Path to the input video")
parser.add_argument("--output", "-o", type=str, default=".", help="Output directory")
args = parser.parse_args()

model = hub.load("https://bit.ly/metrabs_l")

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_name, video_ext = os.path.splitext(os.path.basename(args.video))
out = cv2.VideoWriter(
    os.path.join(args.output, video_name + ".2dkeypoints_multi" + video_ext),
    cv2.VideoWriter_fourcc(*"DIVX"),
    fps,
    (width, height),
)

print("Performing inference...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    preds = model.detect_poses(tf.convert_to_tensor(frame), skeleton="h36m_17")
    pose = preds["poses2d"].numpy()

    if pose.size > 0:
        frame = show2Dpose(frame, pose[0], radius=4)

    out.write(frame)

    # Update progress bar
    progress = cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)
    tqdm.tqdm.write(f"Progress: {progress * 100:.2f}%")

cap.release()
out.release()
cv2.destroyAllWindows()
