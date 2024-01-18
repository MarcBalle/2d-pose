import numpy as np
import cv2


def threshold_keypoints(keypoints, scores, thr=0.3):
    """
    If keypoints were detected with low confidence, their value is set to 0 or nan if this is a core keypoint.
    A threshold of 0.3 corresponds to the threshold used in mmpose for visualization.
    """

    core_joints = [12, 11, 5, 6, 0]  # these are coco joints used to interpolate h36m joints
    thresholded_keypoints = []
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        if score < thr:
            if i in core_joints:  # is some core joint is missing, return empty skeleton
                return None
            thresholded_keypoints.append([0.0, 0.0])
        else:
            thresholded_keypoints.append(kpt)

    return thresholded_keypoints


def skeleton_coco_to_h36m(keypoints, scores, num_keypoint=17):
    """
    Linear interpolation of Human3.6M skeleton joints.
    """
    keypoints = threshold_keypoints(keypoints, scores, thr=0.3)
    if not keypoints:
        return np.array([])
    joints_index = [-1, 12, 14, 16, 11, 13, 15, -2, -3, 0, -4, 5, 7, 9, 6, 8, 10]
    joint = np.zeros((num_keypoint, 2))
    for j in range(17):
        if joints_index[j] >= 0:
            joint[j, :] = keypoints[joints_index[j]]
    joint[0, :] = (joint[1, :] + joint[4, :]) / 2.0
    joint[8, :] = (joint[11, :] + joint[14, :] + joint[9, :]) / 3.0
    joint[7, :] = (joint[0, :] + joint[8, :]) / 2.0
    joint[10, :] = (joint[8, :] * 2.0 + joint[7, :] * 2.0 + joint[9, :]) / 5.0 * 2.5 - joint[7, :] * 1.5

    return joint


def show2Dpose(kps, img):
    if kps.size == 0:
        return img
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img
