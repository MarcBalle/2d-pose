import torch

def skeleton_coco_to_h36m(vec, num_keypoint=17, dim=3):  # Linear interpolation for some joints due to layout difference
    N, _, _ = vec.shape
    joints_index = [-1, 12, 14, 16, 11, 13, 15, -2, -3, 0, -4, 5, 7, 9, 6, 8, 10]
    joint = torch.zeros(N, num_keypoint, dim).to(vec.device)
    for j in range(17):
        if joints_index[j] >= 0:
            joint[:, j, :] = vec[:, joints_index[j], :dim]
    joint[:, 0, :] = (joint[:, 1, :] + joint[:, 4, :]) / 2.0
    joint[:, 8, :] = (joint[:, 11, :] + joint[:, 14, :] + joint[:, 9, :]) / 3.0
    joint[:, 7, :] = (joint[:, 0, :] + joint[:, 8, :]) / 2.0
    joint[:, 10, :] = (joint[:, 8, :] * 2.0 + joint[:, 7, :] * 2.0 + joint[:, 9, :]) / 5.0 * 2.5 - joint[:, 7, :] * 1.5
    return joint