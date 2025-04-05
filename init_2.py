import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Global variables
orb = cv2.ORB_create(3000)  # Increased number of features
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

K = None  # Camera intrinsic matrix
P = None  # Projection matrix

trajectory = []  # To store the trajectory points
ground_truth = []  # To store the ground truth trajectory


def load_calib(file_path):
    """
    Load camera calibration parameters.
    """
    global K, P
    with open(file_path, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]


def load_ground_truth(file_path):
    """
    Load ground truth poses from the KITTI poses.txt file.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            pose = np.array(data).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to 4x4 matrix
            poses.append(pose)
    return poses


def load_images(image_dir):
    """
    Load images from the specified directory.
    """
    image_paths = [os.path.join(image_dir, file) for file in sorted(os.listdir(image_dir))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


def form_transf(R, t):
    """
    Form a 4x4 transformation matrix from rotation and translation.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decomp_essential_mat(E, q1, q2):
    """
    Decompose the essential matrix and resolve the correct pose.
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)

    T1 = form_transf(R1, np.ndarray.flatten(t))
    T2 = form_transf(R2, np.ndarray.flatten(t))
    T3 = form_transf(R1, np.ndarray.flatten(-t))
    T4 = form_transf(R2, np.ndarray.flatten(-t))

    transformations = [T1, T2, T3, T4]
    K_ext = np.concatenate((K, np.zeros((3, 1))), axis=1)

    projections = [K_ext @ T for T in transformations]

    positives = []
    for P, T in zip(projections, transformations):
        hom_Q1 = cv2.triangulatePoints(P, K_ext, q1.T, q2.T)
        hom_Q2 = T @ hom_Q1

        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
        positives.append(total_sum)

    max_index = np.argmax(positives)
    if max_index == 0:
        return R1, np.ndarray.flatten(t)
    elif max_index == 1:
        return R2, np.ndarray.flatten(t)
    elif max_index == 2:
        return R1, np.ndarray.flatten(-t)
    else:
        return R2, np.ndarray.flatten(-t)


def get_matches(img1, img2):
    """
    Extract and match keypoints between two images using KNN-based matching with Lowe's ratio test.
    """
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Use BFMatcher with Hamming distance for ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)  # Find the 2 nearest neighbors

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Keep only matches that pass the ratio test
            good_matches.append(m)

    # Extract matched keypoints
    q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return q1, q2


def main():
    global trajectory, ground_truth

    # Paths to the KITTI dataset
    data_dir = 'KITTI_sequence_1'
    image_dir = os.path.join(data_dir, 'image_l')
    poses_file = os.path.join(data_dir, 'poses.txt')
    calib_file = os.path.join(data_dir, 'calib.txt')

    if not os.path.exists(image_dir) or not os.path.exists(poses_file) or not os.path.exists(calib_file):
        print("Error: Missing required files.")
        return

    # Load calibration, ground truth, and images
    load_calib(calib_file)
    ground_truth = load_ground_truth(poses_file)
    images = load_images(image_dir)

    # Initialize trajectory
    gt_path = []
    estimate_path = []
    cur_pose = np.eye(4)

    # Align the initial pose
    initial_gt_pose = ground_truth[0]
    cur_pose[:3, :3] = initial_gt_pose[:3, :3]  # Align rotation
    cur_pose[:3, 3] = initial_gt_pose[:3, 3]  # Align translation

    for i in tqdm(range(1, len(images)), unit="frame"):
        q1, q2 = get_matches(images[i - 1], images[i])
        E, _ = cv2.findEssentialMat(q1, q2, K)
        R, t = decomp_essential_mat(E, q1, q2)

        transf = form_transf(R, t)
        cur_pose = cur_pose @ np.linalg.inv(transf)

        gt_pose = ground_truth[i]
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimate_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    # Apply offset correction
    gt_path = np.array(gt_path)
    estimate_path = np.array(estimate_path)
    offset = gt_path[0] - estimate_path[0]
    estimate_path += offset

    # Plot the trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(gt_path[:, 0], gt_path[:, 1], label="Ground Truth", color="green")
    plt.plot(estimate_path[:, 0], estimate_path[:, 1], label="Estimated", color="blue")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Visual Odometry Trajectory")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
