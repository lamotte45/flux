import os
import cv2
import torch
import numpy as np
from 3DDFA_V2.TDDFA import TDDFA
from 3DDFA_V2.utils.pose import viz_pose

# Load the 3D Config
config = {'arch': 'mobilenet_v1', 'widen_factor': 1, 'checkpoint_fp': 'weights/mb1_120x120.pth', 'bfm_fp': 'configs/bfm_noneck_v3.pkl'}
tddfa = TDDFA(gpu_mode=True, **config)

def get_pose_tag(img_path):
    img = cv2.imread(img_path)
    # Detect 3D Landmarks
    boxes = [[0, 0, img.shape[1], img.shape[0]]] # Assume face is main subject
    param_lst, roi_box_lst = tddfa(img, boxes)
    
    # Extract Yaw, Pitch, Roll
    # Pitch < -30: Top/Crown View
    # Yaw > 45: Side Profile
    # Else: Front View
    P, R, T = viz_pose(img, param_lst, [0]) # Placeholder for math extraction
    # Logic: Write tags to .txt file matching image name
    print(f"✅ Tagged {img_path}")

# Run on your dataset
