import os
import cv2
import sys
import torch
import pickle
import numpy as np
from torchvision import transforms

sys.path.append('./3DDFA_V2')
from TDDFA import TDDFA
from models.mobilenet_v1 import MobileNet

def get_angles(param):
    p_ = param[:12].reshape(3, 4)
    P = p_[:, :3]
    sy = np.sqrt(P[0, 0] ** 2 + P[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(P[2, 1], P[2, 2])
        yaw = np.arctan2(-P[2, 0], sy)
        roll = np.arctan2(P[1, 0], P[0, 0])
    else:
        pitch = np.arctan2(-P[1, 2], P[1, 1])
        yaw = np.arctan2(-P[2, 0], sy)
        roll = 0
    return pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi

class TDDFA_Fixed(TDDFA):
    def __init__(self, **kvs):
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = 0 if self.gpu_mode else -1
        self.size = 120 
        
        # Load Stats as Numpy (To match model output)
        stats_path = '3DDFA_V2/configs/param_mean_std_62d_120x120.pkl'
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.param_mean = stats.get('mean')
        self.param_std = stats.get('std')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=127.5/255., std=128/255.)
        ])
        
        self.model = MobileNet(widen_factor=kvs.get('widen_factor', 1), num_classes=62)
        checkpoint = torch.load(kvs.get('checkpoint_fp'), map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        
        if self.gpu_mode:
            self.model = self.model.cuda()
            
        self.model.eval()
        from bfm.bfm import BFMModel
        self.bfm = BFMModel(bfm_fp=kvs.get('bfm_fp'), shape_dim=40, exp_dim=10)

config = {
    'widen_factor': 1,
    'checkpoint_fp': '3DDFA_V2/weights/mb1_120x120.pth',
    'bfm_fp': '3DDFA_V2/configs/bfm_noneck_v3.pkl',
    'gpu_mode': True
}
tddfa = TDDFA_Fixed(**config)

def detect_hair_presence(img):
    h, w = img.shape[:2]
    top_region = img[0:int(h*0.35), :]
    return np.var(top_region) > 150

def process_dataset(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} not found")
        return
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🧐 Scanning {len(images)} images...")
    
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        boxes = [[0, 0, img.shape[1], img.shape[0]]]
        
        with torch.no_grad():
            param_lst, _ = tddfa(img, boxes)
        
        tags = ["platinum waves", "kinkystraight hair texture"]
        
        if len(param_lst) == 0:
            angle_tag = "back view, crown focus" if detect_hair_presence(img) else "unknown angle"
        else:
            # param_lst[0] is a numpy array, self.param_std is a numpy array. 
            # The TDDFA.__call__ internal math will now work.
            p, y, r = get_angles(param_lst[0])
            if p < -25: angle_tag = "top view, crown swirl"
            elif abs(y) > 30: angle_tag = "side view, profile view"
            else: angle_tag = "front view"
            tags.append(f"pitch {int(p)} yaw {int(y)}")
            
        tags.append(angle_tag)
        with open(os.path.splitext(img_path)[0] + ".txt", 'w') as f:
            f.write(", ".join(tags))
        print(f"✅ {img_name} -> {angle_tag}")

if __name__ == "__main__":
    process_dataset('dataset_v2/crown_back_angles')
