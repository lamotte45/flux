import torch
import os

print("📦 Initializing 3D Face Alignment Model...")
try:
    # This pulls the 'brain' for head orientation
    model = torch.hub.load('cleardusk/3DDFA_v2', '3ddfa_v2', pretrained=True)
    model.eval()
    print("✅ System Ready: 3D Pose Estimation Active")
except Exception as e:
    print(f"❌ Error loading model: {e}")
