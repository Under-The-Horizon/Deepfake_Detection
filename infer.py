import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs import config
from models.detector import HierarchicalDeepfakeDetector

def get_transforms():
    """Applies the exact same normalization used during training/evaluation."""
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def extract_video_frames(video_path, num_frames=16):
    """
    Extracts evenly spaced frames from a video to get a comprehensive 
    view of the subject throughout the clip.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frames from the video.")

    step = max(1, total_frames // num_frames)
    frames = []
    transform = get_transforms()

    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = transform(image=frame)
            frames.append(augmented['image'])

    cap.release()
    
    if not frames:
        raise ValueError("Frame extraction failed.")
        
    return torch.stack(frames)

def main():
    parser = argparse.ArgumentParser(description="Test a single video for Deepfakes.")
    parser.add_argument("--video", type=str, required=True, help="Path to the .mp4 video file.")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample (default: 16).")
    args = parser.parse_args()

    print(f"\n--- Deepfake Video Inference ---")
    print(f"Target Video: {args.video}")
    print(f"Sampling {args.frames} frames using device: {config.DEVICE}")

    model = HierarchicalDeepfakeDetector(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    if not os.path.exists(config.SAVE_MODEL_PATH):
        print(f"[ERROR] Model weights not found at {config.SAVE_MODEL_PATH}. Train the model first!")
        return
        
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE, weights_only=True))
    model.eval() 
    try:
        video_tensor = extract_video_frames(args.video, num_frames=args.frames)
        video_tensor = video_tensor.to(config.DEVICE)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    with torch.no_grad():
        logits, _ = model(video_tensor)
        
        probabilities = F.softmax(logits, dim=1)
        
        avg_probs = torch.mean(probabilities, dim=0).cpu().numpy()
        
    real_prob = avg_probs[0] * 100
    fake_prob = avg_probs[1] * 100
    
    verdict = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob)
    
    print("\n" + "="*40)
    print(f" VERDICT: {verdict}")
    print(f" CONFIDENCE: {confidence:.2f}%")
    print("="*40)
    print(f" Breakdown -> Real: {real_prob:.2f}% | Fake: {fake_prob:.2f}%\n")

if __name__ == "__main__":
    main()