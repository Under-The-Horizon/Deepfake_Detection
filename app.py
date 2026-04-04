import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

from models.detector import HierarchicalDeepfakeDetector 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = HierarchicalDeepfakeDetector(num_classes=2).to(device)
weights_path = "detector.pth" 

try:
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() 
    print("✅ Model weights loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load weights from {weights_path}. Error: {e}")

video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_and_preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Could not read video file.")

    if total_frames < num_frames:
        frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = video_transform(frame)
            frames.append(frame_tensor)
    cap.release()

    while len(frames) < num_frames:
        frames.append(torch.zeros(3, 224, 224))

    video_tensor = torch.stack(frames)
    return video_tensor

def analyze_video(video_filepath):
    if video_filepath is None:
        return "Please upload a video."
        
    try:
        tensor_frames = extract_and_preprocess_video(video_filepath).to(device)
        
        with torch.no_grad(): 
            logits, _ = model(tensor_frames)
            probabilities = F.softmax(logits, dim=1)
            
            avg_fake_prob = probabilities[:, 1].mean().item() 
        
        if avg_fake_prob > 0.50:
            return f"🚨 FAKE DETECTED\nConfidence: {avg_fake_prob * 100:.2f}%"
        else:
            return f"✅ REAL VIDEO\nConfidence: {(1 - avg_fake_prob) * 100:.2f}%"
            
    except Exception as e:
        return f"❌ An error occurred during analysis: {str(e)}"

print("Starting Web Interface...")
interface = gr.Interface(
    fn=analyze_video,                     
    inputs=gr.Video(label="Upload Video to Analyze"), 
    outputs=gr.Textbox(label="Detection Result", lines=2),     
    title="🔍 Deepfake Detection AI",
    description="Upload an mp4 video. This tool analyzes 16 individual frames using your CViT + GenConViT architecture to detect spatial artifacts."
)

if __name__ == "__main__":
    interface.launch(share=True)