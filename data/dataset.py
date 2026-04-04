import os
import glob
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CelebDFVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_real_limit=None, num_fake_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []
        
        temp_real_paths = []
        for d in [os.path.join(root_dir, 'Celeb-real'), os.path.join(root_dir, 'YouTube-real')]:
            if os.path.exists(d):
                temp_real_paths.extend(glob.glob(os.path.join(d, '*.mp4')))
                
        temp_fake_paths = []
        fake_dir = os.path.join(root_dir, 'Celeb-synthesis')
        if os.path.exists(fake_dir):
            temp_fake_paths.extend(glob.glob(os.path.join(fake_dir, '*.mp4')))
            
        random.seed(42) 
        random.shuffle(temp_real_paths)
        random.shuffle(temp_fake_paths)
        
        selected_real = temp_real_paths[:num_real_limit] if num_real_limit else temp_real_paths
        selected_fake = temp_fake_paths[:num_fake_limit] if num_fake_limit else temp_fake_paths
        
        self.video_paths.extend(selected_real)
        self.labels.extend([0] * len(selected_real))
        self.video_paths.extend(selected_fake)
        self.labels.extend([1] * len(selected_fake))

    def __len__(self):
        return len(self.video_paths)

    def extract_random_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total_frames - 1))
        success, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if success else None

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frame = self.extract_random_frame(video_path)
        
        while frame is None:
            idx = random.randint(0, len(self.video_paths) - 1)
            video_path, label = self.video_paths[idx], self.labels[idx]
            frame = self.extract_random_frame(video_path)

        if self.transform:
            frame = self.transform(image=frame)['image']
            
        return frame, torch.tensor(label, dtype=torch.long)

def get_train_dataloader(data_path, batch_size, num_workers, real_limit, fake_limit):
    train_transforms = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = CelebDFVideoDataset(data_path, train_transforms, real_limit, fake_limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)