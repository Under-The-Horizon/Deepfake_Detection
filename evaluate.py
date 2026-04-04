import os
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs import config
from models.detector import HierarchicalDeepfakeDetector
from data.dataset import CelebDFVideoDataset

def get_test_dataloader():
    test_transforms = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    dataset = CelebDFVideoDataset(
        root_dir=config.DATASET_PATH, 
        transform=test_transforms, 
        num_real_limit=100,  
        num_fake_limit=100   
    )
    
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

def main():
    print(f"Using device: {config.DEVICE}")
    
    model = HierarchicalDeepfakeDetector(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    if not os.path.exists(config.SAVE_MODEL_PATH):
        print(f"\n[ERROR] Could not find saved weights at: {config.SAVE_MODEL_PATH}")
        print("Make sure you run 'python train.py' first to generate the weights file!")
        return
        
    print(f"Loading weights from {config.SAVE_MODEL_PATH}...")
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=config.DEVICE, weights_only=True))
   
    model.eval() 
    
    print("\nLoading Test Dataset...")
    test_loader = get_test_dataloader()
    
    all_preds = []
    all_labels = []
    
    print(f"Starting evaluation on {len(test_loader.dataset)} videos...")
    start_time = time.time()
    
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            logits, _ = model(images)
            
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    eval_time = time.time() - start_time
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds.")
    print("=" * 50)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Precision:         {prec * 100:.2f}%")
    print(f"Recall:            {rec * 100:.2f}%")
    print(f"F1 Score:          {f1 * 100:.2f}%")
    print("=" * 50)
    print("Confusion Matrix:")
    print(f"True Real (TN): {cm[0][0]}  |  False Fake (FP): {cm[0][1]}")
    print(f"False Real (FN): {cm[1][0]}  |  True Fake (TP): {cm[1][1]}")
    print("=" * 50)

if __name__ == "__main__":
    main()
