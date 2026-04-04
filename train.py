import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from configs import config
from data.dataset import get_train_dataloader
from models.detector import HierarchicalDeepfakeDetector
from losses.contrastive import BatchContrastiveLoss

def main():
    parser = argparse.ArgumentParser(description="Train the Deepfake Detector")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, 
                        help='Number of epochs to train (overrides config.py)')
    parser.add_argument('--resume_weights', type=str, default=None, 
                        help='Path to a .pth file to resume training from')
    args = parser.parse_args()
    print(f"Using device: {config.DEVICE}")
    
    print("Loading Dataset...")
    train_loader = get_train_dataloader(
        config.DATASET_PATH, config.BATCH_SIZE, config.NUM_WORKERS, 
        config.NUM_REAL_TRAIN, config.NUM_FAKE_TRAIN
    )
    print(f"Loaded {len(train_loader.dataset)} videos.")

    model = HierarchicalDeepfakeDetector(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    if args.resume_weights:
        if os.path.exists(args.resume_weights):
            print(f"🔄 Resuming training from saved weights: {args.resume_weights}")
            model.load_state_dict(torch.load(args.resume_weights, map_location=config.DEVICE, weights_only=True))
        else:
            print(f"❌ ERROR: Could not find weights at {args.resume_weights}. Starting from scratch.")
           
    class_weights = torch.tensor([7.0, 1.0]).to(config.DEVICE)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    
    criterion_cont = BatchContrastiveLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    print(f"Starting Training for {args.epochs} Epochs...")
    
    best_loss = float('inf') 
    patience = 5             
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        running_ce_loss, running_cont_loss = 0.0, 0.0
        correct, total = 0, 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits, features = model(images)
            
            loss_ce = criterion_ce(logits, labels)
            loss_cont = criterion_cont(features, labels)
            total_loss = loss_ce + (config.LAMBDA_CONTRASTIVE * loss_cont)
            
            total_loss.backward()
            optimizer.step()
            
            running_ce_loss += loss_ce.item()
            running_cont_loss += loss_cont.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_time = time.time() - start_time
        train_acc = 100 * correct / total
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_cont_loss = running_cont_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Acc: {train_acc:.2f}% | CE: {epoch_ce_loss:.4f} | Cont: {epoch_cont_loss:.4f} | Time: {epoch_time:.2f}s")

        if epoch_ce_loss < best_loss:
            best_loss = epoch_ce_loss
            patience_counter = 0
            
            os.makedirs(os.path.dirname(config.SAVE_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            print(f"🌟 Improvement found! Best model saved to {config.SAVE_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"🛑 EARLY STOPPING TRIGGERED! Model hasn't improved in {patience} epochs.")
                break 

    print("Training Script Finished!")

if __name__ == "__main__":
    main()