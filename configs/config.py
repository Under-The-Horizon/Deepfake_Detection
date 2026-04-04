import torch

DATASET_PATH = '/kaggle/input/celeb-df-v2'
SAVE_MODEL_PATH = 'saved_weights/detector.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

MARGIN = 2.0
LAMBDA_CONTRASTIVE = 0.5

NUM_REAL_TRAIN = None
NUM_FAKE_TRAIN = None