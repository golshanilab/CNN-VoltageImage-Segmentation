import torch
import os

DATASET_PATH = os.path.join("dataset","train")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH,"masks")

TEST_SPLIT = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False
#PIN_MEMORY = False

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 2

INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64

THRESHOLD = 0.5

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT,"voltim_seg.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,"plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT,"test_paths.txt"])