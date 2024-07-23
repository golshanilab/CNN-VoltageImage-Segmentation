from voltimseg import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile as tif
import os

def prepare_plot(origImage, origMask, predMask):

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(origImage, cmap="gray")
    ax[0].set_title("Original Image")

    ax[1].imshow(origMask, cmap="gray")
    ax[1].set_title("Original Mask")

    ax[2].imshow(predMask, cmap="gray")
    ax[2].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()

def make_prediction(model, imagePath):
    model.eval()

    with torch.no_grad():

        image = tif.imread(imagePath)
        image = image.astype(np.float32)
        orig = image.copy()

        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)

        gtMask = tif.imread(groundTruthPath)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(config.DEVICE)
        predMask = model(image).squeeze()
        predMask = predMask.cpu().numpy()

        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        prepare_plot(np.sum(orig,axis=0), gtMask.squeeze(), predMask)

        return predMask
    
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_prediction(unet, path)