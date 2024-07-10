from voltimseg.dataset import NeuronSegmentionDataset
from voltimseg.model_2 import UNet
from voltimseg import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# transforms here

trainDS = NeuronSegmentionDataset(imagePaths=trainImages, maskPaths=trainMasks)
testDS = NeuronSegmentionDataset(imagePaths=testImages, maskPaths=testMasks)

print(f"[INFO] training dataset size: {len(trainDS)}")
print(f"[INFO] testing dataset size: {len(testDS)}")

trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()-2)
testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count()-2)

unet = UNet().to(config.DEVICE)
loss_fn = BCEWithLogitsLoss()
opt = RMSprop(unet.parameters(), lr=config.INIT_LR)

trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

H = {"train_loss": [], "test_loss": []}

print("[INFO] training network...")

startTime = time.time()

for e in tqdm(range(config.NUM_EPOCHS)):

    unet.train()
    totalTrainLoss = 0
    totalTestLoss = 0

    for (i, (x,y)) in enumerate(trainLoader):

        (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))


        x = x.float()
        y = y.float()

        print(x.max())
        print(y.max())

        pred = unet(x)
        print(pred.max())
        loss = loss_fn(pred.squeeze(), y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss
    
    with torch.no_grad():

        unet.eval()

        for (x,y) in testLoader:

            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            x = x.float()
            y = y.float()

            pred = unet(x)
            totalTestLoss += loss_fn(pred.squeeze(), y)

    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps

    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

    endTime = time.time()

    print("[INFO] total time takes to train model: {:.2f}s".format(endTime-startTime))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

torch.save(unet, config.MODEL_PATH)
    