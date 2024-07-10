from torch.utils.data import Dataset
import tifffile as tif

class NeuronSegmentionDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):

        image = tif.imread(self.imagePaths[idx])
        mask = tif.imread(self.maskPaths[idx])

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)