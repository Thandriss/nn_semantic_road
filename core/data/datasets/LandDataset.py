import cv2 as cv
import numpy as np
from collections import OrderedDict
from glob import glob
from torch.utils.data import Dataset


class LandDataset(Dataset):
    def __init__(self, root_dir:str, class_labels:list, transforms=None):
        self.root_dir = root_dir
        self.imgs = sorted(glob(root_dir + "/*/images/*"))
        self.masks = sorted(glob(root_dir + "/*/rois/*"))
        self.labels = sorted(glob(root_dir + "/*/labels/*"))
        assert len(self.imgs) == len(self.labels) == len(self.masks)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read image
        img_path = self.imgs[idx]
        image = cv.imread(img_path)

        # Read label
        label_path = self.labels[idx]
        label = cv.imread(label_path)
        label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

        # Read mask (ROI)
        mask_path = self.masks[idx]
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        assert(image.shape[0:1] == label.shape[0:1] == mask.shape[0:1])

        # Prepare data
        if self.transforms:
            image, [label], mask = self.transforms(image, [label], mask)

        return image, label, mask
    
    def calc_class_weights(self, normalize=True) -> dict:
        counts = {}
        for idx in range(len(self.labels)):
            label_path = self.labels[idx]
            label = cv.imread(label_path)
            label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

            class_idxs, class_counts = np.unique(label, return_counts=True)
            for id, count in zip(class_idxs, class_counts):
                if id not in counts:
                    counts[id] = count
                else:
                    counts[id] += count 

        if normalize:
            count_sum = sum(counts.values())
            counts = dict([(key, val / count_sum) for key, val in counts.items()])

        return OrderedDict(sorted(counts.items()))

    def visualize(self, tick_ms=25):
        for i in range(0, self.__len__()):
            image, label, mask = self.__getitem__(i)
            cv.imshow('Image', image.astype(np.uint8))
            cv.imshow('Label', 127*label.astype(np.uint8))
            cv.imshow('Mask', mask.astype(np.uint8))
            cv.waitKey(tick_ms)