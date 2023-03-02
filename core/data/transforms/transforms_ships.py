import torch
import cv2 as cv
import numpy as np


class Resize(object):
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, rects=None, mask=None):
        kx = self.size[0] / image.shape[1]
        ky = self.size[1] / image.shape[0]
        image = cv.resize(image, self.size, interpolation=cv.INTER_AREA)
        if rects is not None:
            for i in range(0, len(rects)):
                rects[i] = np.array([kx*rects[i][0], ky*rects[i][1], kx*rects[i][2], ky*rects[i][3]], dtype=np.int)
        if mask is not None:
            mask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)
        return image, rects, mask


class ConvertFromInts(object):
    def __call__(self, image, rects=None, mask=None):
        image = image.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)
        return image, rects, mask


class Clip(object):
    def __init__(self, mmin:float=0.0, mmax:float=255.0):
        self.min = mmin
        self.max = mmax
        assert self.max >= self.min, "min val must be >= max val"

    def __call__(self, image, rects=None, mask=None):
        image = np.clip(image, self.min, self.max)
        return image, rects, mask


class Normalize(object):
    def __call__(self, image, rects=None, mask=None):
        image = image.astype(np.float32) / 255.0
        return image, rects, mask


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, rects=None, mask=None):
        if self.current == 'BGR' and self.transform == 'GRAY':
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif self.current == 'BGR' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, rects, mask


class ToTensor(object):
    def __init__(self, norm_mask:bool=True):
        self.norm_mask = norm_mask

    def __call__(self, cvimage, rects=None, mask=None):
        if cvimage.ndim == 2:
            cvimage = np.expand_dims(cvimage, axis=2)
        img = torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)

        if rects is not None:
            for i in range(0, len(rects)):
                rects[i] = rects[i].astype(np.float32)
            rects = torch.from_numpy(rects)

        if mask is not None:
            mask = mask.astype(np.float32)
            if self.norm_mask:
                mask = mask / 255.0
            # mask = torch.from_numpy(np.expand_dims(mask, 0))
            mask = torch.from_numpy(mask)

        return img, rects, mask