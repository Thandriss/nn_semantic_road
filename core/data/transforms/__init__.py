from .transforms2 import *
from . import transforms_ships as ts


class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels=None, masks=None):
        for t in self.transforms:
            images, labels, masks = t(images, labels, masks)

        return images, labels, masks


def build_transforms(image_size, is_train:bool=True, to_tensor:bool=True):
    if is_train:
        transform = [
            RandomCrop(min_crop=0.7, probabilty=1.0),
            Resize(image_size),
            RandomRotate(-90.0, 90.0),
            RandomJpeg(min_quality=0.7, probabilty=0.25),
            ConvertFromInts(),
            RandomGamma(probabilty=0.5),
            RandomHue(probabilty=0.1),
            RandomMirror(horizont_prob=0.5, probabilty=0.5),
            Clip()
        ]
    else:
        transform = [
            Resize(image_size),
            ConvertFromInts()
        ]

    if to_tensor:
        transform = transform + [Normalize(), ToTensor(norm_label=False, norm_mask=True)]

    transform = TransformCompose(transform)
    return transform


def build_transforms_ships(image_size, is_train:bool=True, to_tensor:bool=True):
    if is_train:
        transform = [
            ts.Resize(image_size),
            ts.ConvertColor('BGR', 'GRAY'),
            ts.ConvertFromInts(),
            ts.Clip()
        ]
    else:
        transform = [
            ts.Resize(image_size),
            ts.ConvertColor('BGR', 'GRAY'),
            ts.ConvertFromInts(),
        ]

    # TODO: check rects? 0 <= x1 < x2 < W ...

    if to_tensor:
        transform = transform + [ts.Normalize(), ts.ToTensor(norm_mask=True)]

    transform = TransformCompose(transform)
    return transform