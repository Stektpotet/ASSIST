import os
import pickle
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF

from PIL import Image

dataset_directory = "..\\data\\catchsnap"

dataset_raw_directory = "..\\data\\catchsnap\\raw"
def make_split():
    pass

def open_all_in(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    class_idx = []
    class_index = 0
    for dirName, subdirList, fileList in os.walk(path, topdown=True, followlinks=False):
        class_dir_files = [TF.to_tensor(Image.open(os.path.join(dirName, file))) for file in fileList]
        if len(class_dir_files) > 0:
            images.extend(class_dir_files)
            class_idx.extend([class_index] * len(class_dir_files))
            class_index += 1
    return torch.stack(images), torch.tensor(class_idx)


if __name__ == '__main__':
    split = ('test', 'train')
    for s in split:
        with open(dataset_directory + s + '.data', "wb") as f:
            torch.save(open_all_in(os.path.join(dataset_directory, 'split', s)), f, pickle_protocol=4)


