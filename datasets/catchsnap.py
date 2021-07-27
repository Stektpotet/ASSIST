import os
from collections import Callable
from typing import Optional, Any

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
# import os
#
# import torch
# from torch.utils.data import Dataset
#
# class CatchSnap(Dataset):
#     training_file = 'train.data'
#     test_file = 'test.data'
#
#     @property
#     def processed_folder(self):
#         return '..\\data\\catchsnap\\split'
#
#     # def _check_exists(self):
#     #     return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
#     #         os.path.exists(os.path.join(self.processed_folder, self.test_file))
#
#
#     def __init__(self, root, train=True, transform=None, target_transform=None):
#         # self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train  # training set or test set
#
#         # if not self._check_exists():
#         #     raise RuntimeError('Dataset not found.' +
#         #                        ' You can use download=True to download it')
#
#         if self.train:
#             data_file = self.training_file
#         else:
#             data_file = self.test_file
#         self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
#
#     def __getitem__(self, index):
#         raise NotImplementedError


class CatchSnap(ImageFolder):
    __root_dir = os.path.join(os.path.dirname(__file__), "..", "data", "catchsnap", "split")

    def __init__(self,
                 train=True,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None):
        dir = os.path.join(self.__root_dir, 'train' if train else 'test')
        super().__init__(dir, transform, target_transform, loader, is_valid_file)
