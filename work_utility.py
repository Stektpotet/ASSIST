import time

import torch.nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from util.batching_util import augment_batch
from datasets.catchsnap import CatchSnap

transform = T.Compose([
    T.ToTensor(),
    # T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_augmentation_imgwise = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomAffine(
        degrees=180,
        translate=(0.1, 0.1), scale=(0.9, 1.1),
    ),
    T.RandomCrop(32, padding=4),
    transform,
    T.ColorJitter(0.5, 0.5, 0.5, 0.05),
])

transform_augmentation = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
    T.RandomAffine(
        degrees=180,
        translate=(0.1, 0.1), scale=(0.9, 1.1),
    ),
    T.RandomCrop(32, padding=4),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    T.ColorJitter(0.5, 0.5, 0.5, 0.05),
)

def timestamp(start_time, name: str = ""):
    stamp_time = time.time()
    elapsed = stamp_time - start_time
    mins = elapsed // 60
    sec = elapsed % 60
    hours = mins // 60
    mins = mins % 60
    print("{3}:\t{0}:{1}:{2}".format(int(hours), int(mins), sec, name))
    return stamp_time


def pre_augment(n):
    dataset_test = CIFAR10("./data/", train=False, download=True, transform=transform_augmentation_imgwise)
    start_time = time.time()
    for _ in tqdm(range(n)):
        batch = torch.utils.data.dataloader.default_collate([dataset_test[i] for i in range(1024)])
    timestamp(start_time, "pre_augment")

def post_augment(n):
    dataset_test = CIFAR10("./data/", train=False, download=True, transform=transform)

    start_time = time.time()
    for _ in tqdm(range(n)):
        batch = torch.utils.data.dataloader.default_collate([dataset_test[i] for i in range(1024)])

        t_batch = transform_augmentation(batch[0])
    timestamp(start_time, "post_augment")

if __name__ == '__main__':

    catchsnap = CatchSnap(train=False, transform=transform)
    loader = DataLoader(catchsnap, batch_size=8)
    batch = next(loader)

    augment_batch(batch, )

    pass
    # post_augment(10)
    # pre_augment(10)
