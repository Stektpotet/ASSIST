from typing import Tuple, Optional, Sequence, Callable, List, Any

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataloader import default_collate, DataLoader




def _make_data_loader_on_device(dataset: Dataset, device: torch.device, batch_size: Optional[int] = 1,
                               shuffle: bool = False, sampler: Optional[Sampler[int]] = None,
                               batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                               num_workers: int = 0, collate_fn: Optional[Callable[[List], Any]] = None,
                               pin_memory: bool = False, drop_last: bool = False,
                               timeout: float = 0, worker_init_fn: Optional[Callable[[int], None]] = None,
                               multiprocessing_context=None, generator=None,
                               *, prefetch_factor: int = 2,
                               persistent_workers: bool = False):
    if collate_fn is None:
        if batch_sampler is None:  # if auto_collate
            collate_fn = default_collate
        else:
            collate_fn = default_convert

    def wrapped_collate(batch):
        batch = collate_fn(batch)
        return [batch[0].to(device), batch[1].to(device)]

    loader = DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, wrapped_collate, pin_memory,
               drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
               prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    return loader


DataLoaderOnDevice = _make_data_loader_on_device

default_cifar_augmentation = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
    T.RandomAffine(
        degrees=180,
        translate=(0.1, 0.1), scale=(0.9, 1.1),
    ),
    T.RandomCrop(32, padding=4, pad_if_needed=True),
    T.ColorJitter(0.5, 0.5, 0.5, 0.05),
)


def augment_batch(batch: torch.Tensor, augmentation: nn.Module = default_cifar_augmentation) -> Tuple[
    torch.Tensor, torch.Tensor]:
    x, y = batch
    return augmentation(x).contiguous(), y


def create_batch_transforming_collate_fn(batch_transform: torch.nn.Module = None):
    def collate_fn(batch):
        collated = default_collate(batch)
        if batch_transform is not None:
            transformed_images = batch_transform(collated[0])
            transformed = [transformed_images.contiguous(), collated[1]]
            return transformed
        return collated

    return collate_fn


default_augmentor_collate = create_batch_transforming_collate_fn(default_cifar_augmentation)
