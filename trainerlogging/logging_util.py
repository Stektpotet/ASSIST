from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler

from .adapters import LoggerAdapter


@torch.no_grad()
def evaluate(logger, model: nn.Module, loader: DataLoader,
             eval_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    model.eval()
    model_device = next(model.parameters()).device
    accurately_classified = 0
    epoch_losses = torch.empty(len(loader.dataset), dtype=torch.float)
    i = 0
    for x, y in loader:
        x = x.to(model_device)
        y = y.to(model_device)
        model_output = model(x)
        accurate_samples = (model_output.argmax(dim=-1).squeeze() == y.squeeze()).view(-1).int()
        accurately_classified += torch.sum(accurate_samples)
        losses = eval_loss_fn(model_output, y)
        epoch_losses[i:i + loader.batch_size] = losses.detach().squeeze()
        i += loader.batch_size
    model.train()