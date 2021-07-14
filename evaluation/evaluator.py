from dataclasses import dataclass
from typing import Tuple, Callable, Sequence, Union, Dict, Any, Iterable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss

LossFunction = _Loss

@dataclass(frozen=True, eq=False, order=False, unsafe_hash=False)
class EvaluationResult:
    """Performance data from a training model"""
    tag: str
    accuracy: float
    loss_tag: str
    worst_loss: float
    mean_loss: float
    best_loss: float
    quantile_loss: Tuple[Tuple[float, float], ...]


class DatasetEvaluator:
    def __init__(self, tag: str, data_loader: DataLoader, eval_fn: LossFunction,
                 quantiles: Tuple[float, ...] = (0.5, 0.9, 0.99)):
        self.tag = tag
        self.data_loader = data_loader
        self.eval_fn = eval_fn
        self.eval_fn.reduction = "none"
        self.quantiles = torch.tensor((0, *quantiles, 1))

    def evaluate(self, model: nn.Module) -> EvaluationResult:
        model_device = next(model.parameters()).device
        dataset_size = len(self.data_loader.dataset)
        batch_size = self.data_loader.batch_size
        epoch_losses = torch.empty(dataset_size, dtype=torch.float)
        accurately_classified = 0
        i = 0
        model.eval()
        with torch.no_grad():
            for x, y in self.data_loader:
                x = x.to(model_device)
                y = y.to(model_device)
                model_output = model(x)
                accurate_samples = (model_output.argmax(dim=-1).squeeze() == y.squeeze()).int()  # type:torch.Tensor
                accurately_classified += torch.sum(accurate_samples)
                losses = self.eval_fn(model_output, y)
                epoch_losses[i:i + batch_size] = losses.detach().view(-1)
                i += batch_size
        model.train()

        quantiles = tuple(torch.quantile(epoch_losses, self.quantiles))

        return EvaluationResult(
            tag=self.tag,
            accuracy=accurately_classified / dataset_size,
            loss_tag=self.eval_fn.__class__.__name__,
            worst_loss=quantiles[-1],
            mean_loss=epoch_losses.mean().item(),
            best_loss=quantiles[0],
            quantile_loss=tuple((q, v) for q, v in zip(self.quantiles[1:-1], quantiles[1:-1]))
        )


class Evaluator:
    def __init__(self, logger, *evaluators: Tuple[int, DatasetEvaluator]):
        self.logger = logger
        self.evaluators = evaluators

    def evaluate(self, model: nn.Module, epoch: int):
        for interval, evaluator in self.evaluators:
            if epoch % interval != 0:
                continue

            res = evaluator.evaluate(model)
            self.logger.log(data={
                f"{res.tag} Accuracy": res.accuracy,
                f"{res.tag}/{res.loss_tag}/worst loss": res.worst_loss,
                f"{res.tag}/{res.loss_tag}/mean loss": res.mean_loss,
                f"{res.tag}/{res.loss_tag}/best loss": res.best_loss,
            }, step=epoch, commit=False)
            self.logger.log(data={f'{res.tag}/{res.loss_tag}/qloss/q_{q:.4f}': v for q, v in res.quantile_loss}, step=epoch, commit=False)
        self.logger.log({}, step=epoch, commit=True)
