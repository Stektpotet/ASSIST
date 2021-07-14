import random
import warnings
from abc import abstractmethod, ABC
from typing import Callable, Optional, Any, Union, Tuple

import numpy
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, BatchSampler
from tqdm import tqdm

from eventsystem import EventV1, EventV4, EventV2


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class AbstractTrainer(ABC):

    def __init__(self,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 dataset: torch.utils.data.Dataset,
                 sampler: Optional[torch.utils.data.Sampler] = None,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 ):
        self._batch_size = batch_size
        self.loss_fn = loss_fn
        if self.loss_fn.reduction != "none":
            warnings.warn("Trainer changed reduction mode of loss function!")
            self.loss_fn.reduction = 'none'

        self.optimizer = optimizer
        self.train_set = dataset

        self.sampler = sampler

        if self.sampler is None:
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                           num_workers=num_workers, worker_init_fn=seed_worker)
        elif isinstance(self.sampler, BatchSampler):
            self.train_loader = DataLoader(self.train_set, batch_sampler=self.sampler, pin_memory=True, num_workers=num_workers, worker_init_fn=seed_worker)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=self.sampler, pin_memory=True,
                                           num_workers=num_workers, worker_init_fn=seed_worker)

        # ========================================= Inline Behaviour Additions =========================================
        #       https://stackoverflow.com/questions/1092531/event-system-in-python
        # TODO: make events capable of handling kwargs -> this will allow us to not use wandb in here
        _TrainingEvent = EventV1[nn.Module]
        _EpochEvent = EventV2[nn.Module, int]
        _BatchEvent = EventV4[nn.Module, int, int, Tuple[Any, Any]]

        self.on_start_training = _TrainingEvent()
        self.on_end_training = _TrainingEvent()
        self.on_start_epoch = _EpochEvent()
        self.on_end_epoch = _EpochEvent()
        self.on_start_batch = _BatchEvent()
        self.on_end_batch = _BatchEvent()


    def train(self, model: nn.Module, num_epochs: int):
        model_device = next(model.parameters()).device
        self.on_start_training(model)
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.on_start_epoch(model, epoch)
            for batch, (x, y) in enumerate(self.train_loader):
                self.on_start_batch(model, epoch, batch, (x, y))
                self.optimizer.zero_grad()
                self._batch_step(model, x.to(model_device), y.to(model_device))
                self.optimizer.step()
                self.on_end_batch(model, epoch, batch, (x, y))
            self.on_end_epoch(model, epoch)
        self.on_end_training(model)

    @abstractmethod
    def _batch_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
        pass


class ArchetypeTrainer(AbstractTrainer):

    def _batch_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
        model_output = model(x)
        losses = self.loss_fn(model_output, y)
        losses.mean().backward()

class AccumulativeAccuracyFilteringTrainer(ArchetypeTrainer):

    def __init__(self,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 dataset: torch.utils.data.Dataset,
                 sampler: Optional[torch.utils.data.Sampler] = None,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 accumulation_batch_size: Optional[int] = None
                 ):
        if accumulation_batch_size is None:
            accumulation_batch_size = batch_size
        super(AccumulativeAccuracyFilteringTrainer, self).__init__(loss_fn, optimizer, dataset, sampler, accumulation_batch_size, num_workers)
        self._batch_size = batch_size

    @property
    def accumulation_batch_size(self):
        return self.train_loader.batch_size

    def train(self, model: nn.Module, num_epochs: int):
        model_device = next(model.parameters()).device
        sample = self.train_loader.dataset[0]  # type: Tuple[torch.Tensor, int]
        k = 0
        batch_x = torch.empty((self._batch_size, *sample[0].shape), requires_grad=False, device=model_device)
        batch_y = torch.full((self._batch_size,), sample[1], requires_grad=False, device=model_device)

        self.on_start_training(model)


        for epoch in tqdm(range(1, num_epochs + 1)):
            self.on_start_epoch(model, epoch)
            batch = 0
            _accumulation_steps = 0
            _accepted_ratio = 0
            avg_accumulation_steps = 0
            avg_accepted_ratio = 0
            avg_overflow = 0

            last_accumulation_steps = 0
            last_accepted_ratio = 0
            last_overflow = 0

            for x, y in self.train_loader:
                x = x.to(model_device)
                y = y.to(model_device)
                with torch.no_grad():
                    model_output = model(x)
                    error_filter = torch.nonzero(model_output.argmax(dim=-1).squeeze() != y.squeeze()).view(-1)  # type: torch.Tensor

                _accepted_ratio += (len(error_filter) / self.accumulation_batch_size)
                _accumulation_steps += 1

                if k + len(error_filter) < self._batch_size:
                    batch_x[k:k + len(error_filter)] = x[error_filter]
                    batch_y[k:k + len(error_filter)] = y[error_filter]
                else:
                    self.on_start_batch(model, epoch, batch, (batch_x, batch_y))
                    in_limit = (self._batch_size - k)
                    over_limit = len(error_filter) - in_limit
                    batch_x[k:k + in_limit] = x[error_filter][:in_limit]
                    batch_y[k:k + in_limit] = y[error_filter][:in_limit]

                    avg_overflow += over_limit
                    avg_accumulation_steps += _accumulation_steps
                    avg_accepted_ratio += (_accepted_ratio / _accumulation_steps) # range 0-1
                    _accumulation_steps = 0
                    _accepted_ratio = 0

                    model_output = model(x)
                    losses = self.loss_fn(model_output, y)
                    losses.mean().backward()
                    self.optimizer.zero_grad()
                    # TODO: Call superclass _batch_step instead
                    losses = self.loss_fn(model(batch_x), batch_y)  # Bonus, we dont need to calc losses for accurate samples
                    losses.mean().backward()
                    self.optimizer.step()
                    self.on_end_batch(model, epoch, batch, (batch_x, batch_y))
                    batch += 1
                    # batch_x[0:over_limit] = x[error_filter][in_limit:]
                    # batch_y[0:over_limit] = y[error_filter][in_limit:]
                    k = 0 # = over_limit
                    continue
                k += len(error_filter)

            if batch > 0:
                last_overflow = avg_overflow / batch
                last_accumulation_steps = avg_accumulation_steps / batch
                last_accepted_ratio = avg_accepted_ratio / batch

            self.on_end_epoch(model, epoch)
        self.on_end_training(model)

class AccumulativeSoftmaxMarginFilteringTrainer(ArchetypeTrainer):

    def __init__(self,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 dataset: torch.utils.data.Dataset,
                 sampler: Optional[torch.utils.data.Sampler] = None,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 accumulation_batch_size: Optional[int] = None,
                 q: float = 1,
                 margin: float = 0.1
                 ):
        if accumulation_batch_size is None:
            accumulation_batch_size = batch_size
        super(AccumulativeSoftmaxMarginFilteringTrainer, self).__init__(loss_fn, optimizer, dataset, sampler, accumulation_batch_size, num_workers)
        self._batch_size = batch_size
        self.q = max(min(q, 1.0), 0.0)
        self.margin = max(min(margin, 1.0), 0.0)

    @property
    def accumulation_batch_size(self):
        return self.train_loader.batch_size

    def train(self, model: nn.Module, num_epochs: int):
        model_device = next(model.parameters()).device
        sample = self.train_loader.dataset[0]  # type: Tuple[torch.Tensor, int]
        k = 0
        batch_x = torch.empty((self._batch_size, *sample[0].shape), requires_grad=False, device=model_device)
        batch_y = torch.full((self._batch_size,), sample[1], requires_grad=False, device=model_device)

        self.on_start_training(model)


        for epoch in tqdm(range(1, num_epochs + 1)):
            self.on_start_epoch(model, epoch)
            batch = 0
            _accumulation_steps = 0
            _accepted_ratio = 0
            avg_accumulation_steps = 0
            avg_accepted_ratio = 0
            avg_overflow = 0

            last_accumulation_steps = 0
            last_accepted_ratio = 0
            last_overflow = 0
            for x, y in self.train_loader:
                x = x.to(model_device)
                y = y.to(model_device)
                with torch.no_grad():
                    model_output = model(x)
                    beliefs = torch.softmax(model_output, dim=1)
                    sorted_beliefs = torch.sort(beliefs, dim=1).values
                    tops = sorted_beliefs[:, -1]
                    # get the class at q between [0..n-1] of the n belief-sorted classes
                    q_class = int(self.q * (len(sorted_beliefs[0]) - 1) + 0.5)
                    q_values = sorted_beliefs[:, q_class]
                    decision_distance = tops - q_values
                    error_filter = torch.nonzero(decision_distance < self.margin).view(-1)  # type: torch.Tensor

                _accepted_ratio += (len(error_filter) / self.accumulation_batch_size)
                _accumulation_steps += 1

                if k + len(error_filter) < self._batch_size:
                    batch_x[k:k + len(error_filter)] = x[error_filter]
                    batch_y[k:k + len(error_filter)] = y[error_filter]
                else:
                    self.on_start_batch(model, epoch, batch, (batch_x, batch_y))
                    in_limit = (self._batch_size - k)
                    over_limit = len(error_filter) - in_limit
                    batch_x[k:k + in_limit] = x[error_filter][:in_limit]
                    batch_y[k:k + in_limit] = y[error_filter][:in_limit]

                    avg_overflow += over_limit
                    avg_accumulation_steps += _accumulation_steps
                    avg_accepted_ratio += (_accepted_ratio / _accumulation_steps) # range 0-1
                    _accumulation_steps = 0
                    _accepted_ratio = 0

                    model_output = model(x)
                    losses = self.loss_fn(model_output, y)
                    losses.mean().backward()
                    self.optimizer.zero_grad()
                    # TODO: Call superclass _batch_step instead
                    losses = self.loss_fn(model(batch_x), batch_y)  # Bonus, we dont need to calc losses for accurate samples
                    losses.mean().backward()
                    self.optimizer.step()
                    self.on_end_batch(model, epoch, batch, (batch_x, batch_y))
                    batch += 1
                    # batch_x[0:over_limit] = x[error_filter][in_limit:]
                    # batch_y[0:over_limit] = y[error_filter][in_limit:]
                    k = 0 # = over_limit
                    continue
                k += len(error_filter)

            if batch > 0:
                last_overflow = avg_overflow / batch
                last_accumulation_steps = avg_accumulation_steps / batch
                last_accepted_ratio = avg_accepted_ratio / batch

            self.on_end_epoch(model, epoch)
        self.on_end_training(model)
