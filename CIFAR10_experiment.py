import argparse
import pandas as pd
import inspect
import traceback
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, softmax
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from eventsystem.trackable import Trackable
from pytorch_cifar.models3x32x32 import LeNet, FCNet100

from evaluation import Evaluator, DatasetEvaluator
from trainers import AccumulativeAccuracyFilteringTrainer, ArchetypeTrainer, AccumulativeSoftmaxMarginFilteringTrainer
import wandb
from util.image_util import fig2img


def prepare_config(args: argparse.Namespace):
    config = dict(**args.__dict__)
    config["model"] = config.pop("model").__name__
    config["optimizer"] = config.pop("optimizer").__name__
    config["trainer"] = config.pop("trainer").__name__
    return config

def prepare_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=180,
            translate=(0.1, 0.1), scale=(0.9, 1.1),
        ),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.05),
        transform
    ])
    dataset_test = CIFAR10("./data/", train=False, download=True, transform=transform)
    dataset_train = CIFAR10("./data/", download=True, transform=transform_augmented)
    dataset_train_unaugmented = CIFAR10("./data/", download=True, transform=transform)
    return dataset_train, dataset_test, dataset_train_unaugmented




class BeliefMetrics(Trackable):
    def __init__(self, dataloader: DataLoader, num_classes: int = 10, interval: int = 1):
        super().__init__(interval)
        self._interval = interval
        self._dataloader = dataloader
        self._num_classes = num_classes
        self._beliefs = torch.empty((len(self._dataloader.dataset), self._num_classes), requires_grad=False, device='cuda:0')
        self._labels = torch.empty(len(self._dataloader.dataset), dtype=torch.int, requires_grad=False, device='cuda:0')
        self._imgs = []

    def get_belief_dataframe(self):
        sorted_beliefs = torch.sort(self._beliefs, 1).values.detach().cpu()
        return pd.DataFrame({'label': self._labels.detach().cpu().numpy()}).join(pd.DataFrame(sorted_beliefs.numpy()))

    def plot_classwise_distributions(self, path):
        l, b = self._labels.detach().cpu().numpy(), self._beliefs.detach().cpu().numpy()
        df = pd.DataFrame({'label': l}).join(pd.DataFrame(b))
        plots = df.loc[:, df.columns != 'label'].groupby(df['label']).boxplot(layout=(2, 5), sharex=True,
                                                                              figsize=(22, 9), whis=(0, 100))
        f = plots[0].get_figure()
        f.savefig(path+'.png')

    def plot_softmax_distribution(self, model: nn.Module):
        with torch.no_grad():
            i = 0
            for batch, (x, y) in enumerate(self._dataloader):
                batch_beliefs = softmax(model(x.cuda()), 1)
                self._beliefs[i:i + len(batch_beliefs)] = batch_beliefs
                self._labels[i:i + len(y)] = y
                i += len(batch_beliefs)

            sorted_beliefs = torch.sort(self._beliefs, 1).values.detach().cpu()

        l_max, h_min = sorted_beliefs[:, 0].max(), sorted_beliefs[:, 9].min()

        plt.axhline(y=l_max, color="red", linestyle=(0, (5, 5)))
        plt.axhline(y=h_min, color="blue", linestyle=(5, (5, 5)))
        plt.boxplot(torch.transpose(sorted_beliefs, 0, 1), showfliers=False, whis=(0, 100), zorder=1)
        plt.ylabel("Belief Strength")
        plt.xlabel("Classes")
        self._imgs.append(fig2img(plt.gcf()))
        plt.show()

    def save_gif(self, name):
        self._imgs[0].save(name + '.gif', save_all=True, append_images=self._imgs[1:], optimize=False, duration=10, loop=0)

    def __call__(self, model: nn.Module, epoch: int, *args, **kwargs):
        if epoch % self._interval != 0:
            return
        self.plot_softmax_distribution(model)
        # wandb.log({'Belief Strength': plt})

config = {
    'lr': 0.02,
    'epochs': 100,
    'filtering': True
}

if __name__ == '__main__':
    dataset_train, dataset_test, unaugmented = prepare_datasets()

    model = LeNet().cuda()
    if config['filtering']:
        trainer = AccumulativeSoftmaxMarginFilteringTrainer(
            nn.CrossEntropyLoss(reduction='none'),
            SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4),
            dataset=dataset_train, batch_size=2048*4, margin=0.3, q=0.75
        )
    else:
        trainer = ArchetypeTrainer(
            nn.CrossEntropyLoss(reduction='none'),
            SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4),
            dataset=dataset_train, batch_size=2048 * 4
        )

    loader_train = DataLoader(unaugmented, batch_size=2048 * 4, num_workers=2, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=2048 * 4, num_workers=2, pin_memory=True)

    evaluator = Evaluator(
        wandb,
        (5, DatasetEvaluator("Train", loader_train, nn.CrossEntropyLoss(reduction='none'))),
        (1, DatasetEvaluator("Test", loader_test, nn.CrossEntropyLoss(reduction='none')))
    )

    belief_tracker_test = BeliefMetrics(loader_test, interval=10)
    belief_tracker_train = BeliefMetrics(loader_train, interval=10)

    run = wandb.init(project='assist', entity='stektpotet', config=config)
    wandb.watch(model)

    trainer.on_start_training += lambda m: evaluator.evaluate(m, 0)
    trainer.on_start_training += lambda m: belief_tracker_test(m, 0)
    trainer.on_start_training += lambda m: belief_tracker_train(m, 0)
    trainer.on_end_epoch += evaluator.evaluate
    trainer.on_end_epoch += belief_tracker_test
    trainer.on_end_epoch += belief_tracker_train
    trainer.on_end_training += lambda m: wandb.finish()

    trainer.train(model, config['epochs'])
    belief_tracker_test.save_gif('runs/test_'+run.name)
    belief_tracker_train.save_gif('runs/train_'+run.name)
    belief_tracker_test.get_belief_dataframe().to_pickle('runs/test_'+run.name)
    belief_tracker_train.get_belief_dataframe().to_pickle('runs/train_'+run.name)

    belief_tracker_test.plot_classwise_distributions('runs/test_'+run.name)
    belief_tracker_train.plot_classwise_distributions('runs/train_'+run.name)
