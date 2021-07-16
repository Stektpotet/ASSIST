import argparse
import os

import pandas as pd
import inspect
import traceback
from datetime import datetime
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, softmax
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from eventsystem.trackable import Trackable
from experiment_util import make_model, make_trainer, make_scheduler, experiment_argparse, models_1x32x32, \
    models_3x32x32
from pytorch_cifar.models3x32x32 import LeNet, FCNet100

from evaluation import Evaluator, DatasetEvaluator
from trainers import AccumulativeAccuracyFilteringTrainer, ArchetypeTrainer, AccumulativeSoftmaxMarginFilteringTrainer
import wandb
from util.image_util import fig2img



def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser(description='MNIST Experiment')
    args = experiment_argparse(args_parser, models_3x32x32())
    return args

def prepare_config(args: argparse.Namespace, **kwargs: Dict[str, Any]):
    """
    Prepare the config for listing in wandb
    :param args: parsed command line arguments
    :return: clean dictionary of configuration used
    """

    remove_if_equal = {
        'quantiles': [0.5, 0.9, 0.99],
        'accuracy_filtering_sampler': False,
        'accumulation_batch_size': None,
        'scheduler': 'NOPScheduler',
        'nesterov': False,
        'amsgrad': False}


    config = {**args.__dict__, **kwargs}
    config["model"] = config.pop("model").__name__
    config["optimizer"] = config.pop("optimizer").__name__
    config["trainer"] = config.pop("trainer").__name__
    config["scheduler"] = config.pop("scheduler").__name__
    config["Tags"] = config.pop("tags")

    if config['scheduler'] != 'multistep':
        config.pop('milestones')
        if config['scheduler'] == 'NOPScheduler':
            config.pop('scheduler')
            config.pop('gamma')

    if config['optimizer'] != 'adam':
        config.pop('amsgrad')

    if config['trainer'] != 'AccumulativeSoftmaxMarginFilteringTrainer':
        config.pop('margin')
        config.pop('q')

    if config['optimizer'] != 'sgd':
        config.pop('nesterov')
        config.pop('momentum')

    sets_num_classes = {'CIFAR100': 100, 'CIFAR10': 10, 'MNIST': 10}
    if config['num_classes'] == sets_num_classes.get(config.get('dataset', None), 10):
        config.pop('num_classes')

    for k, v in remove_if_equal.items():
        if config.get(k, not v) == v:
            config.pop(k)
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
    def __init__(self, dataloader: DataLoader, num_classes: int = 10, interval: int = 1, gif_duration: float = 5):
        super().__init__(interval)
        self._interval = interval
        self._dataloader = dataloader
        self._num_classes = num_classes
        self._beliefs = torch.empty((len(self._dataloader.dataset), self._num_classes), requires_grad=False, device='cuda:0')
        self._labels = torch.empty(len(self._dataloader.dataset), dtype=torch.int, requires_grad=False, device='cuda:0')
        self._imgs = []
        self._duration = gif_duration

    @property
    def _ms_per_frame(self):
        return self._duration * (1000 / len(self._imgs))

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

    def get_classwise_distributions_figure(self):
        l, b = self._labels.detach().cpu().numpy(), self._beliefs.detach().cpu().numpy()
        df = pd.DataFrame({'label': l}).join(pd.DataFrame(b))
        plots = df.loc[:, df.columns != 'label'].groupby(df['label']).boxplot(layout=(2, 5), sharex=True,
                                                                              figsize=(22, 9), whis=(0, 100))
        return plots[0].get_figure()

    def update(self, model: nn.Module):
        with torch.no_grad():
            i = 0
            for batch, (x, y) in enumerate(self._dataloader):
                batch_beliefs = softmax(model(x.cuda()), 1)
                self._beliefs[i:i + len(batch_beliefs)] = batch_beliefs
                self._labels[i:i + len(y)] = y
                i += len(batch_beliefs)

    def plot_softmax_distribution(self):
        sorted_beliefs = torch.sort(self._beliefs, 1).values.detach().cpu()

        l_max, h_min = sorted_beliefs[:, 0].max(), sorted_beliefs[:, 9].min()

        plt.axhline(y=l_max, color="red", linestyle=(0, (5, 5)))
        plt.axhline(y=h_min, color="blue", linestyle=(5, (5, 5)))
        plt.boxplot(torch.transpose(sorted_beliefs, 0, 1), showfliers=False, whis=(0, 100), zorder=1)
        plt.ylabel("Belief Strength")
        plt.xlabel("Classes")
        plt.show()

    def save_gif(self, name):
        self._imgs[0].save(name + '.gif', save_all=True, append_images=self._imgs[1:], optimize=False, duration=self._ms_per_frame, loop=0)

    def __call__(self, model: nn.Module, epoch: int, *args, **kwargs):
        if epoch % self._interval != 0:
            return
        # self.plot_softmax_distribution(model)
        self.update(model)
        self._imgs.append(fig2img(self.get_classwise_distributions_figure()))
        # wandb.log({'Belief Strength': plt})


if __name__ == '__main__':
    args = parse_args()

    dataset_train, dataset_test, unaugmented = prepare_datasets()

    model = make_model(args.model, args)
    trainer = make_trainer(args.trainer, model, dataset_train, args)
    scheduler = make_scheduler(args.scheduler, trainer.optimizer, args)

    loader_train = DataLoader(unaugmented, batch_size=2048 * 4, num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=2048 * 4, num_workers=args.num_workers, pin_memory=True)

    evaluator = Evaluator(
        wandb,
        (5, DatasetEvaluator("Train", loader_train, nn.CrossEntropyLoss(reduction='none'))),
        (1, DatasetEvaluator("Test", loader_test, nn.CrossEntropyLoss(reduction='none')))
    )

    belief_tracker_test = BeliefMetrics(loader_test, interval=10)
    belief_tracker_train = BeliefMetrics(loader_train, interval=10)

    config = prepare_config(args)

    run = wandb.init(project='assist', entity='stektpotet', config=config)
    wandb.watch(model)

    trainer.on_start_training += lambda m: evaluator.evaluate(m, 0)
    trainer.on_start_training += lambda m: belief_tracker_test(m, 0)
    trainer.on_start_training += lambda m: belief_tracker_train(m, 0)
    trainer.on_end_epoch += evaluator.evaluate
    trainer.on_end_epoch += belief_tracker_test
    trainer.on_end_epoch += belief_tracker_train

    trainer.train(model, config['num_epochs'])

    test_fig = belief_tracker_test.get_classwise_distributions_figure()
    train_fig = belief_tracker_train.get_classwise_distributions_figure()

    wandb.log({'beliefs/test distribution': wandb.Image(test_fig, caption="Test belief distribution")})
    wandb.log({'beliefs/train distribution': wandb.Image(train_fig, caption="Train belief distribution")})

    save_path = os.path.join(wandb.run.dir, "beliefs")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    belief_tracker_test.get_belief_dataframe().to_pickle(os.path.join(save_path, "test_beliefs.pkl"))
    belief_tracker_train.get_belief_dataframe().to_pickle(os.path.join(save_path, "train_beliefs.pkl"))
    belief_tracker_test.plot_classwise_distributions(os.path.join(save_path, "test_beliefs"))
    belief_tracker_train.plot_classwise_distributions(os.path.join(save_path, "train_beliefs"))
    belief_tracker_test.save_gif(os.path.join(save_path, "test_beliefs"))
    belief_tracker_train.save_gif(os.path.join(save_path, "train_beliefs"))
    wandb.save('beliefs/*')

    wandb.finish()