import argparse

from typing import Tuple, Dict, Any, Iterator, List

import torch
import torchvision.transforms as T
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from util.batching_util import default_cifar_augmentation, augment_batch, DataLoaderOnDevice
from datasets.catchsnap import CatchSnap
from datasets.modified_augmentations import ColorJitterExcludingMask, NormalizeExcludingMask
from util.experiment_util import make_model, experiment_argparse, models_3x32x32
from models.SimpleClassifierWBNorm import SimpleClassifierWBNorm

from evaluation import Evaluator, DatasetEvaluator
import wandb


def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser(description='CIFAR10 Experiment')
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

    if config['scheduler'] != 'MultiStepLR':
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
    transform = T.Compose([
        T.ToTensor(),
    ])

    dataset_test = CatchSnap(train=False, transform=transform)
    dataset_train = CatchSnap(transform=transform)
    # dataset_test = CIFAR10("./data/", train=False, download=True, transform=transform)
    # dataset_train = CIFAR10("./data/", download=True, transform=transform)
    return dataset_train, dataset_test

def qmargin_accumulate(loader: DataLoader, model: nn.Module, batch_size: int,
                       augmentation: nn.Module = default_cifar_augmentation,
                       q: float = 1.0, margin: float = 0.4) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        accumulated_data: List[torch.Tensor] = []
        accumulated_labels: List[torch.Tensor] = []
        current_accumulation = 0
        samples_left = len(loader.dataset)

        model.eval()
        for batch in loader:
            x, y = augment_batch(batch, augmentation)
            with torch.no_grad():
                model_output = model(x)
                beliefs = torch.softmax(model_output, dim=1)
                sorted_beliefs = torch.sort(beliefs, dim=1).values
                # sorted_beliefs_idx = torch.argsort(beliefs, dim=1)
                top = sorted_beliefs[:, -1]
                # top_idx = sorted_beliefs_idx[:, -1]  # the highest belief index per sample

                num_classes = len(beliefs[0]) - 1
                # get the class at q between [0..n-1] of the n belief-sorted classes
                q_class = int(q * (num_classes - 1) + 0.5)
                # q_idx = sorted_beliefs_idx[:, q_class]
                qs = sorted_beliefs[:, q_class]
                decision_distance = top - qs
                inside_margin = torch.nonzero(decision_distance < margin).view(-1)

                accumulated_data.append(x[inside_margin])
                accumulated_labels.append(y[inside_margin])
                current_accumulation += len(inside_margin)

            if current_accumulation >= batch_size:
                model.train()
                yield torch.cat(accumulated_data)[:batch_size], torch.cat(accumulated_labels)[:batch_size]
                model.eval()
                samples_left -= current_accumulation

                current_accumulation = 0
                accumulated_data.clear()
                accumulated_labels.clear()

                if samples_left < batch_size:
                    model.train()
                    return
        model.train()


if __name__ == '__main__':
    args = parse_args()
    args.batch_size = 48
    args.num_classes = 23
    args.num_epochs = 400
    args.num_workers = 4

    dataset_train, dataset_test = prepare_datasets()

    model = make_model(SimpleClassifierWBNorm, args)
    device = next(model.parameters()).device

    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.99)
    loss_fn = nn.CrossEntropyLoss()
    # trainer = make_trainer(args.trainer, model, dataset_train, args)
    # scheduler = make_scheduler(args.scheduler, trainer.optimizer, args)


    train_loader = DataLoaderOnDevice(dataset_train, device, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoaderOnDevice(dataset_test, device, batch_size=16, num_workers=args.num_workers)

    evaluator = Evaluator(
        wandb,
        (5, DatasetEvaluator("Train", train_loader, nn.CrossEntropyLoss(reduction='none'))),
        (1, DatasetEvaluator("Test", loader_test, nn.CrossEntropyLoss(reduction='none')))
    )

    config = prepare_config(args)

    run = wandb.init(project='assist', entity='stektpotet', config=config, tags=config['tags'])
    wandb.watch(model)

    aug_transform = nn.Sequential(
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        ColorJitterExcludingMask(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        NormalizeExcludingMask((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )

    evaluator.evaluate(model, 0)
    for epoch in tqdm(range(1, config['num_epochs'] + 1)):
        for batch, (x, y) in enumerate(qmargin_accumulate(train_loader, model, config['batch_size'], aug_transform)):
            optimizer.zero_grad()
            model_output = model(x)
            losses = loss_fn(model_output, y)
            losses.backward()
            optimizer.step()

        scheduler.step()
        evaluator.evaluate(model, epoch)

    wandb.finish()



    # test_fig = belief_tracker_test.get_classwise_distributions_figure()
    # train_fig = belief_tracker_train.get_classwise_distributions_figure()
    #
    # wandb.log({'beliefs/test distribution': wandb.Image(test_fig, caption="Test belief distribution")})
    # wandb.log({'beliefs/train distribution': wandb.Image(train_fig, caption="Train belief distribution")})
    #
    # save_path = os.path.join(wandb.run.dir, "media/images/beliefs")
    #
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #
    # belief_tracker_test.get_belief_dataframe().to_pickle(os.path.join(save_path, "test_beliefs.pkl"))
    # belief_tracker_train.get_belief_dataframe().to_pickle(os.path.join(save_path, "train_beliefs.pkl"))
    # belief_tracker_test.plot_classwise_distributions(os.path.join(save_path, "test_beliefs"))
    # belief_tracker_train.plot_classwise_distributions(os.path.join(save_path, "train_beliefs"))
    # belief_tracker_test.save_gif(os.path.join(save_path, "test_beliefs"))
    # belief_tracker_train.save_gif(os.path.join(save_path, "train_beliefs"))
    # wandb.save('media/images/beliefs/*')
    #
    # wandb.finish()


