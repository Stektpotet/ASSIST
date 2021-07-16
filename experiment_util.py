import string
from argparse import ArgumentParser, Namespace
import random
from typing import List, Type, Dict, TypeVar, Tuple, Generator, Generic, Union, Any, Optional

import numpy
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

# from Samplers import AccuracyFilteredSampler
from arg_utils import from_config
from pytorch_cifar import models3x32x32, models1x32x32
from trainers import AccumulativeAccuracyFilteringTrainer, ArchetypeTrainer, AccumulativeSoftmaxMarginFilteringTrainer, \
    AbstractTrainer

__TModel = TypeVar('__TModel', bound=nn.Module)
__TTrainer = TypeVar('__TTrainer', bound=AbstractTrainer)

def __models() -> Dict[str, __TModel]:
    return {
        'lenet_mnist': models1x32x32.LeNet,
        'vgg11_mnist': models1x32x32.VGG11,
        'vgg13_mnist': models1x32x32.VGG13,
        'vgg16_mnist': models1x32x32.VGG16,
        'vgg19_mnist': models1x32x32.VGG19,
        'resnet18_mnist': models1x32x32.ResNet18,
        'resnet34_mnist': models1x32x32.ResNet34,
        'resnet50_mnist': models1x32x32.ResNet50,
        'resnet101_mnist': models1x32x32.ResNet101,
        'resnet152_mnist': models1x32x32.ResNet152,
        'regnetx200mf_mnist': models1x32x32.RegNetX_200MF,
        'regnetx400mf_mnist': models1x32x32.RegNetX_400MF,
        'regnety400mf_mnist': models1x32x32.RegNetY_400MF,
        'mobilenet_mnist': models1x32x32.MobileNet,
        'mobilenetv2_mnist': models1x32x32.MobileNetV2,
        'resnext29_32x4_mnist': models1x32x32.ResNeXt29_32x4d,
        'resnext29_2x64_mnist': models1x32x32.ResNeXt29_2x64d,
        'resnext29_4x64_mnist': models1x32x32.ResNeXt29_4x64d,
        'resnext29_8x64_mnist': models1x32x32.ResNeXt29_8x64d,
        'simpledla_mnist': models1x32x32.SimpleDLA,
        'densenet121_mnist': models1x32x32.DenseNet121,
        'densenet161_mnist': models1x32x32.DenseNet161,
        'densenet169_mnist': models1x32x32.DenseNet169,
        'densenet201_mnist': models1x32x32.DenseNet201,
        'preactresnet18_mnist': models1x32x32.PreActResNet18,
        'preactresnet34_mnist': models1x32x32.PreActResNet34,
        'preactresnet50_mnist': models1x32x32.PreActResNet50,
        'preactresnet101_mnist': models1x32x32.PreActResNet101,
        'preactresnet152_mnist': models1x32x32.PreActResNet152,
        'dpn26_mnist': models1x32x32.DPN26,
        'dpn92_mnist': models1x32x32.DPN92,
        'dla_mnist': models1x32x32.DLA,
        'efficientnetb0_mnist': models1x32x32.EfficientNetB0,
        'googlenet_mnist': models1x32x32.GoogLeNet,
        'pnasneta_mnist': models1x32x32.PNASNetA,
        'pnasnetb_mnist': models1x32x32.PNASNetB,
        'senet18_mnist': models1x32x32.SENet18,
        'fcnet100_mnist': models1x32x32.FCNet100,
        'fcnet1000_mnist': models1x32x32.FCNet1000,
        'fcnet3000_mnist': models1x32x32.FCNet3000,

        'lenet': models3x32x32.LeNet,
        'vgg11': models3x32x32.VGG11,
        'vgg13': models3x32x32.VGG13,
        'vgg16': models3x32x32.VGG16,
        'vgg19': models3x32x32.VGG19,
        'resnet18': models3x32x32.ResNet18,
        'resnet34': models3x32x32.ResNet34,
        'resnet50': models3x32x32.ResNet50,
        'resnet101': models3x32x32.ResNet101,
        'resnet152': models3x32x32.ResNet152,
        'regnetx200mf': models3x32x32.RegNetX_200MF,
        'regnetx400mf': models3x32x32.RegNetX_400MF,
        'regnety400mf': models3x32x32.RegNetY_400MF,
        'mobilenet': models3x32x32.MobileNet,
        'mobilenetv2': models3x32x32.MobileNetV2,
        'resnext29_32x4': models3x32x32.ResNeXt29_32x4d,
        'resnext29_2x64': models3x32x32.ResNeXt29_2x64d,
        'resnext29_4x64': models3x32x32.ResNeXt29_4x64d,
        'resnext29_8x64': models3x32x32.ResNeXt29_8x64d,
        'simpledla': models3x32x32.SimpleDLA,
        'densenet121': models3x32x32.DenseNet121,
        'densenet161': models3x32x32.DenseNet161,
        'densenet169': models3x32x32.DenseNet169,
        'densenet201': models3x32x32.DenseNet201,
        'preactresnet18': models3x32x32.PreActResNet18,
        'preactresnet34': models3x32x32.PreActResNet34,
        'preactresnet50': models3x32x32.PreActResNet50,
        'preactresnet101': models3x32x32.PreActResNet101,
        'preactresnet152': models3x32x32.PreActResNet152,
        'dpn26': models3x32x32.DPN26,
        'dpn92': models3x32x32.DPN92,
        'dla': models3x32x32.DLA,
        'efficientnetb0': models3x32x32.EfficientNetB0,
        'googlenet': models3x32x32.GoogLeNet,
        'pnasneta': models3x32x32.PNASNetA,
        'pnasnetb': models3x32x32.PNASNetB,
        'senet18': models3x32x32.SENet18,
        'fcnet100': models3x32x32.FCNet100,
        'fcnet1000': models3x32x32.FCNet1000,
        'fcnet3000': models3x32x32.FCNet3000,
    }


def models_3x32x32() -> Tuple[str, ...]:
    """
    Models that can take input shape 3x32x32 and return shape 10 (CIFAR10-like datasets)
    :return:
    """
    return (
        'lenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'regnetx200mf', 'regnetx400mf', 'regnety400mf', 'mobilenet', 'mobilenetv2', 'resnext29_32x4',
        'resnext29_2x64', 'resnext29_4x64', 'resnext29_8x64', 'simpledla', 'densenet121', 'densenet161',
        'densenet169', 'densenet201', 'preactresnet18', 'preactresnet34', 'preactresnet50', 'preactresnet101',
        'preactresnet152', 'dpn26', 'dpn92', 'dla', 'efficientnetb0', 'googlenet', 'pnasneta', 'pnasnetb', 'senet18',
        'fcnet100', 'fcnet1000', 'fcnet3000',
    )


def models_1x32x32() -> Tuple[str, ...]:
    """
    Models that can take input shape 1x32x32 and return shape 10 (MNIST-like datasets)
    :return:
    """
    return (
        'lenet_mnist', 'vgg11_mnist', 'vgg13_mnist', 'vgg16_mnist', 'vgg19_mnist', 'resnet18_mnist', 'resnet34_mnist',
        'resnet50_mnist', 'resnet101_mnist', 'resnet152_mnist', 'regnetx200mf_mnist', 'regnetx400mf_mnist',
        'regnety400mf_mnist', 'mobilenet_mnist', 'mobilenetv2_mnist', 'resnext29_32x4_mnist', 'resnext29_2x64_mnist',
        'resnext29_4x64_mnist', 'resnext29_8x64_mnist', 'simpledla_mnist', 'densenet121_mnist', 'densenet161_mnist',
        'densenet169_mnist', 'densenet201_mnist', 'preactresnet18_mnist', 'preactresnet34_mnist',
        'preactresnet50_mnist', 'preactresnet101_mnist', 'preactresnet152_mnist', 'dpn26_mnist', 'dpn92_mnist',
        'dla_mnist', 'efficientnetb0_mnist', 'googlenet_mnist', 'pnasneta_mnist', 'pnasnetb_mnist', 'senet18_mnist',
        'fcnet100_mnist', 'fcnet1000_mnist', 'fcnet3000_mnist',
    )


def __trainers() -> Dict[str, Type[Union[ArchetypeTrainer,
                                         AccumulativeAccuracyFilteringTrainer,
                                         AccumulativeSoftmaxMarginFilteringTrainer]]]:
    return {
        'archetype': ArchetypeTrainer,
        'accacc': AccumulativeAccuracyFilteringTrainer,
        'qmargin': AccumulativeSoftmaxMarginFilteringTrainer,
    }


def __optimizers() -> Dict[str, Type[torch.optim.Optimizer]]:
    return {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
        'adamax': torch.optim.Adamax,
        'adagrad': torch.optim.Adagrad,
        'adamw': torch.optim.AdamW,
        'asgd': torch.optim.ASGD,
        'lbfgs': torch.optim.LBFGS,
        'rmsprop': torch.optim.RMSprop,
        'rprop': torch.optim.Rprop,
        'sparseadam': torch.optim.SparseAdam
    }

class NOPScheduler:
    # TODO: make this follow the full interface of _LRScheduler?
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        pass
    def step(self):
        pass

def __schedulers() -> Dict[str, Optional[Type[Union[MultiStepLR, ExponentialLR]]]]:
    return {
        'none': NOPScheduler,
        'multistep': torch.optim.lr_scheduler.MultiStepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
    }


def experiment_argparse(parser: ArgumentParser, models: Tuple[str]):
    optimizers = __optimizers()
    trainers = __trainers()
    schedulers = __schedulers()

    parser.add_argument('--seed', default=0, type=int, help='Seed for random generation (used in models3x32x32)')
    parser.add_argument('-e', '--num_epochs', default=300, type=int, help='number of epochs to train for')
    parser.add_argument('-bs', '--batch_size', default=256, type=int, help='number of samples to put in a batch')
    parser.add_argument('-qs', '--quantiles', default=[0.5, 0.9, 0.99], type=float, nargs='*',
                        help='at which quantiles should the loss additionally be plotted')
    parser.add_argument('-o', '--optimizer', default='sgd', type=str, choices=optimizers.keys(),
                        help='Which optimizer to use')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='optimizer learning rate')
    parser.add_argument('--momentum', default=0, type=float, help='optimizer momentum')
    parser.add_argument('--weight_decay', default=0, type=float, help='optimizer weight decay')
    parser.add_argument('-t', '--trainer', default='archetype', type=str, choices=trainers.keys(),
                        help='which trainer to use')
    parser.add_argument('--name', default=parser.description, type=str, help="name of the experiment")
    parser.add_argument('--tags', default=None, type=str, help="tags of the experiment", nargs='+')

    parser.add_argument('-m', '--model', default=models[0], type=str, choices=models, help='model type to train')
    parser.add_argument('-nw', '--num_workers', default=0, type=int, help='number of workers to use in dataloading')
    parser.add_argument('-nc', '--num_classes', default=10, type=int, choices=(10, 100),
                        help='number of classes in the dataset')
    parser.add_argument("-afs", "--accuracy_filtering_sampler", help="use accuracy filtering sampler (not reccomended when augmenting data)",
                        action="store_true")

    parser.add_argument("-abs", "--accumulation_batch_size", default=None, type=int, help="how big should the raw batch be when accumulating filtered batches (when using accacc-trainer)")

    parser.add_argument('--scheduler', default='none', choices=schedulers.keys())
    parser.add_argument('-ms', '--milestones', default=[150, 250], type=int, nargs='*',
                        help='at what milestones should the learning rate scheduler kick in')
    parser.add_argument('--gamma', default=0.1, type=float,
                                 help='what scaling factor should the learning rate scheduler multiply by upon a milestone')

    parser.add_argument('--nesterov', dest='nesterov', action='store_true')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--amsgrad', dest='amsgrad', action='store_true')
    parser.set_defaults(amsgrad=False)


    parser.add_argument('--margin', default=0.4, type=float, help='inclusion margin, from top belief to belief of quantile element.')
    parser.add_argument('-q', '--q', default=1.0, type=float, help='the quantile-th element to compare belief to.')

    args = parser.parse_args()

    args.scheduler = schedulers[args.scheduler]
    args.model = __models()[args.model]
    args.optimizer = optimizers[args.optimizer]
    args.trainer = trainers[args.trainer]
    return args

def make_scheduler(scheduler_type: Type, optimizer, args: Union[Namespace, Dict[str, Any]]):
    if isinstance(args, Namespace):
        args = args.__dict__
    return from_config(scheduler_type, args, optimizer=optimizer)

def make_optimizer(optimizer_type: Type[torch.optim.Optimizer], model_parameters,
                   args: Union[Namespace, Dict[str, Any]], **kwargs) -> torch.optim.Optimizer:
    if isinstance(args, Namespace):
        args = args.__dict__

    return from_config(optimizer_type, args, params=model_parameters, lr=args['learning_rate'], **kwargs)


def make_model(model_type: Type[__TModel], args: Union[Namespace, Dict[str, Any]]) -> __TModel:
    if isinstance(args, Namespace):
        args = args.__dict__
    return _make_model(args['seed'], model_type, args['num_classes'])


def _make_model(seed: int, model_type: Type[__TModel], num_classes: int = 10) -> __TModel:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    m = model_type(num_classes=num_classes)
    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        m = m.cuda()
        print("Using CUDA!")
    return m


def make_trainer(trainer_type: Type[__TTrainer], model: nn.Module, training_dataset, args: Union[Namespace, Dict[str, Any]], **kwargs) -> __TTrainer:
    if isinstance(args, Namespace):
        args = args.__dict__

    optimizer = make_optimizer(args['optimizer'], model.parameters(), args, **kwargs)
    sampler = None
    if args.get('accuracy_filtering_sampler', False):
        print("ACCURACY FILTERING SAMPLER IS NOT USED!!!!")
    #     sampler = AccuracyFilteredSampler(training_dataset, model, nn.CrossEntropyLoss(reduction='none'))

    return from_config(
        trainer_type,
        args,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(reduction='none'),
        dataset=training_dataset,
        sampler=sampler,
        **kwargs
    )
