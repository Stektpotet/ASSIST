from typing import Optional, List, Tuple

import torch
from torch import nn


class FullyConnectedModel(nn.Module):
    @staticmethod
    def _build_layers(size_in, size_out, hidden_layers=None) -> torch.nn.Sequential:
        classifier = torch.nn.Sequential(nn.Flatten())

        activation_fn_mapping = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid()
        }
        if not hidden_layers:
            hidden_layers = []
        layers = []
        n_in = size_in
        for i, (n, a) in enumerate(hidden_layers):
            layer = torch.nn.Linear(n_in, n)
            classifier.add_module(f"Layer#{i}", layer)
            layers.append(layer)

            n_in = n
            act = activation_fn_mapping.get(a, False)
            if act:
                classifier.add_module(f"activation{i}", act)

        output_layer = torch.nn.Linear(n_in, size_out)
        classifier.add_module("Output", output_layer)
        layers.append(output_layer)

        return classifier

    def __init__(self, input_size, output_size, hidden_layers: Optional[List[Tuple[int, str]]] = None):
        super().__init__()
        self._num_layers = 1 if hidden_layers is None else len(hidden_layers) + 1
        self.classifier = self._build_layers(input_size, output_size, hidden_layers)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)


    def save_model(self, savepath):
        torch.save(savepath, self.model.state_dict())

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)



def FCNet3000(num_classes: int = 10):
    return FullyConnectedModel(32*32, num_classes, [(3000, 'relu')])

def FCNet1000(num_classes: int = 10):
    return FullyConnectedModel(32*32, num_classes, [(1000, 'relu')])

def FCNet100(num_classes: int = 10):
    return FullyConnectedModel(32*32, num_classes, [(100, 'relu')])
