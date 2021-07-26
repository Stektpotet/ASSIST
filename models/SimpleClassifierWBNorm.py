from torch import nn


class SimpleClassifierWBNorm(nn.Module):
    def __init__(self, num_classes: int = 23):
        super(SimpleClassifierWBNorm, self).__init__()
        self.__num_classes = num_classes
        self.base_net = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=5*5*32, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.__num_classes)
        ])

    def forward(self, x):
        for layer in self.base_net:
            x = layer(x)
        return x
