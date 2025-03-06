from torch import nn

class FCNCoords(nn.Module):
    def __init__(self):
        super(FCNCoords, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class FullFCNCoords(nn.Module):
    def __init__(self):
        super(FullFCNCoords, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(478 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x