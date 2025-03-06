from torch import nn

class FCN_Blendshapes(nn.Module):
    def __init__(self):
        super(FCN_Blendshapes, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(52, 64), # Saved
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(32), # Saved
            nn.Linear(32, 10), # Saved
            nn.ReLU(),
            nn.Linear(10, 2), # Saved
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x