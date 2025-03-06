from torch import nn

class FCN_BlendshapeManualFeats(nn.Module):
    def __init__(self):
        super(FCN_BlendshapeManualFeats, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(52, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x