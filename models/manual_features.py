from torch import nn

class FCN_ManualFeats(nn.Module):
    def __init__(self):
        super(FCN_ManualFeats, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(29, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 2) # Saved
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class FCN_ManualFeats_Early_Fusion(nn.Module):
    def __init__(self):
        super(FCN_ManualFeats_Early_Fusion, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(29, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59),
            nn.Linear(59, 59), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(59)
        )

    def forward(self, x):
        x = self.layers(x)
        return x