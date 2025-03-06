from torch import nn
from torchvision import models
import timm

class FCN_ManualFeats(nn.Module):
    def __init__(self):
        super(FCN_ManualFeats, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 64), # Saved
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32), # Saved
            nn.ReLU(),
            nn.BatchNorm1d(32), # Saved
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        
        # Replace the last fully connected layer with our own
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512)
            # nn.Linear(512, 2),
            # nn.Softmax(dim=1)
    )
    
    def forward(self, x):
        x = self.resnet(x)
        return x

class MLPMixerWrapper(nn.Module):
    def __init__(self):
        super(MLPMixerWrapper, self).__init__()
        self.mlp_mixer = timm.create_model('mixer_b16_224.miil_in21k_ft_in1k', pretrained=True)
        self.mlp_mixer.head = nn.Linear(in_features=768, out_features=2, bias=True)

    def forward(self, x):
        x = self.mlp_mixer(x)
        return x

    def freeze_weights(self):
        # Freeze all layers
        for param in self.mlp_mixer.parameters():
            param.requires_grad = False

        # Unfreeze the last fully connected layer
        for param in self.mlp_mixer.head.parameters():
            param.requires_grad = True

class EarlyFusionModel(nn.Module):
    def __init__(self):
        super(EarlyFusionModel, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(571, 256),
            nn.Linear(827, 256),
            nn.BatchNorm1d(256),  # BatchNorm here
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),  # Dropout separately

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # BatchNorm here
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)