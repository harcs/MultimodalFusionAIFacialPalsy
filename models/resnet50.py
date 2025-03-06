from torch import nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        
        # Replace the last fully connected layer with our own
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
    )
    
    def forward(self, x):
        x = self.resnet(x)
        return x

    def freeze_weights(self):
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Unfreeze the last CNN layer
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

class ResNet50_Early_Fusion(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_Early_Fusion, self).__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        
        # Replace the last fully connected layer with our own
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512)
            # nn.Linear(512, num_classes)
    )
    
    def forward(self, x):
        x = self.resnet(x)
        return x

    def freeze_weights(self):
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Unfreeze the last CNN layer
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True