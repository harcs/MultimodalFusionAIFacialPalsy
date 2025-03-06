from torch import nn
import timm

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
