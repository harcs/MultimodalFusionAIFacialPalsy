from torch import nn
import fastervit

class FasterVitWrapper(nn.Module):
    def __init__(self):
        super(FasterVitWrapper, self).__init__()
        self.faster_vit = fastervit.create_model('faster_vit_0_224', 
                          pretrained=True,
                          model_path="weights/faster_vit_0.pth.tar")
        # self.faster_vit.head = nn.Linear(in_features=512, out_features=2, bias=True)
        # self.faster_vit.head = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 2),
        #     nn.Softmax(dim=1)
        # )
        # print(self.faster_vit)

    def forward(self, x):
        x = self.faster_vit(x)
        return x

    def freeze_weights(self):
        for param in self.faster_vit.parameters():
            param.requires_grad = False

        counter = 0
        for level in self.faster_vit.levels:
            for name, module in level.named_modules():
                for param in module.parameters():
                    param.requires_grad = True

        # for level in self.faster_vit.levels:
        #     if counter < 2:
        #         counter += 1
        #     else:
        #         print("freezing")
        #         for name, module in level.named_modules():
        #                 for param in module.parameters():
        #                     param.requires_grad = True

        print("Trainable layers in FasterViT:")
        for name, param in self.faster_vit.named_parameters():
            if param.requires_grad:
                print(name)
