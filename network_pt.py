# network module
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class CustomVisionTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True, weights=ViT_B_16_Weights.DEFAULT):
        super(CustomVisionTransformer, self).__init__()
        # Load the Vision Transformer model with specified weights
        if pretrained:
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = vit_b_16(weights=None)

        # Replace the classifier head with a new one for your dataset
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads[0].in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)