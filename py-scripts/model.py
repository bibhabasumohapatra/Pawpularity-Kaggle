import torch
import torch.nn as nn
import timm

model_name = 'swin_base_patch4_window7_224_in22k'

out_dim    = 1

class get_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Sequential(nn.Linear(self.model.head.in_features,768),
                                              nn.Linear(768,256))
        self.last = nn.Linear(256 + 12, 128)
        self.depth1 = nn.Linear(128,64)
        self.depth2 = nn.Linear(64,out_dim)
    def forward(self, image, features):
        x = self.model(image)
        x = self.last(torch.cat([x, features], dim=1))
        x = self.depth1(x)
        x = self.depth2(x)
        return x
