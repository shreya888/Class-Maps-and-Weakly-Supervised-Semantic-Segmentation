import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


class CAMModel(nn.Module):
    def __init__(self, args):
        super(CAMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        resnet18 = torchvision.models.resnet18(pretrained=True)
        layer2 = [module for module in resnet18.layer2.modules() if not isinstance(module, nn.Sequential)]
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', resnet18.conv1),
            ('bn1', resnet18.bn1),
            ('relu', resnet18.relu),
            ('maxpool', resnet18.maxpool),
            ('layer1', resnet18.layer1),
            ('layer2', layer2[0])]))

        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(128, 10)

    def forward(self, img_L, img_S=None):

        if self.args.mode == 'CAM':
            # FORWARD PATH FOR CAM
            f = self.backbone(img_L)
            assert list(f.shape)[1:] == [128, 28, 28]  # Sanity check
            w = self.gap(f)
            w_flat = torch.flatten(w, 1)
            out = self.cls(w_flat)
            return out, f, w_flat

        elif self.args.mode == 'SEG':
            # FORWARD PATH FOR SEG
            f_l = self.backbone(img_L)
            w_l = self.gap(f_l)
            w_l_flat = torch.flatten(w_l, 1)
            out_l = self.cls(w_l_flat)
            if img_S is not None:
                f_s = self.backbone(img_S)
                assert list(f_s.shape)[-2:] == [14, 14]  # Sanity check
                w_s = self.gap(f_s)
                w_s_flat = torch.flatten(w_s, 1)
                out_s = self.cls(w_s_flat)
                return out_l, f_l, w_l_flat, out_s, f_s, w_s_flat
            return out_l, f_l, w_l_flat

        else:
            NotImplementedError
