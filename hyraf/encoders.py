"""Modality encoders. Each returns a [B, D] feature vector."""
import torch
import torch.nn as nn
import torchvision.models as tvm


def _gray_resnet18(frozen_children=2):
    r = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad(): conv1.weight.copy_(r.conv1.weight.mean(dim=1, keepdim=True))
    feats = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
    for i, ch in enumerate(feats.children()):
        if i < frozen_children:
            for p in ch.parameters(): p.requires_grad = False
    return feats


# AMD: scanning-laser fundus image + OCT volume
class FundusEncoder(nn.Module):
    dim = 512

    def __init__(self):
        super().__init__(); self.features = _gray_resnet18(); self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                       # [B, 1, H, W]
        return self.pool(self.features(x)).flatten(1)


class OCTEncoder(nn.Module):
    dim = 512

    def __init__(self):
        super().__init__(); self.features = _gray_resnet18(); self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                       # [B, N, H, W]
        B, N, H, W = x.shape
        f = self.pool(self.features(x.reshape(B * N, 1, H, W))).flatten(1)
        return f.view(B, N, -1).mean(1)


# MIMIC: 12-lead ECG + chest X-ray
class _Block1d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False); self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm1d(cout); self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                         if stride != 1 or cin != cout else nn.Sequential())

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ECGEncoder(nn.Module):
    dim = 512

    def __init__(self, ckpt=None):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False); self.bn1 = nn.BatchNorm1d(64); self.relu = nn.ReLU(inplace=True)
        mk = lambda a, b, s: nn.Sequential(_Block1d(a, b, s), _Block1d(b, b, 1))
        self.layer1, self.layer2, self.layer3, self.layer4 = mk(64, 64, 1), mk(64, 128, 2), mk(128, 256, 2), mk(256, 512, 2)
        if ckpt:
            sd = torch.load(ckpt, map_location='cpu', weights_only=False)
            sd = {k[len('ecg_encoder.'):]: v for k, v in sd.items() if k.startswith('ecg_encoder.') and 'linear' not in k}
            self.load_state_dict(sd, strict=False)

    def forward(self, x):                       # [B, T, 12]
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x)))).mean(-1)


class CXREncoder(nn.Module):
    dim = 256

    def __init__(self):
        super().__init__()
        r = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.proj = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):                       # [B, 3, 224, 224], ImageNet-normalised
        return self.proj(self.body(x.float())).flatten(2).mean(-1)


# ADNI: 3-D MRI + clinical table + biospecimen
class MRIEncoder(nn.Module):
    dim = 512

    def __init__(self, mean=0.52635, std=0.24307):
        super().__init__(); self.mean, self.std = mean, std
        blk = lambda a, b: [nn.Conv3d(a, b, 3, stride=2, padding=1), nn.BatchNorm3d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blk(1, 32), *blk(32, 64), *blk(64, 128), *blk(128, 256), nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Linear(256, 512)

    def forward(self, x):                       # [B, 1, 91, 109, 91]
        return self.fc(self.features((x - self.mean) / self.std).flatten(1))


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=(0.3, 0.2), layers=2):
        super().__init__()
        seq, d = [], in_dim
        for i in range(layers):
            seq += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout[min(i, len(dropout) - 1)])]; d = hidden
        self.net = nn.Sequential(*seq, nn.Linear(d, out_dim)); self.dim = out_dim

    def forward(self, x):
        return self.net(x)


def ClinicalEncoder(in_dim=122):
    return TabularEncoder(in_dim, 256, 256, (0.3, 0.2), layers=2)


def BiospecimenEncoder(in_dim=9):
    return TabularEncoder(in_dim, 64, 64, (0.2,), layers=1)
