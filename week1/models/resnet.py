"""
ResNet for CIFAR-10
Implements the original He et al. (2016) architecture family:
  - ResNet-20  (n=3,  ~0.27M params)
  - ResNet-32  (n=5)
  - ResNet-44  (n=7)
  - ResNet-56  (n=9)
  - ResNet-110 (n=18)

Target: ResNet-20 → ~92-94% on CIFAR-10 (paper: 91.25%, modern training: 93-94%)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Basic building block
# ─────────────────────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Shortcut: option B — 1×1 conv projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ResNet for CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────
class ResNetCIFAR(nn.Module):
    """
    CIFAR-10 ResNet as per He et al. (2016).
    Architecture: 6n+2 layers with 3 stages of [16, 32, 64] filters.
    n=3 → ResNet-20, n=5 → ResNet-32, n=9 → ResNet-56, n=18 → ResNet-110
    """
    def __init__(self, n: int = 3, num_classes: int = 10,
                 block=BasicBlock, width_multiplier: float = 1.0):
        super().__init__()
        base_planes = [16, 32, 64]
        planes = [max(1, int(p * width_multiplier)) for p in base_planes]

        self.in_planes = planes[0]

        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes[0])

        self.layer1 = self._make_layer(block, planes[0], n, stride=1)
        self.layer2 = self._make_layer(block, planes[1], n, stride=2)
        self.layer3 = self._make_layer(block, planes[2], n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(planes[2] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def feature_maps(self, x: torch.Tensor):
        """Return intermediate feature maps (for proxy computation)."""
        feats = {}
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        feats["stem"] = out
        out = self.layer1(out);  feats["layer1"] = out
        out = self.layer2(out);  feats["layer2"] = out
        out = self.layer3(out);  feats["layer3"] = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feats["logits"] = self.fc(out)
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────────────
def resnet20(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """ResNet-20: 6*3+2=20 layers, ~0.27M parameters. Target: 94% on CIFAR-10."""
    return ResNetCIFAR(n=3, num_classes=num_classes, **kwargs)

def resnet32(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(n=5, num_classes=num_classes, **kwargs)

def resnet44(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(n=7, num_classes=num_classes, **kwargs)

def resnet56(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(n=9, num_classes=num_classes, **kwargs)

def resnet110(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(n=18, num_classes=num_classes, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# NAS-compatible cell-based architecture (for NAS-Bench-201 search space)
# ─────────────────────────────────────────────────────────────────────────────
class NASCell(nn.Module):
    """
    A simple cell abstraction compatible with the NAS-Bench-201 op set:
    {'none', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3'}
    Used for building searchable architectures in Week 2+.
    """
    OPS = {
        "none":         lambda C, s: ZeroOp(stride=s),
        "skip_connect": lambda C, s: nn.Identity() if s == 1 else FactorizedReduce(C, C),
        "conv_1x1":     lambda C, s: ConvBNReLU(C, C, 1, s, 0),
        "conv_3x3":     lambda C, s: ConvBNReLU(C, C, 3, s, 1),
        "avg_pool_3x3": lambda C, s: AvgPoolOp(s),
    }

    def __init__(self, op_name: str, C: int, stride: int = 1):
        super().__init__()
        if op_name not in self.OPS:
            raise ValueError(f"Unknown op: {op_name}. Valid: {list(self.OPS)}")
        self.op = self.OPS[op_name](C, stride)

    def forward(self, x):
        return self.op(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, C_in, C_out, k, s, p):
        super().__init__(
            nn.Conv2d(C_in, C_out, k, s, p, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )


class ZeroOp(nn.Module):
    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, 2, 0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, 2, 0, bias=False)
        self.bn    = nn.BatchNorm2d(C_out)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return F.relu(self.bn(out), inplace=True)


class AvgPoolOp(nn.Module):
    def __init__(self, stride: int = 1):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


if __name__ == "__main__":
    model = resnet20()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"ResNet-20 | params: {model.count_parameters():,} | output: {y.shape}")
    # Expected: ~270,000 params, output shape (2, 10)
