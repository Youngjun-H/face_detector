from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic Blocks
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Backbone용: 인덱스 기반 키 (체크포인트: 0, 1, 3, 4)
class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


# Head용: 서브모듈 기반 키 (체크포인트: depthwise_conv, pointwise_conv)
class DepthwiseSeparableHead(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        # depthwise_conv: Conv + BN (체크포인트 구조에 맞춤)
        # 체크포인트 키: depthwise_conv.conv.weight, depthwise_conv.bn.weight
        self.depthwise_conv = nn.Module()
        self.depthwise_conv.conv = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.depthwise_conv.bn = nn.BatchNorm2d(in_ch)
        
        # pointwise_conv: Conv + BN (체크포인트 구조에 맞춤)
        # 체크포인트 키: pointwise_conv.conv.weight, pointwise_conv.bn.weight
        self.pointwise_conv = nn.Module()
        self.pointwise_conv.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pointwise_conv.bn = nn.BatchNorm2d(out_ch)
        
        self.act = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise_conv.conv(x)
        x = self.depthwise_conv.bn(x)
        x = self.act(x)
        x = self.pointwise_conv.conv(x)
        x = self.pointwise_conv.bn(x)
        x = self.act(x)
        return x


# =========================================================
# Backbone (MobileNetV1 – SCRFD-500M)
# =========================================================
class MobileNetV1Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU6(inplace=True),
            ),
            DepthwiseSeparable(16, 16, 1),
        )

        self.layer1 = self._make_stage(16, 40, blocks=2, stride=1)
        self.layer2 = self._make_stage(40, 72, blocks=3, stride=2)
        self.layer3 = self._make_stage(72, 152, blocks=2, stride=2)
        self.layer4 = self._make_stage(152, 288, blocks=6, stride=2)

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = [DepthwiseSeparable(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(DepthwiseSeparable(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)   # 40
        f2 = self.layer2(f1)  # 72
        f3 = self.layer3(f2)  # 152
        f4 = self.layer4(f3)  # 288
        return [f1, f2, f3, f4]


# =========================================================
# Neck (PAFPN)
# =========================================================
class ConvWrap(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=True)

    def forward(self, x):
        return self.conv(x)


class PAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvWrap(72, 16, 1, 1, 0),
            ConvWrap(152, 16, 1, 1, 0),
            ConvWrap(288, 16, 1, 1, 0),
        ])
        self.fpn_convs = nn.ModuleList([
            ConvWrap(16, 16, 3, 1, 1),
            ConvWrap(16, 16, 3, 1, 1),
            ConvWrap(16, 16, 3, 1, 1),
        ])
        self.downsample_convs = nn.ModuleList([
            ConvWrap(16, 16, 3, 2, 1),
            ConvWrap(16, 16, 3, 2, 1),
        ])
        self.pafpn_convs = nn.ModuleList([
            ConvWrap(16, 16, 3, 1, 1),
            ConvWrap(16, 16, 3, 1, 1),
        ])

    def forward(self, feats):
        feats = feats[1:]  # [72,152,288]

        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]

        for i in range(2, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")

        outs = [c(l) for c, l in zip(self.fpn_convs, laterals)]

        for i in range(2):
            outs[i + 1] += self.downsample_convs[i](outs[i])
            outs[i + 1] = self.pafpn_convs[i](outs[i + 1])

        return outs  # stride 8,16,32


# =========================================================
# Integral Module (for DFL)
# =========================================================
class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution."""
    def __init__(self, reg_max=1):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result."""
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


# =========================================================
# SCRFD Head (DFL, anchor=2)
# =========================================================
class SCRFDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.reg_max = 1  # DFL (체크포인트 확인: stride_reg 채널 8 = 4 * (reg_max + 1) = 4 * 2)

        self.integral = Integral(self.reg_max)
        self.cls_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        self.stride_kps = nn.ModuleDict()

        for s in self.strides:
            key = f"({s}, {s})"

            self.cls_stride_convs[key] = nn.Sequential(
                DepthwiseSeparableHead(16, 64, 1),
                DepthwiseSeparableHead(64, 64, 1),
            )

            # ✅ 정확한 채널 수
            self.stride_cls[key] = nn.Conv2d(64, 2, 3, 1, 1)
            self.stride_reg[key] = nn.Conv2d(64, 4 * (self.reg_max + 1), 3, 1, 1)
            self.stride_kps[key] = nn.Conv2d(64, 10 * (self.reg_max + 1), 3, 1, 1)

    def forward(self, feats):
        cls_outs, reg_outs, kps_outs = [], [], []

        for feat, s in zip(feats, self.strides):
            key = f"({s}, {s})"
            x = self.cls_stride_convs[key](feat)

            cls_outs.append(self.stride_cls[key](x))
            reg_outs.append(self.stride_reg[key](x))
            kps_outs.append(self.stride_kps[key](x))

        return cls_outs, reg_outs, kps_outs


# =========================================================
# Full Model
# =========================================================
class SCRFDKPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1Backbone()
        self.neck = PAFPN()
        self.bbox_head = SCRFDHead()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)


# =========================================================
# Checkpoint Loader
# =========================================================
def load_ckpt(model, path, strict=True):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    
    # integral.project 크기 조정
    # 체크포인트는 reg_max=8 (크기 9)이지만 모델은 reg_max=1 (크기 2) 사용
    if "bbox_head.integral.project" in sd:
        ckpt_project = sd["bbox_head.integral.project"]
        model_project = model.bbox_head.integral.project
        
        if ckpt_project.shape != model_project.shape:
            # 체크포인트의 처음 reg_max+1 개 값만 사용
            reg_max = model.bbox_head.reg_max
            sd["bbox_head.integral.project"] = ckpt_project[:reg_max + 1]
            print(f"[load_ckpt] integral.project 크기 조정: {ckpt_project.shape} -> {sd['bbox_head.integral.project'].shape}")
    
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[load_state_dict] strict={strict}")
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    if missing:
        print("Missing keys (first 10):", missing[:10])
    if unexpected:
        print("Unexpected keys (first 10):", unexpected[:10])
    return missing, unexpected


# =========================================================
# Quick Test
# =========================================================
if __name__ == "__main__":
    m = SCRFDKPS().eval()
    load_ckpt(m, "scrfd_500m.pth", strict=True)

    x = torch.randn(1, 3, 640, 640)
    cls, reg, kps = m(x)

    for i in range(3):
        print(
            i,
            cls[i].shape,   # [1,2,H,W]
            reg[i].shape,   # [1,8,H,W]
            kps[i].shape,   # [1,20,H,W]
        )
