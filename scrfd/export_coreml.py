import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from scrfd.scrfd_kps_torch import SCRFDKPS, load_ckpt, DepthwiseSeparable, DepthwiseSeparableHead, ConvWrap
except ImportError:
    from scrfd_kps_torch import SCRFDKPS, load_ckpt, DepthwiseSeparable, DepthwiseSeparableHead, ConvWrap

import numpy as np
import coremltools as ct


# CoreML-safe versions of modules that avoid parentheses/commas in ModuleDict keys.

class CoreMLSCRFDHead(nn.Module):
    """SCRFDHead with CoreML-compatible keys (stride_8 instead of (8, 8))."""

    def __init__(self):
        super().__init__()
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.reg_max = 1

        self.integral_project = nn.Parameter(torch.linspace(0, 1, 2), requires_grad=False)

        self.cls_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        self.stride_kps = nn.ModuleDict()

        for s in self.strides:
            key = f"stride_{s}"
            self.cls_stride_convs[key] = nn.Sequential(
                DepthwiseSeparableHead(16, 64, 1),
                DepthwiseSeparableHead(64, 64, 1),
            )
            self.stride_cls[key] = nn.Conv2d(64, 2, 3, 1, 1)
            self.stride_reg[key] = nn.Conv2d(64, 4 * (self.reg_max + 1), 3, 1, 1)
            self.stride_kps[key] = nn.Conv2d(64, 10 * (self.reg_max + 1), 3, 1, 1)

    def forward(self, feats):
        cls_outs, reg_outs, kps_outs = [], [], []
        for feat, s in zip(feats, self.strides):
            key = f"stride_{s}"
            x = self.cls_stride_convs[key](feat)
            cls_outs.append(self.stride_cls[key](x))
            reg_outs.append(self.stride_reg[key](x))
            kps_outs.append(self.stride_kps[key](x))
        return cls_outs, reg_outs, kps_outs


class CoreMLSCRFDKPS(nn.Module):
    """Full SCRFD model with CoreML-compatible key names."""

    def __init__(self):
        super().__init__()
        # Backbone (MobileNetV1)
        self.stem = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU6(inplace=True),
            ),
            DepthwiseSeparable(16, 16, 1),
        )
        self.layer1 = self._make_stage(16, 40, blocks=2, stride=2)
        self.layer2 = self._make_stage(40, 72, blocks=3, stride=2)
        self.layer3 = self._make_stage(72, 152, blocks=2, stride=2)
        self.layer4 = self._make_stage(152, 288, blocks=6, stride=2)

        # Neck (PAFPN)
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

        # Head
        self.head = CoreMLSCRFDHead()

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = [DepthwiseSeparable(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(DepthwiseSeparable(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Neck
        # Fixed sizes for 640x640 input (layer1 stride=2): f2=80x80, f3=40x40, f4=20x20
        feats = [f2, f3, f4]
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]
        # Top-down: upsample and merge (use fixed sizes to avoid dynamic shape ops)
        upsample_sizes = [(80, 80), (40, 40)]  # targets for laterals[1]->laterals[0], laterals[2]->laterals[1]
        laterals[1] = laterals[1] + F.interpolate(laterals[2], size=upsample_sizes[1], mode="nearest")
        laterals[0] = laterals[0] + F.interpolate(laterals[1], size=upsample_sizes[0], mode="nearest")
        outs = [c(l) for c, l in zip(self.fpn_convs, laterals)]
        for i in range(2):
            outs[i + 1] = outs[i + 1] + self.downsample_convs[i](outs[i])
            outs[i + 1] = self.pafpn_convs[i](outs[i + 1])

        # Head
        cls_outs, reg_outs, kps_outs = self.head(outs)

        # Post-process: flatten and concat all strides
        # Input 640x640, stem stride=2 -> 320x320
        # layer1 stride=2 -> 160x160, layer2 stride=2 -> 80x80,
        # layer3 stride=2 -> 40x40, layer4 stride=2 -> 20x20
        # Neck takes [f2,f3,f4] -> feature maps: 80x80, 40x40, 20x20
        # num_anchors=2, so anchors per stride: 80*80*2=12800, 40*40*2=3200, 20*20*2=800
        anchor_counts = [12800, 3200, 800]

        scores, bboxes, keypoints = [], [], []
        for i in range(3):
            n = anchor_counts[i]

            s = torch.sigmoid(cls_outs[i])
            s = s.permute(0, 2, 3, 1).contiguous()
            s = s.reshape(1, n, 1)
            scores.append(s)

            b = reg_outs[i].permute(0, 2, 3, 1).contiguous()
            b = b.reshape(1, n, 4)
            bboxes.append(b)

            k = kps_outs[i].permute(0, 2, 3, 1).contiguous()
            k = k.reshape(1, n, 10)
            keypoints.append(k)

        return (
            torch.cat(scores, dim=1),
            torch.cat(bboxes, dim=1),
            torch.cat(keypoints, dim=1),
        )


def _transfer_weights(src: SCRFDKPS, dst: CoreMLSCRFDKPS):
    """Copy weights from original model (with '(8, 8)' keys) to CoreML model (with 'stride_8' keys)."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    # Build mapping: original key -> coreml key
    mapping = {}
    for src_key in src_sd:
        dst_key = src_key

        # Flatten backbone path: backbone.stem -> stem, backbone.layerN -> layerN
        dst_key = dst_key.replace("backbone.", "")

        # Flatten neck path: neck.X -> X
        dst_key = dst_key.replace("neck.", "")

        # Flatten head path: bbox_head.X -> head.X
        dst_key = dst_key.replace("bbox_head.", "head.")

        # Replace key format: (8, 8) -> stride_8, etc.
        for s in [8, 16, 32]:
            dst_key = dst_key.replace(f"({s}, {s})", f"stride_{s}")

        # integral.project -> integral_project
        dst_key = dst_key.replace("integral.project", "integral_project")

        mapping[src_key] = dst_key

    new_sd = {}
    for src_key, dst_key in mapping.items():
        if dst_key in dst_sd:
            new_sd[dst_key] = src_sd[src_key]
        else:
            print(f"  Warning: no match for {src_key} -> {dst_key}")

    missing = set(dst_sd.keys()) - set(new_sd.keys())
    if missing:
        print(f"  Missing keys in destination: {missing}")

    dst.load_state_dict(new_sd, strict=True)
    print("  Weight transfer complete.")


def export_coreml():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scrfd_500m_kps.pth")
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_dir, "scrfd_500m_kps.mlpackage")

    input_size = 640

    # Load weights into CoreML-safe model
    # Use SCRFDKPS (stride=1) only for weight extraction, architecture uses stride=2 (matching training)
    src_model = SCRFDKPS()
    load_ckpt(src_model, model_path, strict=True)

    model = CoreMLSCRFDKPS()
    print("Transferring weights...")
    _transfer_weights(src_model, model)
    model.eval()
    del src_model

    # Verify output shapes
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        scores, bboxes, kps = model(dummy)
        total_anchors = 12800 + 3200 + 800  # 16800
        assert scores.shape == (1, total_anchors, 1), f"Unexpected scores shape: {scores.shape}"
        assert bboxes.shape == (1, total_anchors, 4), f"Unexpected bboxes shape: {bboxes.shape}"
        assert kps.shape == (1, total_anchors, 10), f"Unexpected kps shape: {kps.shape}"
        print(f"  Output shapes OK: scores={scores.shape}, bboxes={bboxes.shape}, kps={kps.shape}")

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=1.0 / 128.0,
                bias=[-127.5 / 128.0, -127.5 / 128.0, -127.5 / 128.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="scores", dtype=np.float32),
            ct.TensorType(name="bboxes", dtype=np.float32),
            ct.TensorType(name="keypoints", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.author = "SCRFD"
    mlmodel.short_description = "SCRFD-500M face detector with 5-point keypoints"
    mlmodel.input_description["image"] = "Input image (640x640 RGB)"
    mlmodel.output_description["scores"] = "Face confidence scores (N, 1)"
    mlmodel.output_description["bboxes"] = "Bounding box regressions (N, 4)"
    mlmodel.output_description["keypoints"] = "Facial keypoint regressions (N, 10)"

    mlmodel.save(output_path)
    print(f"Saved CoreML model to {output_path}")

    # Verify
    print("\nVerifying model...")
    spec = mlmodel.get_spec()
    for inp in spec.description.input:
        print(f"  Input: {inp.name}")
    for out in spec.description.output:
        print(f"  Output: {out.name}")


if __name__ == "__main__":
    export_coreml()
