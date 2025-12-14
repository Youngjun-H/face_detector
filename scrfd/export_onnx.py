
import torch
import torch.nn as nn
import os
import sys

# Add path to import scrfd modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from scrfd.scrfd_kps_torch import DepthwiseSeparable, PAFPN, SCRFDHead, load_ckpt as original_load_ckpt
except ImportError:
    from scrfd_kps_torch import DepthwiseSeparable, PAFPN, SCRFDHead, load_ckpt as original_load_ckpt

# Define Corrected Classes exactly as used in inference.py
class CorrectedMobileNetV1Backbone(nn.Module):
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
        # Stride 2 fix
        self.layer1 = self._make_stage(16, 40, blocks=2, stride=2)
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
        f1 = self.layer1(x)   
        f2 = self.layer2(f1)  
        f3 = self.layer3(f2)  
        f4 = self.layer4(f3)  
        return [f1, f2, f3, f4]

class CorrectedSCRFDKPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CorrectedMobileNetV1Backbone()
        self.neck = PAFPN()
        self.bbox_head = SCRFDHead()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)

def export_onnx():
    model_path = '/data/yjhwang/face/scrfd/scrfd/scrfd_500m.pth'
    output_onnx = '/data/yjhwang/face/scrfd/scrfd_500m.onnx'
    
    # Initialize Model
    model = CorrectedSCRFDKPS()
    original_load_ckpt(model, model_path, strict=False)
    model.eval()
    
    # Dummy Input (1, 3, 640, 640)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Dynamic Axes
    # We want dynamic height and width, and potentially batch size
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride8_score': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride8_bbox': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride8_kps': {0: 'batch_size', 2: 'height', 3: 'width'},
        
        'output_stride16_score': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride16_bbox': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride16_kps': {0: 'batch_size', 2: 'height', 3: 'width'},
        
        'output_stride32_score': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride32_bbox': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output_stride32_kps': {0: 'batch_size', 2: 'height', 3: 'width'},
    }
    
    # Output names based on return order: (cls, reg, kps) list of 3 strides
    # The model returns (cls_outs, reg_outs, kps_outs) where each is a list of 3
    # Actually, SCRFDHead.forward returns (cls_outs, reg_outs, kps_outs) which are LISTS.
    # PyTorch ONNX export usually flattens lists of tensors.
    # The order will likely be:
    # cls_out[0], cls_out[1], cls_out[2], reg_out[0], reg_out[1], ...
    
    output_names = [
        'score_8', 'score_16', 'score_32',
        'bbox_8', 'bbox_16', 'bbox_32',
        'kps_8', 'kps_16', 'kps_32'
    ]
    
    # Wrapper to flatten outputs and apply sigmoid/permute matching InsightFace format
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            cls_out, reg_out, kps_out = self.m(x)
            
            # We need to process each stride output
            # cls_out is list of 3 tensors: (B, 2, H, W) -> need (B, H*W*2, 1)
            # reg_out is list of 3 tensors: (B, 8, H, W) -> need (B, H*W*2, 4)
            # kps_out is list of 3 tensors: (B, 20, H, W) -> need (B, H*W*2, 10)
            
            results = []
            
            # Stride 8
            # Score
            score_8 = torch.sigmoid(cls_out[0])
            score_8 = score_8[:, 0:1, :, :] # Take channel 0 (Face) -> (B, 1, H, W) ??
            # Wait, PyTorch model has 2 channels. 0 is Face? 
            # In inference.py: face_score = scores_sigmoid[:, 0]
            # If shape is (B, 2, H, W), slicing [:, 0] gives (B, H, W).
            # We want to keep dims for permute? 
            # Actually simplest is: Permute -> (B, H, W, 2) -> Reshape (B, -1, 2) -> Slice (..., 0) -> (B, -1, 1)
            
            s8 = torch.sigmoid(cls_out[0]) # (B, 2, H, W)
            s8 = s8.permute(0, 2, 3, 1).contiguous() # (B, H, W, 2)
            s8 = s8.view(s8.shape[0], -1, 2) # (B, H*W, 2). Wait, num_anchors=2?
            # No, correct logic:
            # Output channels 2. Num anchors 2.
            # Does channel 2 mean 1 score per anchor? Or 2 scores (fg/bg) per anchor?
            # SCRFDHead: self.stride_cls[key] = nn.Conv2d(64, 2, 3, 1, 1)
            # num_anchors=2.
            # Result 2 channels. This implies 1 channel per anchor (Face Score Only? Or Face/BG?)
            # If 1 channel per anchor, then channels=2 matches num_anchors=2.
            # In inference.py we did: scores_sigmoid = torch.sigmoid(scores).permute(0, 2, 3, 1).reshape(-1, 2)
            # And then face_score = scores_sigmoid[:, 0].
            # This implies we took the score of Anchor 0? 
            # Wait. If there are 2 anchors, we should have 2 scores.
            # If the conv output is 2 channels, and we treat it as 1 score per anchor...
            # Then we should keep BOTH channels?
            # In inference.py: `face_score = scores_sigmoid[:, 0]` -> This is WRONG if we have 2 anchors and channels reps anchors.
            # If 2 channels = 2 classes (BG, Face) for 1 anchor?
            # The conv definition: `self.stride_cls[key] = nn.Conv2d(64, 2, 3, 1, 1)`
            # And `self.num_anchors = 2`.
            # Usually it's `num_anchors * num_classes`.
            # If num_classes=1 (Face), then 2 * 1 = 2 channels. Correct.
            # So channel 0 is Score for Anchor 0. Channel 1 is Score for Anchor 1.
            # My inference.py code: `face_score = scores_sigmoid[:, 0]` <- This only took Anchor 0's score!
            # It discards Anchor 1. This might be why valid detections were lost or it worked partially.
            # If I want to export correctly, I should keep ALL anchor scores.
            # So: (B, 2, H, W) -> Permute (B, H, W, 2) -> Reshape (B, -1, 1)?
            # No, Reshape (B, H*W*2, 1).
            s8 = s8.view(s8.shape[0], -1, 1) # (B, H*W*2, 1)
            
            # BBox
            b8 = reg_out[0] # (B, 8, H, W)
            b8 = b8.permute(0, 2, 3, 1).contiguous() # (B, H, W, 8)
            b8 = b8.view(b8.shape[0], -1, 4) # (B, H*W*2, 4)
            
            # KPS
            k8 = kps_out[0] # (B, 20, H, W)
            k8 = k8.permute(0, 2, 3, 1).contiguous() # (B, H, W, 20)
            k8 = k8.view(k8.shape[0], -1, 10) # (B, H*W*2, 10)
            
            # Stride 16
            s16 = torch.sigmoid(cls_out[1])
            s16 = s16.permute(0, 2, 3, 1).contiguous().view(s16.shape[0], -1, 1)
            b16 = reg_out[1].permute(0, 2, 3, 1).contiguous().view(reg_out[1].shape[0], -1, 4)
            k16 = kps_out[1].permute(0, 2, 3, 1).contiguous().view(kps_out[1].shape[0], -1, 10)
            
            # Stride 32
            s32 = torch.sigmoid(cls_out[2])
            s32 = s32.permute(0, 2, 3, 1).contiguous().view(s32.shape[0], -1, 1)
            b32 = reg_out[2].permute(0, 2, 3, 1).contiguous().view(reg_out[2].shape[0], -1, 4)
            k32 = kps_out[2].permute(0, 2, 3, 1).contiguous().view(kps_out[2].shape[0], -1, 10)
            
            return (s8, s16, s32, b8, b16, b32, k8, k16, k32)
            
    wrapped_model = Wrapper(model)

    print(f"Exporting ONNX to {output_onnx}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_onnx,
        input_names=['input'],
        output_names=output_names,
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'score_8': {0: 'batch', 1: 'num_anchors'},
            'score_16': {0: 'batch', 1: 'num_anchors'},
            'score_32': {0: 'batch', 1: 'num_anchors'},
            'bbox_8': {0: 'batch', 1: 'num_anchors'},
            'bbox_16': {0: 'batch', 1: 'num_anchors'},
            'bbox_32': {0: 'batch', 1: 'num_anchors'},
            'kps_8': {0: 'batch', 1: 'num_anchors'},
            'kps_16': {0: 'batch', 1: 'num_anchors'},
            'kps_32': {0: 'batch', 1: 'num_anchors'},
        }
    )
    print("Success!")

if __name__ == "__main__":
    export_onnx()
