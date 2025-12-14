
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys

# =========================================================
# Corrected Model Definition
# =========================================================
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from scrfd.scrfd_kps_torch import DepthwiseSeparable, PAFPN, SCRFDHead, load_ckpt as original_load_ckpt
except ImportError:
    from scrfd_kps_torch import DepthwiseSeparable, PAFPN, SCRFDHead, load_ckpt as original_load_ckpt

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


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class SCRFDInfer:
    def __init__(self, model_file):
        self.model = CorrectedSCRFDKPS()
        original_load_ckpt(self.model, model_file, strict=False) 
        self.model.eval()
        self.center_cache = {}
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2 # Fixed to 2 based on scrfd_original analysis
        
    def forward(self, img, threshold=0.5):
        # Letterbox Resizing
        input_size = (640, 640)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
        
        with torch.no_grad():
            net_outs = self.model(torch.from_numpy(blob))
        
        scores_list = []
        bboxes_list = []
        kps_list = []

        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = net_outs[0][idx] 
            bbox_preds = net_outs[1][idx] 
            kps_preds = net_outs[2][idx] 
            
            # scores: [B, 2, H, W] -> [B, H, W, 2] -> [-1, 2] (NumAnchors=2)
            # bbox_preds: [B, 8, H, W] -> [B, H, W, 8] -> [-1, 8] -> [-1, 2, 4]
            # kps_preds: [B, 20, H, W] -> [B, H, W, 20] -> [-1, 20] -> [-1, 2, 10]
            
            h, w = bbox_preds.shape[2:]
            anchor_centers = self.get_anchors(h, w, stride) # [N, 2]
            
            # Expand anchors for 2 anchors per location
            # anchor_centers: [N, 2] -> [N, 2, 2] -> [-1, 2]
            anchor_centers = np.stack([anchor_centers]*self.num_anchors, axis=1).reshape(-1, 2)
            
            # Scores
            # Sigmoid is used for detection score
            cls_score = torch.sigmoid(scores).permute(0, 2, 3, 1).reshape(-1, self.num_anchors).detach().numpy()
            cls_score = cls_score.flatten() # [N*2]
            
            # BBox
            # Assuming Direct Regression (No DFL) based on scrfd_original.py
            reg_pred = bbox_preds.permute(0, 2, 3, 1).reshape(-1, self.num_anchors, 4).detach().numpy()
            reg_pred = reg_pred.reshape(-1, 4) # [N*2, 4]
            reg_pred = reg_pred * stride
            
            # KPS
            kps_pred = kps_preds.permute(0, 2, 3, 1).reshape(-1, self.num_anchors, 10).detach().numpy()
            kps_pred = kps_pred.reshape(-1, 10) # [N*2, 10]
            kps_pred = kps_pred * stride
            
            # Decode
            bboxes = distance2bbox(anchor_centers, reg_pred)
            kpss = distance2kps(anchor_centers, kps_pred)
            
            scores_list.append(cls_score)
            bboxes_list.append(bboxes)
            kps_list.append(kpss)
            
        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        kpss = np.concatenate(kps_list, axis=0)
        
        # Threshold
        keep_idxs = np.where(scores > threshold)[0]
        scores = scores[keep_idxs]
        bboxes = bboxes[keep_idxs]
        kpss = kpss[keep_idxs]
        
        # NMS
        dets = np.hstack((bboxes, scores[:, np.newaxis]))
        keep = nms(dets, 0.4)
        
        final_bboxes = dets[keep]
        final_kpss = kpss[keep]
        
        return final_bboxes, final_kpss, det_scale

    def get_anchors(self, h, w, stride):
        key = (h, w, stride)
        if key in self.center_cache:
            return self.center_cache[key]
        
        ys, xs = np.mgrid[:h, :w]
        center_y = (ys * stride).flatten()
        center_x = (xs * stride).flatten()
        centers = np.stack((center_x, center_y), axis=1)
        
        self.center_cache[key] = centers
        return centers

def main():
    img_path = '/data/yjhwang/face/scrfd/iu.png'
    model_path = '/data/yjhwang/face/scrfd/scrfd/scrfd_500m.pth'
    save_path = '/data/yjhwang/face/scrfd/result_iu2.png'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    infer = SCRFDInfer(model_path)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return

    bboxes, kpss, det_scale = infer.forward(img, threshold=0.5)
    
    print(f"Detected {len(bboxes)} faces.")
    
    if len(bboxes) > 0:
        bboxes = bboxes / det_scale
        kpss = kpss / det_scale
        
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        kps = kpss[i]
        
        score = bbox[4]
        x1, y1, x2, y2 = bbox[:4].astype(int)
        
        color = (0, 255, 0)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if kps is not None:
            for j in range(0, len(kps), 2):
                px, py = int(kps[j]), int(kps[j+1])
                cv2.circle(img, (px, py), 2, (0, 0, 255), -1)
                
    cv2.imwrite(save_path, img)
    print(f"Saved result to {save_path}")

if __name__ == "__main__":
    main()
