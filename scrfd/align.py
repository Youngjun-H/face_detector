
import numpy as np
import cv2
import os
import sys
from typing import Optional

# Import SCRFDInfer from inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from inference import SCRFDInfer
except ImportError:
    print("Could not import SCRFDInfer from inference.py. Make sure it is in the same directory.")
    sys.exit(1)

class FaceAligner:
    def __init__(self, crop_size=(112, 112)):
        self.crop_size = crop_size
        # Standard reference points for 112x112 face alignment (ArcFace/InsightFace style)
        self.reference_pts = np.array([
            [38.2946, 51.6963], # left eye
            [73.5318, 51.5014], # right eye
            [56.0252, 71.7366], # nose
            [41.5493, 92.3655], # left mouth
            [70.7299, 92.2041]  # right mouth
        ], dtype=np.float32)

    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        save_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Align face using landmarks."""
        if landmarks is None or len(landmarks) < 5:
            print("Invalid landmarks")
            return None

        facial_pts = landmarks[:5].astype(np.float32)
        
        # estimateAffinePartial2D provides a rigid transformation (rotation + scale + translation)
        tfm, _ = cv2.estimateAffinePartial2D(facial_pts, self.reference_pts)

        if tfm is None:
            print("Transformation estimation failed")
            return None

        aligned_face = cv2.warpAffine(
            image, tfm, (self.crop_size[0], self.crop_size[1]),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # User requested saving aligned face as an image (BGR usually for opencv save)
        # User code converts to RGB at the end: aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        # But cv2.imwrite expects BGR. 
        # I will keep the return as RGB (per user req) but save as BGR.

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            cv2.imwrite(save_path, aligned_face)
            print(f'Aligned face saved to: {save_path}')

        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        return aligned_face_rgb

def main():
    # Settings
    img_path = '/data/yjhwang/face/scrfd/images/parkbogum.jpg' 
    model_path = '/data/yjhwang/face/scrfd/scrfd/scrfd_500m.pth'
    save_path = '/data/yjhwang/face/scrfd/aligned_face.png'


    print(f"Processing: {img_path}")

    # 1. Inference
    detector = SCRFDInfer(model_path)
    img = cv2.imread(img_path)
    
    # 0.5 threshold
    bboxes, kpss, det_scale = detector.forward(img, threshold=0.5)
    
    print(f"Detected {len(bboxes)} faces.")
    
    if len(bboxes) == 0:
        return

    # Scale back landmarks
    if len(bboxes) > 0:
        # bboxes = bboxes / det_scale # Not needed for alignment per se if we use the original image, 
        # BUT detector.forward returns coords on the RESIZED image if I recall correctly?
        # Let's check inference.py logic provided in previous turns.
        # "bboxes = bboxes / det_scale" was in main(). 
        # detector.forward returns final_bboxes, final_kpss (raw on 640x640?)
        # Ah, in the last inference.py, forward returns `return final_bboxes, final_kpss, det_scale`
        # And inside main: `bboxes = bboxes / det_scale`
        # So yes, we need to rescale.
        pass

    bboxes = bboxes / det_scale
    kpss = kpss / det_scale
        

    # 2. Select the largest face (area)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    max_idx = np.argmax(areas)
    
    bbox = bboxes[max_idx]
    best_kps = kpss[max_idx]
    best_kps = best_kps.reshape(-1, 2)
    
    # 3. Align
    aligner = FaceAligner(crop_size=(112, 112))
    # We pass None for save_path initially to get the RGB return for composition
    aligned_rgb = aligner.align_face(img, best_kps, save_path=None)
    
    if aligned_rgb is not None:
        # Create Simple Crop
        x1, y1, x2, y2 = bbox[:4].astype(int)
        # Clamp to image bounds
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        simple_crop = img[y1:y2, x1:x2]
        if simple_crop.size == 0:
            print("Crop failed, bbox outside image")
            return
            
        simple_crop_resized = cv2.resize(simple_crop, (112, 112))
        
        # Convert aligned to BGR for concatenation (since img is BGR)
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        
        # Stack horizontal: [Simple Crop, Aligned]
        combined = np.hstack([simple_crop_resized, aligned_bgr])
        
        # Save
        cv2.imwrite(save_path, combined)
        print(f"Saved comparison to: {save_path} (Left: Simple Crop, Right: Aligned)")
        
    else:
        print("Alignment failed")

if __name__ == "__main__":
    main()
