
import cv2
import os
import sys

# Import SCRFD class from scrfd_original.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrfd_original import SCRFD

def main():
    model_path = '/data/yjhwang/face/scrfd/scrfd_500m.onnx'
    img_path = '/data/yjhwang/face/scrfd/images/iu.png'
    save_path = '/data/yjhwang/face/scrfd/result_onnx_verify.png'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Initialize SCRFD with the new ONNX model
    detector = SCRFD(model_path)
    detector.prepare(0) 
    detector.det_thresh = 0.5 # Set threshold attribute directly

    img = cv2.imread(img_path)
    
    # Run detection (no threshold in call)
    bboxes, kpss = detector.detect(img, input_size=(640, 640))
    
    print(f"Detected {len(bboxes)} faces.")
    
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(int)[0], bbox.astype(int)[1], bbox.astype(int)[2], bbox.astype(int)[3], bbox[4]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
                
    cv2.imwrite(save_path, img)
    print(f"Saved ONNX verification result to {save_path}")

if __name__ == "__main__":
    main()
