# CoreML SCRFD Face Detector

SCRFD-500M 모델을 CoreML로 변환하고 macOS에서 추론하는 방법.

## 1. CoreML 모델 변환

```bash
# 프로젝트 루트에서 실행
.venv/bin/python scrfd/export_coreml.py
```

`scrfd_500m_kps.mlpackage`가 프로젝트 루트에 생성됨.

## 2. 빌드

```bash
clang -fobjc-arc \
  -framework Foundation -framework CoreML -framework AppKit \
  -framework CoreImage -framework ImageIO -framework UniformTypeIdentifiers \
  -framework CoreVideo \
  -o scrfd_detect coreml/main.m coreml/SCRFDInfer.m
```

## 3. 추론

```bash
./scrfd_detect <모델경로> <입력이미지> <출력이미지> [threshold]
```

```bash
# 예시
./scrfd_detect scrfd_500m_kps.mlpackage iu.png result.png 0.5
```

- `threshold` 생략 시 기본값 0.5
- 출력 이미지에 bbox(초록)와 keypoints(빨강)가 표시됨
