#import "SCRFDInfer.h"
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

// =========================================================
// Constants
// =========================================================
static const NSInteger kInputSize = 640;
static const NSInteger kNumAnchors = 2;
static const NSInteger kStrides[] = {8, 16, 32};
// Feature map sizes for 640x640 input (layer1 stride=2): 80, 40, 20
static const NSInteger kFeatSizes[] = {80, 40, 20};
// Anchor counts per stride: 80*80*2, 40*40*2, 20*20*2
static const NSInteger kAnchorCounts[] = {12800, 3200, 800};
static const NSInteger kTotalAnchors = 16800;

// =========================================================
// SCRFDFace
// =========================================================
@implementation SCRFDFace {
    CGPoint _keypoints[5];
}

- (CGPoint)keypointAtIndex:(NSUInteger)index {
    if (index >= 5) return CGPointZero;
    return _keypoints[index];
}

- (void)setKeypoint:(CGPoint)point atIndex:(NSUInteger)index {
    if (index < 5) _keypoints[index] = point;
}

@end

// =========================================================
// Internal detection struct for NMS
// =========================================================
typedef struct {
    float x1, y1, x2, y2, score;
    float kps[10];
} Detection;

// =========================================================
// SCRFDInfer
// =========================================================
@interface SCRFDInfer ()
@property (nonatomic, strong) MLModel *model;
@property (nonatomic) float *anchorCenters; // [kTotalAnchors * 2]
@end

@implementation SCRFDInfer

- (void)dealloc {
    if (_anchorCenters) {
        free(_anchorCenters);
    }
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (!self) return nil;

    NSError *error = nil;
    NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
    NSURL *compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
    if (error) {
        NSLog(@"Failed to compile model: %@", error);
        return nil;
    }

    _model = [MLModel modelWithContentsOfURL:compiledURL error:&error];
    if (error) {
        NSLog(@"Failed to load model: %@", error);
        return nil;
    }

    [self generateAnchors];
    return self;
}

// =========================================================
// Anchor Generation
// =========================================================
- (void)generateAnchors {
    _anchorCenters = (float *)malloc(kTotalAnchors * 2 * sizeof(float));

    NSInteger offset = 0;
    for (NSInteger s = 0; s < 3; s++) {
        NSInteger featH = kFeatSizes[s];
        NSInteger featW = kFeatSizes[s];
        NSInteger stride = kStrides[s];

        for (NSInteger y = 0; y < featH; y++) {
            for (NSInteger x = 0; x < featW; x++) {
                for (NSInteger a = 0; a < kNumAnchors; a++) {
                    _anchorCenters[(offset + a) * 2 + 0] = (float)(x * stride);
                    _anchorCenters[(offset + a) * 2 + 1] = (float)(y * stride);
                }
                offset += kNumAnchors;
            }
        }
    }
}

// =========================================================
// Letterbox + Pixel Buffer (combined, no y-flip issue)
// =========================================================
- (CVPixelBufferRef)createPixelBufferFromImage:(NSImage *)image scale:(float *)outScale {
    // Get actual pixel dimensions from CGImage (not NSImage.size which is in points)
    CGImageRef cgImage = [image CGImageForProposedRect:NULL context:nil hints:nil];
    if (!cgImage) {
        NSLog(@"Failed to get CGImage");
        return NULL;
    }

    size_t imgW = CGImageGetWidth(cgImage);
    size_t imgH = CGImageGetHeight(cgImage);

    // Letterbox: compute new size maintaining aspect ratio
    float imRatio = (float)imgH / (float)imgW;
    float newWidth, newHeight;
    if (imRatio > 1.0f) {
        newHeight = kInputSize;
        newWidth = (float)((int)(newHeight / imRatio));
    } else {
        newWidth = kInputSize;
        newHeight = (float)((int)(newWidth * imRatio));
    }
    *outScale = newHeight / (float)imgH;

    // Create pixel buffer
    CVPixelBufferRef pixelBuffer = NULL;
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
    };
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
                                          kInputSize, kInputSize,
                                          kCVPixelFormatType_32BGRA,
                                          (__bridge CFDictionaryRef)attrs,
                                          &pixelBuffer);
    if (status != kCVReturnSuccess) return NULL;

    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    void *data = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(data, kInputSize, kInputSize,
                                            8, bytesPerRow, colorSpace,
                                            kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);

    // Fill black (entire 640x640)
    CGContextSetRGBFillColor(ctx, 0, 0, 0, 1);
    CGContextFillRect(ctx, CGRectMake(0, 0, kInputSize, kInputSize));

    // Draw image at top-left in CG coords (CG origin = bottom-left, so place at top)
    CGContextDrawImage(ctx, CGRectMake(0, kInputSize - (int)newHeight, (int)newWidth, (int)newHeight), cgImage);

    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

    return pixelBuffer;
}

// =========================================================
// Read MLMultiArray as float32 (handles Float16 conversion)
// =========================================================
- (float *)float32FromMultiArray:(MLMultiArray *)array {
    NSInteger s0 = array.shape.count > 0 ? [array.shape[0] integerValue] : 1;
    NSInteger s1 = array.shape.count > 1 ? [array.shape[1] integerValue] : 1;
    NSInteger s2 = array.shape.count > 2 ? [array.shape[2] integerValue] : 1;
    
    NSInteger st0 = array.strides.count > 0 ? [array.strides[0] integerValue] : 1;
    NSInteger st1 = array.strides.count > 1 ? [array.strides[1] integerValue] : 1;
    NSInteger st2 = array.strides.count > 2 ? [array.strides[2] integerValue] : 1;

    NSInteger count = s0 * s1 * s2;
    float *result = (float *)malloc(count * sizeof(float));

    if (array.dataType == MLMultiArrayDataTypeFloat32) {
        float *ptr = (float *)array.dataPointer;
        NSInteger outIdx = 0;
        for (NSInteger i = 0; i < s0; i++) {
            for (NSInteger j = 0; j < s1; j++) {
                for (NSInteger k = 0; k < s2; k++) {
                    result[outIdx++] = ptr[i * st0 + j * st1 + k * st2];
                }
            }
        }
    } else if (array.dataType == MLMultiArrayDataTypeFloat16) {
        uint16_t *fp16 = (uint16_t *)array.dataPointer;
        NSInteger outIdx = 0;
        for (NSInteger i = 0; i < s0; i++) {
            for (NSInteger j = 0; j < s1; j++) {
                for (NSInteger k = 0; k < s2; k++) {
                    uint16_t h = fp16[i * st0 + j * st1 + k * st2];
                    uint32_t sign = (h >> 15) & 0x1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;

                    uint32_t f;
                    if (exp == 0) {
                        if (mant == 0) {
                            f = sign << 31;
                        } else {
                            exp = 1;
                            while (!(mant & 0x400)) { mant <<= 1; exp--; }
                            mant &= 0x3FF;
                            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                        }
                    } else if (exp == 31) {
                        f = (sign << 31) | 0x7F800000 | (mant << 13);
                    } else {
                        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                    memcpy(&result[outIdx++], &f, sizeof(float));
                }
            }
        }
    } else {
        NSInteger outIdx = 0;
        for (NSInteger i = 0; i < s0; i++) {
            for (NSInteger j = 0; j < s1; j++) {
                for (NSInteger k = 0; k < s2; k++) {
                    NSArray<NSNumber *> *idx = (array.shape.count == 3) ? @[@(i), @(j), @(k)] :
                                               (array.shape.count == 2) ? @[@(i), @(j)] :
                                               @[@(i)];
                    result[outIdx++] = [[array objectForKeyedSubscript:idx] floatValue];
                }
            }
        }
    }

    return result;
}

// =========================================================
// Inference
// =========================================================
- (NSArray<SCRFDFace *> *)detectFacesInImage:(NSImage *)image
                                   threshold:(float)threshold
                                nmsThreshold:(float)nmsThreshold {
    float detScale = 1.0f;
    CVPixelBufferRef pixelBuffer = [self createPixelBufferFromImage:image scale:&detScale];
    if (!pixelBuffer) {
        NSLog(@"Failed to create pixel buffer");
        return @[];
    }

    // Run inference
    NSError *error = nil;
    id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"image": [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer]}
                     error:&error];
    CVPixelBufferRelease(pixelBuffer);

    if (error) {
        NSLog(@"Failed to create input: %@", error);
        return @[];
    }

    id<MLFeatureProvider> output = [self.model predictionFromFeatures:input error:&error];
    if (error) {
        NSLog(@"Inference failed: %@", error);
        return @[];
    }

    // Extract outputs (handle Float16 or Float32)
    MLMultiArray *scoresArr = [output featureValueForName:@"scores"].multiArrayValue;
    MLMultiArray *bboxesArr = [output featureValueForName:@"bboxes"].multiArrayValue;
    MLMultiArray *kpsArr = [output featureValueForName:@"keypoints"].multiArrayValue;

    NSLog(@"Output types: scores=%ld bboxes=%ld kps=%ld",
          (long)scoresArr.dataType, (long)bboxesArr.dataType, (long)kpsArr.dataType);
    NSLog(@"Output shapes: scores=%@ bboxes=%@ kps=%@",
          scoresArr.shape, bboxesArr.shape, kpsArr.shape);

    float *scoresPtr = [self float32FromMultiArray:scoresArr];
    float *bboxesPtr = [self float32FromMultiArray:bboxesArr];
    float *kpsPtr = [self float32FromMultiArray:kpsArr];

    // Decode detections
    NSMutableArray<NSValue *> *detections = [NSMutableArray array];

    NSInteger anchorOffset = 0;
    for (NSInteger s = 0; s < 3; s++) {
        NSInteger stride = kStrides[s];
        NSInteger count = kAnchorCounts[s];

        for (NSInteger i = 0; i < count; i++) {
            NSInteger globalIdx = anchorOffset + i;
            float score = scoresPtr[globalIdx];

            if (score < threshold) continue;

            float cx = _anchorCenters[globalIdx * 2 + 0];
            float cy = _anchorCenters[globalIdx * 2 + 1];

            // Decode bbox: distance * stride
            float left   = bboxesPtr[globalIdx * 4 + 0] * stride;
            float top    = bboxesPtr[globalIdx * 4 + 1] * stride;
            float right  = bboxesPtr[globalIdx * 4 + 2] * stride;
            float bottom = bboxesPtr[globalIdx * 4 + 3] * stride;

            Detection det;
            det.x1 = cx - left;
            det.y1 = cy - top;
            det.x2 = cx + right;
            det.y2 = cy + bottom;
            det.score = score;

            // Decode keypoints: offset * stride
            for (int k = 0; k < 10; k += 2) {
                det.kps[k]     = cx + kpsPtr[globalIdx * 10 + k]     * stride;
                det.kps[k + 1] = cy + kpsPtr[globalIdx * 10 + k + 1] * stride;
            }

            [detections addObject:[NSValue valueWithBytes:&det objCType:@encode(Detection)]];
        }
        anchorOffset += count;
    }

    free(scoresPtr);
    free(bboxesPtr);
    free(kpsPtr);

    NSLog(@"Detections before NMS: %lu", (unsigned long)detections.count);

    // NMS
    NSArray<NSValue *> *kept = [self nms:detections threshold:nmsThreshold];

    // Convert to SCRFDFace
    NSMutableArray<SCRFDFace *> *faces = [NSMutableArray array];
    for (NSValue *val in kept) {
        Detection det;
        [val getValue:&det];

        SCRFDFace *face = [[SCRFDFace alloc] init];
        face.bbox = CGRectMake(det.x1 / detScale, det.y1 / detScale,
                               (det.x2 - det.x1) / detScale,
                               (det.y2 - det.y1) / detScale);
        face.score = det.score;
        for (int k = 0; k < 5; k++) {
            [face setKeypoint:CGPointMake(det.kps[k * 2] / detScale,
                                          det.kps[k * 2 + 1] / detScale)
                      atIndex:k];
        }

        [faces addObject:face];
    }

    return faces;
}

// =========================================================
// NMS
// =========================================================
- (NSArray<NSValue *> *)nms:(NSArray<NSValue *> *)detections threshold:(float)threshold {
    NSArray *sorted = [detections sortedArrayUsingComparator:^NSComparisonResult(NSValue *a, NSValue *b) {
        Detection da, db;
        [a getValue:&da];
        [b getValue:&db];
        if (da.score > db.score) return NSOrderedAscending;
        if (da.score < db.score) return NSOrderedDescending;
        return NSOrderedSame;
    }];

    NSMutableArray<NSValue *> *kept = [NSMutableArray array];
    BOOL *suppressed = (BOOL *)calloc(sorted.count, sizeof(BOOL));

    for (NSUInteger i = 0; i < sorted.count; i++) {
        if (suppressed[i]) continue;

        Detection di;
        [sorted[i] getValue:&di];
        [kept addObject:sorted[i]];

        float areaI = (di.x2 - di.x1 + 1) * (di.y2 - di.y1 + 1);

        for (NSUInteger j = i + 1; j < sorted.count; j++) {
            if (suppressed[j]) continue;

            Detection dj;
            [sorted[j] getValue:&dj];

            float xx1 = fmaxf(di.x1, dj.x1);
            float yy1 = fmaxf(di.y1, dj.y1);
            float xx2 = fminf(di.x2, dj.x2);
            float yy2 = fminf(di.y2, dj.y2);

            float w = fmaxf(0.0f, xx2 - xx1 + 1);
            float h = fmaxf(0.0f, yy2 - yy1 + 1);
            float inter = w * h;

            float areaJ = (dj.x2 - dj.x1 + 1) * (dj.y2 - dj.y1 + 1);
            float iou = inter / (areaI + areaJ - inter);

            if (iou > threshold) {
                suppressed[j] = YES;
            }
        }
    }

    free(suppressed);
    return kept;
}

// =========================================================
// Detect and Draw
// =========================================================
- (BOOL)detectAndDrawInFile:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                  threshold:(float)threshold {
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:inputPath];
    if (!image) {
        NSLog(@"Failed to load image: %@", inputPath);
        return NO;
    }

    NSArray<SCRFDFace *> *faces = [self detectFacesInImage:image
                                                threshold:threshold
                                             nmsThreshold:0.4f];

    NSLog(@"Detected %lu faces.", (unsigned long)faces.count);

    // Draw on a bitmap using CG (top-left origin, avoids AppKit coordinate confusion)
    CGImageRef cgRef = [image CGImageForProposedRect:NULL context:nil hints:nil];
    if (!cgRef) {
        NSLog(@"Failed to get CGImage");
        return NO;
    }

    size_t imgW = CGImageGetWidth(cgRef);
    size_t imgH = CGImageGetHeight(cgRef);

    // Create RGBA bitmap context
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(NULL, imgW, imgH, 8, imgW * 4, colorSpace,
                                            kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(colorSpace);

    // Draw original image (CG origin = bottom-left, no flip needed)
    CGContextDrawImage(ctx, CGRectMake(0, 0, imgW, imgH), cgRef);

    // Draw detections
    // face.bbox is in top-left origin, convert to CG bottom-left origin
    for (SCRFDFace *face in faces) {
        CGRect bbox = face.bbox;

        // Convert top-left origin → CG bottom-left origin
        CGFloat cgY = (CGFloat)imgH - bbox.origin.y - bbox.size.height;
        CGRect cgRect = CGRectMake(bbox.origin.x, cgY, bbox.size.width, bbox.size.height);

        // Draw bbox (green, 2px)
        CGContextSetRGBStrokeColor(ctx, 0, 1, 0, 1);
        CGContextSetLineWidth(ctx, 2.0);
        CGContextStrokeRect(ctx, cgRect);

        // Draw keypoints (red circles, radius 3)
        CGContextSetRGBFillColor(ctx, 1, 0, 0, 1);
        for (int k = 0; k < 5; k++) {
            CGPoint kp = [face keypointAtIndex:k];
            CGFloat kpCgY = (CGFloat)imgH - kp.y;
            CGFloat r = 3.0;
            CGContextFillEllipseInRect(ctx, CGRectMake(kp.x - r, kpCgY - r, r * 2, r * 2));
        }

        NSLog(@"  Face: score=%.3f bbox=(%.0f,%.0f,%.0f,%.0f)",
              face.score, bbox.origin.x, bbox.origin.y,
              bbox.size.width, bbox.size.height);
    }

    // Save as PNG
    CGImageRef resultImage = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);

    CFURLRef url = (__bridge CFURLRef)[NSURL fileURLWithPath:outputPath];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL);
    if (!dest) {
        NSLog(@"Failed to create image destination");
        CGImageRelease(resultImage);
        return NO;
    }
    CGImageDestinationAddImage(dest, resultImage, NULL);
    BOOL ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(resultImage);

    if (ok) {
        NSLog(@"Saved result to %@", outputPath);
    }
    return ok;
}

@end
