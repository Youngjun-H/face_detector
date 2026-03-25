#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <AppKit/AppKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface SCRFDFace : NSObject
@property (nonatomic) CGRect bbox;
@property (nonatomic) float score;

// left_eye, right_eye, nose, left_mouth, right_mouth
- (CGPoint)keypointAtIndex:(NSUInteger)index;
- (void)setKeypoint:(CGPoint)point atIndex:(NSUInteger)index;

@end

@interface SCRFDInfer : NSObject

- (nullable instancetype)initWithModelPath:(NSString *)modelPath;
- (NSArray<SCRFDFace *> *)detectFacesInImage:(NSImage *)image
                                   threshold:(float)threshold
                                nmsThreshold:(float)nmsThreshold;
- (BOOL)detectAndDrawInFile:(NSString *)inputPath
                 outputPath:(NSString *)outputPath
                  threshold:(float)threshold;

@end

NS_ASSUME_NONNULL_END
