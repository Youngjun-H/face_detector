#import <Foundation/Foundation.h>
#import "SCRFDInfer.h"

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) {
            NSLog(@"Usage: scrfd_detect <model.mlpackage> <input_image> <output_image> [threshold]");
            return 1;
        }

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSString *inputPath = [NSString stringWithUTF8String:argv[2]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[3]];
        float threshold = (argc > 4) ? atof(argv[4]) : 0.5f;

        SCRFDInfer *infer = [[SCRFDInfer alloc] initWithModelPath:modelPath];
        if (!infer) {
            NSLog(@"Failed to initialize model");
            return 1;
        }

        BOOL ok = [infer detectAndDrawInFile:inputPath
                                  outputPath:outputPath
                                   threshold:threshold];
        return ok ? 0 : 1;
    }
}
