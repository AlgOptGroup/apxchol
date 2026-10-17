#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace apxchol::detail::metal {

inline std::string describe(NSError* error) {
    const char* text = error ? error.localizedDescription.UTF8String : nullptr;
    return text ? text : "unknown Metal error";
}

struct compiled_library {
    id<MTLLibrary> library;
    bool contract_pragma;
};

inline compiled_library compile_library(id<MTLDevice> device, const char* kernels) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        options.languageVersion = MTLLanguageVersion3_2;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* error = nil;
    for (bool contract : {true, false}) {
        std::string source = contract ? "#pragma METAL fp contract(off)\n" : "";
        source += kernels;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                    options:options error:&error];
        if (library != nil) return {library, contract};
    }
    throw std::runtime_error("Metal kernel compilation failed: " + describe(error));
}

} // namespace apxchol::detail::metal
