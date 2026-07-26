// image_io_apple.cpp — PNG/JPEG/HEIC/etc. decoding via macOS ImageIO.
#include "image_io.hpp"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

RGBImage load_platform_rgb_image(const std::string& path) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8*>(path.data()), path.size(), false);
    if (!url) throw std::runtime_error("image: cannot represent path");
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) throw std::runtime_error("image: ImageIO cannot decode '" + path + "'");
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) throw std::runtime_error("image: ImageIO found no image frame");

    const size_t width = CGImageGetWidth(image);
    const size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0 || width > std::numeric_limits<size_t>::max() / height || width * height > std::numeric_limits<size_t>::max() / 4) {
        CGImageRelease(image);
        throw std::runtime_error("image: invalid decoded dimensions");
    }

    std::vector<uint8_t> rgba(width * height * 4);
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const CGBitmapInfo bitmap_info = CGBitmapInfo(uint32_t(kCGImageAlphaPremultipliedLast) | uint32_t(kCGBitmapByteOrder32Big));
    CGContextRef context = CGBitmapContextCreate(rgba.data(), width, height, 8, width * 4, color_space, bitmap_info);
    CGColorSpaceRelease(color_space);
    if (!context) {
        CGImageRelease(image);
        throw std::runtime_error("image: cannot create RGB conversion context");
    }
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGImageRelease(image);

    std::vector<uint8_t> rgb(width * height * 3);
    for (size_t pixel = 0; pixel < width * height; ++pixel) {
        rgb[pixel * 3 + 0] = rgba[pixel * 4 + 0];
        rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
        rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    return RGBImage(width, height, std::move(rgb));
}
