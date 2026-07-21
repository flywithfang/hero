// image_io.hpp — file decoding at the application boundary.
//
// Model code accepts RGBImage and has no dependency on file formats.  The
// built-in chat frontend deliberately starts with the dependency-free PPM
// formats; PNG/JPEG support can be added here without touching any encoder or
// decoder assembly.
#pragma once
#include "multimodal.hpp"
#include <cctype>
#include <fstream>

#if defined(__APPLE__)
RGBImage load_platform_rgb_image(const std::string& path);
#endif

inline std::string ppm_word(std::istream& input) {
    std::string word;
    for (;;) {
        input >> std::ws;
        if (input.peek() != '#') break;
        std::string comment;
        std::getline(input, comment);
    }
    input >> word;
    if (!input) throw std::runtime_error("image: invalid PPM header");
    return word;
}

inline RGBImage load_rgb_image(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("image: cannot open '" + path + "'");
    char signature[2]{};
    input.read(signature, 2);
    input.clear();
    input.seekg(0);
    if (!(signature[0] == 'P' && (signature[1] == '6' || signature[1] == '3'))) {
#if defined(__APPLE__)
        return load_platform_rgb_image(path);
#else
        throw std::runtime_error("image: unsupported format (expected RGB PPM P6/P3)");
#endif
    }
    const std::string magic = ppm_word(input);
    const size_t width = std::stoull(ppm_word(input));
    const size_t height = std::stoull(ppm_word(input));
    const size_t maximum = std::stoull(ppm_word(input));
    if (maximum == 0 || maximum > 255)
        throw std::runtime_error("image: unsupported PPM sample range");
    if (width == 0 || height == 0 || width > std::numeric_limits<size_t>::max() / height ||
        width * height > std::numeric_limits<size_t>::max() / 3)
        throw std::runtime_error("image: invalid dimensions");

    std::vector<uint8_t> pixels(width * height * 3);
    if (magic == "P6") {
        char separator = 0;
        input.get(separator);
        if (!input || !std::isspace(static_cast<unsigned char>(separator)))
            throw std::runtime_error("image: malformed P6 header separator");
        input.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(pixels.size()));
        if (input.gcount() != std::streamsize(pixels.size()))
            throw std::runtime_error("image: truncated PPM pixel data");
        if (maximum != 255)
            for (uint8_t& value : pixels) value = uint8_t(size_t(value) * 255 / maximum);
    } else {
        for (uint8_t& value : pixels) {
            const size_t sample = std::stoull(ppm_word(input));
            if (sample > maximum) throw std::runtime_error("image: PPM sample exceeds range");
            value = uint8_t(sample * 255 / maximum);
        }
    }
    return RGBImage(width, height, std::move(pixels));
}
