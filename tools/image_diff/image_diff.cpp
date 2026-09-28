// image_diff -- compare two images pixel by pixel (visual regression checks).
//
//   image_diff reference.png candidate.png [--tolerance N] [--diff out.png]
//
// Prints the number of differing pixels, the largest per-channel difference
// and the PSNR.  A pixel differs when any channel differs by more than N
// (default 0).  Exit code: 0 identical within tolerance, 1 different,
// 2 usage or I/O error.

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Image {
    int            width  = 0;
    int            height = 0;
    unsigned char* pixels = nullptr; // RGBA8

    ~Image() { stbi_image_free(pixels); }
};

bool load(const char* path, Image& image) {
    int channels = 0;
    image.pixels = stbi_load(path, &image.width, &image.height, &channels, 4);
    if (!image.pixels) {
        std::fprintf(stderr, "image_diff: cannot read %s: %s\n", path, stbi_failure_reason());
        return false;
    }
    return true;
}

int usage() {
    std::fprintf(stderr, "usage: image_diff reference.png candidate.png [--tolerance N] [--diff out.png]\n");
    return 2;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) return usage();
    int tolerance = 0;
    std::string diffPath;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--diff") == 0 && i + 1 < argc) {
            diffPath = argv[++i];
        } else {
            return usage();
        }
    }

    Image a, b;
    if (!load(argv[1], a) || !load(argv[2], b)) return 2;
    if (a.width != b.width || a.height != b.height) {
        std::printf("size mismatch: %dx%d vs %dx%d\n", a.width, a.height, b.width, b.height);
        return 1;
    }

    const size_t pixelCount = static_cast<size_t>(a.width) * static_cast<size_t>(a.height);
    std::vector<unsigned char> diff(diffPath.empty() ? 0 : pixelCount * 4, 0);
    size_t differing = 0;
    int maxDelta = 0;
    double squaredError = 0.0;
    for (size_t p = 0; p < pixelCount; ++p) {
        int pixelDelta = 0;
        for (int c = 0; c < 3; ++c) { // alpha is always opaque in captures
            const int d = std::abs(int(a.pixels[p * 4 + c]) - int(b.pixels[p * 4 + c]));
            pixelDelta = std::max(pixelDelta, d);
            squaredError += double(d) * double(d);
        }
        maxDelta = std::max(maxDelta, pixelDelta);
        if (pixelDelta > tolerance) {
            ++differing;
            if (!diff.empty()) {
                diff[p * 4 + 0] = 255;
                diff[p * 4 + 3] = 255;
            }
        } else if (!diff.empty()) {
            // Dimmed reference so differences stand out in context.
            for (int c = 0; c < 3; ++c) diff[p * 4 + c] = static_cast<unsigned char>(a.pixels[p * 4 + c] / 4);
            diff[p * 4 + 3] = 255;
        }
    }

    const double mse = squaredError / (double(pixelCount) * 3.0);
    const double psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : INFINITY;
    std::printf("%zu of %zu pixels differ (tolerance %d), max delta %d, PSNR %.2f dB\n", differing, pixelCount,
                tolerance, maxDelta, psnr);

    if (!diffPath.empty() && !stbi_write_png(diffPath.c_str(), a.width, a.height, 4, diff.data(), a.width * 4)) {
        std::fprintf(stderr, "image_diff: cannot write %s\n", diffPath.c_str());
        return 2;
    }
    return differing == 0 ? 0 : 1;
}
