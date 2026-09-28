#include <doctest/doctest.h>

#include "null_texture_manager.h"

using namespace phosphor;
using phosphor::test::NullTextureManager;

TEST_CASE("default textures are created once") {
    NullTextureManager tm;
    tm.createDefaultTextures();
    tm.createDefaultTextures();
    CHECK(tm.textureCount() == 4);
    CHECK(tm.getDefaultWhite() != tm.getDefaultNormal());
    CHECK(tm.uploads[tm.getDefaultNormal()].rgba == std::vector<u8>{128, 128, 255, 255});
    CHECK(tm.uploads[tm.getDefaultWhite()].sRGB);
    CHECK_FALSE(tm.uploads[tm.getDefaultMR()].sRGB);
}

TEST_CASE("pixel data is expanded to RGBA") {
    NullTextureManager tm;

    const u8 gray[] = {10, 20};
    const u32 g = tm.loadTextureFromMemory(gray, 2, 1, 1, false);
    CHECK(tm.uploads[g].rgba == std::vector<u8>{10, 10, 10, 255, 20, 20, 20, 255});

    const u8 grayAlpha[] = {10, 99};
    const u32 ga = tm.loadTextureFromMemory(grayAlpha, 1, 1, 2, false);
    CHECK(tm.uploads[ga].rgba == std::vector<u8>{10, 10, 10, 99});

    const u8 rgb[] = {1, 2, 3};
    const u32 c = tm.loadTextureFromMemory(rgb, 1, 1, 3, true);
    CHECK(tm.uploads[c].rgba == std::vector<u8>{1, 2, 3, 255});
}

TEST_CASE("invalid input falls back to the default white texture") {
    NullTextureManager tm;
    tm.createDefaultTextures();
    CHECK(tm.loadTextureFromMemory(nullptr, 1, 1, 4) == tm.getDefaultWhite());
    CHECK(tm.loadTexture("/nonexistent/texture.png") == tm.getDefaultWhite());
    CHECK(tm.textureCount() == 4);
}
