#pragma once

// F15.1 (ticket 04, review): file-system callbacks for tinygltf that keep every
// external image AND buffer URI inside the asset directory, before anything is
// opened.  Installed with TinyGLTF::SetFsCallbacks (v2.9.7 needs all five).
// Rules: the glTF file itself is readable; any other path must be the asset
// directory plus a relative path without `..` segments, absolute forms,
// schemes or symlink escapes (canonical prefix); tinygltf's current-directory
// fallback never matches unless the asset directory is the current directory.
// Portable (no Metal).

#include <functional>
#include <string>
#include <vector>

namespace tinygltf {
class TinyGLTF;
} // namespace tinygltf

namespace phosphor {

class ContainedFileSystem {
public:
    /// `mainFile` is the glTF/GLB path handed to the loader (its directory is
    /// the asset directory).
    explicit ContainedFileSystem(std::string mainFile);
    ContainedFileSystem(const ContainedFileSystem&) = delete;
    ContainedFileSystem& operator=(const ContainedFileSystem&) = delete;

    /// Install the callbacks on `loader`; this object must outlive its use.
    /// Returns false (with `error`) when tinygltf rejects the callback set.
    [[nodiscard]] bool install(tinygltf::TinyGLTF& loader, std::string& error);

    /// Directory of the glTF file, with a trailing separator (empty = cwd).
    [[nodiscard]] const std::string& assetDirectory() const { return baseDir_; }
    /// One message per refused URI, naming it (deduplicated, in order).
    [[nodiscard]] const std::vector<std::string>& refusals() const { return refusals_; }
    /// Every path actually opened for reading, in order.
    [[nodiscard]] const std::vector<std::string>& reads() const { return reads_; }
    /// Called with each path right before it is read (tests count reads here).
    void setReadObserver(std::function<void(const std::string&)> observer) { observer_ = std::move(observer); }

    /// Policy: may `path` (as tinygltf passes it) be touched?  `silent` is set
    /// for paths that are simply not candidates (the current-directory
    /// fallback) rather than escapes.
    [[nodiscard]] bool allowed(const std::string& path, std::string& error, bool& silent) const;

private:
    void refuse(const std::string& message);

    std::string                                mainFile_;
    std::string                                baseDir_;
    std::vector<std::string>                   refusals_;
    std::vector<std::string>                   reads_;
    std::function<void(const std::string&)>    observer_;
};

} // namespace phosphor
