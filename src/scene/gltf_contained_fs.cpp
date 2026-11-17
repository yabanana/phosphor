#include "scene/gltf_contained_fs.h"

#include "scene/gltf_material_extensions.h"

#include <tiny_gltf.h> // declarations only; the implementation lives in gltf_loader.cpp

#include <algorithm>

namespace phosphor {

ContainedFileSystem::ContainedFileSystem(std::string mainFile) : mainFile_(std::move(mainFile)) {
    const size_t slash = mainFile_.find_last_of("/\\");
    baseDir_ = slash == std::string::npos ? std::string{} : mainFile_.substr(0, slash + 1);
}

bool ContainedFileSystem::allowed(const std::string& path, std::string& error, bool& silent) const {
    silent = false;
    if (path == mainFile_) return true;
    if (!baseDir_.empty() && path.rfind(baseDir_, 0) != 0) {
        // tinygltf's "." search path: not a candidate, never an alternative root.
        silent = true;
        error  = "'" + path + "' is not inside the asset directory";
        return false;
    }
    return isContainedRelativePath(baseDir_, path.substr(baseDir_.size()), error);
}

void ContainedFileSystem::refuse(const std::string& message) {
    if (std::find(refusals_.begin(), refusals_.end(), message) == refusals_.end()) refusals_.push_back(message);
}

bool ContainedFileSystem::install(tinygltf::TinyGLTF& loader, std::string& error) {
    tinygltf::FsCallbacks fs;
    fs.user_data = this;
    // No `~` or %APPDATA% expansion: the path is used exactly as joined.
    fs.ExpandFilePath = [](const std::string& path, void*) { return path; };
    fs.FileExists = [this](const std::string& path, void*) {
        std::string why;
        bool silent = false;
        if (!allowed(path, why, silent)) {
            if (!silent) refuse(why);
            return false;
        }
        return tinygltf::FileExists(path, nullptr);
    };
    fs.GetFileSizeInBytes = [this](size_t* size, std::string* err, const std::string& path, void*) {
        std::string why;
        bool silent = false;
        if (!allowed(path, why, silent)) {
            if (!silent) refuse(why);
            if (err) *err += why + "\n";
            return false;
        }
        return tinygltf::GetFileSizeInBytes(size, err, path, nullptr);
    };
    fs.ReadWholeFile = [this](std::vector<unsigned char>* out, std::string* err, const std::string& path, void*) {
        std::string why;
        bool silent = false;
        if (!allowed(path, why, silent)) {
            if (!silent) refuse(why);
            if (err) *err += why + "\n";
            return false;
        }
        reads_.push_back(path);
        if (observer_) observer_(path);
        return tinygltf::ReadWholeFile(out, err, path, nullptr);
    };
    fs.WriteWholeFile = [](std::string* err, const std::string& path, const std::vector<unsigned char>&, void*) {
        if (err) *err += "writing '" + path + "' is not allowed while loading\n";
        return false;
    };
    return loader.SetFsCallbacks(std::move(fs), &error);
}

} // namespace phosphor
