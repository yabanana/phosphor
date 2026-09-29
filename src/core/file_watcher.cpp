#include "core/file_watcher.h"

namespace phosphor {

// F3 contract stub: implemented by the F3.6 work package.
FileWatcher::FileWatcher(std::string directory, std::vector<std::string> extensions)
    : directory_(std::move(directory)), extensions_(std::move(extensions)) {}
bool FileWatcher::poll() { return false; }

} // namespace phosphor
