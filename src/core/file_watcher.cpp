#include "core/file_watcher.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <system_error>

namespace phosphor {

namespace fs = std::filesystem;

FileWatcher::FileWatcher(std::string directory, std::vector<std::string> extensions)
    : directory_(std::move(directory)), extensions_(std::move(extensions)) {}

bool FileWatcher::poll() {
    changed_.clear();

    std::error_code ec;
    fs::directory_iterator it(directory_, ec);
    if (ec) {
        // Missing / unreadable directory: no change, keep the previous state.
        // A first poll on a missing directory still primes an empty baseline
        // so files that appear later count as added.
        primed_ = true;
        return false;
    }

    std::vector<FileState> now;
    for (const fs::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) break;
        std::error_code fe;
        if (!it->is_regular_file(fe) || fe) continue;
        const fs::path& p = it->path();
        const std::string ext = p.extension().string();
        if (std::find(extensions_.begin(), extensions_.end(), ext) == extensions_.end()) continue;
        const auto mt = fs::last_write_time(p, fe);
        if (fe) continue; // vanished between listing and stat
        const auto size = fs::file_size(p, fe);
        if (fe) continue;
        FileState s;
        s.name  = p.filename().string();
        s.size  = static_cast<u64>(size);
        s.mtime = static_cast<i64>(mt.time_since_epoch().count());
        now.push_back(std::move(s));
    }
    std::sort(now.begin(), now.end(),
              [](const FileState& a, const FileState& b) { return a.name < b.name; });

    if (primed_) {
        // Merge the two sorted lists.
        std::size_t i = 0, j = 0;
        while (i < files_.size() || j < now.size()) {
            if (j == now.size() || (i < files_.size() && files_[i].name < now[j].name)) {
                changed_.push_back(files_[i++].name); // removed
            } else if (i == files_.size() || now[j].name < files_[i].name) {
                changed_.push_back(now[j++].name); // added
            } else {
                if (files_[i].size != now[j].size || files_[i].mtime != now[j].mtime)
                    changed_.push_back(now[j].name);
                ++i;
                ++j;
            }
        }
    }
    files_  = std::move(now);
    primed_ = true;
    return !changed_.empty();
}

} // namespace phosphor
