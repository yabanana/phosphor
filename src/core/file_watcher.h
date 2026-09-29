#pragma once

#include "core/types.h"

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// FileWatcher -- polling change detector for a directory (F3.6 hot reload).
//
// Portable (std::filesystem): each poll() stats the regular files directly in
// `directory` whose extension is in `extensions` (e.g. {".metal", ".h"}) and
// compares size + mtime with the previous poll.  The first poll() records the
// baseline and reports no change.  Added and removed files count as changes.
// It is meant to run on a utility thread (a handful of stat calls), never in
// the frame.
// ---------------------------------------------------------------------------

class FileWatcher {
public:
    FileWatcher(std::string directory, std::vector<std::string> extensions);

    /// True if something changed since the previous poll (see above).
    bool poll();
    /// Paths (relative to the directory) that changed in the last poll().
    [[nodiscard]] const std::vector<std::string>& changed() const { return changed_; }
    [[nodiscard]] const std::string& directory() const { return directory_; }

private:
    struct FileState {
        std::string name;
        u64 size  = 0;
        i64 mtime = 0; // file_time_type ticks
    };

    std::string              directory_;
    std::vector<std::string> extensions_;
    std::vector<FileState>   files_; // sorted by name
    std::vector<std::string> changed_;
    bool                     primed_ = false;
};

} // namespace phosphor
