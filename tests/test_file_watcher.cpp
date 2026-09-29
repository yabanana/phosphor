#include "core/file_watcher.h"
#include "core/process.h"

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace phosphor;
namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(bool create = true) {
        static int counter = 0;
        path = fs::temp_directory_path() /
               ("phosphor_fw_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::error_code ec;
        fs::remove_all(path, ec);
        if (create) fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    [[nodiscard]] std::string str() const { return path.string(); }
};

void writeFile(const fs::path& p, const std::string& content) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << content;
}

using Names = std::vector<std::string>;

} // namespace

TEST_CASE("FileWatcher: first poll is a baseline") {
    TempDir d;
    writeFile(d.path / "a.metal", "x");
    FileWatcher w(d.str(), {".metal"});
    CHECK_FALSE(w.poll());
    CHECK(w.changed().empty());
    CHECK_FALSE(w.poll());
    CHECK(w.directory() == d.str());
}

TEST_CASE("FileWatcher: added, modified (size), modified (mtime), removed") {
    TempDir d;
    FileWatcher w(d.str(), {".metal", ".h"});
    CHECK_FALSE(w.poll());

    writeFile(d.path / "b.metal", "1");
    writeFile(d.path / "a.h", "1");
    CHECK(w.poll());
    CHECK(w.changed() == Names{"a.h", "b.metal"}); // sorted
    CHECK_FALSE(w.poll());

    writeFile(d.path / "b.metal", "12345"); // size change
    CHECK(w.poll());
    CHECK(w.changed() == Names{"b.metal"});
    CHECK_FALSE(w.poll());

    // Same size, explicit mtime bump.
    const auto t = fs::last_write_time(d.path / "a.h");
    fs::last_write_time(d.path / "a.h", t + std::chrono::seconds(10));
    CHECK(w.poll());
    CHECK(w.changed() == Names{"a.h"});
    CHECK_FALSE(w.poll());

    fs::remove(d.path / "b.metal");
    CHECK(w.poll());
    CHECK(w.changed() == Names{"b.metal"});
    CHECK_FALSE(w.poll());
    CHECK(w.changed().empty());
}

TEST_CASE("FileWatcher: extension filter, no recursion, only regular files") {
    TempDir d;
    FileWatcher w(d.str(), {".metal"});
    CHECK_FALSE(w.poll());

    writeFile(d.path / "x.txt", "1");
    writeFile(d.path / "y.METAL", "1"); // case-sensitive
    writeFile(d.path / "metal", "1");
    fs::create_directories(d.path / "sub.metal");     // directory with matching name
    writeFile(d.path / "sub.metal" / "z.metal", "1"); // nested
    CHECK_FALSE(w.poll());

    writeFile(d.path / "ok.metal", "1");
    CHECK(w.poll());
    CHECK(w.changed() == Names{"ok.metal"});
}

TEST_CASE("FileWatcher: missing directory") {
    TempDir d(false); // not created
    FileWatcher w(d.str(), {".metal"});
    CHECK_FALSE(w.poll());
    CHECK_FALSE(w.poll());
    CHECK(w.changed().empty());

    fs::create_directories(d.path);
    writeFile(d.path / "late.metal", "1");
    CHECK(w.poll());
    CHECK(w.changed() == Names{"late.metal"});
}

TEST_CASE("runProcess: output and exit code") {
    const ProcessResult r = runProcess({"/bin/sh", "-c", "echo out; echo err 1>&2; exit 3"});
    CHECK(r.exitCode == 3);
    CHECK(r.output.find("out") != std::string::npos);
    CHECK(r.output.find("err") != std::string::npos);
}

TEST_CASE("runProcess: success, PATH lookup, no shell") {
    CHECK(runProcess({"true"}).exitCode == 0);
    const ProcessResult r = runProcess({"echo", "a;b $HOME"});
    CHECK(r.exitCode == 0);
    CHECK(r.output == "a;b $HOME\n");
}

TEST_CASE("runProcess: nonexistent program fails") {
    // -1 (spawn failed) or 127 (child exec failed) depending on the libc.
    const ProcessResult r = runProcess({"phosphor-no-such-program-xyz"});
    CHECK(r.exitCode != 0);
    CHECK(runProcess({}).exitCode == -1);
}

TEST_CASE("runProcess: killed by signal") {
    const ProcessResult r = runProcess({"/bin/sh", "-c", "kill -9 $$"});
    CHECK(r.exitCode == 128 + 9);
}

TEST_CASE("runProcess: large output does not deadlock") {
    const ProcessResult r = runProcess({"/bin/sh", "-c", "yes 0123456789abcdef | head -n 200000"});
    CHECK(r.exitCode == 0);
    CHECK(r.output.size() == 200000u * 17u);
}
