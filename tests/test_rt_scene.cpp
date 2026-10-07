#include <doctest/doctest.h>
#include "renderer/rt_scene.h"

using namespace phosphor;

TEST_CASE("RT lifecycle selects build refit and compaction under independent budgets") {
    RtScene scene;
    scene.request(0, 1, 1);
    scene.request(1, 1, 1);
    CHECK(scene.mesh(0).state == RtBlasState::Pending);
    auto work = scene.plan({1, 0, 0});
    REQUIRE(work.size() == 1);
    CHECK(work[0].kind == RtWorkKind::Build);
    CHECK(work[0].mesh == 0);
    scene.markBuilt(0, 100, 4096, 0);
    scene.request(0, 1, 2);
    work = scene.plan({0, 1, 0});
    REQUIRE(work.size() == 1);
    CHECK(work[0].kind == RtWorkKind::Refit);
    const auto generation = scene.tableGeneration();
    scene.markRefitted(0, 1);
    CHECK(scene.tableGeneration() == generation);
    scene.queueCompaction(0, 2048);
    CHECK(scene.mesh(0).state == RtBlasState::CompactionQueued);
    work = scene.plan({0, 0, 1});
    REQUIRE(work.size() == 1);
    CHECK(work[0].kind == RtWorkKind::Compact);
    CHECK(scene.markCompacted(0, work[0].version, 101, 2048, 2));
    CHECK(scene.mesh(0).state == RtBlasState::Compacted);
    CHECK(scene.tableGeneration() == generation + 1);
    scene.request(0, 2, 3);
    work = scene.plan({1, 0, 0});
    REQUIRE(work.size() == 1);
    CHECK(work[0].kind == RtWorkKind::Build);
    CHECK_THROWS(scene.markRefitted(0, 3));
}

TEST_CASE("RT compaction waits until every old TLAS snapshot is replaced") {
    RtScene scene(3);
    scene.request(0, 1, 1);
    scene.markBuilt(0, 100, 4096, 0);
    scene.commitSnapshot(0, 10, 0);
    scene.commitSnapshot(1, 10, 1);
    // Slot 2 was never used and must not hold an artificial version/frame.
    const auto oldVersion = scene.mesh(0).version;
    scene.queueCompaction(0, 2048);
    REQUIRE(scene.markCompacted(0, oldVersion, 101, 2048, 2));
    scene.completeFrame(2);
    CHECK(scene.collectRetired().empty());
    scene.commitSnapshot(0, 10, 3);
    CHECK(scene.collectRetired().empty()); // Slot 1 still names oldVersion.
    scene.clearSnapshot(1);
    auto retired = scene.collectRetired();
    REQUIRE(retired.size() == 1);
    CHECK(retired[0].resourceID == 100);
    CHECK(retired[0].version == oldVersion);
    CHECK(retired[0].lastReader == 2); // Copy frame, beyond last TLAS reader.
    CHECK(scene.collectRetired().empty());
}

TEST_CASE("RT retired snapshot can be cleared only after its last reader completes") {
    RtScene scene(2);
    scene.request(0, 1, 1);
    scene.markBuilt(0, 101, 4096, 0);
    scene.commitSnapshot(0, 10, 0);
    CHECK_THROWS(scene.clearSnapshot(0)); // Frame zero is not implicitly done.
    scene.completeFrame(0);
    scene.commitSnapshot(0, 10, 5);
    scene.remove(0, 2); // Saved lastReader, not removal frame, controls lifetime.
    CHECK_THROWS(scene.clearSnapshot(0));
    scene.completeFrame(4);
    CHECK(scene.collectRetired().empty());
    scene.completeFrame(5);
    scene.clearSnapshot(0);
    auto retired = scene.collectRetired();
    REQUIRE(retired.size() == 1);
    CHECK(retired[0].lastReader == 5);
}

TEST_CASE("RT unused snapshots never prevent retirement but copy completion still does") {
    RtScene scene(3);
    scene.request(0, 1, 1);
    scene.markBuilt(0, 1, 4096, 0);
    scene.queueCompaction(0, 1024);
    CHECK(scene.markCompacted(0, scene.mesh(0).version, 2, 1024, 9));
    scene.completeFrame(8);
    CHECK(scene.collectRetired().empty());
    scene.completeFrame(9);
    REQUIRE(scene.collectRetired().size() == 1);
}

TEST_CASE("RT stale compaction results cannot replace rebuilt resources") {
    RtScene scene;
    scene.request(0, 1, 1);
    scene.markBuilt(0, 1, 4096, 0);
    scene.queueCompaction(0, 1024);
    const auto obsolete = scene.mesh(0).version;
    scene.request(0, 2, 2);
    scene.markBuilt(0, 2, 8192, 1);
    scene.queueCompaction(0, 2048);
    CHECK_FALSE(scene.markCompacted(0, obsolete, 3, 1024, 2));
    CHECK(scene.mesh(0).resourceID == 2);
    CHECK(scene.mesh(0).state == RtBlasState::CompactionQueued);
    CHECK(scene.retiredCount() == 1);
}

TEST_CASE("RT vertex revisions cancel stale compaction queries") {
    RtScene scene;
    scene.request(0, 1, 1);
    scene.markBuilt(0, 1, 4096, 0);
    scene.queueCompaction(0, 1024);
    scene.request(0, 1, 2);
    CHECK(scene.mesh(0).state == RtBlasState::Built);
    CHECK_FALSE(scene.markCompacted(0, scene.mesh(0).version, 2, 1024, 1));
    REQUIRE(scene.plan().size() == 1);
    CHECK(scene.plan()[0].kind == RtWorkKind::Refit);
}

TEST_CASE("RT TLAS rebuild policy tracks capacity table publication and explicit interval") {
    RtScene scene;
    scene.request(0, 1, 1);
    scene.markBuilt(0, 100, 4096, 0);
    CHECK(scene.tlasAction(0, 0, 0) == RtTlasAction::None);
    CHECK(scene.tlasAction(0, 100, 0) == RtTlasAction::Build);
    scene.commitSnapshot(0, 100, 0);
    scene.completeFrame(0);
    CHECK(scene.tlasAction(0, 100, 3) == RtTlasAction::Refit);
    CHECK(scene.tlasAction(0, 101, 3) == RtTlasAction::Build);
    CHECK(scene.tlasAction(0, 100, 1000, 0) == RtTlasAction::Refit);
    CHECK(scene.tlasAction(0, 100, 12, 12) == RtTlasAction::Build);
    scene.commitSnapshot(0, 100, 12, RtTlasAction::Build);
    scene.completeFrame(12);
    CHECK(scene.tlasAction(0, 100, 13, 12) == RtTlasAction::Refit);
    scene.queueCompaction(0, 2048);
    scene.markCompacted(0, scene.mesh(0).version, 101, 2048, 13);
    CHECK(scene.tlasAction(0, 100, 14) == RtTlasAction::Build);
    CHECK_THROWS(scene.commitSnapshot(0, 100, 14, RtTlasAction::Refit));
}

TEST_CASE("RT registry enforces ownership and safe ring replacement") {
    RtScene scene(1);
    CHECK_THROWS(RtScene(0));
    scene.request(0, 1, 1);
    CHECK_THROWS(scene.markBuilt(0, 0, 4096, 0));
    CHECK_THROWS(scene.markBuilt(0, 1, 0, 0));
    scene.markBuilt(0, 1, 4096, 0);
    CHECK_THROWS(scene.markBuilt(0, 1, 4096, 1));
    CHECK_THROWS(scene.queueCompaction(0, 8192));
    scene.commitSnapshot(0, 1, 0);
    CHECK_THROWS(scene.commitSnapshot(0, 1, 1));
    scene.completeFrame(0);
    scene.commitSnapshot(0, 0, 1);
    CHECK(scene.tlasAction(0, 1, 2) == RtTlasAction::Build);
    scene.remove(0, 2);
    scene.request(1, 1, 1);
    CHECK_THROWS(scene.markBuilt(1, 1, 4096, 3)); // Retired ID still owned.
    scene.completeFrame(2);
    CHECK(scene.collectRetired().size() == 1);
    CHECK_NOTHROW(scene.markBuilt(1, 1, 4096, 3));
    CHECK_THROWS(scene.completeFrame(1));
}

TEST_CASE("RT output work vectors reuse preallocated capacity") {
    RtScene scene;
    scene.request(0, 1, 1);
    std::vector<RtWork> work;
    work.reserve(8);
    const auto* address = work.data();
    scene.plan({}, work);
    CHECK(work.data() == address);
    scene.markBuilt(0, 10, 4096, 0);
    scene.plan({}, work);
    CHECK(work.empty());
    CHECK_FALSE(scene.hasWork());
    CHECK(work.data() == address);
}

TEST_CASE("RT refit invalidates old compaction query and snapshots retain the same resource ID") {
    RtScene scene(2);
    scene.request(0, 1, 1);
    scene.markBuilt(0, 100, 4096, 0);
    scene.commitSnapshot(0, 10, 0);
    const auto beforeRefit = scene.mesh(0).version;
    scene.request(0, 1, 2);
    scene.markRefitted(0, 1);
    const auto afterRefit = scene.mesh(0).version;
    CHECK(afterRefit != beforeRefit);
    CHECK_FALSE(scene.queueCompaction(0, 2048, beforeRefit));
    CHECK(scene.queueCompaction(0, 2048, afterRefit));
    CHECK_FALSE(scene.markCompacted(0, beforeRefit, 101, 2048, 2));
    CHECK(scene.markCompacted(0, afterRefit, 101, 2048, 2));
    scene.completeFrame(2);
    CHECK(scene.collectRetired().empty()); // Slot 0 carries the pre-refit version, SAME resource.
    scene.clearSnapshot(0);
    auto retired = scene.collectRetired();
    REQUIRE(retired.size() == 1);
    CHECK(retired[0].resourceID == 100);
}

TEST_CASE("RT explicit rebuild request does not falsify topology revision") {
    RtScene scene;
    scene.request(0, 9, 10);
    scene.markBuilt(0, 100, 4096, 0);
    scene.requestRebuild(0);
    const auto work = scene.plan();
    REQUIRE(work.size() == 1);
    CHECK(work[0].kind == RtWorkKind::Build);
    CHECK(work[0].topologyRevision == 9);
    CHECK_THROWS(scene.markRefitted(0, 1));
    scene.markBuilt(0, 101, 4096, 1);
    CHECK_FALSE(scene.hasWork());
    CHECK(scene.mesh(0).topologyRevision == 9);
}
