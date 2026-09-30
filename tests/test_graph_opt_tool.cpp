#include <doctest/doctest.h>

#include "graph_opt/plan_merge.h"

using namespace phosphor;
using namespace phosphor::rg;

namespace {
GraphPlan plan(const char* family, u64 key) {
    GraphPlan p;
    p.family = family;
    p.key    = key;
    return p;
}
} // namespace

TEST_CASE("graph_opt mergePlans: replaces same family, keeps others, appends new") {
    const std::vector<GraphPlan> existing{plan("a", 1), plan("b", 2), plan("c", 3)};
    const std::vector<GraphPlan> updates{plan("b", 20), plan("d", 40)};
    const auto merged = mergePlans(existing, updates);
    REQUIRE(merged.size() == 4);
    CHECK(merged[0].family == "a");
    CHECK(merged[0].key == 1);
    CHECK(merged[1].family == "b");
    CHECK(merged[1].key == 20);
    CHECK(merged[2].family == "c");
    CHECK(merged[3].family == "d");
}

TEST_CASE("graph_opt mergePlans: empty inputs and duplicates") {
    CHECK(mergePlans({}, {}).empty());
    CHECK(mergePlans({plan("a", 1)}, {}).size() == 1);
    const auto m = mergePlans({plan("a", 1), plan("a", 2)}, {plan("a", 9), plan("a", 10)});
    REQUIRE(m.size() == 1);
    CHECK(m[0].key == 10); // last update wins, stale duplicates dropped
    const auto n = mergePlans({}, {plan("x", 1), plan("x", 2), plan("y", 3)});
    REQUIRE(n.size() == 2);
    CHECK(n[0].key == 2);
}

TEST_CASE("graph_opt mergePlans: result round-trips through the plan file") {
    const auto merged = mergePlans({plan("a", 1)}, {plan("b", 2)});
    std::vector<GraphPlan> back;
    std::string error;
    REQUIRE_MESSAGE(fromJson(toJson(merged), back, &error), error);
    CHECK(back.size() == 2);
}
