#include "renderer/stochastic_sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace phosphor::di {
u32 stochasticHash(u32 v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}
float whiteSample(u32 x, u32 y, u32 frame, u32 dimension, u32 seed) {
    const u32 bits = stochasticHash(seed ^ stochasticHash(x) ^ stochasticHash(y + 0x9e3779b9u) ^
                                    stochasticHash(frame + 0x632be5abu) ^ stochasticHash(dimension + 0x85157af5u));
    return float(bits >> 8) * (1.0f / 16777216.0f);
}
float StbnMask::sample(u32 x, u32 y, u32 frame, u32 dimension) const {
    if (!config.width || !config.height || !config.frames || !config.dimensions || ranks.empty())
        return whiteSample(x, y, frame, dimension, config.seed);
    const size_t n = size_t(config.width) * config.height * config.frames;
    const size_t index = (((size_t(dimension % config.dimensions) * config.frames + frame % config.frames) *
                          config.height + y % config.height) * config.width + x % config.width);
    // Cranley-Patterson random rotation per dimension AND temporal block preserves
    // uniform marginals while retaining ranks within each generated mask.
    const float rank = float((double(ranks[index]) + 0.5) / n);
    const float rotation = whiteSample(0, 0, frame / config.frames, dimension, config.seed ^ 0xa511e9b3u);
    return rank + rotation - std::floor(rank + rotation);
}

StbnMask generateStbn(const StbnConfig& c) {
    if (!c.width || !c.height || !c.frames || !c.dimensions || !std::isfinite(c.sigmaSpatial) ||
        !std::isfinite(c.sigmaTemporal) || c.sigmaSpatial <= 0 || c.sigmaTemporal <= 0)
        throw std::invalid_argument("invalid STBN configuration");
    const u64 count64 = u64(c.width) * c.height * c.frames;
    // Defined safety bound for the exact CPU generator, not a quality preset.
    if (count64 < 4 || count64 > 16384 || count64 * c.dimensions > 1048576)
        throw std::length_error("STBN generator budget exceeded");
    const u32 n = u32(count64), density = std::max(1u, n / 10u);
    StbnMask result{c, std::vector<u32>(size_t(n) * c.dimensions), true};
    struct Neighbor { u32 index; double energy; };
    std::vector<std::vector<Neighbor>> neighbors(n);
    auto distance = [](u32 a, u32 b, u32 extent) { const u32 d = a > b ? a - b : b - a; return std::min(d, extent - d); };
    // Same spatial plane OR same pixel through time. Include self-energy so
    // occupied-cluster and empty-void ordering agree with void-and-cluster.
    for (u32 i = 0; i < n; ++i) {
        const u32 ix = i % c.width, iy = (i / c.width) % c.height, it = i / (c.width * c.height);
        for (u32 j = 0; j < n; ++j) {
            const u32 jx = j % c.width, jy = (j / c.width) % c.height, jt = j / (c.width * c.height);
            double exponent = 0;
            if (it == jt) {
                const double dx = distance(ix, jx, c.width), dy = distance(iy, jy, c.height);
                exponent = (dx * dx + dy * dy) / (2 * c.sigmaSpatial * c.sigmaSpatial);
            } else if (ix == jx && iy == jy) {
                const double dt = distance(it, jt, c.frames);
                exponent = dt * dt / (2 * c.sigmaTemporal * c.sigmaTemporal);
            } else continue;
            neighbors[i].push_back({j, std::exp(-exponent)});
        }
    }
    for (u32 dimension = 0; dimension < c.dimensions; ++dimension) {
        std::vector<u8> occupied(n, 0);
        std::vector<double> energy(n, 0);
        std::vector<u32> order(n); std::iota(order.begin(), order.end(), 0);
        u32 random = stochasticHash(c.seed ^ stochasticHash(dimension));
        for (u32 i = n - 1; i > 0; --i) {
            random = stochasticHash(random + 0x9e3779b9u);
            const u32 j = u32((u64(random) * (i + 1u)) >> 32);
            std::swap(order[i], order[j]);
        }
        auto toggle = [&](u32 index, bool enabled) {
            occupied[index] = enabled;
            const double sign = enabled ? 1 : -1;
            for (const auto& neighbor : neighbors[index]) energy[neighbor.index] += sign * neighbor.energy;
        };
        auto extreme = [&](bool findCluster) {
            u32 best = ~0u;
            double value = findCluster ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            // Deterministic shuffled tie-breaking, independent per dimension.
            for (u32 index : order) {
                if (bool(occupied[index]) != findCluster) continue;
                if ((findCluster && energy[index] > value) || (!findCluster && energy[index] < value)) {
                    best = index; value = energy[index];
                }
            }
            return best;
        };
        for (u32 i = 0; i < density; ++i) toggle(order[i], true);
        bool converged = false;
        for (u32 iteration = 0; iteration < c.maxRelaxationSwaps; ++iteration) {
            const u32 cluster = extreme(true);
            toggle(cluster, false);
            const u32 voidIndex = extreme(false);
            toggle(voidIndex, true);
            if (voidIndex == cluster) { converged = true; break; }
        }
        result.relaxationConverged &= converged;
        const auto initial = occupied;
        const auto initialEnergy = energy;
        auto* rank = result.ranks.data() + size_t(dimension) * n;
        for (u32 remaining = density; remaining > 0; --remaining) {
            const u32 index = extreme(true); rank[index] = remaining - 1; toggle(index, false);
        }
        occupied = initial; energy = initialEnergy;
        for (u32 next = density; next < n / 2; ++next) {
            const u32 index = extreme(false); rank[index] = next; toggle(index, true);
        }
        // Complement at half occupancy; remove tightest complementary cluster
        // to assign upper ranks. Adding voids all the way to full would not
        // produce the complementary blue-noise distributions.
        for (u32 i = 0; i < n; ++i) occupied[i] = occupied[i] ? 0 : 1;
        std::fill(energy.begin(), energy.end(), 0);
        for (u32 i = 0; i < n; ++i) if (occupied[i])
            for (const auto& neighbor : neighbors[i]) energy[neighbor.index] += neighbor.energy;
        for (u32 next = n / 2; next < n; ++next) {
            const u32 index = extreme(true); rank[index] = next; toggle(index, false);
        }
    }
    return result;
}
} // namespace phosphor::di
