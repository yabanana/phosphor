#include "core/memory/tlsf_allocator.h"

#include <algorithm>
#include <bit>
#include <limits>

namespace phosphor {

namespace {

u32 floorLog2(u64 v) { return 63u - static_cast<u32>(std::countl_zero(v)); }

bool isPowerOfTwo(u64 v) { return v != 0 && (v & (v - 1)) == 0; }

u64 alignUp(u64 v, u64 alignment) { return (v + alignment - 1) & ~(alignment - 1); }

} // namespace

TlsfAllocator::TlsfAllocator(u64 capacity) : capacity_(capacity) {
    reset();
}

void TlsfAllocator::reset() {
    nodes_.clear();
    freeNodes_.clear();
    flBitmap_ = 0;
    slBitmaps_.fill(0);
    for (auto& row : heads_) row.fill(NONE);
    usedBytes_       = 0;
    allocationCount_ = 0;
    if (capacity_ == 0) return;

    const u32 node = newNode();
    nodes_[node].offset = 0;
    nodes_[node].size   = capacity_;
    insertFree(node);
}

// Sizes below SL_COUNT map linearly into first level 0; larger sizes use the
// position of their top bit (first level) and the next SL_BITS bits (second).
TlsfAllocator::Index TlsfAllocator::mappingInsert(u64 size) {
    if (size < SL_COUNT) return {0, static_cast<u32>(size)};
    const u32 log2 = floorLog2(size);
    return {log2 - SL_BITS + 1, static_cast<u32>((size >> (log2 - SL_BITS)) ^ SL_COUNT)};
}

// Round the size up to the next list boundary, so that any block in the
// resulting list is large enough (good-fit search without scanning).
bool TlsfAllocator::mappingSearch(u64 size, Index& index) {
    if (size >= SL_COUNT) {
        const u64 round = (u64{1} << (floorLog2(size) - SL_BITS)) - 1;
        if (size > std::numeric_limits<u64>::max() - round) return false;
        size += round;
    }
    index = mappingInsert(size);
    return index.fl < FL_COUNT;
}

u32 TlsfAllocator::newNode() {
    u32 node;
    if (!freeNodes_.empty()) {
        node = freeNodes_.back();
        freeNodes_.pop_back();
        nodes_[node] = Node{};
    } else {
        node = static_cast<u32>(nodes_.size());
        nodes_.emplace_back();
    }
    nodes_[node].live = true;
    return node;
}

void TlsfAllocator::releaseNode(u32 node) {
    nodes_[node].live = false;
    freeNodes_.push_back(node);
}

void TlsfAllocator::insertFree(u32 node) {
    Node& n = nodes_[node];
    const Index idx = mappingInsert(n.size);
    const u32 head = heads_[idx.fl][idx.sl];
    n.isFree   = true;
    n.prevFree = NONE;
    n.nextFree = head;
    if (head != NONE) nodes_[head].prevFree = node;
    heads_[idx.fl][idx.sl] = node;
    flBitmap_ |= u64{1} << idx.fl;
    slBitmaps_[idx.fl] |= 1u << idx.sl;
}

void TlsfAllocator::removeFree(u32 node) {
    Node& n = nodes_[node];
    if (n.prevFree != NONE) {
        nodes_[n.prevFree].nextFree = n.nextFree;
    } else {
        const Index idx = mappingInsert(n.size);
        heads_[idx.fl][idx.sl] = n.nextFree;
        if (n.nextFree == NONE) {
            slBitmaps_[idx.fl] &= ~(1u << idx.sl);
            if (slBitmaps_[idx.fl] == 0) flBitmap_ &= ~(u64{1} << idx.fl);
        }
    }
    if (n.nextFree != NONE) nodes_[n.nextFree].prevFree = n.prevFree;
    n.isFree   = false;
    n.prevFree = NONE;
    n.nextFree = NONE;
}

u32 TlsfAllocator::findFree(Index& index) const {
    // Same first level, second level >= requested.
    u32 slMap = slBitmaps_[index.fl] & (~0u << index.sl);
    if (slMap == 0) {
        // Any larger first level.
        const u64 flMap = index.fl + 1 < 64 ? flBitmap_ & (~u64{0} << (index.fl + 1)) : 0;
        if (flMap == 0) return NONE;
        index.fl = static_cast<u32>(std::countr_zero(flMap));
        slMap    = slBitmaps_[index.fl];
    }
    index.sl = static_cast<u32>(std::countr_zero(slMap));
    return heads_[index.fl][index.sl];
}

u32 TlsfAllocator::splitTail(u32 node, u64 size) {
    if (nodes_[node].size == size) return NONE;
    const u32 tail = newNode(); // may reallocate nodes_: index, don't hold references
    Node& n = nodes_[node];
    Node& t = nodes_[tail];
    t.offset   = n.offset + size;
    t.size     = n.size - size;
    t.prevPhys = node;
    t.nextPhys = n.nextPhys;
    if (n.nextPhys != NONE) nodes_[n.nextPhys].prevPhys = tail;
    n.nextPhys = tail;
    n.size     = size;
    return tail;
}

TlsfAllocator::Allocation TlsfAllocator::allocate(u64 size, u64 alignment) {
    if (size == 0 || !isPowerOfTwo(alignment) || size > capacity_) return {};

    // Searching for size + alignment - 1 guarantees an aligned fit in any
    // block of the list found, whatever the block's offset.
    const u64 slack = alignment - 1;
    if (size > std::numeric_limits<u64>::max() - slack) return {};
    Index idx;
    if (!mappingSearch(size + slack, idx)) return {};
    u32 node = findFree(idx);
    if (node == NONE) return {};
    removeFree(node);

    // Front padding up to the aligned offset becomes its own free block.
    const u64 padding = alignUp(nodes_[node].offset, alignment) - nodes_[node].offset;
    if (padding > 0) {
        const u32 aligned = splitTail(node, padding);
        insertFree(node); // the padding; its previous neighbour is used, so no merge
        node = aligned;
    }
    const u32 tail = splitTail(node, size);
    if (tail != NONE) insertFree(tail); // next neighbour was used: no merge needed

    Node& n  = nodes_[node];
    n.request = size;
    usedBytes_ += n.size;
    ++allocationCount_;
    return {n.offset, size, node};
}

void TlsfAllocator::free(Handle handle) {
    if (handle >= nodes_.size() || !nodes_[handle].live || nodes_[handle].isFree) return;

    u32 node = handle;
    usedBytes_ -= nodes_[node].size;
    --allocationCount_;
    nodes_[node].request = 0;

    // Merge with the next physical block.
    const u32 next = nodes_[node].nextPhys;
    if (next != NONE && nodes_[next].isFree) {
        removeFree(next);
        nodes_[node].size += nodes_[next].size;
        nodes_[node].nextPhys = nodes_[next].nextPhys;
        if (nodes_[next].nextPhys != NONE) nodes_[nodes_[next].nextPhys].prevPhys = node;
        releaseNode(next);
    }
    // Merge into the previous physical block.
    const u32 prev = nodes_[node].prevPhys;
    if (prev != NONE && nodes_[prev].isFree) {
        removeFree(prev);
        nodes_[prev].size += nodes_[node].size;
        nodes_[prev].nextPhys = nodes_[node].nextPhys;
        if (nodes_[node].nextPhys != NONE) nodes_[nodes_[node].nextPhys].prevPhys = prev;
        releaseNode(node);
        node = prev;
    }
    insertFree(node);
}

TlsfAllocator::Stats TlsfAllocator::stats() const {
    Stats s;
    s.capacity        = capacity_;
    s.usedBytes       = usedBytes_;
    s.freeBytes       = capacity_ - usedBytes_;
    s.allocationCount = allocationCount_;
    // O(free blocks): meant for the UI and tests, not the allocation path.
    for (u32 fl = FL_COUNT; fl-- > 0;) {
        if (!(flBitmap_ & (u64{1} << fl))) continue;
        for (u32 sl = 0; sl < SL_COUNT; ++sl) {
            for (u32 n = heads_[fl][sl]; n != NONE; n = nodes_[n].nextFree) {
                ++s.freeBlockCount;
                s.largestFreeBlock = std::max(s.largestFreeBlock, nodes_[n].size);
            }
        }
    }
    return s;
}

bool TlsfAllocator::validate(std::string* error) const {
    auto fail = [&](const std::string& what) {
        if (error) *error = what;
        return false;
    };
    if (capacity_ == 0) return nodes_.empty() ? true : fail("nodes in an empty range");

    // Physical chain: starts at offset 0, contiguous, covers the range.
    u32 first = NONE;
    for (u32 i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].live && nodes_[i].prevPhys == NONE) {
            if (first != NONE) return fail("more than one physical head");
            first = i;
        }
    }
    if (first == NONE) return fail("no physical head");

    u64 offset = 0, used = 0;
    u32 count = 0, freeBlocks = 0, liveNodes = 0;
    for (u32 i = 0; i < nodes_.size(); ++i) liveNodes += nodes_[i].live ? 1 : 0;
    u32 prev = NONE;
    for (u32 n = first; n != NONE; prev = n, n = nodes_[n].nextPhys) {
        const Node& b = nodes_[n];
        if (!b.live) return fail("dead node in physical chain");
        if (b.prevPhys != prev) return fail("broken prevPhys link");
        if (b.offset != offset) return fail("gap or overlap at offset " + std::to_string(b.offset));
        if (b.size == 0) return fail("zero-sized block");
        if (b.isFree) {
            if (prev != NONE && nodes_[prev].isFree) return fail("adjacent free blocks not merged");
            ++freeBlocks;
        } else {
            if (b.request == 0 || b.request > b.size) return fail("used block smaller than its request");
            used += b.size;
            ++count;
        }
        offset += b.size;
        if (--liveNodes, liveNodes > nodes_.size()) return fail("cycle in physical chain");
    }
    if (offset != capacity_) return fail("blocks do not cover the range");
    if (liveNodes != 0) return fail("live node outside the physical chain");
    if (used != usedBytes_ || count != allocationCount_) return fail("usage counters out of sync");

    // Free lists: every entry free, in the list its size maps to, bitmaps exact.
    u32 listed = 0;
    for (u32 fl = 0; fl < FL_COUNT; ++fl) {
        for (u32 sl = 0; sl < SL_COUNT; ++sl) {
            const bool bit = (slBitmaps_[fl] >> sl) & 1u;
            if (bit != (heads_[fl][sl] != NONE)) return fail("second-level bitmap out of sync");
            u32 prevFree = NONE;
            for (u32 n = heads_[fl][sl]; n != NONE; prevFree = n, n = nodes_[n].nextFree) {
                const Node& b = nodes_[n];
                const Index idx = mappingInsert(b.size);
                if (!b.isFree || !b.live) return fail("non-free block in a free list");
                if (idx.fl != fl || idx.sl != sl) return fail("free block in the wrong list");
                if (b.prevFree != prevFree) return fail("broken prevFree link");
                if (++listed > freeBlocks) return fail("free lists longer than the free blocks");
            }
        }
        if (((flBitmap_ >> fl) & 1u) != (slBitmaps_[fl] != 0)) return fail("first-level bitmap out of sync");
    }
    if (listed != freeBlocks) return fail("free block missing from the free lists");
    return true;
}

} // namespace phosphor
