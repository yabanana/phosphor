// timestamp.metal -- anchor of the commit-start timestamps (F4.1).
//
// Each commit starts with a compute encoder holding one 1-thread dispatch of
// this empty kernel and a timestamp.  An encoder with only the timestamp is
// dropped by the driver (the timestamp is never written, measured), and
// MTL4::CommandBuffer::writeTimestampIntoHeap grows driver bookkeeping on
// reused command buffers forever (measured), hence this anchor.

#include <metal_stdlib>

kernel void timestamp_anchor() {}
