#pragma once
// F15 closure (COLOR_0): function-constant presence of the per-vertex colour (interpolants of the vertex / mesh stages, FrameConstants::vertexColors).
// The reserved constant VERTEX_COLOR (shaders/variants.def) is undefined => 0 in every pipeline of a scene without COLOR_0, so everything behind
// kVertexColorOn compiles out.  Needs the generated forward_variants.generated.metal.h (kVertexColor).
constant bool kVertexColorOn = kVertexColor != 0u;
