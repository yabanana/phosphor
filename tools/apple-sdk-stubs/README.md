# Apple SDK stubs (Linux syntax checking only)

Minimal stand-ins for the handful of Apple system headers that metal-cpp
includes. They let `clang++ -fsyntax-only` type-check Metal host code on a
Linux machine (for example a cloud AI session with no Mac). Nothing built with
these stubs can link or run; the real build always uses the macOS SDK.

Used by `tools/check_metal_syntax.sh`.
