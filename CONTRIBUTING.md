# Contributing to Phosphor

[GitHub Issues](https://github.com/yabanana/phosphor/issues) is where we track
reproducible bugs, device results, and scoped changes. Use [Discussions](https://github.com/yabanana/phosphor/discussions)
for questions, capture walkthroughs, and proposals that need feedback.

## First contributions

| If you have | Start here | What to share |
| --- | --- | --- |
| A physical M3 or M4 Mac | [Run Many Lights](#many-lights-on-a-physical-mac) | Chip, macOS and Xcode versions, commit, and benchmark JSON |
| A Linux machine with Clang | [Check the portable build](docs/RENDERER.md#building-on-linux-no-gpu) | Distribution, compiler version, and the first command that fails or needs clarification |
| Metal or MSL experience | [Capture one pass](#inspect-one-metal-pass) | Pass name, time, resolution, and one observation from the capture |

Find open work tagged [good first issue](https://github.com/yabanana/phosphor/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or [help wanted](https://github.com/yabanana/phosphor/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
For a code change, leave a note on the issue so others can see who is working
on it.

## Many Lights on a physical Mac

Build Release from the [quickstart](README.md#run-on-a-mac). Run the same
workload below at 1280×720. It uses a procedural scene, so you can run it
without downloading assets. Use the default physical GPU capabilities.

```sh
./build/phosphor --bench 5 --offscreen --resolution 1280x720 \
  --render-path visibility --lighting restir --rt on \
  --frames 256 --warmup 128 --fixed-timestep --no-ui --no-vsync \
  --gpu-timing-serial --report build/many-lights.json
```

Post the JSON and include the output of `git rev-parse --short HEAD` and
`xcodebuild -version`, your chip and GPU core count, macOS version, and
whether the Mac was on AC power. A repeat under the same conditions helps
show run-to-run spread. If it fails, post the first error and exact command.
These are exploratory measurements. The [hardware validation policy](docs/plans/HARDWARE_VALIDATION.md)
defines what counts as certification.

## Inspect one Metal pass

Start with the Many Lights command above. Add
`MTL_CAPTURE_ENABLED=1` before the executable and
`--gpu-capture --gpu-capture-frame 140 --gpu-capture-dir build/captures`
to the arguments. Open the trace in Xcode and inspect the ReSTIR DI passes.
Share the pass name, one observation about its workload, and the chip, OS,
commit, and resolution. Use the ordinary benchmark run for timing comparisons.

## Pull requests

Target `main` in the [official repository](https://github.com/yabanana/phosphor).
Link the issue, describe the change, and list what you ran. Run
`ctest --test-dir build --output-on-failure` for portable changes. For
renderer changes, include the bench and flags, device, and a capture or
benchmark as appropriate. Report the checks you were able to run. Link the
phase ID if the change implements an existing roadmap task.

Submit code and documentation that you have the right to contribute under the
[project license](LICENSE). Keep any applicable third-party notices.
