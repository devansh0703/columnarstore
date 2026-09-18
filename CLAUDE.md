# CLAUDE.md

Project: ColumnarStore — SIMD-accelerated columnar storage library for analytical workloads (C++20, CMake).

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
/usr/bin/ctest --test-dir build --output-on-failure
```

- Debug builds enable ASan/UBSan; use `/usr/bin/ctest` — the user's `~/.local/bin/ctest` shim is broken (missing python cmake module).
- CI lint job builds with `-Werror`; never introduce new compiler warnings.
- Requires: C++20 compiler, CMake 3.20+, libzstd-dev, liblz4-dev, libgtest-dev, libbenchmark-dev.
