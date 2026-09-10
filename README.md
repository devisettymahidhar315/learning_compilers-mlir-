# MLIR

Notes for building and learning **MLIR** (Multi-Level Intermediate Representation).

After the build, start at **[Getting started](docs/getting-started.md)**.

| | |
| --- | --- |
| **This page** | Clone llvm-project, CMake + Ninja, `check-mlir` |
| [Getting started](docs/getting-started.md) | How to read IR, run `mlir-opt` / lit, and the rest of the notes |

---

## Build MLIR

Needs **CMake**, **Ninja**, **clang**, **lld**, and (optional) **ccache**. This only enables the MLIR project.

```bash
git clone https://github.com/llvm/llvm-project.git

mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON

cmake --build . --target check-mlir
```

`check-mlir` builds MLIR and runs the MLIR lit tests. Tools land in `llvm-project/build/bin/` (`mlir-opt`, `llvm-lit`, `FileCheck`).

The first configure + `check-mlir` takes a long time.

| CMake flag | Meaning |
| --- | --- |
| `-G Ninja` | Use the Ninja generator |
| `-DLLVM_ENABLE_PROJECTS=mlir` | Build MLIR (not Clang, etc.) |
| `-DLLVM_BUILD_EXAMPLES=ON` | Build MLIR examples |
| `-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU"` | Host CPU + NVIDIA + AMD GPU backends |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Keep asserts (useful while learning) |
| `-DCMAKE_C_COMPILER=clang` / `clang++` | Compile LLVM/MLIR with Clang |
| `-DLLVM_ENABLE_LLD=ON` | Link with lld (faster than ld) |
| `-DLLVM_CCACHE_BUILD=ON` | Cache compiles for rebuilds |

**Next:** [Getting started](docs/getting-started.md) — run the tools, then read concepts and how to write `.mlir`.
