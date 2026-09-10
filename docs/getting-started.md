# Getting started

[Build MLIR](../README.md) first. Then follow this page.

| Step | Page | What you learn |
| --- | --- | --- |
| 1 | [MLIR concepts](README-summary.md) | Why IR, SSA, dialects, operations, passes, traits |
| 2 | [Writing MLIR](README-writing-mlir.md) | `module`, functions, types, `return`, FileCheck, lit |

```mermaid
flowchart LR
  A[".mlir file"] --> B["mlir-opt"]
  B --> C["printed IR"]
  C --> D["FileCheck"]
  D --> E["lit PASS / FAIL"]
```

---

## Suggested order

1. **Build** — [README](../README.md) (`check-mlir` finished, binaries in `build/bin/`)
2. **Concepts** — [MLIR concepts](README-summary.md)  
   IR → SSA → why MLIR → dialects → ops → passes → traits
3. **Writing** — [Writing MLIR](README-writing-mlir.md)  
   `module`, functions, types, `return`
4. **Testing** — same writing page, FileCheck + lit sections  
   `// CHECK` + `// RUN:` + `llvm-lit`

---

## Run the tools

From a build as in the [README](../README.md):

```bash
# See the printed IR
/home/llvm-project/build/bin/mlir-opt /home/llvm-project/mlir/test/a.mlir

# Fold constants (optimize, same dialect)
/home/llvm-project/build/bin/mlir-opt /home/llvm-project/mlir/test/a.mlir --canonicalize

# lit + FileCheck
/home/llvm-project/build/bin/llvm-lit -v /home/llvm-project/mlir/test/a.mlir
```

If your clone is not under `/home`, use `llvm-project/build/bin/mlir-opt` and a `.mlir` file under `mlir/test/`.

| Tool | Job |
| --- | --- |
| **mlir-opt** | Parse, optimize, or lower, then print IR |
| **FileCheck** | Match `// CHECK` patterns against that printout |
| **lit** | Run `// RUN:` lines on many files and report PASS/FAIL |

---

## Symbols in a `.mlir` file

| Prefix | Meaning | Example |
| --- | --- | --- |
| `@` | Symbol (function name) | `@add` |
| `%` | SSA value | `%0` |
| `^` | Block label | `^bb0` |
| `:` | Type | `: i32` |
| `->` | Function result type | `-> i32` |

---

## Notes

| Page | Covers |
| --- | --- |
| [MLIR concepts](README-summary.md) | IR, SSA, dialects (`arith`, `func`, `scf`, `tensor`, `memref`, `linalg`, `affine`), operations, optimize vs lower, traits |
| [Writing MLIR](README-writing-mlir.md) | How to write a file, `func.call`, FileCheck directives, lit `RUN` lines |
