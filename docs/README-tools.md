# MLIR tools

[Getting started](getting-started.md) · [MLIR concepts](README-summary.md) · [Writing MLIR](README-writing-mlir.md) · [Build](../README.md)

After [build](../README.md), binaries are in `llvm-project/build/bin/`.

- [Where they live](#where-they-live)
- [Map](#map)
- [Optimize vs lower vs translate](#optimize-vs-lower-vs-translate)
- [How they fit](#how-they-fit)
- [mlir-opt](#1-mlir-opt)
- [Debugging the pipeline](#debugging-the-pipeline)
- [mlir-translate](#2-mlir-translate)
- [mlir-cpu-runner](#3-mlir-cpu-runner--mlir-runner)
- [FileCheck and lit](#4-filecheck-and-lit)
- [mlir-tblgen](#5-mlir-tblgen)
- [mlir-lsp-server](#6-mlir-lsp-server)
- [Other](#7-other)
- [Commands on this machine](#8-commands-on-this-machine)

---

## Where they live

```text
llvm-project/build/bin/
  mlir-opt
  mlir-translate
  mlir-cpu-runner   (or mlir-runner, depending on LLVM version)
  mlir-tblgen
  mlir-lsp-server
  llvm-lit
  FileCheck
```

`mlir-opt` **transforms** IR. It does not run your program. To execute, lower to the `llvm` dialect, then use **mlir-cpu-runner** (JIT) or **mlir-translate** → LLVM IR → `lli` / `llc`.

---

## Map

| Tool | Job | You use it to |
| --- | --- | --- |
| **mlir-opt** | Parse, run passes, print IR | Optimize or lower `.mlir` |
| **mlir-translate** | Convert between formats | MLIR `llvm` dialect ↔ LLVM IR |
| **mlir-cpu-runner** | JIT-execute | See a numeric result |
| **FileCheck** | Match `// CHECK` lines | Assert on printed IR |
| **llvm-lit** | Run `// RUN:` lines | PASS / FAIL a folder of tests |
| **mlir-tblgen** | `.td` → C++ | Write a dialect later |
| **mlir-lsp-server** | Language server | Hover / go-to in the editor |

They do not overlap: **opt** rewrites (optimize *and* lower), **translate** changes format, **runner** executes, **lit** orchestrates tests.

---

## Optimize vs lower vs translate

Passes are mostly **two kinds**. Both run on **mlir-opt**.

| Kind | What it does | Still MLIR? | Tool |
| --- | --- | --- | --- |
| **Optimize** | Simplify IR at the **same** level | Yes, same dialects | **mlir-opt** |
| **Lower** | Rewrite high-level ops into **lower** dialects | Yes, different dialects | **mlir-opt** |

**mlir-translate is not lowering.** Lowering is `scf` → `cf` → `llvm` *inside MLIR*. Translate only changes **format** after that: MLIR `llvm` dialect text ↔ LLVM IR (`.ll`). It does not run `--canonicalize` or `--convert-scf-to-cf`.

```text
optimize     mlir-opt --canonicalize --cse --symbol-dce
lower        mlir-opt --convert-scf-to-cf --convert-arith-to-llvm
format       mlir-translate --mlir-to-llvmir     ← not a pass
```

Same idea as [passes](README-summary.md#3-pass).

---

## How they fit

```text
.mlir
  │
  ├─ mlir-opt                  parse / optimize / lower, print IR
  │     │
  │     ├─ FileCheck           does this dump match CHECK lines?
  │     └─ llvm-lit            run many such files
  │
  ├─ mlir-opt → llvm dialect
  │     ├─ mlir-translate --mlir-to-llvmir     → LLVM IR (.ll)
  │     └─ mlir-cpu-runner                     → JIT, print a result
  │
  └─ mlir-tblgen               only when you define ops in TableGen
```

```text
write IR → mlir-opt (same or lower dialect) → print
                                              → translate to LLVM IR
                                              → JIT with the runner
```

---

## 1. mlir-opt

The main tool. Reads `.mlir`, optionally runs **passes** (optimize *or* lower), prints IR (or writes `-o`). `--help` lists registered passes. Names match what you later put in a `PassManager`.

### Parse and print (round-trip)

No pass flags = parse, verify, print. Confirms the file is valid.

```bash
mlir-opt a.mlir
mlir-opt a.mlir -o out.mlir
```

`mlir-opt` often pretty-prints: it may insert an explicit `module { }` and shorten `func.call` to `call`.

### Pass flags

```bash
mlir-opt a.mlir --canonicalize
mlir-opt a.mlir --cse
mlir-opt a.mlir --symbol-dce
mlir-opt a.mlir --convert-scf-to-cf
```

**Optimize** (same dialects, simpler IR):

**`--canonicalize`**  
Folds and simplifies ops using their canonical patterns. Dialects stay the same: `20 + 30` becomes a constant `50`, redundant ops disappear. Run it often, including again after a lowering pass.

**`--cse`**  
Common subexpression elimination: if two ops compute the same thing, keep one and reuse the SSA value. Safe for ops without side effects (`Pure`). Does not change dialect.

**`--symbol-dce`**  
Dead **symbol** elimination, not SSA-value DCE. Deletes unused `private` functions (and other symbols) that nothing in the module references. Public functions stay, even if unused in this file, because a later linker or caller might need them.

```mlir
module {
  func.func private @dead() {   // never called → removed by --symbol-dce
    return
  }
  func.func @keep() -> i32 {    // public → kept
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}
```

`--canonicalize` can drop dead **ops** inside a function (`Pure` values nobody uses). `--symbol-dce` drops dead **functions**. They complement each other.

**Lower** (high dialect → lower dialect, still MLIR, still mlir-opt):

**`--convert-scf-to-cf`**  
Rewrites structured control flow into a CFG. `scf.if` / `scf.for` become `cf.cond_br` / `cf.br` and blocks. Same program, lower abstraction.

**`--convert-arith-to-llvm`**  
Rewrites scalar math into the `llvm` dialect. `arith.addi` becomes `llvm.add` (and similarly for other arith ops). Needed before `mlir-translate --mlir-to-llvmir`.

**`--convert-func-to-llvm`**  
Rewrites `func.func` / `func.return` / `func.call` into `llvm.func` / `llvm.return` / `llvm.call`. Together with arith (and cf) conversion, this is how a module becomes “LLVM dialect MLIR.”

**`--one-shot-bufferize`**  
Bufferization: tensor **values** become memref **buffers**. After this, later passes can emit loads, stores, and allocs. It is lowering of representation, not a format change.

**`--reconcile-unrealized-casts`**  
Conversion often inserts `builtin.unrealized_conversion_cast` as a temporary bridge between types. This pass removes the ones that now match. Run it last in a “to LLVM dialect” pipeline.

### Pipeline (several passes, in order)

```bash
mlir-opt a.mlir --canonicalize --cse --symbol-dce --convert-scf-to-cf
```

Or one pipeline string (order is left to right, nested on a parent op):

```bash
mlir-opt a.mlir --pass-pipeline='builtin.module(canonicalize,cse,symbol-dce,convert-scf-to-cf)'
```

Optimize, then lower, then optimize again at the new level.

### Debugging the pipeline

Two different questions: **which passes will run**, and **what did the IR look like after each one**.

**`--dump-pass-pipeline`**  
Prints the pass manager as text: the list of passes, nested on which op (`builtin.module`, `func.func`, …). It does **not** print IR. Use it to confirm order, nesting, and that the flags you passed actually built the pipeline you think you built.

```bash
mlir-opt a.mlir --canonicalize --cse --symbol-dce --dump-pass-pipeline
```

```text
Pass Manager with 1 passes:
builtin.module(
  canonicalize,
  cse,
  symbol-dce
)
```

**`--mlir-print-ir-after-all`**  
After **every** pass, prints a banner (`// -----// IR Dump After Canonicalizer`) and the IR at that point. This is how you see which pass folded a constant, deleted a function, or introduced `llvm.*` ops. Output can be large; scroll dump-to-dump.

```bash
mlir-opt a.mlir --canonicalize --cse --symbol-dce --mlir-print-ir-after-all
```

```text
// -----// IR Dump After Canonicalizer (builtin.module op)
module {
  func.func @keep() -> i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}
// -----// IR Dump After CSE (builtin.module op)
...
```

Use them together: `--dump-pass-pipeline` is the **plan**; `--mlir-print-ir-after-all` is the **trace**.

Related:

**`--mlir-print-ir-before-all`**  
Same dumps, but **before** each pass. Pair with after-all when a pass crashes: before-all is the IR that went in.

**`--mlir-print-ir-after-change`**  
Like after-all, but skips passes that left the IR unchanged. Quieter when you only care about passes that did work.

**`--mlir-print-ir-after=cse`** / **`--mlir-print-ir-before=cse`**  
Dump only around one pass name. Use this when after-all is too noisy.

**`--mlir-disable-threading`**  
Runs passes on one thread so dumps do not interleave. Turn this on whenever you print IR around passes.

```bash
mlir-opt a.mlir --canonicalize --cse --symbol-dce \
  --dump-pass-pipeline --mlir-print-ir-after-all --mlir-disable-threading
```

### Other mlir-opt flags

**`-o file`**  
Write the resulting IR to a file instead of stdout. Useful when you will feed that file to another tool or to FileCheck via `%t`.

**`--split-input-file`**  
Treat `// -----` as a separator: each chunk is parsed and run independently. MLIR tests pack many small cases in one file this way.

**`--verify-diagnostics`**  
For tests that *expect* a verifier or pass error. FileCheck then matches `expected-error` comments instead of requiring a clean parse.

**`--show-dialects`**  
Lists dialects linked into this `mlir-opt`. If a dialect is missing here, the binary cannot parse those ops.

**`--allow-unregistered-dialect`**  
Parse ops whose dialect is not registered (`foo.bar`). Fine for sketches; the verifier knows much less about those ops.

```bash
mlir-opt --show-dialects
```

### It does not execute

```text
arith.constant 20 + 30   →  canonicalize  →  arith.constant 50
```

That is a **compile-time** fold. Nobody printed `50` as a program result. To run the function, see [mlir-cpu-runner](#3-mlir-cpu-runner--mlir-runner).

---

## 2. mlir-translate

**mlir-opt** stays in MLIR (maybe a lower dialect). **mlir-translate** changes **format**. It does not optimize or lower dialects.

**`--mlir-to-llvmir`**  
Emit LLVM IR text (`.ll`) from MLIR. The input must already be the **`llvm` dialect** (and LLVM types). Translate will not run `--convert-arith-to-llvm` for you; do that with `mlir-opt` first.

**`--import-llvm`**  
The opposite direction: LLVM IR → MLIR `llvm` dialect. Use it when you already have `.ll` and want to inspect or pass it through MLIR. You still get low-level `llvm.*` ops, not `linalg` or `arith`.

```bash
mlir-opt a.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
| mlir-translate --mlir-to-llvmir
```

If you still have `scf` / `cf` / `memref`, convert those first (`--convert-scf-to-cf`, `--convert-cf-to-llvm`, …). A leftover high-level op → translate error.

Then LLVM tools: `lli` (JIT the `.ll`) or `llc` (object / assembly).

```text
.mlir → mlir-opt (to llvm dialect) → mlir-translate → .ll → lli / llc
```

---

## 3. mlir-cpu-runner / mlir-runner

JIT-executes IR that has been lowered to the **`llvm` dialect**. Prints the **return value** of an entry function.

The binary is named `mlir-cpu-runner` or `mlir-runner` depending on LLVM version. Same role.

```bash
mlir-opt a.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
| mlir-cpu-runner -e add -entry-point-result=i32
```

**`-e name`**  
Which function to JIT and call (the symbol, like `@add`). That function is the program’s entry for this run.

**`-entry-point-result=i32`**  
The result type of that entry function (`i32`, `f32`, `void`, …). It must match the IR; `@add() -> i32` with `-entry-point-result=f32` fails.

**`-shared-libs=…`**  
Extra runtime `.so` files, usually `libmlir_c_runner_utils` and `libmlir_runner_utils`. Needed when the IR calls print / unranked-memref helpers.

The entry function must match `-entry-point-result`. `@add() -> i32` with `-entry-point-result=f32` fails.

Need `printmemref` / unranked memref helpers? Pass the runner utils from the same build:

```text
-shared-libs=build/lib/libmlir_c_runner_utils.so,build/lib/libmlir_runner_utils.so
```

**opt vs runner:** opt rewrites and prints IR. Runner executes and prints a **value**.

---

## 4. FileCheck and lit

Testing the **printed IR**, not running the program.

| Tool | Job |
| --- | --- |
| **mlir-opt** | Produce the dump |
| **FileCheck** | Match `// CHECK` / `CHECK-LABEL` / `CHECK-NEXT` |
| **llvm-lit** | Find files, run `// RUN:`, report PASS / FAIL |

```text
// RUN: mlir-opt %s --canonicalize | FileCheck %s
```

Full directives, regex SSA captures, and lit reports: [Writing MLIR — FileCheck](README-writing-mlir.md#7-filecheck--not-mlir-syntax) and [lit](README-writing-mlir.md#8-lit--llvm-integrated-tester).

Keep tests under `mlir/test/` so `lit.cfg.py` is found.

---

## 5. mlir-tblgen

Ops, types, and dialects are often declared in **TableGen** (`.td`). `mlir-tblgen` generates C++ (op classes, builders, verifiers).

You do not need this to **write and run** `.mlir` files. You need it when you **add a dialect**.

```text
MyOps.td  →  mlir-tblgen  →  MyOps.h.inc / MyOps.cpp.inc  →  compile into a pass / tool
```

Typical invocations (CMake already wraps these):

```bash
mlir-tblgen --gen-op-decls MyOps.td
mlir-tblgen --gen-op-defs MyOps.td
mlir-tblgen --gen-dialect-decls MyOps.td
```

`--help` lists generators. The `.td` style for `arith.addi` is in [traits](README-summary.md#4-trait).

---

## 6. mlir-lsp-server

Language server for `.mlir` in the editor (diagnostics, hover, go-to). Point the MLIR / LLVM VS Code (or similar) extension at:

```text
llvm-project/build/bin/mlir-lsp-server
```

Same parser and verifier as `mlir-opt`. It does not run passes unless you wire that in the editor.

---

## 7. Other

| Tool | When you need it |
| --- | --- |
| **mlir-reduce** | Shrink a crashing `.mlir` toward a minimal repro |
| **mlir-pdll** | Compile PDLL rewrite patterns (alternative to C++ `RewritePattern`) |
| **FileCheck** / **llvm-lit** | Always, once you test IR |

---

## 8. Commands on this machine

Tools are in the LLVM build `bin` directory. If the clone is not under `/home`, swap the prefix.

```bash
BIN=/home/llvm-project/build/bin
MLIR=/home/llvm-project/mlir/test/a.mlir

# parse + print
$BIN/mlir-opt $MLIR

# optimize (same dialect)
$BIN/mlir-opt $MLIR --canonicalize
$BIN/mlir-opt $MLIR --canonicalize --cse --symbol-dce

# which passes will run (no IR dump)
$BIN/mlir-opt $MLIR --canonicalize --cse --symbol-dce --dump-pass-pipeline

# IR after every pass
$BIN/mlir-opt $MLIR --canonicalize --cse --symbol-dce \
  --mlir-print-ir-after-all --mlir-disable-threading

# dialects this binary knows
$BIN/mlir-opt --show-dialects

# lower toward LLVM, then LLVM IR text
$BIN/mlir-opt $MLIR \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
| $BIN/mlir-translate --mlir-to-llvmir

# JIT a function named @add that returns i32
$BIN/mlir-opt $MLIR \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
| $BIN/mlir-cpu-runner -e add -entry-point-result=i32

# lit + FileCheck (needs lit.cfg.py nearby)
$BIN/llvm-lit -v $MLIR
```

If `mlir-cpu-runner` is missing, try `$BIN/mlir-runner` with the same flags.
