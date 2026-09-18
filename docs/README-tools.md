# MLIR tools

[Getting started](getting-started.md) · [MLIR concepts](README-summary.md) · [Writing MLIR](README-writing-mlir.md) · [Build](../README.md)

After [build](../README.md), binaries are in `llvm-project/build/bin/`.

- [Where they live](#where-they-live)
- [Map](#map)
- [Optimize vs lower vs translate](#optimize-vs-lower-vs-translate)
- [How they fit](#how-they-fit)
- [mlir-opt](#1-mlir-opt)
- [CSE vs canonicalize](#cse-vs-canonicalize)
- [inline](#inline)
- [Pass manager](#pass-manager)
- [Debugging the pipeline](#debugging-the-pipeline)
- [mlir-translate](#2-mlir-translate)
- [mlir-cpu-runner](#3-mlir-cpu-runner--mlir-runner)
- [FileCheck and lit](#4-filecheck-and-lit)
- [mlir-tblgen](#5-mlir-tblgen)
- [mlir-lsp-server](#6-mlir-lsp-server)
- [Other](#7-other)
- [Commands on this machine](#8-commands-on-this-machine)
- [Cheat sheet](#9-cheat-sheet)

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
optimize     mlir-opt --canonicalize --cse --inline --symbol-dce
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
mlir-opt a.mlir --inline
mlir-opt a.mlir --symbol-dce
mlir-opt a.mlir --convert-scf-to-cf
```

**Optimize** (same dialects, simpler IR):

**`--canonicalize`**  
Folds and simplifies ops using their canonical patterns. Dialects stay the same: `20 + 30` becomes a constant `50`, redundant ops disappear. Run it often, including again after a lowering pass.

**`--cse`**  
Common subexpression elimination: if two ops compute the same thing, keep one and reuse the SSA value. Safe for ops without side effects (`Pure`). Does not change dialect.

### CSE vs canonicalize

Same level (optimize), different job. CSE looks **across** ops. Canonicalize looks **at** one op and folds it.

| | **CSE** | **Canonicalize** |
| --- | --- | --- |
| Core job | Merge identical ops | Simplify / fold individual ops |
| `x + 0` → `x` | No | Yes |
| Constant folding (`10 + 10` → `20`) | No | Yes |
| Dedup two identical `muli` / `addi` | Yes | No |
| Compares ops against each other | Yes | No |

#### Example: duplicate constants (`10 + 10`)

```mlir
module {
  func.func @add() -> (i32, i32) {
    %0 = arith.constant 10 : i32
    %1 = arith.constant 10 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = arith.addi %0, %1 : i32
    return %2, %3 : i32, i32
  }
}
```

Two constants that happen to be `10`, and two **identical** adds of those values. CSE and canonicalize do not do the same rewrite.

**`--cse`** — merge duplicates, do **not** fold `10 + 10`:

```bash
mlir-opt a.mlir --cse
```

```mlir
module {
  func.func @add() -> (i32, i32) {
    %c10_i32 = arith.constant 10 : i32
    %0 = arith.addi %c10_i32, %c10_i32 : i32
    return %0, %0 : i32, i32
  }
}
```

`%0` and `%1` were the same constant → one `%c10_i32`. `%2` and `%3` were the same add → one `%0`, returned twice. The add is still there; CSE never computes `20`.

**`--canonicalize`** — fold each add, do **not** keep a redundant `addi`:

```bash
mlir-opt a.mlir --canonicalize
```

```mlir
module {
  func.func @add() -> (i32, i32) {
    %c20_i32 = arith.constant 20 : i32
    return %c20_i32, %c20_i32 : i32, i32
  }
}
```

Each `arith.addi` of two `10`s becomes `arith.constant 20`. The two result values are that same constant. Canonicalize did not “dedup the two adds”; it simplified each add on its own, and the adds disappeared.

#### Example: duplicate `muli` and `x + 0`

This is the opposite split: CSE can merge the two multiplies; canonicalize can fold `x + 0`. Neither pass does the other job.

```mlir
module {
  func.func @demo(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %a = arith.muli %x, %x : i32
    %b = arith.muli %x, %x : i32
    %s = arith.addi %a, %c0 : i32
    %t = arith.addi %s, %b : i32
    return %t : i32
  }
}
```

`mlir-opt` may print `%x` as `%arg0` and `%c0` as `%c0_i32`. Same IR.

| Op in source | `--cse` | `--canonicalize` |
| --- | --- | --- |
| duplicate `muli %x, %x` | merged | left as-is |
| `addi %a, 0` (`x + 0`) | left as-is | folded to `%a` |

**`--cse`** — one multiply; the `+ 0` stays:

```bash
mlir-opt a.mlir --cse
```

```mlir
module {
  func.func @demo(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg0, %arg0 : i32
    %1 = arith.addi %0, %c0_i32 : i32
    %2 = arith.addi %1, %0 : i32
    return %2 : i32
  }
}
```

`%a` and `%b` were the same `muli` → one `%0`, used twice (`%1` and `%2`). CSE does not know that `%x + 0` is `%x`, so `%c0_i32` and that `addi` remain. `%t` is still “(%a + 0) + %a”.

**`--canonicalize`** — drop `+ 0`; two multiplies stay:

```bash
mlir-opt a.mlir --canonicalize
```

```mlir
module {
  func.func @demo(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    %1 = arith.muli %arg0, %arg0 : i32
    %2 = arith.addi %0, %1 : i32
    return %2 : i32
  }
}
```

`arith.addi %a, %c0` is the identity `x + 0` → replaced by `%a`. `%c0` has no users left, so it disappears. Canonicalize never compares the two `muli` ops, so `%x * %x` is still computed twice and then added.

Together they finish the job: `--canonicalize --cse` folds `+ 0` **and** keeps a single `muli`. Canonicalize first is the usual habit.

### inline

**`--inline`**  
Replace a **call** with a copy of the callee’s body. That removes call overhead and exposes the callee’s ops to later `--canonicalize` / `--cse`. It is still an optimize pass: dialects stay the same.

Inlining needs a **caller** and a **callee**. A file with only arithmetic and no `func.call` / `call` does nothing — there is no call site to replace.

| | `--inline` | `--symbol-dce` |
| --- | --- | --- |
| Job | Copy the callee body into the caller | Delete unused **private** symbols |
| Needs | A call op | A function nobody references |
| Deletes the original function? | **No** | Yes, if `private` and unused |

```mlir
module {
  func.func @square(%x: i32) -> i32 {
    %0 = arith.muli %x, %x : i32
    return %0 : i32
  }
  func.func @main(%a: i32) -> i32 {
    %r = func.call @square(%a) : (i32) -> i32
    return %r : i32
  }
}
```

`@main` calls `@square`. That call is what `--inline` rewrites.

```bash
mlir-opt a.mlir --inline
```

```mlir
module {
  func.func @square(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }
  func.func @main(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

What changed, line by line:

1. The `func.call @square` inside `@main` is **gone**.
2. In its place, `--inline` **spliced** `@square`’s body: `arith.muli` of the argument. `%a` (printed `%arg0`) is the operand that used to be passed to the call.
3. `@square` itself is **still in the module**. Inlining copies; it does not delete.

After this, nothing in the file calls `@square` anymore. It is unused **in this module**, but it is still **public** (no `private`), so a later linker could still use it. `--inline` does not care either way.

To actually drop `@square`, mark it private and run dead-symbol elimination:

```bash
mlir-opt a.mlir --inline --symbol-dce
```

`--inline` first (call → body in `@main`). `--symbol-dce` second (unused `private @square` → deleted). Public `@square` survives `--symbol-dce` even after inlining.

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

`--canonicalize` can drop dead **ops** inside a function (`Pure` values nobody uses). `--inline` copies a callee into a caller and leaves the original function. `--symbol-dce` drops dead **functions** (if they are `private`).

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

### Pass manager

A **pass** is one rewrite. A **pass manager** (`mlir::PassManager`) is what **runs a list of passes**, in order, on the **right ops**. `mlir-opt --canonicalize --cse` does not “call canonicalize then cse by magic”; it **builds a pass manager** and `run`s it on the parsed module. `--dump-pass-pipeline` prints that manager. A **pipeline** is the list (and nesting) you put in it.

```text
pass           one rewrite (canonicalize, cse, convert-scf-to-cf)
pipeline       ordered list, possibly nested
PassManager    the runner  (C++ object; mlir-opt builds one for you)
```

Same idea as [passes](README-summary.md#3-pass): dialect = vocabulary, pass = rewrite, **pass manager = the schedule**.

#### Nesting — which op a pass sees

MLIR is nested ops. A pass is scheduled on **one parent op type**. It does not blindly walk the whole file unless you nest it that way.

```text
builtin.module {                 ← outer manager usually lives here
  func.func @a { ... }           ← can nest a manager on each func.func
  func.func @b { ... }
}
```

```text
builtin.module(                  # runs once on the module
  canonicalize,
  func.func(                     # cloned/run for every func.func
    cse
  ),
  inline,                        # needs the module symbol table
  symbol-dce
)
```

| Nesting | What the pass can see |
| --- | --- |
| `builtin.module(cse)` | The whole module as one unit |
| `func.func(cse)` | One function at a time; `@a` and `@b` cannot CSE against each other |
| `symbol-dce` / `inline` | Must be on the **module** — they use `@` symbols, not one function’s body |

`--inline` looks up `@square` in the module. `--symbol-dce` deletes unused `private` functions. Neither belongs inside `func.func(...)`. `--cse` is usually nested on `func.func` so each function is a separate CSE problem.

That is why `--dump-pass-pipeline` prints **parentheses**, not a flat list. The dump is the pass manager.

#### Two ways to fill the manager

**Flags** (what you have been typing). `mlir-opt` wraps them in a default outer `builtin.module(...)` and picks a nest per pass:

```bash
mlir-opt a.mlir --canonicalize --cse --inline --symbol-dce
```

**Explicit pipeline string** — you control the nest:

```bash
mlir-opt a.mlir --pass-pipeline='builtin.module(canonicalize,func.func(cse),inline,symbol-dce)'
```

Order is **left to right**. Nested `func.func(...)` runs in full on `@a`, then on `@b`, before the next sibling at module level.

Always confirm with:

```bash
mlir-opt a.mlir --canonicalize --cse --inline --symbol-dce --dump-pass-pipeline
```

If the dump’s nesting is not what you meant, use `--pass-pipeline=` instead of a pile of flags.

#### C++ (when you write a tool)

`mlir-opt` is a CLI over this. A custom tool does the same in code:

```cpp
mlir::PassManager pm(context);

// module-level
pm.addPass(mlir::createCanonicalizerPass());

// nested: one CSE run per func.func  →  func.func(cse)
mlir::OpPassManager &fnPM = pm.nest<mlir::func::FuncOp>();
fnPM.addPass(mlir::createCSEPass());

pm.addPass(mlir::createInlinerPass());
pm.addPass(mlir::createSymbolDCEPass());

if (failed(pm.run(moduleOp)))
  return failure();
```

| C++ | Text pipeline |
| --- | --- |
| `pm.addPass(...)` | a name inside `builtin.module(...)` |
| `pm.nest<func::FuncOp>()` | `func.func(...)` |
| `pm.run(moduleOp)` | what `mlir-opt` does after parse |

`OpPassManager` is the nested manager (one op type). `PassManager` is the top-level manager (usually on `builtin.module`).

#### Pass kinds (enough to read dumps)

| Kind | Role |
| --- | --- |
| **Transformation** | Changes IR (optimize or lower) |
| **Analysis** | Computes facts (liveness, aliases); later passes may query them |
| **Operation pass** | Restricted to one op type (`func.func`, `builtin.module`, …) |

With assertions on, the manager **verifies IR after each pass** (`--verify-each`). A broken pass fails there instead of producing mystery IR later.

Failure: if a pass returns failure, `pm.run` fails and `mlir-opt` exits nonzero. Debug with `--dump-pass-pipeline` (the plan) and `--mlir-print-ir-after-all` (the trace) below.

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
$BIN/mlir-opt $MLIR --inline

# which passes will run (no IR dump)
$BIN/mlir-opt $MLIR --canonicalize --cse --symbol-dce --dump-pass-pipeline

# explicit pass manager (nest CSE on each function)
$BIN/mlir-opt $MLIR --pass-pipeline='builtin.module(canonicalize,func.func(cse),inline,symbol-dce)' \
  --dump-pass-pipeline

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

---

## 9. Cheat sheet

### Tools

| Tool | What it does | Output format |
| --- | --- | --- |
| **mlir-opt** | Runs passes: transform, optimize, lower | Still MLIR (textual) |
| **mlir-translate** | Converts MLIR to another representation (`--mlir-to-llvmir`) | Leaves MLIR → LLVM IR (`.ll`) |
| **mlir-tblgen** | Generates C++ boilerplate from `.td` (TableGen) files | C++ `.h.inc` / `.cpp.inc` |
| **mlir-cpu-runner** | JIT-executes lowered IR | A numeric result, not IR |

### Common `mlir-opt` flags

| Flag | Name | What it does | Example |
| --- | --- | --- | --- |
| `--cse` | Common subexpression elimination | Merges identical ops into one (dedup only) | two `muli %x, %x` → one |
| `--canonicalize` | Canonicalize | Fold, identities (`x+0→x`), DCE, canonical form | `10+10` → `20`; drops `+0` |
| `--inline` | Inliner | Replaces a call with the callee’s body | `call @square(%a)` → `muli %a, %a` |
| `--symbol-dce` | Symbol DCE | Deletes **private** unreferenced functions/symbols | removes unused `private @foo` |
| `--convert-scf-to-cf` | SCF → CF | Lowers `scf.for` / `scf.if` to branches | only affects `scf` ops |
| `--convert-arith-to-llvm` / `--convert-func-to-llvm` | Lower to LLVM dialect | Ops become `llvm.*` (**still MLIR**) | `arith.addi` → `llvm.add` |
| `--convert-to-llvm` | Convert to LLVM | One flag that lowers several dialects toward LLVM dialect | still MLIR, not `.ll` |

`--convert-*-to-llvm` is **lowering** (`mlir-opt`). `--mlir-to-llvmir` is **format** (`mlir-translate`).

### CSE vs canonicalize

| Aspect | `--cse` | `--canonicalize` |
| --- | --- | --- |
| Core job | Deduplicate identical ops | Simplify / fold individual ops |
| `x + 0` → `x` | No | Yes |
| Constant folding (`10+10` → `20`) | No | Yes |
| Merge two identical `muli` | Yes | No |
| Compares ops against each other | Yes | No |

Worked dumps: [CSE vs canonicalize](#cse-vs-canonicalize).

### Operation-level vs symbol-level

| Level | Passes | Needs |
| --- | --- | --- |
| Operation (inside a function) | `--cse`, `--canonicalize` | foldable or duplicate ops |
| Function call | `--inline` | at least one `call` |
| Symbol / whole function | `--symbol-dce` | `private` **and** unreferenced |

### Pipeline construction

| Command form | Who builds the pipeline | Order comes from |
| --- | --- | --- |
| `--cse --canonicalize …` (separate flags) | `mlir-opt` auto-assembles and wraps in `builtin.module(...)` | order of flags on the command line |
| `--pass-pipeline='builtin.module(...)'` | You write the structure, nest, and options | order you type in the string |

`builtin.module(...)` is the **anchor op** the passes run on. Both forms create a **pass manager**; one is auto-built, one is hand-written. Details: [Pass manager](#pass-manager).

### Inspection / debugging

| Flag | Without it | With it |
| --- | --- | --- |
| `--dump-pass-pipeline` | assembled pipeline not shown | prints the `builtin.module(...)` pipeline (then still runs) |
| `--mlir-print-ir-after-all` | no intermediate IR | prints the IR after every pass |
| `--mlir-print-ir-after-change` | every pass dumps | dump only when IR actually changed |
| `--mlir-disable-threading` | dumps may interleave | one thread, readable dumps |

`--dump-pass-pipeline` = the **plan**. `--mlir-print-ir-after-all` = the **trace**.

### Visibility (inlining + DCE)

| Visibility | Can `--symbol-dce` remove it? | Why |
| --- | --- | --- |
| **public** (default) | Never | outside code might call it |
| **private** | Yes, if unreferenced | compiler knows it is internal-only |

`--inline` never deletes the original function. After inlining, a **private** unused callee is what `--symbol-dce` can drop.
