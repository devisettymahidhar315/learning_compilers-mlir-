# Writing MLIR

[Getting started](getting-started.md) · [MLIR concepts](README-summary.md) · [Build](../README.md)

- [module vs no module](#1-module--vs-no-module)
- [Smallest example](#2-smallest-example-line-by-line)
- [Functions](#3-function-declaration--definition)
- [Calling a function](#4-calling-a-function)
- [Cheat sheet](#5-cheat-sheet)
- [Common mistakes](#6-common-mistakes)
- [FileCheck](#7-filecheck--not-mlir-syntax)
- [lit](#8-lit--llvm-integrated-tester)
- [Commands on this machine](#9-commands-to-run-on-this-machine)

---

## 1. `module { }` vs no module

A **module** (`builtin.module`) is the top-level container: one compilation unit and one **symbol table** for names like `@add`.

**With** an explicit module:

```mlir
module {
  func.func @add() -> i32 {
    %0 = arith.constant 20 : i32
    return %0 : i32
  }
}
```

**Without** writing `module`:

```mlir
func.func @add() -> i32 {
  %0 = arith.constant 20 : i32
  return %0 : i32
}
```

These parse to the **same IR**. If you omit `module { }`, the parser inserts an implicit module. `mlir-opt` usually prints it back.

| Explicit `module` | No `module` in the file |
| --- | --- |
| Compilation unit is visible | Shorter examples |
| Can set module attributes | Still a module after parse |
| Can nest modules | Parser wraps top-level ops |
| `@add` is found via `func.call @add` | Same, after the implicit wrap |

A module does not run the IR. It only groups it.

---

## 2. Smallest example, line by line

```mlir
module {
  func.func @add() -> i32 {
    %0 = arith.constant 20 : i32
    return %0 : i32
  }
}
```

Mixed dialects is normal:

| Text | Dialect / op | Role |
| --- | --- | --- |
| `module` | `builtin.module` | Container |
| `func.func` | func | Define a function |
| `arith.constant` | arith | SSA constant (**operation**, not a function) |
| `return` | `func.return` | Terminator of the function body |

### Why `return` is compulsory

Every finished block needs a **terminator**. For `func.func` that is `func.return`.

```mlir
func.func @add() -> i32 {
  %0 = arith.constant 20 : i32
  return %0 : i32          // must return one i32
}

func.func @do_nothing() {
  return                   // void: still required
}
```

Omit `return` → verifier: block has no terminator.  
`-> i32` plus bare `return` → invalid (count and types must match).

### Types on every value

MLIR is strongly typed. The type is part of the syntax.

```mlir
%0 = arith.constant 20 : i32
return %0 : i32
func.func @add() -> i32

%2 = func.call @add(%0, %1) : (i32, i32) -> i32
```

---

## 3. Function declaration / definition

```mlir
func.func @name(%arg0: type, %arg1: type) -> result_type {
  return ...
}
```

### `func.func`

The op that defines or declares a function. `{ ... }` is a region (usually one block).

### `@name` (symbol, not SSA)

| Prefix | Meaning | Example |
| --- | --- | --- |
| `@` | Symbol (looked up in the module) | `@add` |
| `%` | SSA value | `%0` |
| `^` | Block label | `^bb0` |

```mlir
func.call @add() : () -> i32
```

### Arguments

Each argument is `%name : type`.

```mlir
func.func @add() -> i32
func.func @add(%arg0: i32, %arg1: i32) -> i32
```

These are SSA **block arguments** from the caller. Assigned once; do not write `%arg0 = ...` again.

```mlir
func.func @add(%0: i32, %1: i32) -> i32 {
  %3 = arith.addi %0, %1 : i32
  return %3 : i32
}
```

### Result type

| Syntax | Meaning |
| --- | --- |
| `-> i32` | One result |
| `-> (i32, f32)` | Two results |
| (omit `->`) | Void |

`return` must match exactly.

```mlir
func.func @pair() -> (i32, i32) {
  %a = arith.constant 1 : i32
  %b = arith.constant 2 : i32
  return %a, %b : i32, i32
}
```

### Body

New value ⇒ new `%name` (SSA). Last op must be `return` (or another terminator after lowering to `cf`).

### Declaration without a body

```mlir
func.func private @extern_add(i32, i32) -> i32
```

Definitions with a body are **public** by default. `private` = only used inside this module.

---

## 4. Calling a function

```mlir
%2 = func.call @add(%0, %1) : (i32, i32) -> i32
```

| Piece | Meaning |
| --- | --- |
| `func.call` | The op |
| `@add` | Callee symbol |
| `(%0, %1)` | SSA arguments |
| `(i32, i32) -> i32` | Must match the callee |

```mlir
module {
  func.func @add(%0: i32, %1: i32) -> i32 {
    %3 = arith.addi %0, %1 : i32
    return %3 : i32
  }

  func.func @num() -> i32 {
    %0 = arith.constant 20 : i32
    %1 = arith.constant 30 : i32
    %2 = func.call @add(%0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
}
```

`@add` and `@num` share the module’s symbol table. That is a main use of `module`.

`mlir-opt` often prints `call` instead of `func.call`.

---

## 5. Cheat sheet

```mlir
module {                              // optional in text; always there after parse
  func.func @add(%arg0: i32) -> i32 { // symbol @add, one arg, one result
    %c = arith.constant 1 : i32
    %s = arith.addi %arg0, %c : i32
    return %s : i32                   // terminator, matches -> i32
  }
}
```

`@` symbol · `%` SSA value · `:` type · `->` result type · `()` args or call operands

---

## 6. Common mistakes

```mlir
// missing return — no terminator
func.func @add() -> i32 {
  %0 = arith.constant 20 : i32
}

// return type mismatch
func.func @add() -> i32 {
  return
}

// SSA name reused
%0 = arith.constant 20 : i32
%0 = arith.constant 30 : i32

// missing type
%0 = arith.constant 20

// wrong call type (if @add takes two i32s)
func.call @add(%0, %1) : () -> i32

// symbol not in this module
func.call @missing() : () -> i32
```

---

## 7. FileCheck — not MLIR syntax

`// CHECK...` lines are **comments**. MLIR ignores them. They are instructions for **FileCheck**.

1. A `RUN` line starts a tool (usually `mlir-opt`)
2. That tool prints IR
3. FileCheck looks for your patterns in that printout
4. All found in the right place → **PASS**; otherwise **FAIL**

```text
// RUN: mlir-opt %s | FileCheck %s
```

| Token | Meaning |
| --- | --- |
| `%s` | This file |
| `mlir-opt %s` | Parse / print (add flags to run passes) |
| `FileCheck %s` | Read CHECK lines from this same file |

Colon with **no space**: `// CHECK-LABEL:` not `// CHECK-LABEL :`.

### The four directives

| Directive | Where it matches |
| --- | --- |
| **CHECK-LABEL** | Unique section start, usually a function. Later checks stay in this section. |
| **CHECK** | Some **later** line (skips lines in between) |
| **CHECK-SAME** | The **same** line as the previous match |
| **CHECK-NEXT** | The **immediately next** line |

```mlir
module {
  // CHECK-LABEL: func.func @add
  func.func @add() -> i32 {
    %0 = arith.constant 20 : i32
    return %0 : i32
  }

  // CHECK-LABEL: func.func @aa
  // CHECK: arith.addi
  // CHECK-SAME: i32
  // CHECK-NEXT: return
  func.func @aa() -> i32 {
    %0 = func.call @add() : () -> i32
    %1 = arith.addi %0, %0 : i32
    return %1 : i32
  }
}
```

FileCheck on the printed IR:

1. Find `func.func @add` (anchor)
2. Jump to `func.func @aa`
3. Skip `call`, find `arith.addi`
4. Same line must contain `i32`
5. Next line must contain `return`

Later: `CHECK-NOT`, `CHECK-DAG`, `CHECK-EMPTY`.

Regex so you do not hard-code SSA names (`%0` vs `%c20`):

```text
// CHECK: %[[A:.*]] = arith.constant 20 : i32
// CHECK-NEXT: return %[[A]] : i32
```

| | |
| --- | --- |
| MLIR ops | `func.func`, `arith.constant`, `return` — the program |
| FileCheck | `// CHECK-LABEL:`, `// CHECK:` — the test |
| RUN line | `// RUN: mlir-opt %s \| FileCheck %s` — how lit runs it |

---

## 8. lit — LLVM Integrated Tester

Many `.mlir` files → you cannot check every dump by hand. **lit** finds tests, runs `RUN` lines, prints PASS / FAIL.

| Tool | Job |
| --- | --- |
| **mlir-opt** | Transform IR (optimize, lower, or parse + print) |
| **FileCheck** | Verify the printed IR |
| **lit** | Discover files, run `RUN` lines, report results |

They do not overlap.

### RUN line (top of file)

```text
// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s --canonicalize | FileCheck %s
// RUN: mlir-opt %s --convert-scf-to-cf | FileCheck %s
```

A comment. MLIR ignores it; **lit** executes it. Several `RUN` lines: all must succeed. No extra flags = parse and print (round-trip).

### How you run it

```bash
llvm-lit -v a.mlir
llvm-lit -v path/to/folder
```

`-v` = verbose. A folder needs `lit.cfg.py` (already in `mlir/test/`). That is why the report says `MLIR :: a.mlir`.

### The report

```text
-- Testing: 1 tests, 1 workers --
PASS: MLIR :: a.mlir (1 of 1)

Total Discovered Tests: 1
  Passed: 1 (100.00%)
```

```text
a.mlir → lit reads RUN → mlir-opt prints IR → FileCheck matches → PASS/FAIL
```

Other substitutions: `%S` source dir, `%t` temp file, `%T` temp dir.

```text
// RUN: mlir-opt %s --canonicalize -o %t
// RUN: FileCheck %s < %t
```

---

## 9. Commands to run on this machine

Tools are in the LLVM build `bin` directory.

| Tool | What it does | Command |
| --- | --- | --- |
| **mlir-opt** | Parse / print IR (add flags to optimize or lower). No FileCheck. | `/home/llvm-project/build/bin/mlir-opt /home/llvm-project/mlir/test/a.mlir` |
| **llvm-lit** | Reads `// RUN:`, runs `mlir-opt \| FileCheck`, prints PASS/FAIL | `/home/llvm-project/build/bin/llvm-lit -v /home/llvm-project/mlir/test/a.mlir` |

```bash
# printed IR
/home/llvm-project/build/bin/mlir-opt /home/llvm-project/mlir/test/a.mlir

# with a pass
/home/llvm-project/build/bin/mlir-opt /home/llvm-project/mlir/test/a.mlir --canonicalize

# lit + FileCheck
/home/llvm-project/build/bin/llvm-lit -v /home/llvm-project/mlir/test/a.mlir
```

**lit** needs `lit.cfg.py` nearby — keep the test under `/home/llvm-project/mlir/test/`.
