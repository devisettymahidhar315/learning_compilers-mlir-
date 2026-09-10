# MLIR concepts

[Getting started](getting-started.md) · [Writing MLIR](README-writing-mlir.md) · [Build](../README.md)

**MLIR** = Multi-Level Intermediate Representation. File extension: `.mlir`.

A `.mlir` file is the **textual** form. The compiler also holds the same IR **in memory** as C++ objects. There is a bytecode form too; `.mlir` is the usual text form.

- [Why IR?](#why-ir)
- [Benefits of IR](#benefits-of-ir)
- [What is SSA?](#what-is-ssa)
- [Why MLIR?](#why-mlir)
- [What is MLIR?](#what-is-mlir)
- [Dialect](#1-dialect)
- [Operation](#2-operation)
- [Pass](#3-pass)
- [Trait](#4-trait)

---

## Why IR?

One frontend language + one target = one pipeline. Easy.

Five languages × six targets **without** a shared IR = **30** pipelines. Hard.

With a shared IR, every frontend lowers to that IR, and that IR lowers to each target: **5 + 6 = 11** paths.

This is the **N × M** compiler problem: N languages × M targets becomes **N + M** if they share an IR.

---

## Benefits of IR

1. **Machine independent** — The IR is not tied to one CPU or GPU. Hardware details appear only at the backend (x86, ARM, GPU, …).
2. **Easier optimization** — Optimize the shared IR once. SSA lets passes reason at compile time (constant folding, DCE, CSE, later tiling / fusion).
3. **Reusable backend** — One backend per target. Every frontend that lowers to the same IR reuses it. That is the N + M payoff.

---

## What is SSA?

**Static Single Assignment:** each value is assigned **exactly once**. You do not reuse the same name for a new value.

```text
# without SSA          # with SSA
x = 1                  x1 = 1
x = x + 2              x2 = x1 + 2
```

If a name can be assigned many times, the optimizer often cannot tell which value is live except at runtime. With SSA it knows at compile time. LLVM IR, MLIR, and most modern IRs follow SSA.

---

## Why MLIR?

Older path:

```text
source → frontend → LLVM IR → machine code
```

LLVM IR is **low-level** (load, store, add, branch). Excellent for CPU codegen. For ML and GPU, lowering there too early **drops the meaning** (“this is a matmul”). After that you mostly see loads and stores.

MLIR keeps **several abstraction levels** in one framework: optimize while the high-level meaning is still there, then lower step by step.

**History:** started at Google (TensorFlow / Chris Lattner), now developed in **llvm-project**.

---

## What is MLIR?

MLIR is an IR **and** a framework for many IRs (**dialects**) at different levels. Values follow SSA.

| | |
| --- | --- |
| **LLVM IR** | Mostly one low abstraction level |
| **MLIR** | Many levels in one system |

A `.mlir` file does **not** belong to one dialect. Mixing dialects in one module is normal (`func.func` + `arith.addi` in the same function). That is how **progressive lowering** works: some ops stay high-level while others are already lowered, until everything is in the `llvm` dialect.

---

## 1. Dialect

A **dialect** is a namespace of related **operations, types, and attributes**. In text: `dialect.operation`.

```text
arith.constant
arith.addi
memref.load
linalg.matmul
func.func
```

Usually defined in **TableGen** (`.td`), which generates C++. You can also write C++ by hand.

### Popular dialects

#### arith

Scalar math. No tensors, no memory. Suffix `i` = integer, `f` = float.

`constant`, `addi` / `addf`, `subi` / `subf`, `muli` / `mulf`, `divsi` / `divui` / `divf`, comparisons, shifts.

```mlir
%2 = arith.addi %0, %1 : i32
```

#### func

These are **operations**, not “sub functions.”

| Op | Role |
| --- | --- |
| `func.func` | Define a function (`@add`). Body is a region. |
| `func.call` | Call a function; results are SSA values. |
| `func.return` | Terminator. In text you often write `return`. |

#### scf (Structured Control Flow)

Nested regions, not raw goto: `scf.if`, `scf.for`, `scf.while`, `scf.yield`. Later can lower to `cf` (`cf.br`, `cf.cond_br`).

#### tensor

| | |
| --- | --- |
| **Type** `tensor<4x8xf32>` | SSA **value**. No buffer, no address. “What to compute.” |
| **Dialect** | `empty`, `extract`, `insert`, reshape, … |

You do not store in place. An op produces a **new** tensor value.

#### memref

Buffer in memory: alloc / load / **store** / dealloc. “Where it lives.”

**tensor → memref** is a later lowering called **bufferization**, not something memref does automatically.

#### linalg (Linear Algebra)

Whole kernels: `matmul`, `conv_2d`, `fill`, `generic`. `arith.muli` is one scalar multiply; `linalg.matmul` is still visibly a matmul (tile / fuse / GPU). Can sit on tensor or memref types.

#### affine

Affine = linear form + constant, compile-time coefficients (`2 * d0 + d1 + 8`). Not `d0 * d1`.

`affine.for`, `affine.if`, `affine.load`, `affine.store`, `affine.apply`. Better dependence analysis than a general `scf.for`.

#### Other names

`math`, `vector`, `cf`, `llvm`, `gpu`, `bufferization`.

### Dialect comparison

| Dialect | What it is for | Level | Works on | Typical ops |
| --- | --- | --- | --- | --- |
| **arith** | Scalar math | Low / mid | `i32`, `f32`, index | `constant`, `addi`, `muli` |
| **func** | Define / call / return | Wrapper | Functions | `func`, `call`, `return` |
| **scf** | Structured if / for / while | Mid | Regions | `if`, `for`, `yield` |
| **tensor** | Tensor **values** | High | `tensor<…>` | `empty`, `extract` |
| **memref** | Memory **buffers** | Mid / low | `memref<…>` | `alloc`, `load`, `store` |
| **linalg** | Whole kernels | High / mid | tensor or memref | `matmul`, `generic` |
| **affine** | Affine loops / maps | Mid | memref + affine indices | `for`, `load`, `store` |
| **cf** | Unstructured branches | Low | CFG | `br`, `cond_br` |
| **llvm** | Last MLIR stop | Lowest | LLVM types | `llvm.add`, `llvm.load` |
| **math** | Math functions | Mid | Scalars / vectors | `sin`, `exp` |
| **vector** | SIMD | Mid / low | `vector<…>` | `transfer_read` |
| **gpu** | GPU launch | Mid | GPU kernels | `launch` |
| **bufferization** | tensor → memref | Conversion | tensor in, memref out | `to_memref` |

### Easy “vs” pairs

| Pair | Difference |
| --- | --- |
| **arith vs linalg** | One scalar add vs a whole matmul the compiler still recognizes |
| **tensor vs memref** | Value (no address) vs buffer (alloc / load / store) |
| **scf vs affine vs cf** | General structured loops vs analyzable affine loops vs goto-style branches |
| **func vs scf** | Whole function vs control flow *inside* a function |

### Lowering (high → low)

```text
func + linalg / tensor     this is a matmul on tensors
        ↓
   bufferization           tensors become memrefs
        ↓
linalg / affine / scf / memref    loops over memory
        ↓
       cf                  branches
        ↓
      llvm                 then LLVM IR → machine code
```

---

## 2. Operation

An **operation** is the main IR entity. Almost everything is an op: `module`, `func.func`, `arith.addi`, `func.return`.

LLVM splits Function / BasicBlock / Instruction. MLIR is **nested ops**.

| Part | Meaning | Example |
| --- | --- | --- |
| Name | `dialect.opname` | `arith.addi` |
| Results | SSA values produced | `%2` |
| Operands | SSA values used | `%0`, `%1` |
| Type | On results / operands | `: i32` |
| Attributes | Compile-time data | `20` on `arith.constant` |
| Regions | Nested bodies | body of `func.func`, `scf.if` |
| Successors | Branch targets | `cf.br` (not arith) |

```text
%2 = arith.addi %0, %1 : i32
 ^    ^          ^        ^
 |    op name    operands type
 result (assigned once)
```

A function definition is still an operation. Its payload is a **region**.

---

## 3. Pass

A **pass** walks the IR and does a **transformation**, **optimization**, or **analysis**.

| | Optimize | Lower |
| --- | --- | --- |
| Goal | Better IR, **same** level | Next **lower** dialect |
| Dialects | Usually stay | Change (`linalg` → `scf` → `cf` → `llvm`) |
| Example pass | `canonicalize`, `cse` | `convert-scf-to-cf` |
| Example rewrite | `20+30` → `50` | `scf.if` → `cf.cond_br` |

**Dialect = vocabulary. Pass = rewrite.**

Run with `mlir-opt` (or a PassManager). A **pipeline** is a list of passes.

### Optimize

Same dialect. Meaning unchanged; IR smaller or simpler.

`canonicalize`, `cse`, `dce`, inliner, fusion, tiling, …

```mlir
# before
%0 = arith.constant 20 : i32
%1 = arith.constant 30 : i32
%2 = arith.addi %0, %1 : i32

# after canonicalize
%2 = arith.constant 50 : i32
```

### Lower (dialect conversion)

High-level ops disappear; lower ops appear.

| Pass | Rewrite |
| --- | --- |
| `one-shot-bufferize` | tensor → memref |
| `convert-linalg-to-loops` | `linalg.matmul` → `scf.for` + load/store |
| `convert-scf-to-cf` | `scf.if` → `cf.cond_br` |
| `convert-arith-to-llvm` | `arith.addi` → `llvm.add` |

```mlir
# before
scf.if %cond { ... }

# after
cf.cond_br %cond, ^then, ^else
```

Optimize, then lower, then optimize again at the new level.

An **analysis** pass may only compute facts (liveness, aliasing).

Inside a pass: **RewritePattern** = one local rule; **ConversionTarget** = which ops are still legal.

```text
operation   arith.addi
dialect     arith
pass        canonicalize, convert-scf-to-cf
```

---

## 4. Trait

MLIR **transforms** IR. `mlir-opt` does not execute your program. Typical run path:

```text
.mlir → passes → llvm dialect → LLVM IR → JIT or machine code
```

A **trait** is a **property or restriction** on an op. The name says what it does (`addi`). Traits say what is legal to assume.

```tablegen
def AddIOp : Arith_Op<"addi", [Pure, Commutative, SameOperandsAndResultType]>
```

Passes query traits instead of hard-coding every op.

| | |
| --- | --- |
| **Property** | Semantic fact: `Pure`, `Commutative` |
| **Restriction** | Verifier rule: `Terminator`, `HasParent`, `SameOperandsAndResultType` |

### Common traits

| Trait | Meaning |
| --- | --- |
| **Pure** | No side effects; same inputs → same output. `arith.addi` yes; `memref.store` no. Enables DCE / CSE / hoist. |
| **Commutative** | `a+b == b+a`. `addi` yes; `subi` no. Canonicalize can sort operands. |
| **Terminator** | Last op in a block. `func.return`, `scf.yield`, `cf.br`. Omit `return` → verifier error. |
| **IsolatedFromAbove** | Region cannot capture outer SSA. `func.func` has this; `scf.for` does not. |
| **HasParent** | Only legal inside a parent (`func.return` only in `func.func`). |
| **SameOperandsAndResultType** | Operands and result share a type. |

Others you will see: `ConstantLike`, `SingleBlock`, `AffineScope`, `Idempotent`, `ElementwiseMappable`.

| Trait | Used by |
| --- | --- |
| Pure | canonicalize, cse, dce, hoist |
| Commutative | canonicalize |
| Terminator | verifier, CFG, `convert-scf-to-cf` |
| IsolatedFromAbove | inliner, treating a region as a unit |

**Trait vs interface:** a trait stamps a property; an interface is a set of methods a pass can call.

```text
operation   the entity
dialect     its namespace
pass        the rewrite
trait       the contract the rewrite relies on
```
