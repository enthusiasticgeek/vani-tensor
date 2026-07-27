# vani-tensor — TODO

> Compiler builtins that already exist and must NOT be reimplemented:
> `abs` `push` `pop` `len` `set` `vec`
>
> Depends on vani-matrix (`mat_mul_rect`) -- v0.1.0's only Kosh dependency.

---

## v0.1.0 — Implemented ✓

### Shape and index utilities (5 functions)
- [x] `tensor_size` -- product of shape entries
- [x] `tensor_strides` -- row-major (C-order) strides
- [x] `tensor_flat_index`, `tensor_unflatten_index` -- multi-index <-> flat
      offset, validated as an exhaustive round trip over every offset in a
      (2,3,4) tensor, not just one hand-picked example
- [x] `tensor_shapes_equal`

### Construction (4 functions)
- [x] `tensor_zeros`, `tensor_ones`, `tensor_full`, `tensor_copy`

### Indexing (2 functions)
- [x] `tensor_get`, `tensor_set`

### Reshape (1 function)
- [x] `tensor_reshape` -- data-preserving copy under a new shape (reshape
      never moves elements in a flat row-major layout, so this is a
      validated copy, not a real transformation)

### Elementwise arithmetic and reductions (8 functions)
- [x] `tensor_add`, `tensor_sub`, `tensor_mul` (Hadamard), `tensor_scale`
- [x] `tensor_sum`, `tensor_mean`, `tensor_max`, `tensor_min`

### Broadcasting (1 function)
- [x] `tensor_broadcast_add_last_axis` -- adds a length-`shape[last]` vector
      to every slice along the last axis (the "bias vector added to every
      row of a batch" pattern); full N-D NumPy-style broadcasting rules are
      out of scope for v0.1.0, see Future

### Axis permutation (1 function)
- [x] `tensor_permute` -- general N-D transpose via an odometer-style
      counter loop (no recursion). Validated against every one of a (2,3,4)
      tensor's 24 elements under a (1,0,2) permutation, not just a 2D
      transpose spot-check

### Contraction (1 function)
- [x] `tensor_contract_last_first` -- contracts `a`'s last axis against
      `b`'s first axis by reusing vani-matrix's `mat_mul_rect` directly
      (a row-major tensor's data is already exactly the matrix
      `mat_mul_rect` expects -- see README's "Contraction reuses
      vani-matrix" section). Validated both as an ordinary 2D matmul and,
      composed, as a 3D contract-against-ones-vector == groupwise-sum
      consistency check

### Tests and examples
- [x] `tests/test_shape_index.vani` -- size/strides/flat-index, exhaustive
      flatten/unflatten round trip, shapes_equal
- [x] `tests/test_construction_arithmetic.vani` -- zeros/ones/full/copy,
      get/set, reshape, all elementwise ops and reductions
- [x] `tests/test_broadcast_permute_contract.vani` -- broadcast, 2D and
      exhaustive 3D permute, 2D-matmul-equivalence and 3D-groupwise-sum
      contraction checks
- [x] `examples/batch_normalize_demo.vani` -- broadcast a per-feature bias
      over a batch, summary stats, transpose
- [x] `examples/contraction_demo.vani` -- (2,3,4) contracted with (4,2) as
      a batched-matmul-style operation

### Safety annotations
- [x] `#[bounded_stack(bytes=N)]` on every function, budgets set to `vanic
      check`'s exact reported worst-case (largest: `tensor_permute` at 450
      bytes, since it composes `tensor_strides` + `tensor_flat_index` inside
      its main loop; no recursion anywhere in this library)

---

## v0.1.4 (2026-07-27)

- [x] `tensor_map(a, f)` -- elementwise apply for any caller-defined
      top-level `fn(f64) -> f64`, same convention as vani-calculus's
      single-variable functions. A compiler builtin like `sqrt` isn't
      itself addressable as a function pointer -- confirmed by trying it
      directly first and getting a real type error (`argument 2 ... got
      i64`), so the test wraps it in a one-line named fn instead; the
      function's own doc comment now says so explicitly rather than
      claiming builtins work directly. `#[bounded_stack(bytes = 128)]`,
      `vanic check`'s exact reported worst-case; no unrelated WCET drift
      found in this package (checked, since two sibling packages had it
      this session).
- [x] `tests/test_construction_arithmetic.vani` extended: `tensor_map`
      with a wrapped builtin (`sqrt`) and a caller-defined function.
      Full suite + `vanic audit-safety` re-verified on both backends.

## Future

No v0.2.0 is currently planned. Candidates if a concrete need shows up: full
NumPy-style N-D broadcasting (rank-padding + per-axis size-1 stretching,
not just the last-axis-vector case v0.1.0 covers), and general Einstein
summation (`tensor_contract_last_first` only covers the single most common
contraction pattern).
