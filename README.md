# vani-tensor

N-dimensional array library for the [vāṇī compiler](https://github.com/enthusiasticgeek/vani-compiler).

Depends on [vani-matrix](https://github.com/enthusiasticgeek/vani-matrix) for
`mat_mul_rect`, which `tensor_contract_last_first` reuses directly.

**API reference / tutorial:** <https://enthusiasticgeek.github.io/vani-tensor/>

## Add to your project

```toml
# vani.toml
[deps]
tensor = { registry = "kosh", version = "^0.1" }
```

```sh
vanic add tensor
vanic build
```

## What's included (v0.1.0 — complete; see TODO.md)

| Module | Functions |
|---|---|
| Shape and index utilities | `tensor_size`, `tensor_strides`, `tensor_flat_index`, `tensor_unflatten_index`, `tensor_shapes_equal` |
| Construction | `tensor_zeros`, `tensor_ones`, `tensor_full`, `tensor_copy` |
| Indexing | `tensor_get`, `tensor_set` |
| Reshape | `tensor_reshape` |
| Elementwise arithmetic / reductions | `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_scale`, `tensor_sum`, `tensor_mean`, `tensor_max`, `tensor_min` |
| Broadcasting | `tensor_broadcast_add_last_axis` |
| Axis permutation | `tensor_permute` |
| Contraction | `tensor_contract_last_first` |

## Encoding

A tensor is a flat row-major `Vec<f64>` plus an explicit `Vec<i64>` shape --
no hidden metadata, no `Tensor` struct bundling the two together. This
matches vani-matrix's flat-`Vec<f64>`-plus-explicit-dims convention exactly:
**a rank-2 tensor's data `Vec` is byte-for-byte the same layout as a
vani-matrix matrix**, so a 2D tensor can be passed straight to any
vani-matrix function and vice versa.

Multi-index `(i0, i1, ..., i_{n-1})` maps to flat offset `sum(i_d *
stride_d)`, where `stride_{n-1} = 1` and `stride_d = stride_{d+1} *
shape[d+1]` (last axis varies fastest — C order, same as NumPy's default).

## Contraction reuses vani-matrix, doesn't reimplement it

`tensor_contract_last_first(a, a_shape, b, b_shape)` contracts `a`'s last
axis against `b`'s first axis (requires `a_shape[last] == b_shape[0]`) --
the N-D generalization of matrix multiply. With this library's row-major
layout, `a`'s data is already exactly an `(outer_a x K)` matrix and `b`'s
data is already exactly a `(K x inner_b)` matrix, so this function is
**literally `vani-matrix`'s `mat_mul_rect`** under a shape-computing
wrapper -- no new multiply loop, and it inherits `mat_mul_rect`'s tested
correctness. See `tests/test_broadcast_permute_contract.vani` for a
composed check (contracting against a ones-vector must equal a groupwise
sum) plus an exhaustive 24-element cross-check for `tensor_permute`.

## What this library does NOT provide

These are already vāṇī compiler builtins — call them directly, no import needed:

`abs` `push` `pop` `len` `set` `vec`

## License

MIT
