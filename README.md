# HGEMV — Matrix-Vector Multiplication on CDNA

This new project is the continuation of
[SGEMV_RDNA3](https://github.com/deicidia/SGEMV_RDNA3) on CDNA architecture in
half precision (FP16). 

## Metric 


## Environment


### Cache hierarchy


## Kernel variants

| Variant | Geometry | Reduction | LDS/block | VGPR |
|---|---|---|---|---|
| `hgemv_block` | 1 block (256 thr) per row | LDS tree, then wave shuffles | - | - |
| `hgemv_wave64` | 1 wave (64 thr) per row | wave shuffles | - | - |

## Results

| Variant | Time | BW_eff | % of - |
|---|---|---|---|
| `hgemv_block` | - | - | - | 
| `hgemv_wave64` | - | - | - |

## Reproducing

Variants are declared once, in `include/variants.hpp`:

```cpp
{"block",   hgemv_block,    256}, // name, kernel, threads per row
{"wave64",  hgemv_wave64,   64},
```

Both the benchmark and the test binary read that table, and both derive the
launch geometry from `threads_per_row` rather than spelling it out. Adding a
variant means one line here and one kernel. The harness does not change, and
the grid cannot drift apart between the two binaries.

```bash
make bench                   # every variant
make bench VARIANT=naive     # one, or several: VARIANT="naive block"
make test                    # correctness: every variant x 5 shapes
make resources               # LDS, VGPR, spills, occupancy
make isa                     # demangled GCN assembly
make clean
```