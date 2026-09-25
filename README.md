# CUDA Learning Lab

Working through the CUDA project ladder — from vector addition to a toy flash-attention-style kernel — as foundational GPU knowledge underpinning the AI infra work in the companion repo, `ai-homelab-infra`.

Full step-by-step guide with explanations and resources: [`docs/cuda-learning-guide.md`](docs/cuda-learning-guide.md)

Used AI for suggesting/walking through these projects.

## Progress

### Tier 1 — Fundamentals
- [ ] 01 — Vector addition
- [ ] 02 — Basic parallel matrix multiplication
- [ ] 03 — Parallel reduction (all 7 versions)
- [ ] 04 — Image grayscale + box blur

### Tier 2 — Memory Hierarchy
- [ ] 05 — Optimized matmul (shared-memory tiling)
- [ ] 06 — Matrix transpose (bank-conflict avoidance)
- [ ] 07 — 1D/2D convolution (halo cells)
- [ ] 08 — Parallel prefix sum / scan (Blelloch)

### Tier 3 — Classic Algorithms
- [ ] 09 — Bitonic sort
- [ ] 10 — Radix sort
- [ ] 11 — Histogram with atomics
- [ ] 12 — Sparse matrix-vector multiply (SpMV)
- [ ] 13 — Small N-body simulation

### Tier 4 — Advanced
- [ ] 14 — Tiled matmul toward cuBLAS performance
- [ ] 15 — Fused online softmax
- [ ] 16 — LayerNorm/RMSNorm kernel
- [ ] 17 — Quantized INT8 GEMM (toy)
- [ ] 18 — Toy flash-attention-style kernel
- [ ] 19 — Multi-stream overlap

## Repo structure

```
cuda-learning-lab/
├── README.md
├── docs/
│   └── cuda-learning-guide.md
├── tier1-fundamentals/
│   ├── 01-vector-add/
│   ├── 02-matmul-naive/
│   ├── 03-reduction/
│   └── 04-image-blur/
├── tier2-memory-hierarchy/
│   ├── 05-matmul-tiled/
│   ├── 06-transpose/
│   ├── 07-convolution/
│   └── 08-prefix-sum/
├── tier3-classic-algorithms/
│   ├── 09-bitonic-sort/
│   ├── 10-radix-sort/
│   ├── 11-histogram/
│   ├── 12-spmv/
│   └── 13-nbody/
├── tier4-advanced/
│   ├── 14-matmul-cublas-comparison/
│   ├── 15-online-softmax/
│   ├── 16-layernorm/
│   ├── 17-int8-gemm/
│   ├── 18-flash-attention-toy/
│   └── 19-multistream/
└── .gitignore
```

Each project folder holds its own `.cu` source, a small `Makefile` or one-line `nvcc` build command in a comment header, and a `notes.md` with what I measured (timing, occupancy, bandwidth) and what I've learned

## Hardware

- GPU: NVIDIA GTX 1650 (mobile, 4GB VRAM, Turing, compute capability 7.5, `sm_75`)
- Build target: `nvcc -arch=sm_75 -O3`

## Profiling

Every project ran through the standard loop before being marked done:
```bash
compute-sanitizer ./project        # correctness first
nsys profile -o report ./project   # timeline-level bottleneck check
ncu --set full -o kreport ./project  # kernel-level deep dive
```
See `docs/cuda-learning-guide.md` Section 3 for the full walkthrough.
