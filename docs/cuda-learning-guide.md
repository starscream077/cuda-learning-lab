# CUDA Learning Guide — From Zero to Flash-Attention-Style Kernels

A step-by-step companion for working through the CUDA project ladder. 

---

## Dualboot

I'm using Ubuntu in dual-boot, installed on my personal computer. Here are a few steps I've followed to set it up:

- Create a Windows recovery drive (mandatory, I had issues with copying system files, sfc scan and clean up image didn't work. Luckily I had a drive which I created right after I got the machine)
- Shrink disks to create new partitions. I couldn't go beyond 85 Gb on my 512 Gb disk (this [video](https://youtu.be/TxDNyBA83lE?si=4-Rd_6cCfdSUhcwD) helped in shrinking more)
- Disable Intel RST - needed for Ubuntu install (booted Windows in safe more, entered BIOS setup, set SATA operation to AHCI)
- Install Ubuntu using live USB

## Section 0 — Setup (do this once)

- [ ] Confirm driver + CUDA toolkit are installed: `nvidia-smi` (driver) and `nvcc --version` (toolkit). If `nvcc` isn't found, install with `sudo apt install nvidia-cuda-toolkit`, or the newer versions from NVIDIA's CUDA Toolkit downloads page.
- [ ] Editor: VS Code with the "Nsight Visual Studio Code Edition" extension gives syntax highlighting + debugging for `.cu` files.
- [ ] Compile-and-run loop for every project:
  ```bash
  nvcc -arch=sm_75 -O3 my_kernel.cu -o my_kernel   # sm_75 = Turing = GTX 1650
  ./my_kernel
  ```
- [ ] Verify card and its specs (need these numbers for tuning later):
  ```bash
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
  ```
  For GTX 1650 (mobile): expecting ~4GB VRAM, compute capability 7.5, 1024 CUDA cores, 32 threads/warp, max 1024 threads/block.

**Core mental model:**
- **Thread** → smallest unit of execution, runs kernel function once.
- **Block** → a group of threads (up to 1024) that can share fast on-chip "shared memory" and synchronize with `__syncthreads()`.
- **Grid** → all the blocks launched for one kernel call.
- **Warp** → a group of 32 threads that physically execute in lockstep on the hardware. This is why divergent branching (`if` statements that split a warp's threads down different paths) is expensive — it's the single most important hardware fact in CUDA.
- **Kernel** → the `__global__` function that runs on the GPU, launched from the CPU with `kernel<<<gridDim, blockDim>>>(args)`.

---

## Tier 1 — Fundamentals

### Project 1: Vector Addition

**Things to learn:** the CPU↔GPU round trip (allocate, copy, launch, copy back, free) and thread indexing.

**Steps:**
1. Allocate two input arrays `A`, `B` and one output array `C` on the **host** (CPU, regular `malloc`/`new`).
2. Allocate matching arrays on the **device** (GPU) with `cudaMalloc(&d_A, size)`.
3. Copy host data to device with `cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice)`.
4. Write the kernel — each thread computes exactly one output element:
   ```cpp
   __global__ void vecAdd(float* A, float* B, float* C, int n) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) C[i] = A[i] + B[i];
   }
   ```
5. Launch it: `vecAdd<<<(n+255)/256, 256>>>(d_A, d_B, d_C, n);` — the `(n+255)/256` is the standard "round up" pattern so we always launch enough blocks to cover `n` elements.
6. Copy the result back: `cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost)`.
7. Free device memory with `cudaFree`.
8. Verify correctness against a simple CPU loop computing the same thing.

**Watch for:** forgetting the `if (i < n)` bounds check — when `n` isn't a multiple of block size, extra threads will write out of bounds without it.

---

### Project 2: Basic Parallel Matrix Multiplication

**Things to learn:** 2D thread indexing, first taste of a memory-bound kernel.

**Steps:**
1. Represent matrices as flat 1D arrays (row-major): element `(row, col)` of an `N×N` matrix lives at `A[row*N + col]`.
2. Use 2D grid/block dimensions so one thread computes one output element `C[row][col]`:
   ```cpp
   __global__ void matMul(float* A, float* B, float* C, int N) {
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int col = blockIdx.x * blockDim.x + threadIdx.x;
       if (row < N && col < N) {
           float sum = 0;
           for (int k = 0; k < N; k++)
               sum += A[row*N + k] * B[k*N + col];
           C[row*N + col] = sum;
       }
   }
   ```
3. Launch with a 2D block, e.g. `dim3 block(16,16); dim3 grid((N+15)/16,(N+15)/16); matMul<<<grid,block>>>(...)`.
4. Verify against a CPU triple-loop for a small N (say 256) before trying larger sizes.
5. Time it with `cudaEvent_t` (start/stop events) and note the GFLOPS — this becomes baseline for Project 5.

**Watch for:** this version reads `A` and `B` from slow global memory `2N` times per output element — that redundancy is exactly what shared-memory tiling (Project 5) fixes. Don't optimize yet; just get it correct and note the timing.

---

### Project 3: Parallel Reduction (Sum/Max of an Array)

**Things to learn:** the single most important optimization progression in CUDA — Mark Harris's classic "7 kernels" walkthrough. Implement each version in sequence, don't skip to the end.

**Steps (implement all 7, in order, timing each):**
1. **v0 — naive, divergent branching:** each thread checks `if (tid % (2*s) == 0)` to decide whether to add — causes warp divergence.
2. **v1 — non-divergent branching:** same idea, but restructure indexing (`int index = 2*s*tid`) so active threads are contiguous within a warp.
3. **v2 — avoid shared-memory bank conflicts:** change the stride pattern so consecutive threads access non-conflicting banks.
4. **v3 — sequential addressing:** reverse the loop so `s` starts at `blockDim.x/2` and halves each iteration — this is the version most tutorials call "the good one."
5. **v4 — first add during load:** halve the number of blocks by having each thread load and add two elements before the reduction loop even starts.
6. **v5 — unroll the last warp:** once `s <= 32`, I'll be down to one warp — no `__syncthreads()` needed since a warp is already in lockstep; unroll this manually.
7. **v6 — fully unrolled + template on block size:** use C++ templates so the compiler unrolls the entire loop at compile time.
8. For each version, print elapsed time and computed **effective bandwidth** (bytes read+written / time) — watch the number climb toward card's peak memory bandwidth as we go from v0 to v6.

**Resource for this one specifically:** Mark Harris's original NVIDIA slide deck "Optimizing Parallel Reduction in CUDA" is the canonical reference — https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf; nearly every course references this exact progression.

---

### Project 4: Image Grayscale + Box Blur (2D Stencil)

**Things to learn:** 2D data over a 1D or 2D thread grid, and boundary/edge handling.

**Steps:**
1. Load a small image (use `stb_image.h`, a single-header public-domain library — just drop the `.h` file next to `.cu` file) into a host `unsigned char*` buffer.
2. Grayscale kernel: one thread per pixel, `gray = 0.299*R + 0.587*G + 0.114*B`.
3. Box blur kernel: each thread reads a small neighborhood (e.g. 3×3 or 5×5) around its pixel and averages — this is first kernel where each thread reads *multiple* nearby memory locations, foreshadowing why shared-memory tiling matters for stencils too.
4. Handle image edges by clamping the neighborhood indices to valid range (`max(0, min(width-1, x))`) rather than reading out of bounds.
5. Write the result back out with `stb_image_write.h` and open it to visually confirm correctness — a big advantage of this project is one can *see* bugs immediately (a shifted or discolored image is obviously wrong).

---

## Tier 2 — Memory Hierarchy Mastery

### Project 5: Optimized Matmul with Shared-Memory Tiling

**Things to learn:** the core CUDA optimization pattern — cooperatively load a tile into fast shared memory once, then every thread in the block reuses it many times instead of re-reading slow global memory.

**Steps:**
1. Pick a tile size, e.g. `TILE = 16`.
2. Declare shared memory for one tile of `A` and one tile of `B`: `__shared__ float As[TILE][TILE]; __shared__ float Bs[TILE][TILE];`
3. Loop over tiles along the shared dimension: each iteration, every thread in the block cooperatively loads one element of `A`'s tile and one element of `B`'s tile into shared memory.
4. Call `__syncthreads()` after the load — this is mandatory; without it, some threads will start computing before others finish loading.
5. Each thread computes its partial dot-product using the shared-memory tile (fast on-chip reads instead of global memory).
6. Call `__syncthreads()` again before loading the *next* tile (so no thread overwrites shared memory that another thread still needs).
7. Compare timing against Project 2's naive version on the same matrix size — should see a real, measurable speedup
8. Try `TILE = 32` and compare — bigger tiles mean fewer global memory reads but more shared memory per block, which can reduce how many blocks fit per SM (occupancy trade-off will be measured properly in the profiling section).

---

### Project 6: Matrix Transpose with Bank-Conflict Avoidance

**Things to learn:** shared memory bank conflicts — the classic "add one column of padding" trick.

**Steps:**
1. Naive version: each thread reads `in[row][col]` and writes `out[col][row]` directly to global memory — works, but the *write* pattern is uncoalesced (adjacent threads write to far-apart addresses), which is slow.
2. Shared-memory version: load a tile into shared memory in coalesced (row-major) order, `__syncthreads()`, then write it back out in coalesced order too — the transpose happens "for free" inside shared memory, where the access pattern doesn't need to be coalesced.
3. Declare the shared tile as `__shared__ float tile[TILE][TILE+1]` — **that `+1` is the entire lesson of this project.** Without it, threads accessing the tile column-wise all land in the same shared-memory bank (a 32-way bank conflict), serializing what should be a parallel access. The padding shifts each row by one element so column-wise access spreads across banks.
4. Profile both the naive and padded versions with `ncu`'s bank-conflict metric (see Section 3) and confirm the conflict count drops to near zero with padding.

---

### Project 7: 1D/2D Convolution with Halo Cells

**Things to learn:** shared-memory tiling for stencil operations, where tiles need to overlap ("halo" regions) because each output depends on neighboring inputs.

**Steps:**
1. Pick a small filter, e.g. a 5-tap 1D Gaussian kernel, or a 3×3 2D kernel.
2. Each block loads its tile of input data into shared memory **plus a halo border** equal to half the filter width on each side — e.g. for a 5-tap filter, each block needs 2 extra elements on each edge.
3. Assign halo-loading duty to the threads at the edges of the block (or have every thread load one "main" element plus conditionally load a halo element).
4. `__syncthreads()`, then every thread computes its convolution output purely from shared memory.
5. Verify against a naive global-memory version and against a CPU reference.
6. This project is genuinely the mechanical ancestor of a CNN convolution layer — worth pausing to notice that.

---

### Project 8: Parallel Prefix Sum / Scan (Blelloch Algorithm)

**Things to learn:** the hardest "small" algorithm on this list — work-efficient parallel scan via up-sweep/down-sweep, and multi-block scans via a "scan of block sums" pattern.

**Steps:**
1. Start with the **naive Hillis-Steele scan** (simpler, not work-efficient, but a good warm-up): each step, add element `i - 2^d` to element `i`, for increasing `d`. Implement this first within a single block using shared memory.
2. Then implement the **Blelloch work-efficient scan**: build a balanced binary tree over the data.
   - **Up-sweep (reduce) phase:** iteratively add pairs of elements, halving active threads each step, until we have the total sum at the last element.
   - Set the last element to zero (this makes it an *exclusive* scan).
   - **Down-sweep phase:** iteratively swap and add back down the tree, reconstructing all prefix sums.
3. Get this correct for a single block (≤1024 elements) first.
4. **Multi-block extension:** scan each block independently, collect each block's total into a small array, scan *that* array (recursively, or just on the CPU if it's small), then add each block's offset back into its elements. This "scan of sums" pattern is used constantly in real GPU libraries (it's literally how `thrust::exclusive_scan` and radix sort's histogram step work).

**Honest expectation-setting:** this is the hardest project in Tier 2. Budget real time for it.

---

## Tier 3 — Classic Parallel Algorithms

### Project 9: Bitonic Sort

**Things to learn:** a sorting network — a fixed, data-independent sequence of compare-and-swap operations, which maps onto SIMD/GPU hardware because every thread does the same comparison structure regardless of data values.

**Steps:**
1. Understand the concept first (on paper, with 8 elements) before coding: a bitonic sequence is one that monotonically increases then decreases (or vice versa). A **bitonic merge** turns a bitonic sequence into a sorted one via `log(n)` compare-swap stages. Building a full **bitonic sort** recursively creates bitonic sequences of increasing size (2, 4, 8, ... n) and merges each.
2. Array size must be a power of 2 for the basic version — pad with `+infinity` sentinel values if needed.
3. The kernel takes two parameters per launch: `j` (comparison distance within this stage) and `k` (size of the bitonic sequence being built). Each thread computes its partner index via XOR: `partner = i ^ j`, and compares-swaps with it based on whether `(i & k) == 0` (ascending) or not (descending).
4. Launch the kernel `O(log²n)` times from the host, in a loop over `k` and nested loop over `j` — each launch is one stage of the sorting network.
5. Verify against `std::sort` on the CPU for correctness, then time against it for a sense of GPU sorting speedup at different array sizes (bitonic sort's `O(n log²n)` complexity means it only wins over good CPU sorts at large `n`, which is itself an instructive result).

---

### Project 10: Radix Sort

**Things to learn:** non-comparison sorting via repeated histogram + scan + scatter — and a direct contrast against bitonic sort's approach.

**Steps:**
1. Pick a small radix, e.g. 1 bit at a time (simplest to implement) or 4 bits (faster, more complex).
2. Per bit position (LSB to MSB), for each pass:
   - **Predicate step:** each thread computes whether its element has a 0 or 1 in the current bit.
   - **Histogram/count step:** count how many 0s and 1s exist (reuse Project 8 scan here).
   - **Scan step:** exclusive-scan the predicate array to get each element's *destination index* within its bucket (0s go first, 1s go after all 0s).
   - **Scatter step:** each thread writes its element to its computed destination index.
3. Repeat for all bit positions (32 passes for a 32-bit integer, 1 bit at a time) — each pass produces an array sorted by that bit, and the stability of the scan-based scatter ensures earlier passes' ordering is preserved.
4. Compare against bitonic sort implementation on the same data — radix sort should win clearly at larger array sizes since it's `O(n)` per pass instead of `O(n log²n)` total.

---

### Project 11: Histogram with Atomics

**Things to learn:** atomic operations, contention cost, and the privatization pattern (per-block private copies merged at the end) that fixes it.

**Steps:**
1. **Naive version:** every thread does `atomicAdd(&histogram[bin], 1)` directly on one global histogram array. Simple, correct, but every thread targeting a popular bin serializes against every other thread hitting that same bin.
2. Time it, and try skewed input data (e.g. mostly one value) vs. uniform random data — should see the skewed case run much slower, purely from atomic contention.
3. **Privatized version:** declare a `__shared__` histogram array per block, initialize it to zero, have each thread `atomicAdd` into the **shared-memory** copy (contention is now only within one block's 32-1024 threads, and shared-memory atomics are much faster than global-memory ones), then after `__syncthreads()`, have threads cooperatively add each block's shared histogram into the final global one (one global atomic add per bin per block, instead of one per element).
4. Compare timing of naive vs. privatized on the same skewed data — this is one of the more dramatic before/after speedups in the whole list, worth genuinely feeling the difference.

---

### Project 12: Sparse Matrix-Vector Multiply (SpMV)

**Things to learn:** irregular/data-dependent memory access, as a deliberate contrast to every regular-access-pattern kernel so far.

**Steps:**
1. Pick the CSR (Compressed Sparse Row) format: three arrays — `values[]` (nonzero values), `col_idx[]` (column index of each nonzero), `row_ptr[]` (start index into the above two arrays for each row).
2. Generate a small sparse test matrix (e.g. a random matrix with 95% zeros) and build its CSR representation on the CPU.
3. Kernel: one thread per row — each thread loops from `row_ptr[row]` to `row_ptr[row+1]`, accumulating `values[k] * x[col_idx[k]]`.
4. Notice (and profile, per Section 3) that this kernel has **irregular workload per thread** (rows can have very different numbers of nonzeros) and **irregular memory access** (`x[col_idx[k]]` is a scattered read) — both hurt performance compared to every dense kernel written so far. This mismatch between "one thread per row" and actual GPU efficiency is exactly why real SpMV libraries use fancier schemes (e.g., one *warp* per row) — implementing the naive version first is what makes that motivation click.

---

### Project 13: Small N-Body Simulation

**Things to learn:** a compute-bound kernel (as opposed to the mostly memory-bound kernels above), and tile-based force computation.

**Steps:**
1. Represent each body with position `(x,y,z)`, velocity, and mass.
2. Naive kernel: one thread per body, each thread loops over *all other* bodies computing gravitational force contributions, then integrates position/velocity — `O(n²)` total work.
3. Time this for increasing `n` (e.g. 1024, 4096, 16384) and notice it stays compute-bound throughout — unlike memory-bound kernels, adding more arithmetic per body barely changes the wall-clock time relative to adding more bodies.
4. Optional tiled version: load a tile of bodies into shared memory (same pattern as Project 5's matmul tiling) so each block reuses the same body data for multiple force calculations instead of re-reading global memory per pair.

---

## Tier 4 — Advanced

### Project 14: Push Tiled Matmul Toward cuBLAS Performance

**Steps:**
1. Add **register blocking**: instead of each thread computing one output element, have each thread compute a small `2×2` or `4×4` sub-tile of outputs, keeping those partial sums in registers (fastest possible storage) across the whole tile loop.
2. Add **double buffering**: while computing on the current shared-memory tile, issue the load for the *next* tile so the memory fetch overlaps with compute instead of blocking on it.
3. Benchmark against `cublasSgemm` (link `-lcublas`) on identical matrix sizes — compute kernel's GFLOPS as a percentage of cuBLAS's. Getting to 30–50% of cuBLAS by hand is a genuinely good result for a self-written kernel; cuBLAS uses assembly-level tuning we won't match by hand, and that gap is itself the lesson.

### Project 15: Fused Online Softmax

**Steps:**
1. Naive softmax needs two passes: one to find the max (for numerical stability), one to compute `exp(x - max)` and sum, plus a third to divide. Implement this first for correctness.
2. Implement the **online (single-pass) softmax**: maintain a running max `m` and running sum `l`, and when a new, larger max is found, *rescale* the running sum by `exp(old_m - new_m)` before continuing — this is the trick that lets us process data in one pass without knowing the max in advance.
3. This exact technique is the numerical core of flash-attention (Project 18) — implementing it standalone first, on a plain 1D array, makes the attention version far less intimidating later.

### Project 16: LayerNorm/RMSNorm Kernel

**Steps:**
1. For each row of a `[batch, features]` tensor, compute mean and variance (LayerNorm) or just mean-square (RMSNorm, what most modern LLMs use), using a block-level reduction (reuse Project 3 skills directly).
2. Normalize: `(x - mean) / sqrt(variance + eps)`, then scale/shift by learned parameters `gamma`/`beta`.
3. Fuse it into as few kernel launches as possible — ideally one kernel per row, using shared memory for the reduction, rather than separate kernels for mean/variance/normalize.

### Project 17: Quantized INT8 GEMM (Toy Version)

**Steps:**
1. Quantize two small `float` matrices to `int8` on the host: pick a scale factor `s` such that `int8_val = round(float_val / s)`, clamped to `[-127, 127]`.
2. Kernel: accumulate the matmul in `int32` (to avoid overflow) using the `int8` inputs.
3. Dequantize the output: `float_result = int32_accum * scale_A * scale_B`.
4. Compare accuracy against the full-precision result on the same input — this hands-on look at quantization error is exactly what's happening under the hood every time we run a Q4/Q8 model through Ollama on a 4GB card.

### Project 18: Toy Flash-Attention-Style Kernel

**Steps:**
1. Implement standard (non-flash) attention first for a small sequence length: `scores = Q @ Kᵀ`, softmax over scores, `output = softmax(scores) @ V` — using matmul and softmax kernels from earlier as building blocks. Get this numerically correct against a CPU/PyTorch reference before optimizing.
2. Now the flash-attention idea: instead of materializing the full `N×N` score matrix, **tile** over blocks of K/V — for each Q tile, loop over K/V tiles, maintaining a running (online-softmax, Project 15) output accumulator, so the full `N×N` matrix is never stored, only `O(tile_size²)` at a time.
3. This is genuinely the capstone — expect it to take multiple sessions. The payoff is that by this point every piece (tiling from Project 5, online softmax from Project 15, shared-memory management from Projects 5–7) is something we've already built once before, standalone.

### Project 19: Multi-Stream Overlap

**Steps:**
1. Create two or more CUDA streams: `cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);`
2. Use `cudaMemcpyAsync` (not the blocking `cudaMemcpy`) and launch kernels with a stream argument: `kernel<<<grid, block, 0, s1>>>(...)`.
3. Split a larger workload into chunks, and pipeline: while chunk 1 computes on the GPU, chunk 2's data is being copied over — asynchronously, on a different stream.
4. Requires **pinned host memory** (`cudaMallocHost` instead of regular `malloc`) for the async copies to actually overlap with compute — this is a common gotcha, regular pageable memory silently falls back to synchronous behavior.
5. Profile with `nsys` (Section 3) and *look at the timeline* — we want to visually see the memcpy and compute bars overlapping instead of running back-to-back. This is the one project in the whole list that's validated visually in the profiler rather than by a speed number alone.

---

## Section 3 — Profiling & Tuning Walkthrough

Apply this exact loop to **every** project above once it's correct:

1. **Correctness first.** Compare against a CPU reference or `compute-sanitizer` before touching performance.
   ```bash
   compute-sanitizer ./my_kernel
   ```
2. **Timeline view — `nsys`.** Confirms whether kernel is even the bottleneck (vs. memcpy or launch overhead).
   ```bash
   nsys profile -o report ./my_kernel
   nsys stats report.nsys-rep
   ```
3. **Kernel deep-dive — `ncu`.** The single most important question: is the kernel memory-bound or compute-bound?
   ```bash
   ncu --set full -o kernel_report ./my_kernel
   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./my_kernel
   ```
   Whichever percentage is closer to 100% tells which class of optimization is worth implementing.
4. **Check occupancy** (achieved vs. theoretical, in the `ncu` report) — low occupancy usually traces to register pressure or shared-memory usage per block. `nvcc --ptxas-options=-v my_kernel.cu` prints registers-per-thread at compile time.
5. **Check bank conflicts** for any kernel using shared memory: `l1tex__data_bank_conflicts` metric in `ncu`.
6. **Change one thing, re-measure.** Resist stacking multiple changes before re-profiling — each project to teach which specific change mattered.

---

## Documentation & Video Resources

| Resource | Best for |
|---|---|
| [freeCodeCamp — CUDA Programming Course (12 hrs, YouTube, by Elliot Arledge)](https://www.youtube.com/results?search_query=freecodecamp+cuda+programming+course) | Full beginner-to-intermediate video course, closely matches this project ladder's structure |
| [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | The official reference — dense, but authoritative on execution model, memory hierarchy, and API details |
| [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | Directly maps to the profiling/tuning section above — coalescing, occupancy, shared memory |
| *CUDA by Example* (Sanders & Kandrot) | The classic beginner book — gentler pacing than the official docs, great alongside Tier 1–2 |
| *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj) | The standard textbook once past basics — used in most university GPU courses, covers everything through Tier 3 rigorously |
| [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/) | Reference for every metric mentioned in Section 3 |
| Mark Harris — "Optimizing Parallel Reduction in CUDA" | The canonical source for Project 3's 7-version progression |
| NVIDIA Developer Blog (developer.nvidia.com/blog) | Search per-topic — has dedicated posts on tiling, bank conflicts, streams, and quantized GEMM that pair well with Tier 2–4 |

**How to use the video course alongside this guide:** the freeCodeCamp course covers roughly Tier 1 and the start of Tier 2 in its early hours, then shifts toward deep-learning-specific CUDA (cuBLAS, cuDNN concepts) in its later hours — good alignment with Tier 4. Use it for the "watch someone type it and explain it live" experience, and use this guide for the structured step list and the parts (Tiers 3–4 classic algorithms) it doesn't cover in as much depth.
