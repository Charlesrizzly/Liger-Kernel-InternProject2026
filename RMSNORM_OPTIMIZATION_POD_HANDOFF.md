# CuTe DSL RMSNorm Optimization: Pod Agent Instructions

You are taking over an active performance-engineering task in the Liger Kernel
repository. Work directly on the GPU pod so that profiling, implementation,
correctness testing, and benchmarking happen in the same environment.

## Mission

Optimize the CuTe DSL implementation of RMSNorm, with the primary focus on the
backward pass on NVIDIA B200. The goal is to match or beat the existing Triton
RMSNorm while preserving correctness, memory usage, H100 performance, and the
forward-pass improvements already achieved.

Do not optimize a synthetic proxy while regressing the actual repository
benchmark. Use hardware profiles to identify the dominant cost, make one
measurable change at a time, and retain only variants that improve the real
benchmark.

## Safety and Working Rules

- Use only the repository, installed development tools, and approved NVIDIA
  profiling tools already available on the pod.
- Do not install or run unreviewed third-party plugins, hooks, MCP integrations,
  or internet scripts.
- Do not upload source code, profiles, or benchmark data to third-party services.
- Do not delete or overwrite unrelated user changes or benchmark data.
- Do not commit generated CSV, PNG, TXT, `.ncu-rep`, `.nsys-rep`, SQLite, or
  optimization-workspace files unless explicitly requested.
- Keep source changes surgical and consistent with existing CuTe DSL patterns.
- Do not weaken correctness tolerances merely to make a new variant pass.
- Do not change the benchmark solely to make CuTe appear faster. If a benchmark
  quirk is found, report it and measure both the existing benchmark and a clean
  isolated path.

## Repository and Branch

Repository on the pod:

```bash
cd /shared/user/kenemuo/Liger-Kernel-InternProject2026
```

Active branch:

```text
kenemuo/rmsnorm-cutedsl-fused-bwd
```

Synchronize without creating an accidental merge commit:

```bash
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

The latest source/profiling commit before this handoff is:

```text
df0ac43 Profile benchmark-matched RMSNorm weights
```

The handoff document itself may appear as a newer documentation-only commit.

## Environment Previously Observed

- PyTorch: `2.12.0.1+cu130`
- CUDA runtime: `13.0`
- Triton: `3.7.0`
- `nvidia-cutlass-dsl`: `4.5.2`
- Target GPUs: H100 and B200

Confirm the actual pod before profiling:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("sms:", torch.cuda.get_device_properties(0).multi_processor_count)
PY

ncu --version
nsys --version
```

## What Has Already Been Done

### 1. Benchmark visualization

The visualizer was updated to render grouped comparisons and automatically tag
output filenames with the GPU:

- `8d6876d` - grouped benchmark bar charts
- `ec8317d` - GPU-tagged visualization filenames

### 2. First fused backward

An early fused backward used one persistent CTA per SM, large FP32 dW
accumulators, and repeated X/dY/W loads. It was poor on B200, reaching roughly
3.7-3.8x Triton backward latency in some measurements.

### 3. Shared-staged Blackwell-oriented attempt

Commit `e250a3f` added:

- a vectorized forward path;
- shared-memory staging for backward;
- architecture-aware persistent CTA counts;
- safe fallback handling for unsupported alignment and mixed-dtype cases.

This improved some paths, and B200 forward at `T=8192` beat Triton by about 12%,
but backward still remained approximately 2.6-2.8x Triton. The design had copied
the high-level staging idea from Quack without its complete prefetch pipeline and
selective register-retention strategy.

### 4. Triton-shaped register-resident backward

Commit `cfe2590` replaced the shared-staged backward fast path with a structure
that closely mirrors Triton:

- exactly one CTA per SM;
- contiguous persistent row ranges;
- 4, 8, or 16 warps based on hidden size;
- X and dY loaded directly from global memory into registers once per row;
- X and dY retained across the row reduction and reused for dX and dW;
- W and FP32 dW accumulators retained across the persistent row loop;
- no X/dY shared-memory staging;
- no `cp.async` pipeline;
- no staging waits or trailing per-row barrier;
- one FP32 dW partial row per SM, followed by PyTorch reduction and cast;
- mixed BF16 activation / FP32 weight cases allowed on the fast path.

This was the largest successful improvement so far:

- H100 token backward improved by about 30%.
- H100 token full pass improved by about 21%.
- B200 token backward improved by about 25%.
- B200 token full pass improved by about 12%.
- B200 model-config full pass improved by about 41%.
- B200 model-config full-pass geometric mean reached approximately 1.083x
  Triton.
- The best B200 model full-pass result was only about 6.4% slower than Triton.
- Memory remained equal to Triton.

The remaining gap was still substantial in backward:

- H100 backward remained approximately 2x Triton in the token sweep.
- B200 model-config backward remained approximately 1.42x Triton.

### 5. Correctness and test fixes

- `64b606e` applied the repository-standard BF16 dW tolerance.
- `1dc9594` corrected the tolerance selection to depend on activation dtype,
  rather than weight dtype.

CuTe and Triton reduce dW rows in different orders. Forward and dX remain under
strict tolerances; only BF16-activation dW uses the established looser gradient
tolerance.

### 6. Focused profiler

- `d54945b` added `benchmark/scripts/profile_rms_norm_backward.py`.
- `df0ac43` corrected it to default to the FP32 RMSNorm weight used by the real
  benchmark and added repeated iterations for stable Nsight Systems analysis.

The profiler:

- selects CuTe with `LIGER_KERNEL_IMPL=cutedsl`;
- selects Triton with an empty `LIGER_KERNEL_IMPL`;
- warms up compilation before capture;
- profiles direct backend forward/backward functions;
- places the backward region inside CUDA profiler and NVTX ranges;
- supports activation dtype, weight dtype, rows, hidden size, and iteration
  count.

## Important Profiling Discovery

The first H100/B200 profiles accidentally used BF16 weights. Those reports are
useful for understanding code generation, but they do **not** match the actual
benchmark.

The repository benchmark creates `LigerRMSNorm` and calls `.to(device)` without
a dtype, so:

- activations are BF16;
- the RMSNorm weight remains FP32.

This mixed BF16-activation / FP32-weight path must be the primary profiling and
optimization target.

### Existing BF16-weight NCU results

These results show that the generated CuTe kernel is already competitive when
the weight is BF16:

| GPU | Kernel | Main backward | Registers/thread | Achieved occupancy | Spills |
|---|---|---:|---:|---:|---:|
| H100 | CuTe | 49.25 us | 115 | 12.52% | 0 |
| H100 | Triton | 49.22 us | 113 | 12.48% | 0 |
| B200 | CuTe | 40.16 us | 123 | 12.46% | 0 |
| B200 | Triton | 37.92 us | 99 | 12.56% | 0 |

On B200:

- CuTe reached about 1.69 TB/s versus Triton's 1.79 TB/s.
- CuTe used 123 registers/thread versus Triton's 99.
- Neither kernel spilled.
- Both launched 148 blocks of 256 threads: one eight-warp CTA per SM.
- Both therefore exposed only about eight active warps per SM, or 12.5%
  achieved occupancy.
- The CuTe dW reduction and BF16 cast were approximately the same cost as
  Triton's.

The BF16-weight main kernel is therefore only about 6% slower on B200 and equal
on H100. It cannot explain the approximately 2x end-to-end token-backward gap.
The next profiles must use FP32 weights.

## Current Benchmark Evidence

For BF16 activations with hidden size 4096, the recent token-sweep medians were
approximately:

### H100

| Rows | Triton backward | CuTe backward |
|---:|---:|---:|
| 1024 | 0.1021 ms | 0.2045 ms |
| 2048 | 0.1034 ms | 0.2071 ms |
| 4096 | 0.1042 ms | 0.2044 ms |
| 8192 | 0.1135 ms | 0.2128 ms |

### B200

| Rows | Triton backward | CuTe backward |
|---:|---:|---:|
| 1024 | 0.0618 ms | 0.1271 ms |
| 2048 | 0.0615 ms | 0.1252 ms |
| 4096 | 0.0635 ms | 0.1278 ms |
| 8192 | 0.0951 ms | 0.1260 ms |

CuTe remains almost flat as rows increase. This strongly suggests a fixed
dispatch, wrapper, launch, or epilogue floor in addition to any kernel-level
inefficiency.

## Important Benchmark Quirks

Read `benchmark/scripts/utils.py`, especially `run_speed_benchmark`.

The backward benchmark:

```python
y = fwd_fn()
do = torch.randn_like(y)
triton.testing.do_bench(
    lambda: y.backward(do, retain_graph=True),
    grad_to_none=input_tensors,
)
```

Consequences:

1. It measures the PyTorch autograd wrapper, not only the direct backend call.
2. It repeatedly reuses the same retained graph.
3. The CuTe in-place backward may overwrite the upstream `do` buffer.
4. Only the input tensor is included in `grad_to_none`.
5. The layer weight gradient may accumulate across benchmark repetitions,
   adding `AccumulateGrad` work.

Do not assume the direct profiling harness and the benchmark measure identical
paths. First profile the direct mixed-dtype implementation. If that is close to
Triton but the benchmark remains slow, profile an exact reproduction of the
autograd benchmark path.

## Key Source Files

### CuTe implementation

`src/liger_kernel/ops/cutedsl/ops/rms_norm.py`

Important areas:

- `_cta_reduce_sum_warp0`
- `_rms_norm_fwd_vector_kernel`
- `_rms_norm_bwd_fused_vector_kernel`
- `_rms_norm_bwd_fused_host`
- `_launch_bwd_fused`
- `rms_norm_forward`
- `rms_norm_backward`
- `LigerRMSNormFunction`

### Launch helpers

`src/liger_kernel/ops/cutedsl/ops/rms_norm_fastpath.py`

Important helpers:

- `fast_path_vector_width`
- `triton_backward_warp_count`

The common vector width is currently constrained by the largest participating
element type. BF16 activations plus an FP32 weight therefore select four
elements per vector. This may cause BF16 X/dY traffic to use only 8-byte chunks
instead of 16-byte chunks and is a high-priority mixed-dtype hypothesis.

### Triton reference

`src/liger_kernel/ops/rms_norm.py`

Important areas:

- `_rms_norm_backward_kernel`
- `rms_norm_backward`
- the one-program-per-SM launch and final dW reduction

Use Triton as the execution-structure reference, but do not assume its launch
parameters are automatically optimal for CuTe code generation on B200.

### Tensor conversion helper

`src/liger_kernel/ops/cutedsl/ops/utils.py`

`to_cute_tensor` currently performs:

```python
from_dlpack(t.detach(), assumed_align=...)
ct.mark_layout_dynamic(...)
```

The fused backward converts six tensors on every invocation. Repeated DLPack and
dynamic-layout construction is a leading fixed-overhead hypothesis.

### Tests

- `test/transformers/test_cutedsl_rms_norm.py`
- `test/transformers/test_cutedsl_rms_norm_fastpath.py`

### Benchmarks and profiling

- `benchmark/scripts/benchmark_rms_norm.py`
- `benchmark/scripts/run_cutedsl_compare.py`
- `benchmark/scripts/profile_rms_norm_backward.py`
- `benchmark/scripts/utils.py`
- `benchmark/benchmarks_visualizer.py`

## Phase 1: Establish a Correct Baseline

Do not modify the kernel before completing this phase.

### 1. Run correctness tests

The parent `test/conftest.py` may fail while importing Transformers because the
pod's torchvision environment is intentionally incomplete. Bypass that unrelated
conftest:

```bash
python -m pytest \
  test/transformers/test_cutedsl_rms_norm_fastpath.py \
  test/transformers/test_cutedsl_rms_norm.py \
  -xvs --confcutdir=test/transformers -o addopts=""
```

The mixed-weight test currently covers cache separation at hidden size 512.
Before finalizing any mixed-dtype optimization, add or run benchmark-sized
BF16-activation / FP32-weight parity checks at hidden sizes 4096 and 8192.

### 2. Profile benchmark-matched FP32 weights

Profile both CuTe and Triton.

Required shapes:

1. `rows=4096`, `hidden_size=4096` on H100 and B200 for an apples-to-apples
   architecture comparison.
2. `rows=2048`, `hidden_size=8192` on B200 for the wide-model,
   register-pressure-sensitive case.

For Nsight Compute, use one iteration. For Nsight Systems, use 20 iterations so
capture startup overhead is amortized.

Example for B200 `4096 x 4096`:

```bash
GPU=b200
ROWS=4096
HIDDEN=4096
OUT=/tmp/rmsnorm-profile-${GPU}-wfp32
SCRIPT="$PWD/benchmark/scripts/profile_rms_norm_backward.py"
mkdir -p "$OUT"

for IMPL in cutedsl triton; do
  if [ "$IMPL" = cutedsl ]; then BACKEND=cutedsl; else BACKEND=""; fi
  BASE="$OUT/${GPU}_${IMPL}_m${ROWS}_n${HIDDEN}_wfp32"

  LIGER_KERNEL_IMPL="$BACKEND" nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o "$BASE" \
    python "$SCRIPT" \
      --rows "$ROWS" \
      --hidden-size "$HIDDEN" \
      --dtype bfloat16 \
      --weight-dtype float32 \
      --iterations 20

  LIGER_KERNEL_IMPL="$BACKEND" ncu \
    --profile-from-start off \
    --clock-control none \
    --force-overwrite \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    -o "$BASE" \
    python "$SCRIPT" \
      --rows "$ROWS" \
      --hidden-size "$HIDDEN" \
      --dtype bfloat16 \
      --weight-dtype float32 \
      --iterations 1

  nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum \
    "$BASE.nsys-rep" \
    > "${GPU}_${IMPL}_m${ROWS}_n${HIDDEN}_wfp32_nsys.txt"

  ncu --import "$BASE.ncu-rep" --page details --csv \
    > "${GPU}_${IMPL}_m${ROWS}_n${HIDDEN}_wfp32_ncu.csv"
done
```

For the B200 wide case, rerun with:

```bash
ROWS=2048
HIDDEN=8192
```

### 3. Record these quantities

For every CuTe/Triton pair, record:

- NVTX range total and per-iteration duration;
- CUDA API launch time;
- idle gaps between the main kernel, dW reduction, and cast;
- main backward kernel duration;
- dW partial-reduction duration;
- final cast duration;
- registers per thread;
- local-memory loads/stores and spill requests;
- achieved and theoretical occupancy;
- active warps per SM;
- DRAM, L2, and L1/TEX throughput;
- long-scoreboard stall contribution;
- barrier stall contribution;
- block size, grid size, and waves per SM.

Also compute:

```text
wrapper/fixed cost =
    measured end-to-end time
    - main kernel time
    - dW reduction time
    - cast time
```

### 4. Reproduce the real benchmark

From the repository root:

```bash
cd benchmark/scripts

python run_cutedsl_compare.py \
  --kernel rms_norm \
  --source b200 \
  --sweep-mode token_length \
  --overwrite

python run_cutedsl_compare.py \
  --kernel rms_norm \
  --source b200 \
  --sweep-mode model_config \
  --overwrite
```

Use `--source h100` on H100.

Do not commit the resulting CSVs.

## Phase 2: Diagnose Before Editing

Classify the result using the following decision tree.

### Case A: FP32-weight CuTe main kernel is materially slower

If the CuTe main kernel is more than about 10% slower than Triton, investigate
kernel code generation first.

Priority hypotheses:

1. **Mixed-dtype vector width**
   - Current common width is four elements because W is FP32.
   - X and dY are BF16 and may therefore use only 8-byte transactions.
   - Investigate independent vector layouts: 8-wide BF16 X/dY access and 4-wide
     FP32 W access while preserving the same column ownership.
   - Do not introduce illegal `cp.async` widths.

2. **Register pressure**
   - BF16-weight B200 CuTe already used 123 registers/thread versus Triton's 99.
   - Measure FP32-weight register usage and spills at hidden sizes 4096 and 8192.
   - Reduce unnecessary live ranges, duplicate fragments, conversions, and
     address temporaries before changing the algorithm.

3. **Warp count**
   - Parameter tuning must precede structural changes.
   - Test legal 4/8/16-warp variants manually; do not add `@triton.autotune`.
   - Record compilation success, registers, spills, kernel duration, and full
     backward duration for every variant.

4. **CTA multiplicity**
   - Test one versus two CTAs per SM only after measuring register occupancy.
   - Two CTAs may improve latency hiding but doubles the number of dW partials
     and can make the final reduction slower.
   - Do not return to the old 2-4x-SM shared-staged design without new evidence.

5. **Reduction and barriers**
   - The persistent kernel performs CTA-wide row reductions.
   - Use stall metrics to determine whether barriers are actually significant.
   - Do not remove synchronization required for in-place dY-to-dX safety.

### Case B: Main kernels are close but direct CuTe end-to-end is slower

Investigate host and launch overhead:

1. Time each `to_cute_tensor` call separately.
2. Time the cached compiled callable directly versus `_launch_bwd_fused`.
3. Determine how much time is spent in:
   - `detach`;
   - DLPack conversion;
   - dynamic-layout marking;
   - current-stream conversion;
   - the CuTe compiled-function/FFI invocation;
   - PyTorch allocation of `dW_partial`.
4. Explore safe reuse of conversion/layout metadata only if tensor pointer,
   dtype, shape, stride, device, alignment, stream, and lifetime remain correct.
5. Never cache a wrapper that can silently retain a stale tensor pointer.

### Case C: Direct paths are close but the repository benchmark is slower

Profile the exact autograd path:

```python
y = layer(x)
do = torch.randn_like(y)
for _ in range(iterations):
    y.backward(do, retain_graph=True)
```

Match the benchmark's gradient-clearing behavior. Separate:

- Python autograd-engine overhead;
- CuTe `LigerRMSNormFunction.backward`;
- DLPack/layout conversion;
- main kernel;
- dW reduction/cast;
- leaf-gradient accumulation.

If weight-gradient accumulation is a meaningful benchmark artifact, report it,
but still optimize against the existing benchmark unless the project owner
approves a benchmark correction.

### Case D: dW reduction or cast is dominant

CuTe and Triton currently use the same high-level pattern:

```python
dW = dW_partial.sum(dim=0).to(W.dtype)
```

Only redesign this if profiles show a CuTe-specific difference. Potential
approaches such as atomics, an in-kernel final reduction, or fewer partial rows
must be evaluated for numerical stability and contention. Do not assume fusion
is automatically faster.

## Phase 3: Controlled Optimization Loop

For each variant:

1. State one explicit hypothesis.
2. Save the baseline measurements before editing.
3. Make the smallest code change that tests the hypothesis.
4. Run a benchmark-sized BF16-activation / FP32-weight parity smoke test.
5. Run the focused B200 microbenchmark/profile.
6. Compare against the immediately preceding baseline.
7. Reject the variant if it does not improve the target or violates a guardrail.
8. Record:
   - source change;
   - shape;
   - kernel duration;
   - full backward duration;
   - registers/thread;
   - spills;
   - occupancy;
   - forward impact;
   - H100 impact;
   - correctness result.

Start with parameter tuning, then proceed to structural changes. Stop after two
consecutive variants improve the target by less than 1%, unless a new profile
identifies a different bottleneck.

## Performance Guardrails

Reject a variant when:

- a non-target metric regresses by more than 5%;
- forward regresses by more than 5%;
- H100 materially regresses to improve only B200;
- one pass regresses by more than 10% for a marginal gain elsewhere;
- memory usage increases without a justified and measured speed gain;
- any correctness test fails;
- a mixed BF16/FP32 case falls back unexpectedly;
- irregular dimensions or non-affine RMSNorm break.

The existing B200 forward win at long sequence length must be preserved.

## Correctness Requirements

Cover:

- FP32 and BF16 activations;
- BF16 and FP32 weights;
- Llama, Gemma, and no-cast modes;
- affine and non-affine operation;
- in-place and out-of-place backward;
- hidden sizes 512, 1023, 2050, 4096, and 8192;
- mixed-weight compilation in the same process to detect cache-key collisions;
- partial final CTAs and rows not divisible by the SM count.

Run the full targeted suite after selecting a winner:

```bash
python -m pytest \
  test/transformers/test_cutedsl_rms_norm_fastpath.py \
  test/transformers/test_cutedsl_rms_norm.py \
  -xvs --confcutdir=test/transformers -o addopts=""

make checkstyle
```

## Final Benchmark and Visualization

After correctness and checkstyle pass, rerun token-length and model-config
benchmarks on B200 and H100.

From `benchmark/`:

```bash
python benchmarks_visualizer.py \
  --kernel-name rms_norm \
  --metric-name speed \
  --sweep-mode token_length \
  --data-file data/all_benchmark_data_cutedsl_rms_norm_b200.csv \
  --overwrite

python benchmarks_visualizer.py \
  --kernel-name rms_norm \
  --metric-name speed \
  --sweep-mode model_config \
  --data-file data/all_benchmark_data_cutedsl_rms_norm_b200.csv \
  --overwrite

python benchmarks_visualizer.py \
  --kernel-name rms_norm \
  --metric-name memory \
  --sweep-mode token_length \
  --data-file data/all_benchmark_data_cutedsl_rms_norm_b200.csv \
  --overwrite

python benchmarks_visualizer.py \
  --kernel-name rms_norm \
  --metric-name memory \
  --sweep-mode model_config \
  --data-file data/all_benchmark_data_cutedsl_rms_norm_b200.csv \
  --overwrite
```

Repeat with the H100 CSV.

## Commit Discipline

Before committing:

```bash
git status --short
git diff --check
git diff -- src/liger_kernel/ops/cutedsl/ops/rms_norm.py
git diff -- src/liger_kernel/ops/cutedsl/ops/rms_norm_fastpath.py
git diff -- test/transformers/test_cutedsl_rms_norm.py
git diff -- test/transformers/test_cutedsl_rms_norm_fastpath.py
```

Stage only intentional source and test changes. Exclude benchmark data, plots,
Nsight reports, generated SQLite files, and temporary scripts.

Do not commit an optimization until:

- correctness passes;
- checkstyle passes;
- B200 improvement is measured;
- H100 and forward guardrails are checked;
- the result is compared with Triton using the same dtype and shape.

## Expected Final Report

When finished, report:

1. The measured root cause of the remaining backward gap.
2. The profiles that prove the diagnosis.
3. Every variant tried and why it was kept or rejected.
4. Before/after B200 and H100 tables for forward, backward, and full pass.
5. Registers, spills, occupancy, bandwidth, and stall changes.
6. Correctness and checkstyle results.
7. Files changed and the final commit hash.
8. Remaining limitations or shapes where Triton is still faster.

The central question is not simply "how can occupancy be increased?" Both
current kernels intentionally launch one CTA per SM. Determine whether the real
remaining cost is mixed-dtype code generation, CuTe wrapper/FFI overhead,
autograd integration, or the dW epilogue, and optimize the component that the
measurements identify.
