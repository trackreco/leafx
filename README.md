# leafx

leafx generates, from MkFitCore's sources as they are, a copy of mkFit's leaf code for a port to use,
for example a GPU port. The copy covers:

- Matriplex;
- the track propagation;
- the Kalman update;
- the mini propagators;
- vdt.

In the copy:

- every function that works on Matriplexes, and every struct with Matriplex members, is a template on
  the lane count `N`;
- every function carries a host/device annotation, `MKFIT_HOST_DEVICE`;
- everything lives in a target namespace, `mkfit::portable`.

MkFitCore uses N = NN = 8 lanes per Matriplex on the CPU. A GPU port uses the same code with N = 1, one
track per thread. MkFitCore keeps its own code and build. The port owns the configuration and the
generated copy, and regenerates the copy when MkFitCore changes.

```
# a host copy (what test/run-test.sh compiles against MkFitCore)
python3 leafx.py configs/mkfitcore-host.toml   -o OUT --source-root CMSSW/src
# a device copy: also Matriplex and vdt, every function host/device
LEAFX_VDT_ROOT=DIR_WITH_vdt python3 leafx.py configs/mkfitcore-device.toml -o OUT --source-root CMSSW/src
# exit 1 if OUT is stale against the sources
python3 leafx.py CONFIG -o OUT --source-root CMSSW/src --check
# cross-check the parse with libclang
python3 leafx.py CONFIG -o OUT --source-root CMSSW/src --verify -I VDT_INCLUDE
```

Requirements:
- Python 3.11 or newer (for `tomllib`), and nothing else;
- `clang-format`, to format the output;
- `python3-clang` (libclang), only for `--verify`.

The source tree can also be given as `$LEAFX_SOURCE_ROOT`.

The same two configurations work on MkFitCore of cms-sw master (the base of SegmentLinking/cmssw#290)
and on trackreco/cmssw `mkfinder-v2p2-perf` (see Tests).

## What it writes

| file | what |
|---|---|
| `leafx_types.h` | the annotation macros; the prelude includes; the Matriplex dimension constants and types as alias templates, read from `Matrix.h`: `template <int N> using MPlexLV = Matriplex::Matriplex<float, LL, 1, N>;` |
| one header per source file | the transformed functions and classes |
| `leafx_kernels/*.ah` | copies of the generated matrix kernels the functions include, made safe for any N |
| `Matriplex/`, `vdt/` (device configuration) | copies of the whole headers, every function annotated, in `mkfit::portable::Matriplex` and `mkfit::portable::vdt` |
| `leafx_all.h` | all of the above, in dependency order |
| `leafx_report.txt` | errors; the external names the copy uses; every decision that was not mechanical |

Every generated file starts with a stamp naming its source files and their git blobs.

## The rules

**Functions**

| rule | example |
|---|---|
| `template <int N>` on functions whose signature has a Matriplex | `void f(const MPlexLV& p)` -> `template <int N> MKFIT_HOST_DEVICE void f(const MPlexLV<N>& p)` |
| functions that need N but cannot deduce it from their arguments are templated anyway, and calls from generated code pass `<N>` | `MultFull<N>(...)` |
| other functions are annotated and made `inline` | `float getS(float ...)` -> `MKFIT_HOST_DEVICE inline float getS(...)` |
| `NN` -> `N`; `const idx_t N = NN;` is removed | `for (int n = 0; n < NN; ++n)` -> `n < N` |
| aliases on Matriplexes become alias templates | `using MPF = MPlexQF;` -> `template <int N> using MPF = MPlexQF<N>;` |
| arrays with a runtime size get size `N`, reported | `float r0[nmax - nmin];` -> `float r0[N];` |
| a pointer parameter to a Matriplex is not deduced, so a `nullptr` argument works | `const MPlexQI* p = nullptr` -> `const leafx_nd<MPlexQI<N>>* p = nullptr` |
| namespace-scope Matriplexes (dummy outputs) become a local variable in each function that uses them, so no global is shared between threads | `dummy_chi2` |
| `throw X;` -> `LEAFX_THROW(X);` | the throw on the host, `__trap()` in CUDA device code, `abort()` in HIP device code |

**Files and namespaces**

| rule | example |
|---|---|
| a file-scope `#include` can be spliced in, before parsing | `splice = ["KalmanUtilsMPlex.icc"]` |
| the anonymous namespace and file-scope functions become `detail_<file>` | |
| calls into `detail_<file>` and into the generated API are fully qualified | qualifying also switches argument-dependent lookup off. With N = NN, that lookup through `mkfit::PropagationFlags` would otherwise find MkFitCore's own non-template function and prefer it. |
| source namespaces map under the target | `namespace_map = { "mkfit" = "", "mkfit::mini_propagators" = "mini_propagators" }` |
| `mkfit::X` -> `X`; `using namespace mkfit;` is dropped | so that a declaration in the target namespace can shadow `X` |
| `#if` blocks on configured macros are resolved; debug calls are removed | `#ifdef DEBUG`, `#ifdef UNUSED`; `dprint(...)` |
| enums and namespace-scope `constexpr` scalars are kept as they are; class forward declarations follow their class; other namespace-scope declarations are dropped, and reported | |

**Classes**

| rule | example |
|---|---|
| a class with a data member of a Matriplex type becomes a template on N, and its name a type that takes `<N>` everywhere | `struct StatePlex { MPF x, y, z; ... }` -> `template <int N> struct StatePlex { MPF<N> x, y, z; ... }` |
| in a class that is not templated, a member function that takes a Matriplex becomes a member template | `State(const MPlexLV& par, int ti)` |
| out-of-class definitions get the template header and the qualification | `int InitialStatePlex::propagate_to_r(...)` -> `template <int N> ... InitialStatePlex<N>::propagate_to_r(...)` |
| members of a templated base, used unqualified in a derived class, get `this->`: a dependent base is not searched by unqualified lookup | `px *= pt;` -> `this->px *= pt;` |
| member functions are annotated, except those defaulted or deleted | |

**Matrix kernels.** The `.ah` kernels choose an AVX path by `#ifdef MPLEX_INTRINSICS`, which cannot
depend on N. That path steps a whole vector of lanes at a time, so at N = 1 it would write 8 floats into
a one-lane Matriplex. In the copy the intrinsics branch is under `if constexpr (N % width == 0)`:
N = NN takes the same intrinsics as MkFitCore, and N = 1 takes the plain loop. MatriplexSym's 3×3 and
6×6 multiplies have the same switch, and get the same treatment in the library copy.

**Library mode** (`[[library]]`, for Matriplex and vdt in the device configuration):
- the whole header is copied, and every function, free and member, is annotated;
- its namespaces move under the target, and its include guard is renamed, so the copy and the original
  can both be included;
- namespace-scope constant tables get `MKFIT_DEVICE_CONST`, which is `__device__` in the device
  compilation pass.

## The device prelude: what a device build provides

The copy refers to MkFitCore's names unqualified, from inside `mkfit::portable`:
`Config::`, `Const::`, `vdt::`, `TrackerInfo`, `PropagationFlags`, `hipo()` and so on.
- On the host these resolve to MkFitCore's own.
- A device build declares its own versions in `mkfit::portable`, and they take precedence.

`leafx_report.txt` lists every such name, with its use count. The device configuration includes
`leafx-device-prelude.h`, written by the consumer. `test/gpu/leafx-device-prelude.h` is a working
example of about 110 lines:
- it includes the generated Matriplex and vdt, and defines `MPLEX_VDT` as MkFitCore's `Matrix.h` does;
- it takes the `constexpr` `Config::` and `Const::` from MkFitCore's `Config.h`;
- it fixes `Config::usePtMultScat`, which MkFitCore sets at run time; the GPU test checks the value
  against the host run;
- it provides host/device `bFieldFromZR`, `hipo`, `isFinite`, `getPhi`, `getTheta` and `sincos4`;
- it provides `TrackerInfo` (the material map on a flat array, `material_checked()` as in MkFitCore),
  `PropagationFlags`, `SVector3` and `ModuleInfo` (the module plane for the mini propagators), and
  `cpe_func` as a function pointer instead of a `std::function`.

## Markup in the sources (optional)

Comments, so that no compiler ever sees them:

| markup | effect |
|---|---|
| `//@leafx skip` before a function or member | it is left out of the copy |
| `//@leafx begin-skip` ... `//@leafx end-skip` | the lines in between are left out |
| `//@leafx size(EXPR)` on an array declaration | the array gets size `EXPR` |

Each piece of markup has a configuration equivalent:
- `skip = [...]` per file, which takes `name` or `Class::member`;
- `host_only = [...]`: kept, but not annotated;
- `undefine`, `strip_calls`.

MkFitCore needs no markup at all. `test/markup/` shows every rule and every piece of markup on a small,
self-contained sample.

## Tests

### Host: `test/run-test.sh`

It takes these from the environment:
- `CMSSW_SRC`;
- `MKFIT_BUILD`: a standalone mkFit build of that tree;
- `ROOTSYS`;
- `MODE`;
- `DUMP`: writes inputs and outputs for the GPU test.

`test/test_leaf.cc` runs 16 functions on 7,574 random tracks each (1000 batches of 8, every 7th batch
partial), with the parametrised field and the phase-2 material map (D121):
- `propagateHelixToRMPlex`, `ToZMPlex`, `ToPlaneMPlex`, `propagateLineToRMPlex`;
- `kalmanUpdate`, `kalmanComputeChi2`, their endcap versions, `kalmanUpdatePlane`,
  `kalmanPropagateAndUpdateAndChi2Plane`, `kalmanPropagateAndComputeChi2Plane`,
  `kalmanOperationPlaneLocal` (with its optional outputs);
- `mini_propagators`: `propagate_to_r`, `propagate_to_z`, a two-point `Hermite3D`, and the per-hit
  `InitialState::propagate_to_plane`.

It compares every output float bitwise: MkFitCore at NN = 8 against the copy at N = 8, and against the
copy at N = 1, lane by lane.

| MODE | reference | result (2026-10-13, trackreco/cmssw `mkfinder-v2p2-perf`, gcc 15, `-mavx`) |
|---|---|---|
| `strict`: the build's flags with `-Ofast` replaced by `-O2 -fno-fast-math` | MkFitCore's sources, compiled into the test | **0 differing floats**, in all 32 comparisons (16 functions × N = 8 and N = 1) |
| `ofast`: the build's own flags | `libMkFitCore.so` | rounding-sized differences (below) |

**The transformation is exact.** Under `-Ofast` the compiler optimises the same arithmetic differently in
different code contexts: vectorised against scalar math functions, and reassociation.
- In the propagation, the worst covariance difference is 1e-2 of sqrt(C_ii C_jj), at N = 1 in the
  iterative propagation to R.
- In the Kalman updates, the same source compiled for 8 lanes and for 1, both under `-Ofast`, already
  differs by up to 3.3e9 of sqrt(C_ii C_jj) on 30 of 7,574 tracks. The update (I − KH)C subtracts large
  numbers to get small ones, and that amplifies last-bit differences on the ill-conditioned updates among
  these random inputs.

### GPU: `test/gpu/run-gpu-test.sh`

It generates the device copy, compiles `test_gpu.cu` with nvcc, and runs every function at N = 1, one
track per thread. It compares with the host copy's outputs from the dump.

Result: RTX 4070 (sm_89), CUDA 13.3, gcc 13 host compiler, inside `cmssw-el9`; `--fmad=false`:

| | result |
|---|---|
| nvcc's host compilation vs gcc 15's | **bitwise identical**, all 16 functions |
| GPU vs host: plane and line propagation, `kalmanUpdate`, `kalmanComputeChi2`, their endcap versions, the three Plex mini propagators | **bitwise identical** (9 of 16) |
| GPU vs host: propagation to R and Z, the plane Kalman updates, the per-hit plane propagation | rounding level: the device's `std::sin`, `std::cos` and `std::atan2` are other implementations than glibc's |
| GPU vs host: `kalmanPropagateAndUpdateAndChi2Plane` | 16 tracks above 1e-2 of sqrt(C_ii C_jj) (3 of them with a negative variance on the host already); fewer than the CPU shows between 8 lanes and 1 under `-Ofast` (30) |

With nvcc's default `--fmad=true` every function differs at rounding level.

### The release tree: `test/release/instantiate.cc`

It instantiates the main entry points with the release signatures. The copy generated from cms-sw
master's MkFitCore by the same two configurations compiles with gcc at N = 1 and N = 8, and with nvcc
in a CUDA kernel at N = 1, with no host-function-from-device warnings.

The functional tests above are written for the `mkfinder-v2p2-perf` signatures.

## Not done yet, and caveats

- **Random inputs, not reconstructed tracks.** Some of them are ill-conditioned updates.
- **Plain CUDA kernels.** An Alpaka kernel calls the same functions; this was not built here.
- **HIP** is not tested; the rules are written for it.
- **No performance measurement.** The kernel times the GPU test prints are not a benchmark.
- **Heuristics to keep an eye on:**
  - the automatic sizing of runtime-sized arrays to `N`;
  - the call qualification, which recognises a call by `name(` or `name<`;
  - the class templating, which is decided by data members;
  - `this->`, which is added to base members unless they are a parameter of the function (a local
    variable with the same name as a base member would be rewritten wrongly);
  - the dependency report, which counts a name as external only where it is used as a call.
- **`--verify`** needs the include paths of everything the sources include; with ROOT's SMatrix missing
  (MiniPropagators via TrackState.h) libclang takes minutes.

## License

Apache License 2.0, as the rest of trackreco.
