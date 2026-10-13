# leafx

leafx generates a copy of mkFit's leaf code for a port to use: the Matriplex physics, so far the
track propagation. It reads MkFitCore's sources unchanged, and writes code in which:

- every function that works on Matriplexes is a template on the lane count `N`;
- every function carries a host/device annotation, `MKFIT_HOST_DEVICE`;
- everything lives in a target namespace, `mkfit::portable`.

MkFitCore uses N = NN = 8 lanes per Matriplex on the CPU. A GPU port uses the same code with N = 1,
one track per thread. MkFitCore keeps its own code and build. The port owns the configuration and
the generated copy, and regenerates the copy when MkFitCore changes.

```
python3 leafx.py configs/mkfitcore-host.toml -o OUT --source-root CMSSW/src            # generate
python3 leafx.py configs/mkfitcore-host.toml -o OUT --source-root CMSSW/src --check    # exit 1 if OUT is stale
python3 leafx.py configs/mkfitcore-host.toml -o OUT --source-root CMSSW/src --verify -I VDT_INCLUDE
python3 leafx.py test/markup/sample.toml -o OUT                                          # the markup sample
```

Requirements:
- Python 3.11 or newer (for `tomllib`), and nothing else;
- `clang-format`, to format the output;
- `python3-clang` (libclang), only for `--verify`.

`--verify` cross-checks the tokenizer's function boundaries against libclang's. The source tree can
also be given as `$LEAFX_SOURCE_ROOT`.

## What it writes

| file | what |
|---|---|
| `leafx_types.h` | the Matriplex types as alias templates, read from `Matrix.h`: `template <int N> using MPlexLV = Matriplex::Matriplex<float, 6, 1, N>;`. Also the annotation macro and a helper for non-deduced parameters. |
| one header per source file | the transformed functions |
| `leafx_kernels/*.ah` | copies of the generated matrix kernels the functions include, made safe for any N (see below) |
| `leafx_all.h` | all of the above, in dependency order |
| `leafx_report.txt` | errors; the external names the copy uses; every non-mechanical decision |

Every generated file starts with a stamp naming its source files and their git blobs.

## The rules

| rule | example |
|---|---|
| `template <int N>` on functions whose signature has a Matriplex | `void f(const MPlexLV& p)` -> `template <int N> MKFIT_HOST_DEVICE void f(const MPlexLV<N>& p)` |
| other functions: annotated and `inline` | `float getS(float ...)` -> `MKFIT_HOST_DEVICE inline float getS(...)` |
| `NN` -> `N`; `const idx_t N = NN;` removed | `for (int n = 0; n < NN; ++n)` -> `n < N` |
| file-level aliases on Matriplexes become alias templates | `using MPF = MPlexQF;` -> `template <int N> using MPF = MPlexQF<N>;` |
| arrays with a runtime size get size `N`, and the change is reported | `float r0[nmax - nmin];` -> `float r0[N];` |
| pointer parameters to a Matriplex are not deduced, so a `nullptr` argument works | `const MPlexQI* p = nullptr` -> `const leafx_nd<MPlexQI<N>>* p = nullptr` |
| the anonymous namespace becomes `detail_<file>`; calls into it and into the generated API are fully qualified | qualifying also switches argument-dependent lookup off. With N = NN, that lookup through `mkfit::PropagationFlags` would otherwise find MkFitCore's own non-template function and prefer it. |
| `mkfit::X` -> `X`; `using namespace mkfit;` is dropped | so that a declaration in the target namespace can shadow `X` |
| `#if` blocks on configured macros are resolved; debug calls are removed | `#ifdef DEBUG`, `#ifdef UNUSED`; `dprint(...)` |
| matrix kernels (`.ah`) are copied, with the intrinsics branch under `if constexpr` on N | The kernels choose an AVX path with `#ifdef MPLEX_INTRINSICS`, which cannot depend on N. That path steps a whole vector of lanes at a time, so at N = 1 it would write 8 floats into a one-lane Matriplex. In the copy, N = NN takes the same intrinsics as MkFitCore, and N = 1 the plain loop. |
| forward declarations of classes and namespace-scope variables are dropped, and reported | `class PropagationFlags;` |

**Name resolution is left to the consumer.** The copy refers to `Config::`, `Const::`, `vdt::`,
`TrackerInfo`, `PropagationFlags`, `hipo()` and so on unqualified, from inside the target namespace.
- On the host these resolve to MkFitCore's own.
- A device build declares its own versions in the target namespace, and they take precedence.

The report lists every such name with its use count; that list is what a device build has to provide.
For the propagation there are 32, among them `Config::matScale`, `Config::matScaleFwdPix`,
`Config::matElossVarScale` and `Config::usePtMultScat`, which `applyMaterialEffects()` reads.

## Markup in the sources (optional)

Comments, so that no compiler ever sees them:

| markup | effect |
|---|---|
| `//@leafx skip` before a function | the function is left out of the copy |
| `//@leafx begin-skip` ... `//@leafx end-skip` | the lines in between are left out |
| `//@leafx size(EXPR)` on an array declaration | the array gets size `EXPR` in the copy |

Each piece of markup has a configuration equivalent: `skip = [...]` per file, `undefine`,
`strip_calls`. The propagation needs no markup at all. `test/markup/` shows every rule and every piece
of markup on a small, self-contained sample.

## Test

`test/run-test.sh` takes these from the environment:
- `CMSSW_SRC`: the src/ tree;
- `MKFIT_BUILD`: a standalone mkFit build of that tree;
- `ROOTSYS`: if the build uses ROOT;
- `MODE`.

`test/test_prop.cc` runs `propagateHelixToRMPlex`, `propagateHelixToZMPlex`,
`propagateHelixToPlaneMPlex` and `propagateLineToRMPlex` on 15,145 random tracks each: 2000 batches of
8, every 7th batch partial. The parametrised field and the phase-2 material map (D121) are on. It
compares every output float bitwise, among three versions:
- MkFitCore at NN = 8;
- the copy at N = 8;
- the copy at N = 1, lane by lane.

In `strict` mode the test fails on any difference.

Result, 2026-10-13, MkFitCore of trackreco/cmssw `mkfinder-v2p2-perf`, gcc 15, `-mavx`:

| MODE | reference | result |
|---|---|---|
| `strict`: the build's flags with `-Ofast` replaced by `-O2 -fno-fast-math` | MkFitCore's sources, compiled into the test | **0 differing floats of 408,915**, in every comparison, for all 4 functions |
| `ofast`: the build's own flags | `libMkFitCore.so` | the copy at N = 8: plane and line bitwise identical; R and Z differ in 1.7 % and 4.7 % of floats, worst 5.5e-6 of sqrt(C_ii C_jj). The copy at N = 1: worst 1e-2 of sqrt(C_ii C_jj), for 444 of 15,145 tracks above 1e-4, in the iterative propagation to R. |
| `ofast-src` | MkFitCore's sources, compiled with `-Ofast` | same as `ofast` |

**The transformation is exact.** Under `-Ofast` the compiler optimises the same arithmetic differently
in different code contexts: vectorised against scalar math functions, and reassociation. Which
optimisation causes each difference has not been determined.

## Not done yet

- **The Kalman update.** It needs the same rules, plus a rule or markup for two things: its
  namespace-scope `dummy_*` globals and its `std::function` CPE callback.
- **A device compile.** Matriplex itself needs a device copy: annotating class members is not written
  yet. A device build also needs a set of the names in the report.
- **Classes in leaf files** are reported as errors and not transformed.
- **Heuristics to keep an eye on:**
  - the automatic sizing of runtime-sized arrays to `N`;
  - the call qualification, which recognises a call by `name(` or `name<`;
  - the dependency report, which counts a name as external only where it is used as a call.

## License

Apache License 2.0, as the rest of trackreco.
