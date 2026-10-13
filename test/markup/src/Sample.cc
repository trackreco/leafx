// A sample leaf file with every rule and every piece of markup leafx knows.
#include <vdt/sincos.h>
#include "Matrix.h"
#include "Debug.h"

namespace {
  using namespace mkfit;

  using MPF = MPlexQF;  // a file-level alias: becomes a template on N

  // A matrix kernel helper: N comes from the template now, the .ah gets a lane-guarded copy.
  void MultSample(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    typedef float T;
    const Matriplex::idx_t N = NN;
    const T* a = A.fArray;
    T* c = C.fArray;
    ASSUME_ALIGNED(a, 64);
#include "SampleKernel.ah"
  }

  // No Matriplex in the signature: annotated and inline, not a template.
  float scalar_helper(float x) { return x * x; }

  void lanes_impl(const MPlexLV& par, MPlexQF& out, const int nmin, const int nmax) {
    float tmp[nmax - nmin];  // runtime size: becomes [N] automatically (reported)
    float other[nmax - nmin];  //@leafx size(N + 1)
    for (int n = nmin; n < nmax; ++n) {
      tmp[n - nmin] = scalar_helper(par.constAt(n, 3, 0));
      other[n - nmin] = tmp[n - nmin];
      out[n] = other[n - nmin];
    }
    dprint("not on the device");  // debug statement: removed
  }
}  // namespace

namespace mkfit {

  class PropagationFlags;  // forward declaration: dropped, the consumer decides what it is

  void sample(const MPlexLV& par, MPlexQF& out, const int N_proc, const MPlexQI* mask = nullptr) {
    MPF tmp;  // file-level alias, now MPF<N>
    lanes_impl(par, tmp, 0, NN);
    for (int n = 0; n < NN; ++n)
      out[n] = (mask && mask->constAt(n, 0, 0)) ? 0.f : mkfit::hipo(tmp[n], 1.f);
#ifdef DEBUG
    printf("resolved away: DEBUG is in `undefine`\n");
#endif
    //@leafx begin-skip
    std::cout << "host-only diagnostics, dropped from the copy\n";
    //@leafx end-skip
  }

  //@leafx skip
  void host_only_dump(const MPlexLV& par) { std::cout << par.constAt(0, 0, 0) << "\n"; }

}  // namespace mkfit
