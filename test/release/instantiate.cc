// leafx on the release MkFitCore (cms-sw master, the base of SegmentLinking/cmssw#290): instantiates
// the generated copy's main entry points with the release signatures, at N = 1 and N = NN on the host,
// and at N = 1 in a CUDA kernel when compiled with nvcc. A compile test: it checks that the copy
// generated from that tree is complete and well-formed; the functional tests run on the tree they
// were written for.

#include "leafx_all.h"

namespace P = mkfit::portable;

// Names the device prelude declares in mkfit::portable and the host takes from mkfit: looked up
// unqualified from inside the namespace, as the generated code does.
namespace mkfit::portable {
  using Test_PF = PropagationFlags;
  using Test_MI = ModuleInfo;
}  // namespace mkfit::portable

template <int N>
struct Args {
  P::MPlexLS<N> err, oerr;
  P::MPlexLV<N> par, opar;
  P::MPlexQI<N> chg, fail;
  P::MPlexQF<N> rad, chi2;
  P::MPlexHV<N> pnt, nrm, dir, mspar;
  P::MPlexHS<N> mserr;
};

template <int N>
MKFIT_HOST_DEVICE void instantiate(Args<N>& a, const P::Test_PF& pf) {
  namespace MP = P::mini_propagators;
  P::propagateHelixToRMPlex<N>(a.err, a.par, a.chg, a.rad, a.oerr, a.opar, a.fail, N, pf);
  P::propagateHelixToZMPlex<N>(a.err, a.par, a.chg, a.rad, a.oerr, a.opar, a.fail, N, pf);
  P::propagateHelixToPlaneMPlex<N>(a.err, a.par, a.chg, a.pnt, a.nrm, a.oerr, a.opar, a.fail, N, pf);
  P::propagateLineToRMPlex<N>(a.err, a.par, a.mserr, a.mspar, a.oerr, a.opar, N);
  P::kalmanUpdate<N>(a.err, a.par, a.mserr, a.mspar, a.oerr, a.opar, N);
  P::kalmanComputeChi2<N>(a.err, a.par, a.chg, a.mserr, a.mspar, a.chi2, N);
  P::kalmanUpdateEndcap<N>(a.err, a.par, a.mserr, a.mspar, a.oerr, a.opar, N);
  P::kalmanPropagateAndUpdateAndChi2Plane<N>(a.err, a.par, a.chg, a.mserr, a.mspar, a.nrm, a.dir, a.pnt, a.oerr,
                                             a.opar, a.fail, a.chi2, N, pf, true);
  P::kalmanPropagateAndComputeChi2Plane<N>(a.err, a.par, a.chg, a.mserr, a.mspar, a.nrm, a.dir, a.pnt, a.chi2,
                                           a.opar, a.fail, N, pf, true);
  P::kalmanOperationPlaneLocal<N>(P::KFO_Update_Params | P::KFO_Calculate_Chi2, a.err, a.par, a.chg, a.mserr,
                                  a.mspar, a.nrm, a.dir, a.pnt, a.oerr, a.opar, a.chi2, N);
  MP::InitialStatePlex<N> isp(a.par, a.chg);
  MP::StatePlex<N> sp;
  isp.propagate_to_r(MP::PA_Exact, a.rad, sp, true, N);
  isp.propagate_to_z(MP::PA_Exact, a.rad, sp, true, N);
  MP::InitialState is(a.par, a.chg, 0);
  MP::State c;
  P::Test_MI mi;
  is.propagate_to_plane(MP::PA_Line, mi, c, true);
}

#ifdef __CUDACC__
__global__ void kernel(Args<1>* a, P::Test_PF pf) { instantiate<1>(a[threadIdx.x], pf); }
#endif

int main() {
  P::Test_PF pf;
  Args<1> a1;
  Args<8> a8;
  instantiate<1>(a1, pf);
  instantiate<8>(a8, pf);
#ifdef __CUDACC__
  kernel<<<1, 1>>>(nullptr, pf);
#endif
  return 0;
}
