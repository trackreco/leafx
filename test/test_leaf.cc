// leafx functional test: MkFitCore's propagation and Kalman update (NN lanes) against the leafx copy
// instantiated with N = NN and with N = 1, on random tracks, with the parametrised field and the
// phase-2 material map. Compares every output float bitwise.
//
//   test_leaf GEOMETRY_FILE [N_BATCHES] [strict] [dump=FILE]
//
// With `strict` the exit status is 1 on any difference (meaningful only when the reference and the copy
// are compiled without fast math). With `dump=FILE` it writes, for the GPU test (test/gpu/), the
// material map, the runtime Config values, and per function the inputs of every track and the
// outputs of the generated copy at N = 1.

#include "RecoTracker/MkFitCore/src/PropagationMPlex.h"  // reference: MkFitCore, NN lanes
#include "RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "leafx_all.h"  // generated: mkfit::portable, templated on N

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace mkfit;
namespace P = mkfit::portable;

// ---------------------------------------------------------------------------
// Inputs and outputs of one batch, for N lanes. P::MPlexXX<NN> is MkFitCore's MPlexXX.

template <int N>
struct In {
  P::MPlexLS<N> err;
  P::MPlexLV<N> par;
  P::MPlexQI<N> chg;
  P::MPlexQF<N> rad;           // target r or z
  P::MPlexHV<N> pnt, nrm, dir;  // a plane: point, normal, direction in the plane
  P::MPlexHS<N> mserr;         // a hit
  P::MPlexHV<N> mspar;

  template <int M>
  void lane_from(const In<M>& b, int n) {
    copy(b.err, n, err);
    copy(b.par, n, par);
    copy(b.chg, n, chg);
    copy(b.rad, n, rad);
    copy(b.pnt, n, pnt);
    copy(b.nrm, n, nrm);
    copy(b.dir, n, dir);
    copy(b.mserr, n, mserr);
    copy(b.mspar, n, mspar);
  }
  void write(FILE* f) const {
    w(f, err);
    w(f, par);
    w(f, chg);
    w(f, rad);
    w(f, pnt);
    w(f, nrm);
    w(f, dir);
    w(f, mserr);
    w(f, mspar);
  }
  template <class M>
  static void w(FILE* f, const M& m) {
    fwrite(m.fArray, sizeof(m.fArray[0]), M::kTotSize, f);
  }
  template <class Big, class Small>
  static void copy(const Big& b, int n, Small& s) {
    typename Big::value_type tmp[Big::kSize];
    b.copyOut(n, tmp);
    s.copyIn(0, tmp);
  }
};

template <int N>
struct Out {
  P::MPlexLS<N> err;
  P::MPlexLV<N> par;
  P::MPlexQI<N> fail, chg;
  P::MPlexQF<N> chi2, x1, x2;

  void zero(const P::MPlexQI<N>& in_chg) {
    err.setVal(0.f);
    par.setVal(0.f);
    fail.setVal(0);
    chg = in_chg;
    chi2.setVal(0.f);
    x1.setVal(0.f);
    x2.setVal(0.f);
  }
  template <int M>
  void lane_to(Out<M>& b, int n) const {
    copy(err, b.err, n);
    copy(par, b.par, n);
    copy(fail, b.fail, n);
    copy(chg, b.chg, n);
    copy(chi2, b.chi2, n);
    copy(x1, b.x1, n);
    copy(x2, b.x2, n);
  }
  void write(FILE* f) const {
    In<N>::w(f, err);
    In<N>::w(f, par);
    In<N>::w(f, fail);
    In<N>::w(f, chg);
    In<N>::w(f, chi2);
    In<N>::w(f, x1);
    In<N>::w(f, x2);
  }
  template <class Small, class Big>
  static void copy(const Small& s, Big& b, int n) {
    typename Small::value_type tmp[Small::kSize];
    s.copyOut(0, tmp);
    b.copyIn(n, tmp);
  }
};

// ---------------------------------------------------------------------------
// Comparison

struct Cmp {
  long n_float = 0, n_diff = 0, n_tracks = 0, n_tracks_diff = 0, n_int_diff = 0;
  double max_rel = 0;             // parameters, chi2: |dx| / max(|x|, 1e-3)
  double max_relcov = 0;          // covariance: |dC_ij| / sqrt(C_ii C_jj)
  long n_trk_cov[3] = {0, 0, 0};  // tracks whose worst covariance difference is > 1e-6, 1e-4, 1e-2

  void add(const Out<NN>& a, const Out<NN>& b, int n_proc) {
    for (int n = 0; n < n_proc; ++n) {
      bool any = false;
      double trk_cov = 0;
      auto cmp = [&](float x, float y, double scale, double& mx) {
        ++n_float;
        if (std::memcmp(&x, &y, sizeof(float)) == 0 || (std::isnan(x) && std::isnan(y)))
          return;
        ++n_diff;
        any = true;
        const double d = std::abs(double(x) - double(y));
        if (scale > 0 && d / scale > mx)
          mx = d / scale;
      };
      auto rel = [&](float x, float y) { cmp(x, y, std::max(std::abs(double(x)), 1e-3), max_rel); };
      for (int i = 0; i < 6; ++i)
        rel(a.par.constAt(n, i, 0), b.par.constAt(n, i, 0));
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          cmp(a.err.constAt(n, i, j), b.err.constAt(n, i, j),
              std::sqrt(std::abs(double(a.err.constAt(n, i, i)) * a.err.constAt(n, j, j))), trk_cov);
      rel(a.chi2.constAt(n, 0, 0), b.chi2.constAt(n, 0, 0));
      rel(a.x1.constAt(n, 0, 0), b.x1.constAt(n, 0, 0));
      rel(a.x2.constAt(n, 0, 0), b.x2.constAt(n, 0, 0));
      if (a.fail.constAt(n, 0, 0) != b.fail.constAt(n, 0, 0) || a.chg.constAt(n, 0, 0) != b.chg.constAt(n, 0, 0)) {
        ++n_int_diff;
        any = true;
      }
      max_relcov = std::max(max_relcov, trk_cov);
      for (int k = 0; k < 3; ++k)
        n_trk_cov[k] += trk_cov > std::pow(10., -6 + 2 * k);
      ++n_tracks;
      n_tracks_diff += any;
    }
  }
  void print(const char* what) const {
    printf("  %-30s tracks %6ld, differing %6ld | floats differing %6ld of %6ld | max: par %.1e, cov %.1e | int %ld\n",
           what, n_tracks, n_tracks_diff, n_diff, n_float, max_rel, max_relcov, n_int_diff);
    printf("  %-30s tracks with cov difference > 1e-6: %ld, > 1e-4: %ld, > 1e-2: %ld\n", "", n_trk_cov[0],
           n_trk_cov[1], n_trk_cov[2]);
  }
};

// ---------------------------------------------------------------------------
// Random input

struct Gen {
  std::mt19937 rng;
  Gen(unsigned seed) : rng(seed) {}
  float u(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }
  float g(float s) { return std::normal_distribution<float>(0.f, s)(rng); }

  void fill(In<NN>& in, bool forward) {
    for (int n = 0; n < NN; ++n) {
      const float x = g(0.01f), y = g(0.01f), z = u(-10.f, 10.f);
      const float pt = std::exp(u(std::log(0.5f), std::log(50.f)));
      const float phi = u(-3.14159f, 3.14159f);
      const float eta = forward ? (u(0.f, 1.f) < 0.5f ? -1.f : 1.f) * u(1.2f, 2.8f) : u(-2.5f, 2.5f);
      const float theta = 2.f * std::atan(std::exp(-eta));
      const float par[6] = {x, y, z, 1.f / pt, phi, theta};
      for (int i = 0; i < 6; ++i)
        in.par(n, i, 0) = par[i];
      in.chg(n, 0, 0) = u(0.f, 1.f) < 0.5f ? -1 : 1;
      // covariance: diagonal plus a few correlations, positive definite
      float d[6] = {g(0.01f) * g(0.01f) + 1e-4f, 1e-4f + 1e-5f * u(0, 1), 1e-3f + 1e-3f * u(0, 1),
                    std::pow(0.02f / pt, 2.f) * u(0.5f, 1.5f), 1e-6f * u(0.5f, 2.f), 1e-6f * u(0.5f, 2.f)};
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          in.err(n, i, j) = (i == j) ? d[i] : 0.f;
      in.err(n, 4, 0) = 0.2f * std::sqrt(d[4] * d[0]);
      in.err(n, 4, 3) = 0.3f * std::sqrt(d[4] * d[3]);
      in.err(n, 5, 2) = 0.2f * std::sqrt(d[5] * d[2]);
      // target r (barrel) or z (forward)
      in.rad(n, 0, 0) = forward ? z + (eta > 0 ? 1.f : -1.f) * u(5.f, 120.f) : u(3.f, 110.f);
      // a plane through a point on the straight line, tilted in (r, z); a direction in it
      const float dist = u(3.f, 60.f);
      const float dir[3] = {std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta)};
      const float p[3] = {x + dist * dir[0], y + dist * dir[1], z + dist * dir[2]};
      const float r = std::hypot(p[0], p[1]), tau = u(-1.f, 1.f);
      const float nv[3] = {std::cos(tau) * p[0] / r, std::cos(tau) * p[1] / r, std::sin(tau)};
      float dv[3] = {-nv[1], nv[0], 0.f};  // n x z, normalised
      const float dl = std::hypot(dv[0], dv[1]);
      for (int i = 0; i < 2; ++i)
        dv[i] /= dl;
      const float wv[3] = {nv[1] * dv[2] - nv[2] * dv[1], nv[2] * dv[0] - nv[0] * dv[2], nv[0] * dv[1] - nv[1] * dv[0]};
      for (int i = 0; i < 3; ++i) {
        in.pnt(n, i, 0) = p[i];
        in.nrm(n, i, 0) = nv[i];
        in.dir(n, i, 0) = dv[i];
      }
      // a hit near p, in the plane: fine along dir, coarse along the strip (w)
      const float su = 0.003f, sv = u(0.02f, 1.f);
      const float hu = g(su), hv = g(sv);
      for (int i = 0; i < 3; ++i)
        in.mspar(n, i, 0) = p[i] + hu * dv[i] + hv * wv[i];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j)
          in.mserr(n, i, j) = su * su * dv[i] * dv[j] + sv * sv * wv[i] * wv[j] + 1e-8f * nv[i] * nv[j];
    }
  }
};

// ---------------------------------------------------------------------------
// The functions under test

enum Fn {
  F_R,
  F_Z,
  F_Plane,
  F_Line,
  F_KUpd,
  F_KChi2,
  F_KUpdEndcap,
  F_KChi2Endcap,
  F_KUpdPlane,
  F_KPropUpdChi2Plane,
  F_KPropChi2Plane,
  F_KOpPlaneLocal,
  F_Count
};
const char* fn_name[F_Count] = {"propagateHelixToRMPlex",
                                "propagateHelixToZMPlex",
                                "propagateHelixToPlaneMPlex",
                                "propagateLineToRMPlex",
                                "kalmanUpdate",
                                "kalmanComputeChi2",
                                "kalmanUpdateEndcap",
                                "kalmanComputeChi2Endcap",
                                "kalmanUpdatePlane",
                                "kalmanPropagateAndUpdateAndChi2Plane",
                                "kalmanPropagateAndComputeChi2Plane",
                                "kalmanOperationPlaneLocal (+detV, chi2Acc)"};
bool fn_forward(int f) { return f == F_Z || f == F_KUpdEndcap || f == F_KChi2Endcap; }

// LIB = true: MkFitCore's functions (N = NN only); false: the generated copy.
template <bool LIB, int N>
void run(Fn f, const In<N>& in, Out<N>& o, int np, const PropagationFlags& pf) {
#define CALL(fn, ...)            \
  if constexpr (LIB)             \
    mkfit::fn(__VA_ARGS__);      \
  else                           \
    P::fn<N>(__VA_ARGS__)
  const int kfo = KFO_Update_Params | KFO_Calculate_Chi2 | KFO_Local_Cov;
  switch (f) {
    case F_R:
      CALL(propagateHelixToRMPlex, in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Z:
      CALL(propagateHelixToZMPlex, in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Plane:
      CALL(propagateHelixToPlaneMPlex, in.err, in.par, in.chg, in.pnt, in.nrm, nullptr, o.err, o.par, o.fail, np, pf);
      break;
    case F_Line:
      CALL(propagateLineToRMPlex, in.err, in.par, in.mserr, in.mspar, o.err, o.par, np);
      break;
    case F_KUpd:
      CALL(kalmanUpdate, in.err, in.par, in.mserr, in.mspar, o.err, o.par, np);
      break;
    case F_KChi2:
      CALL(kalmanComputeChi2, in.err, in.par, in.chg, in.mserr, in.mspar, o.chi2, np);
      break;
    case F_KUpdEndcap:
      CALL(kalmanUpdateEndcap, in.err, in.par, in.mserr, in.mspar, o.err, o.par, np);
      break;
    case F_KChi2Endcap:
      CALL(kalmanComputeChi2Endcap, in.err, in.par, in.chg, in.mserr, in.mspar, o.chi2, np);
      break;
    case F_KUpdPlane:
      CALL(kalmanUpdatePlane, in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt, o.err, o.par, np);
      break;
    case F_KPropUpdChi2Plane:
      CALL(kalmanPropagateAndUpdateAndChi2Plane, in.err, in.par, o.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt,
           o.err, o.par, o.fail, o.chi2, np, pf, true);
      break;
    case F_KPropChi2Plane:
      CALL(kalmanPropagateAndComputeChi2Plane, in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt,
           o.chi2, o.par, o.fail, np, pf, true);
      break;
    case F_KOpPlaneLocal:
      CALL(kalmanOperationPlaneLocal, kfo, in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt, o.err,
           o.par, o.chi2, np, nullptr, nullptr, true, &o.x1, &o.x2, 4.0f);
      break;
    default:
      break;
  }
#undef CALL
}

int main(int argc, char* argv[]) {
  const std::string geom = argc > 1 ? argv[1] : "CMS-phase2.bin";
  const int n_batches = argc > 2 ? std::atoi(argv[2]) : 2000;
  bool strict = false;
  std::string dump_file;
  for (int a = 3; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "strict")
      strict = true;
    else if (arg.rfind("dump=", 0) == 0)
      dump_file = arg.substr(5);
  }
  FILE* dump = dump_file.empty() ? nullptr : fopen(dump_file.c_str(), "wb");

  TrackerInfo ti;
  ti.read_bin_file(geom);
  Config::usePtMultScat = true;  // as the phase-2 geometry plugin sets it
  PropagationFlags pf(PF_use_param_b_field | PF_apply_material);
  pf.tracker_info = &ti;
  printf("geometry %s, material bins %d x %d; %d batches of %d tracks per function\n", geom.c_str(),
         ti.mat_nbins_z(), ti.mat_nbins_r(), n_batches, NN);
  if (dump) {
    // header: magic, number of functions; Config; the material map
    const int head[2] = {0x6c656166, F_Count};
    fwrite(head, sizeof(int), 2, dump);
#ifdef MKFIT_STANDALONE
    const float cfgv[4] = {float(Config::usePtMultScat), Config::matScale, Config::matElossVarScale,
                           Config::matScaleFwdPix};
#else
    const float cfgv[4] = {float(Config::usePtMultScat), Config::matScale, Config::matElossVarScale, 1.f};
#endif
    fwrite(cfgv, sizeof(float), 4, dump);
    const int nz = ti.mat_nbins_z(), nr = ti.mat_nbins_r();
    const float fac[2] = {nz / ti.mat_range_z(), nr / ti.mat_range_r()};
    fwrite(&nz, sizeof(int), 1, dump);
    fwrite(&nr, sizeof(int), 1, dump);
    fwrite(fac, sizeof(float), 2, dump);
    for (int iz = 0; iz < nz; ++iz)
      for (int ir = 0; ir < nr; ++ir) {
        const float m[2] = {ti.material_bbxi(iz, ir), ti.material_radl(iz, ir)};
        fwrite(m, sizeof(float), 2, dump);
      }
  }

  int status = 0;
  for (int f = 0; f < F_Count; ++f) {
    Gen gen(12345 + f);
    Cmp c8, c1, c18;
    long n_failed = 0;
    std::vector<In<1>> d_in;
    std::vector<Out<1>> d_out;
    for (int b = 0; b < n_batches; ++b) {
      In<NN> in;
      gen.fill(in, fn_forward(f));
      const int np = (b % 7 == 6) ? 5 : NN;  // some partial batches
      Out<NN> lib, g8, g1;
      lib.zero(in.chg);
      g8.zero(in.chg);
      g1.zero(in.chg);
      run<true>(Fn(f), in, lib, np, pf);
      run<false>(Fn(f), in, g8, np, pf);
      for (int n = 0; n < np; ++n) {
        In<1> i1;
        Out<1> o1;
        i1.lane_from(in, n);
        o1.zero(i1.chg);
        run<false>(Fn(f), i1, o1, 1, pf);
        o1.lane_to(g1, n);
        if (dump) {
          d_in.push_back(i1);
          d_out.push_back(o1);
        }
      }
      for (int n = 0; n < np; ++n)
        n_failed += lib.fail.constAt(n, 0, 0) != 0;
      c8.add(lib, g8, np);
      c1.add(lib, g1, np);
      c18.add(g8, g1, np);
    }
    if (dump) {
      const int nt[2] = {f, int(d_in.size())};
      fwrite(nt, sizeof(int), 2, dump);
      for (size_t t = 0; t < d_in.size(); ++t) {
        d_in[t].write(dump);
        d_out[t].write(dump);
      }
    }
    printf("%s (failed for %ld tracks)\n", fn_name[f], n_failed);
    c8.print("generated N=NN vs MkFitCore");
    c1.print("generated N=1  vs MkFitCore");
    if (strict && (c8.n_diff || c8.n_int_diff || c1.n_diff || c1.n_int_diff))
      status = 1;
  }
  if (dump) {
    fclose(dump);
    printf("dumped to %s\n", dump_file.c_str());
  }
  return status;
}
