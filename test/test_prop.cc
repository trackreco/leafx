// leafx functional test: MkFitCore's propagation (libMkFitCore, NN lanes) against the
// leafx-generated copy instantiated with N = NN and with N = 1, on random tracks, with the
// parametrised field and the phase-2 material map. Compares every output float bitwise.

#include "RecoTracker/MkFitCore/src/PropagationMPlex.h"  // reference: MkFitCore, NN lanes
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
// One batch of NN inputs, and the outputs of the three implementations

struct In {
  MPlexLS err;
  MPlexLV par;
  MPlexQI chg;
  MPlexQF rad;  // target r or z
  MPlexHV pnt, nrm;
  MPlexHS mserr;  // propagateLineToRMPlex
  MPlexHV mspar;
};

struct Out {
  MPlexLS err;
  MPlexLV par;
  MPlexQI fail;
  void zero() {
    err.setVal(0.f);
    par.setVal(0.f);
    fail.setVal(0);
  }
};

// lane copy NN <-> 1
template <class Big, class Small>
void lane_out(const Big& b, int n, Small& s) {
  typename Big::value_type tmp[Big::kSize];
  b.copyOut(n, tmp);
  s.copyIn(0, tmp);
}
template <class Big, class Small>
void lane_in(Big& b, int n, const Small& s) {
  typename Big::value_type tmp[Big::kSize];
  s.copyOut(0, tmp);
  b.copyIn(n, tmp);
}

// ---------------------------------------------------------------------------
// Comparison

struct Cmp {
  long n_float = 0, n_diff = 0, n_tracks = 0, n_tracks_diff = 0, n_fail_diff = 0;
  double max_rel = 0;    // parameters: |dx| / max(|x|, 1e-3)
  double max_relcov = 0;  // covariance: |dC_ij| / sqrt(C_ii C_jj)
  long n_trk_cov[3] = {0, 0, 0};  // tracks whose worst covariance difference is > 1e-6, 1e-4, 1e-2
  void add(const Out& a, const Out& b, int n_proc) {
    for (int n = 0; n < n_proc; ++n) {
      bool any = false;
      double trk_cov = 0, trk_par = 0;
      auto cmp = [&](float x, float y, double scale, double& mx) {
        ++n_float;
        if (std::memcmp(&x, &y, sizeof(float)) == 0)
          return;
        if (std::isnan(x) && std::isnan(y))
          return;
        ++n_diff;
        any = true;
        double d = std::abs(double(x) - double(y));
        if (scale > 0 && d / scale > mx)
          mx = d / scale;
      };
      for (int i = 0; i < 6; ++i)
        cmp(a.par.constAt(n, i, 0), b.par.constAt(n, i, 0), std::max(std::abs(double(a.par.constAt(n, i, 0))), 1e-3),
            max_rel);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          cmp(a.err.constAt(n, i, j), b.err.constAt(n, i, j),
              std::sqrt(std::abs(double(a.err.constAt(n, i, i)) * a.err.constAt(n, j, j))), trk_cov);
      max_relcov = std::max(max_relcov, trk_cov);
      for (int k = 0; k < 3; ++k)
        n_trk_cov[k] += trk_cov > std::pow(10., -6 + 2 * k);
      (void)trk_par;
      if (a.fail.constAt(n, 0, 0) != b.fail.constAt(n, 0, 0)) {
        ++n_fail_diff;
        any = true;
      }
      ++n_tracks;
      n_tracks_diff += any;
    }
  }
  void print(const char* what) const {
    printf("  %-34s tracks %6ld, differing %6ld | floats differing %6ld of %6ld | max: par %.1e, cov %.1e | fail flag %ld\n",
           what, n_tracks, n_tracks_diff, n_diff, n_float, max_rel, max_relcov, n_fail_diff);
    printf("  %-34s tracks with cov difference > 1e-6: %ld, > 1e-4: %ld, > 1e-2: %ld\n", "", n_trk_cov[0],
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

  void fill(In& in, int mode) {
    for (int n = 0; n < NN; ++n) {
      const float x = g(0.01f), y = g(0.01f), z = u(-10.f, 10.f);
      const float pt = std::exp(u(std::log(0.5f), std::log(50.f)));
      const float phi = u(-3.14159f, 3.14159f);
      const float eta = (mode == 2) ? (u(0.f, 1.f) < 0.5f ? -1.f : 1.f) * u(1.2f, 2.8f) : u(-2.5f, 2.5f);
      const float theta = 2.f * std::atan(std::exp(-eta));
      const float par[6] = {x, y, z, 1.f / pt, phi, theta};
      for (int i = 0; i < 6; ++i)
        in.par(n, i, 0) = par[i];
      in.chg(n, 0, 0) = u(0.f, 1.f) < 0.5f ? -1 : 1;
      // covariance: diagonal + a few correlations, positive definite
      float d[6] = {g(0.01f) * g(0.01f) + 1e-4f, 1e-4f + 1e-5f * u(0, 1), 1e-3f + 1e-3f * u(0, 1),
                    std::pow(0.02f / pt, 2.f) * u(0.5f, 1.5f), 1e-6f * u(0.5f, 2.f), 1e-6f * u(0.5f, 2.f)};
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          in.err(n, i, j) = (i == j) ? d[i] : 0.f;
      in.err(n, 4, 0) = 0.2f * std::sqrt(d[4] * d[0]);
      in.err(n, 4, 3) = 0.3f * std::sqrt(d[4] * d[3]);
      in.err(n, 5, 2) = 0.2f * std::sqrt(d[5] * d[2]);
      // targets
      if (mode == 2) {
        in.rad(n, 0, 0) = z + (eta > 0 ? 1.f : -1.f) * u(5.f, 120.f);  // z plane
      } else {
        in.rad(n, 0, 0) = u(3.f, 110.f);  // radius
      }
      const float dist = u(3.f, 60.f);
      const float dir[3] = {std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta)};
      const float p[3] = {x + dist * dir[0], y + dist * dir[1], z + dist * dir[2]};
      const float r = std::hypot(p[0], p[1]), tau = u(-1.f, 1.f);
      for (int i = 0; i < 3; ++i)
        in.pnt(n, i, 0) = p[i];
      in.nrm(n, 0, 0) = std::cos(tau) * p[0] / r;
      in.nrm(n, 1, 0) = std::cos(tau) * p[1] / r;
      in.nrm(n, 2, 0) = std::sin(tau);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j)
          in.mserr(n, i, j) = (i == j) ? 1e-4f : 0.f;
      for (int i = 0; i < 3; ++i)
        in.mspar(n, i, 0) = p[i];
    }
  }
};

// ---------------------------------------------------------------------------
// The functions under test, each in three forms

enum Fn { F_R, F_Z, F_Plane, F_Line, F_Count };
const char* fn_name[F_Count] = {"propagateHelixToRMPlex", "propagateHelixToZMPlex", "propagateHelixToPlaneMPlex",
                                "propagateLineToRMPlex"};

void run_lib(Fn f, const In& in, Out& o, int np, const PropagationFlags& pf) {
  switch (f) {
    case F_R:
      mkfit::propagateHelixToRMPlex(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Z:
      mkfit::propagateHelixToZMPlex(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Plane:
      mkfit::propagateHelixToPlaneMPlex(in.err, in.par, in.chg, in.pnt, in.nrm, nullptr, o.err, o.par, o.fail, np, pf);
      break;
    case F_Line:
      mkfit::propagateLineToRMPlex(in.err, in.par, in.mserr, in.mspar, o.err, o.par, np);
      break;
    default:
      break;
  }
}

template <int N, class I, class O>
void run_gen(Fn f, const I& in, O& o, int np, const PropagationFlags& pf) {
  switch (f) {
    case F_R:
      P::propagateHelixToRMPlex<N>(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Z:
      P::propagateHelixToZMPlex<N>(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, np, pf);
      break;
    case F_Plane:
      P::propagateHelixToPlaneMPlex<N>(in.err, in.par, in.chg, in.pnt, in.nrm, nullptr, o.err, o.par, o.fail, np, pf);
      break;
    case F_Line:
      P::propagateLineToRMPlex<N>(in.err, in.par, in.mserr, in.mspar, o.err, o.par, np);
      break;
    default:
      break;
  }
}

struct In1 {
  P::MPlexLS<1> err;
  P::MPlexLV<1> par;
  P::MPlexQI<1> chg;
  P::MPlexQF<1> rad;
  P::MPlexHV<1> pnt, nrm;
  P::MPlexHS<1> mserr;
  P::MPlexHV<1> mspar;
};
struct Out1 {
  P::MPlexLS<1> err;
  P::MPlexLV<1> par;
  P::MPlexQI<1> fail;
};

int main(int argc, char* argv[]) {
  const std::string geom = argc > 1 ? argv[1] : "CMS-phase2.bin";
  const int n_batches = argc > 2 ? std::atoi(argv[2]) : 2000;
  // strict: fail on any difference (only meaningful when the reference and the copy are compiled
  // without fast math); otherwise only report.
  const bool strict = argc > 3 && std::string(argv[3]) == "strict";

  TrackerInfo ti;
  ti.read_bin_file(geom);
  Config::usePtMultScat = true;  // as the phase-2 geometry plugin sets it
  PropagationFlags pf(PF_use_param_b_field | PF_apply_material);
  pf.tracker_info = &ti;
  printf("geometry %s, material bins %d x %d; %d batches of %d tracks per function\n", geom.c_str(),
         ti.mat_nbins_z(), ti.mat_nbins_r(), n_batches, NN);

  int status = 0;
  for (int f = 0; f < F_Count; ++f) {
    Gen gen(12345 + f);
    Cmp c8, c1, c18;
    long n_failed = 0;
    for (int b = 0; b < n_batches; ++b) {
      In in;
      gen.fill(in, f == F_Z ? 2 : 1);
      const int np = (b % 7 == 6) ? 5 : NN;  // some partial batches
      Out lib, g8, g1;
      lib.zero();
      g8.zero();
      g1.zero();
      run_lib(Fn(f), in, lib, np, pf);
      run_gen<NN>(Fn(f), in, g8, np, pf);
      for (int n = 0; n < np; ++n) {
        In1 i1;
        Out1 o1;
        lane_out(in.err, n, i1.err);
        lane_out(in.par, n, i1.par);
        lane_out(in.chg, n, i1.chg);
        lane_out(in.rad, n, i1.rad);
        lane_out(in.pnt, n, i1.pnt);
        lane_out(in.nrm, n, i1.nrm);
        lane_out(in.mserr, n, i1.mserr);
        lane_out(in.mspar, n, i1.mspar);
        o1.err.setVal(0.f);
        o1.par.setVal(0.f);
        o1.fail.setVal(0);
        run_gen<1>(Fn(f), i1, o1, 1, pf);
        lane_in(g1.err, n, o1.err);
        lane_in(g1.par, n, o1.par);
        lane_in(g1.fail, n, o1.fail);
      }
      for (int n = 0; n < np; ++n)
        n_failed += lib.fail.constAt(n, 0, 0) != 0;
      c8.add(lib, g8, np);
      c1.add(lib, g1, np);
      c18.add(g8, g1, np);
    }
    printf("%s (propagation failed for %ld tracks)\n", fn_name[f], n_failed);
    c8.print("generated N=NN  vs MkFitCore");
    c1.print("generated N=1   vs MkFitCore");
    c18.print("generated N=1   vs generated N=NN");
    if (strict && (c8.n_diff || c8.n_fail_diff || c1.n_diff || c1.n_fail_diff))
      status = 1;
  }
  return status;
}
