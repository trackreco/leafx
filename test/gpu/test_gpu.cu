// leafx GPU test: the leafx device copy (configs/mkfitcore-device.toml) run on the GPU, one track per
// thread (N = 1), and on the host, against the dump written by test_leaf (dump=FILE): the outputs of the
// host copy at N = 1, which test_leaf has compared with MkFitCore.
//
//   test_gpu DUMP_FILE

#include "leafx_all.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace P = mkfit::portable;

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                           \
    }                                                                                    \
  } while (0)

struct In1 {
  P::MPlexLS<1> err;
  P::MPlexLV<1> par;
  P::MPlexQI<1> chg;
  P::MPlexQF<1> rad;
  P::MPlexHV<1> pnt, nrm, dir;
  P::MPlexHS<1> mserr;
  P::MPlexHV<1> mspar;
};
struct Out1 {
  P::MPlexLS<1> err;
  P::MPlexLV<1> par;
  P::MPlexQI<1> fail, chg;
  P::MPlexQF<1> chi2, x1, x2;
};

template <class M>
bool rd(FILE* f, M& m) {
  return fread(m.fArray, sizeof(m.fArray[0]), M::kTotSize, f) == size_t(M::kTotSize);
}
bool read_in(FILE* f, In1& i) {
  return rd(f, i.err) && rd(f, i.par) && rd(f, i.chg) && rd(f, i.rad) && rd(f, i.pnt) && rd(f, i.nrm) &&
         rd(f, i.dir) && rd(f, i.mserr) && rd(f, i.mspar);
}
bool read_out(FILE* f, Out1& o) {
  return rd(f, o.err) && rd(f, o.par) && rd(f, o.fail) && rd(f, o.chg) && rd(f, o.chi2) && rd(f, o.x1) &&
         rd(f, o.x2);
}

// The functions under test, in test_leaf's order.
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

MKFIT_HOST_DEVICE void run1(int f, const In1& in, Out1& o, const P::PropagationFlags& pf) {
  const int kfo = P::KFO_Update_Params | P::KFO_Calculate_Chi2 | P::KFO_Local_Cov;
  o.err.setVal(0.f);
  o.par.setVal(0.f);
  o.fail.setVal(0);
  o.chg = in.chg;
  o.chi2.setVal(0.f);
  o.x1.setVal(0.f);
  o.x2.setVal(0.f);
  switch (f) {
    case F_R:
      P::propagateHelixToRMPlex<1>(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, 1, pf);
      break;
    case F_Z:
      P::propagateHelixToZMPlex<1>(in.err, in.par, in.chg, in.rad, o.err, o.par, o.fail, 1, pf);
      break;
    case F_Plane:
      P::propagateHelixToPlaneMPlex<1>(in.err, in.par, in.chg, in.pnt, in.nrm, nullptr, o.err, o.par, o.fail, 1, pf);
      break;
    case F_Line:
      P::propagateLineToRMPlex<1>(in.err, in.par, in.mserr, in.mspar, o.err, o.par, 1);
      break;
    case F_KUpd:
      P::kalmanUpdate<1>(in.err, in.par, in.mserr, in.mspar, o.err, o.par, 1);
      break;
    case F_KChi2:
      P::kalmanComputeChi2<1>(in.err, in.par, in.chg, in.mserr, in.mspar, o.chi2, 1);
      break;
    case F_KUpdEndcap:
      P::kalmanUpdateEndcap<1>(in.err, in.par, in.mserr, in.mspar, o.err, o.par, 1);
      break;
    case F_KChi2Endcap:
      P::kalmanComputeChi2Endcap<1>(in.err, in.par, in.chg, in.mserr, in.mspar, o.chi2, 1);
      break;
    case F_KUpdPlane:
      P::kalmanUpdatePlane<1>(in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt, o.err, o.par, 1);
      break;
    case F_KPropUpdChi2Plane:
      P::kalmanPropagateAndUpdateAndChi2Plane<1>(in.err, in.par, o.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt,
                                                 o.err, o.par, o.fail, o.chi2, 1, pf, true);
      break;
    case F_KPropChi2Plane:
      P::kalmanPropagateAndComputeChi2Plane<1>(in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt,
                                               o.chi2, o.par, o.fail, 1, pf, true);
      break;
    case F_KOpPlaneLocal:
      P::kalmanOperationPlaneLocal<1>(kfo, in.err, in.par, in.chg, in.mserr, in.mspar, in.nrm, in.dir, in.pnt, o.err,
                                      o.par, o.chi2, 1, nullptr, nullptr, true, &o.x1, &o.x2, 4.0f);
      break;
  }
}

__global__ void kernel(int f, const In1* in, Out1* out, int n, P::PropagationFlags pf) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    run1(f, in[i], out[i], pf);
}

struct Cmp {
  long n_float = 0, n_diff = 0, n_tracks_diff = 0, n_int_diff = 0;
  double max_par = 0, max_cov = 0;
  long n_trk_cov[3] = {0, 0, 0};
  void add(const Out1& a, const Out1& b) {
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
    auto rel = [&](float x, float y) { cmp(x, y, std::max(std::abs(double(x)), 1e-3), max_par); };
    for (int i = 0; i < 6; ++i)
      rel(a.par.constAt(0, i, 0), b.par.constAt(0, i, 0));
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        cmp(a.err.constAt(0, i, j), b.err.constAt(0, i, j),
            std::sqrt(std::abs(double(a.err.constAt(0, i, i)) * a.err.constAt(0, j, j))), trk_cov);
    rel(a.chi2[0], b.chi2[0]);
    rel(a.x1[0], b.x1[0]);
    rel(a.x2[0], b.x2[0]);
    if (a.fail[0] != b.fail[0] || a.chg[0] != b.chg[0]) {
      ++n_int_diff;
      any = true;
    }
    max_cov = std::max(max_cov, trk_cov);
    for (int k = 0; k < 3; ++k)
      n_trk_cov[k] += trk_cov > std::pow(10., -6 + 2 * k);
    n_tracks_diff += any;
  }
  void print(const char* what, long n_tracks) const {
    printf("  %-24s tracks %6ld, differing %6ld | floats differing %6ld of %6ld | max: par %.1e, cov %.1e | int %ld"
           " | cov > 1e-6: %ld, > 1e-4: %ld, > 1e-2: %ld\n",
           what, n_tracks, n_tracks_diff, n_diff, n_float, max_par, max_cov, n_int_diff, n_trk_cov[0],
           n_trk_cov[1], n_trk_cov[2]);
  }
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: test_gpu DUMP_FILE\n");
    return 2;
  }
  FILE* f = fopen(argv[1], "rb");
  if (!f) {
    perror(argv[1]);
    return 2;
  }
  int head[2];
  float cfgv[4];
  fread(head, sizeof(int), 2, f);
  fread(cfgv, sizeof(float), 4, f);
  if (head[0] != 0x6c656166 || head[1] != F_Count) {
    fprintf(stderr, "not a test_leaf dump with %d functions\n", F_Count);
    return 2;
  }
  // the prelude fixes these at compile time; they must be what the host test ran with
  const float expect[4] = {float(P::Config::usePtMultScat), P::Config::matScale, P::Config::matElossVarScale, 1.f};
  for (int i = 0; i < 4; ++i)
    if (cfgv[i] != expect[i]) {
      fprintf(stderr, "Config mismatch, item %d: host test %g, device prelude %g\n", i, cfgv[i], expect[i]);
      return 2;
    }

  P::TrackerInfo ti_h;
  float fac[2];
  fread(&ti_h.nz, sizeof(int), 1, f);
  fread(&ti_h.nr, sizeof(int), 1, f);
  fread(fac, sizeof(float), 2, f);
  ti_h.fac_z = fac[0];
  ti_h.fac_r = fac[1];
  std::vector<P::TrackerInfo::Material> mat(size_t(ti_h.nz) * ti_h.nr);
  fread(mat.data(), sizeof(P::TrackerInfo::Material), mat.size(), f);
  ti_h.mat = mat.data();

  // the same material map in device memory
  P::TrackerInfo ti_d = ti_h;
  P::TrackerInfo::Material* mat_d;
  CUDA_CHECK(cudaMalloc(&mat_d, mat.size() * sizeof(mat[0])));
  CUDA_CHECK(cudaMemcpy(mat_d, mat.data(), mat.size() * sizeof(mat[0]), cudaMemcpyHostToDevice));
  ti_d.mat = mat_d;
  P::TrackerInfo* ti_dp;
  CUDA_CHECK(cudaMalloc(&ti_dp, sizeof(ti_d)));
  CUDA_CHECK(cudaMemcpy(ti_dp, &ti_d, sizeof(ti_d), cudaMemcpyHostToDevice));

  P::PropagationFlags pf_h, pf_d;
  pf_h.use_param_b_field = pf_d.use_param_b_field = true;
  pf_h.apply_material = pf_d.apply_material = true;
  pf_h.tracker_info = &ti_h;
  pf_d.tracker_info = ti_dp;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s, sm_%d%d; material bins %d x %d\n", prop.name, prop.major, prop.minor, ti_h.nz, ti_h.nr);

  for (int fn = 0; fn < F_Count; ++fn) {
    int nt[2];
    fread(nt, sizeof(int), 2, f);
    const int n = nt[1];
    std::vector<In1> in(n);
    std::vector<Out1> ref(n), host(n), dev(n);
    for (int t = 0; t < n; ++t)
      if (!read_in(f, in[t]) || !read_out(f, ref[t])) {
        fprintf(stderr, "short dump\n");
        return 2;
      }
    for (int t = 0; t < n; ++t)
      run1(fn, in[t], host[t], pf_h);

    In1* in_d;
    Out1* out_d;
    CUDA_CHECK(cudaMalloc(&in_d, n * sizeof(In1)));
    CUDA_CHECK(cudaMalloc(&out_d, n * sizeof(Out1)));
    CUDA_CHECK(cudaMemcpy(in_d, in.data(), n * sizeof(In1), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kernel<<<(n + 127) / 128, 128>>>(fn, in_d, out_d, n, pf_d);
    cudaEventRecord(e1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    CUDA_CHECK(cudaMemcpy(dev.data(), out_d, n * sizeof(Out1), cudaMemcpyDeviceToHost));
    cudaFree(in_d);
    cudaFree(out_d);

    Cmp c_host, c_dev, c_dh;
    for (int t = 0; t < n; ++t) {
      c_host.add(ref[t], host[t]);
      c_dev.add(ref[t], dev[t]);
      c_dh.add(host[t], dev[t]);
    }
    if (getenv("LEAFX_GPU_WORST")) {  // the track with the largest covariance difference
      int tw = -1;
      double w = 0;
      for (int t = 0; t < n; ++t)
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j <= i; ++j) {
            const double sc = std::sqrt(std::abs(double(host[t].err.constAt(0, i, i)) * host[t].err.constAt(0, j, j)));
            const double d = std::abs(double(host[t].err.constAt(0, i, j)) - dev[t].err.constAt(0, i, j));
            if (sc > 0 && d / sc > w) {
              w = d / sc;
              tw = t;
            }
          }
      int n_big = 0, n_big_bad = 0;  // tracks above 1e-2; of those, with a negative variance on the host
      for (int t = 0; t < n; ++t) {
        double wt = 0;
        bool bad = false;
        for (int i = 0; i < 6; ++i) {
          bad |= host[t].err.constAt(0, i, i) < 0;
          for (int j = 0; j <= i; ++j) {
            const double sc = std::sqrt(std::abs(double(host[t].err.constAt(0, i, i)) * host[t].err.constAt(0, j, j)));
            if (sc > 0)
              wt = std::max(wt, std::abs(double(host[t].err.constAt(0, i, j)) - dev[t].err.constAt(0, i, j)) / sc);
          }
        }
        if (wt > 1e-2) {
          ++n_big;
          n_big_bad += bad;
        }
      }
      printf("  tracks with cov difference > 1e-2: %d, of which with a negative variance on the host: %d\n", n_big,
             n_big_bad);
      if (tw >= 0) {
        const auto& h = host[tw];
        const auto& d = dev[tw];
        const float dx = in[tw].mspar[0] - h.par[0], dy = in[tw].mspar[1] - h.par[1], dz = in[tw].mspar[2] - h.par[2];
        printf("  worst track %d (%.2g): chi2 host %g dev %g; 1/pT %g; hit-to-updated-state %g cm\n", tw, w, h.chi2[0],
               d.chi2[0], in[tw].par[3], std::sqrt(dx * dx + dy * dy + dz * dz));
        for (int i = 0; i < 6; ++i)
          printf("    err[%d][%d] host %-12g dev %-12g\n", i, i, h.err.constAt(0, i, i), d.err.constAt(0, i, i));
      }
    }
    if (getenv("LEAFX_GPU_DEBUG"))
      for (int t = 0; t < n; ++t)
        if (std::memcmp(host[t].par.fArray, dev[t].par.fArray, sizeof(host[t].par.fArray)) ||
            std::memcmp(host[t].err.fArray, dev[t].err.fArray, sizeof(host[t].err.fArray))) {
          printf("  first differing track %d: fail host %d dev %d\n", t, host[t].fail[0], dev[t].fail[0]);
          for (int i = 0; i < 6; ++i)
            printf("    par[%d] in %-13g host %-13g dev %-13g | err[%d][%d] in %-11g host %-11g dev %g\n", i,
                   in[t].par.constAt(0, i, 0), host[t].par.constAt(0, i, 0), dev[t].par.constAt(0, i, 0), i, i,
                   in[t].err.constAt(0, i, i), host[t].err.constAt(0, i, i), dev[t].err.constAt(0, i, i));
          break;
        }
    printf("%s (%d tracks, kernel %.3f ms)\n", fn_name[fn], n, ms);
    c_host.print("nvcc host vs gcc host", n);
    c_dev.print("GPU vs gcc host", n);
    c_dh.print("GPU vs nvcc host", n);
  }
  fclose(f);
  return 0;
}
