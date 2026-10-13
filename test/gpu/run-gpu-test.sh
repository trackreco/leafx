#!/bin/bash
# GPU test: generate the device copy, compile test_gpu.cu with nvcc, and run it on a dump written by
# test_leaf (test/run-test.sh with DUMP=FILE).
#
# Environment:
#   CMSSW_SRC        the src/ tree with RecoTracker/MkFitCore (required)
#   LEAFX_VDT_ROOT   the directory that contains vdt/ (required)
#   DUMP             the dump file (required)
#   NVCC             nvcc (default: from PATH); NVCC_FLAGS extra flags (e.g. --fmad=false)
#   CUDA_ARCH        default sm_89
set -e
HERE=$(cd $(dirname $0)/../.. && pwd)
: ${CMSSW_SRC:?} ${LEAFX_VDT_ROOT:?} ${DUMP:?}
GEN=${GEN:-$HERE/test/out-device}
BIN=$HERE/test/bin
mkdir -p $BIN
# SKIP_GEN=1: the copy in $GEN was generated already (e.g. outside a container whose Python is < 3.11)
[ -n "$SKIP_GEN" ] || python3 $HERE/leafx.py $HERE/configs/mkfitcore-device.toml -o $GEN --source-root $CMSSW_SRC
# the generated copy comes first on the include path: its vdt/ and Matriplex/ shadow the originals
${NVCC:-nvcc} -std=c++20 -O2 -arch=${CUDA_ARCH:-sm_89} --expt-relaxed-constexpr \
  -Xcudafe --diag_suppress=unrecognized_pragma -Xcompiler -fno-fast-math ${NVCC_FLAGS} \
  -I$GEN -I$HERE/test/gpu -I$CMSSW_SRC \
  -o $BIN/test_gpu $HERE/test/gpu/test_gpu.cu
$BIN/test_gpu $DUMP
