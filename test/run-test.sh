#!/bin/bash
# Functional test: MkFitCore's propagation and Kalman update against the leafx copy, at N = NN and at N = 1.
#
# Environment:
#   CMSSW_SRC    the src/ tree with RecoTracker/MkFitCore (required)
#   MKFIT_BUILD  a standalone mkFit build of that tree: libMkFitCore.so and friends (required)
#   ROOTSYS      the ROOT the standalone build used, if it was built with ROOT
#   GEOM         a phase-2 geometry file (default: $MKFIT_BUILD/CMS-phase2.bin)
#   NBATCH       batches of NN tracks per function (default 2000)
#   DUMP         write the dump for the GPU test (test/gpu/) to this file
#   MODE         strict:    -O2 -fno-fast-math; the reference is MkFitCore's propagation sources compiled
#                           into the test with the same flags, so any difference is the copy's.
#                ofast:     the build's own flags; the reference is libMkFitCore as built.
#                ofast-src: the build's flags; the reference compiled from source into the test.
#   MKFIT_DEFS, MKFIT_OPT  the build's -D flags and optimisation flags; by default from `make echo-aclic`.
#                The -D flags must be the build's: they change class layouts.
set -e
HERE=$(cd $(dirname $0)/.. && pwd)
: ${CMSSW_SRC:?set CMSSW_SRC to the src/ tree}
: ${MKFIT_BUILD:?set MKFIT_BUILD to the standalone build directory}
MODE=${MODE:-strict}
GEN=${GEN:-$HERE/test/out-host}
BIN=$HERE/test/bin

if [ -n "$ROOTSYS" ]; then
  export PATH=$ROOTSYS/bin:$PATH
  export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH=$MKFIT_BUILD:$LD_LIBRARY_PATH

if [ -z "$MKFIT_DEFS" -o -z "$MKFIT_OPT" ]; then
  ACLIC=$(make -s --no-print-directory -C $MKFIT_BUILD echo-aclic)
  MKFIT_DEFS=${MKFIT_DEFS:-$(echo "$ACLIC" | sed -n 's/^ACLIC_DEFS=//p')}
  MKFIT_OPT=${MKFIT_OPT:-$(echo "$ACLIC" | sed -n 's/^ACLIC_OPT=//p')}
fi
case $MODE in
  strict) OPT=$(echo $MKFIT_OPT | sed 's/-Ofast/-O2 -fno-fast-math/') ;;
  ofast | ofast-src) OPT=$MKFIT_OPT ;;
  *) echo "unknown MODE $MODE"; exit 1 ;;
esac

# generate the copy from the tree under test
python3 $HERE/leafx.py $HERE/configs/mkfitcore-host.toml -o $GEN --source-root $CMSSW_SRC

CPPFLAGS="-I$GEN -I$CMSSW_SRC -I$MKFIT_BUILD/mkFit-external $MKFIT_DEFS"
[ -n "$ROOTSYS" ] && CPPFLAGS="$CPPFLAGS -I$(root-config --incdir)"
CXXFLAGS="$OPT -std=c++20 -pthread -Wall -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-psabi -Wno-vla"

REFSRC=""
if [ $MODE != ofast ]; then
  for f in PropagationMPlex PropagationMPlexCommon PropagationMPlexEndcap PropagationMPlexPlane KalmanUtilsMPlex MiniPropagators; do
    REFSRC="$REFSRC $CMSSW_SRC/RecoTracker/MkFitCore/src/$f.cc"
  done
fi

LIBS="-L$MKFIT_BUILD -Wl,-rpath,$MKFIT_BUILD"
for l in MkFitCore MkFitCMS MkFitRootDataFormats; do
  [ -e $MKFIT_BUILD/lib$l.so ] && LIBS="$LIBS -l$l"
done
[ -n "$ROOTSYS" ] && LIBS="$LIBS $(root-config --libs) -lROOTDataFrame -lROOTEve"
LIBS="$LIBS -ltbb"

mkdir -p $BIN
echo "MODE=$MODE OPT=$OPT"
g++ $CPPFLAGS $CXXFLAGS -o $BIN/test_leaf_$MODE $HERE/test/test_leaf.cc $REFSRC $LIBS
$BIN/test_leaf_$MODE ${GEOM:-$MKFIT_BUILD/CMS-phase2.bin} ${NBATCH:-2000} $MODE ${DUMP:+dump=$DUMP}
