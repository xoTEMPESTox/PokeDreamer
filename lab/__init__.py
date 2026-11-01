"""PokeDreamer lab code.

Windows + this conda env: numpy is conda-forge (MKL in <env>/Library/bin) and torch bundles Intel OpenMP.
Run without `conda activate`, numpy.linalg cannot find its BLAS DLL (crash 0xc06d007f, e.g. inside matplotlib);
with it on PATH, MKL's default threading layer loads LLVM libomp.dll next to torch's libiomp5md.dll (OMP Error #15).
So: put Library/bin on PATH and keep numpy's MKL single-threaded. Must run before numpy's first linalg call.
"""
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ["PATH"] = os.path.join(sys.prefix, "Library", "bin") + os.pathsep + os.environ.get("PATH", "")
