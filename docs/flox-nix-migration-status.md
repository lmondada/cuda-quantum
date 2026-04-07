# Flox/Nix Dependency Migration — Handoff Document

## Overview

This document captures the status of an ongoing migration of cuda-quantum's build
dependencies from bash install scripts (`scripts/`) to a reproducible
[Flox](https://flox.dev) environment. The goal is that `flox activate` provides
a fully configured shell where `cmake -B build && ninja -C build` just works.

---

## Platform & Environment

- **OS**: Linux x86_64 (WSL2 on Windows — `Linux 6.6.87.2-microsoft-standard-WSL2`)
- **Flox version**: 1.10.0
- **Nix version**: 2.31.2+1
- **Branch**: `lm/flox`
- **Key constraint**: nixpkgs-unstable defaults to GCC 15, which **cannot build LLVM 16**.
  The LLVM flake therefore pins to an older nixpkgs commit (`e6f23dc`) that provides
  `llvmPackages_16` with GCC 13.

---

## What Has Been Done

### 1. Flox environment bootstrapped — ✅ DONE

File: `.flox/env/manifest.toml`

The following nixpkgs catalog packages are installed and verified working:

| Package | Notes |
|---|---|
| `cmake` | v4.1.2 |
| `ninja` | v1.13.2 |
| `gcc` | v15.2.0 |
| `git`, `pkg-config`, `python3`, `doxygen` | All working |
| `zlib` | With `dev` + `static` outputs (headers present) |
| `openssl` | With `dev` output (headers present) |
| `curl` | With `dev` output (headers present) |
| `openblas` | With `dev` output |

The activation hook sets all `*_INSTALL_PREFIX` env vars to `$FLOX_ENV` (the merged
symlink tree), so `CMakeLists.txt`'s custom lookup logic finds them automatically.

**BLAS note**: The original scripts build a custom BLAS from Fortran source that
produces `libblas.a`. OpenBLAS provides only `libblas.so`. The hook sets
`BLAS_LIBRARIES=$FLOX_ENV/lib/libblas.so` directly to avoid the static-lib lookup
in `CMakeLists.txt:191`.

### 2. NVIDIA GPU libraries (cuQuantum, cuTensor) — ✅ DONE (catalog packages)

These were initially attempted as custom flakes (downloading NVIDIA tarballs), but
it turned out the Flox catalog already has suitable packages:

- `cuquantum`: `flox-cuda/cudaPackages.cuquantum` at **25.09.0.7**
  (scripts use 26.01.0.4 — minor version difference; `CUSTATEVEC_ROOT` etc. just
  need the directory, no strict version check in CMakeLists.txt)
- `cutensor`: `flox-cuda/cudaPackages.libcutensor` at **2.3.1.0** (exact match)

Both are restricted to `systems = ["x86_64-linux", "aarch64-linux"]`.
`allow.unfree = true` is set in `[options]`.

### 3. AWS SDK C++ — ✅ DONE (catalog package)

`aws-sdk-cpp` is available directly from the nixpkgs catalog. The scripts build
only `braket;s3-crt;sts` components, but the full nixpkgs package is fine since
CMake only links what it needs.

### 4. QRMI — ✅ DONE (custom flake)

File: `.flox/flakes/qrmi/flake.nix`

Pre-built binary from GitHub releases. Uses `autoPatchelfHook` to fix RPATHs.
Restricted to `x86_64-linux` (upstream only ships this architecture).

- URL: `https://github.com/qiskit-community/qrmi/releases/download/v0.12.0/libqrmi-0.12.0-el8-x86_64.tar.gz`
- Hash: `sha256-KYYVDU9V4fZWa+8W2fs4l8oE3X6qaBhl9+8kTymKZ0Y=`

### 5. LLVM/MLIR/Clang — ⚠️ IN PROGRESS (custom flake, not yet verified to build)

File: `.flox/flakes/llvm-cudaq/flake.nix`

This is the most complex dependency. The approach (revised by the user) is:

1. **Pin nixpkgs** to `e6f23dc08d3624daab7094b701aa3954923c6bbb` — a commit that
   still has `llvmPackages_16` (nixpkgs-unstable has dropped it).
2. **Fetch the exact LLVM commit** from the `tpls/llvm` submodule
   (`7cbf1a2591520c2491aa35339f227775f4d3adf6`) as a flake input.
3. **Apply 5 patches** from `.flox/flakes/llvm-cudaq/patches/`:
   - `CompilerRTUtils.cmake.diff` — fixes compiler-rt targeting when built as runtime
   - `fix_region_simplification.diff` — MLIR region simplification fix
   - `idempotent_option_category.diff` — option category idempotency fix
   - `llvm_pr71968_mod.diff` — upstream PR backport
   - `mlir_python_sources_install.diff` — adds `mlir-python-sources` distribution
     target missing in LLVM 16 (cherry-pick of upstream commit `9494bd84df3c`)
4. **Use `llvmPackages_16.override { monorepoSrc = patchedMonorepo; }`** — this
   reuses nixpkgs' existing LLVM 16 derivations (with GCC 13 stdenv) rather than
   building from scratch.
5. **Output a `symlinkJoin`** merging `llvm`, `clang-unwrapped`, `mlir`, `lld`,
   `openmp`, `compiler-rt`, `libcxx`, `libunwind` into one prefix — matching the
   single-directory layout `CMakeLists.txt` expects via `LLVM_INSTALL_PREFIX`.

**Current status**: The flake has been rewritten to use `llvmPackages_16.override`
(user's revision). Nix dry-run confirms it would build these derivations:
- `llvm-16.0.6.drv`
- `mlir-16.0.6.drv`
- `clang-at-least-16-LLVMgold-path.patch.drv`
- `clang-16.0.6.drv`

A full build has **not been attempted yet** with this new approach. The previous
`stdenv.mkDerivation`-from-scratch approach hit several issues (Perl, NumPy, build
directory confusion) and was replaced by the user.

**To test**: Run:
```bash
nix --extra-experimental-features 'nix-command flakes' build \
  'path:/path/to/cuda-quantum/.flox/flakes/llvm-cudaq' --no-link
```

### 6. MLIR Python bindings — ⚠️ IN PROGRESS (custom flake, not yet tested)

File: `.flox/flakes/mlir-python-cudaq/flake.nix`

A separate flake that builds MLIR with Python bindings enabled, for multiple Python
versions (3.10, 3.11, 3.12, 3.13). It imports `llvm-cudaq` as an input, reuses the
patched monorepo source, and overrides `llvm16.mlir` with `MLIR_ENABLE_BINDINGS_PYTHON=ON`.

Not yet tested.

---

## File Layout

```
.flox/
  env/
    manifest.toml          ← main flox config (packages + hook)
  flakes/
    llvm-cudaq/
      flake.nix            ← LLVM 16 with cuda-quantum patches
      flake.lock           ← pinned: nixpkgs e6f23dc, llvm 7cbf1a2
      patches/
        CompilerRTUtils.cmake.diff
        fix_region_simplification.diff
        idempotent_option_category.diff
        llvm_pr71968_mod.diff
        mlir_python_sources_install.diff   ← extracted from upstream commit 9494bd84
    mlir-python-cudaq/
      flake.nix            ← MLIR Python bindings, multiple Python versions
    qrmi/
      flake.nix            ← pre-built QRMI binary, x86_64-linux only
```

The custom `cuquantum` and `cutensor` flakes that existed earlier in the session
were **replaced** by catalog packages (`flox-cuda/cudaPackages.*`) and deleted.

The custom `aws-sdk-cudaq` flake was also replaced by the catalog `aws-sdk-cpp`.

---

## What Still Needs to Be Done

### Priority 1: Verify LLVM flake builds

```bash
cd /path/to/cuda-quantum
nix --extra-experimental-features 'nix-command flakes' build \
  'path:.flox/flakes/llvm-cudaq' --no-link
```

Expected components to appear in the output:
- `bin/llvm-config`, `bin/clang`, `bin/clang++`, `bin/mlir-opt`, `bin/lld`
- `lib/libMLIR*.a` (static MLIR libs)
- `lib/cmake/llvm/`, `lib/cmake/mlir/`, `lib/cmake/clang/` (cmake configs)

If it fails, check for:
- Missing nixpkgs patches for LLVM 16 at the pinned commit
- pybind11 or numpy version incompatibilities with the pinned nixpkgs

### Priority 2: Verify the manifest activates cleanly

```bash
FLOX_DISABLE_METRICS=true flox activate -- bash -c '
  echo "cmake: $(cmake --version | head -1)"
  echo "LLVM_INSTALL_PREFIX=$LLVM_INSTALL_PREFIX"
  llvm-config --version
  mlir-opt --version
'
```

This requires the `llvm-cudaq` flake to be in the `[install]` section. It is
already uncommented in the manifest.

### Priority 3: Verify MLIR Python bindings flake

```bash
nix --extra-experimental-features 'nix-command flakes' build \
  'path:.flox/flakes/mlir-python-cudaq#python311' --no-link
```

### Priority 4: Test end-to-end CMake configure

```bash
FLOX_DISABLE_METRICS=true flox activate -- bash -c '
  cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAQ_ENABLE_PYTHON=ON \
    -DCUDAQ_BUILD_TESTS=ON \
    -DCUDAQ_DISABLE_CPP_FRONTEND=OFF
'
```

Watch for: CMake failing to find LLVM/MLIR (`LLVM_DIR` not found), BLAS issues,
OpenSSL/curl cmake config path issues (curl's `CURLConfig.cmake` may not be in
the standard location — may need `-DCURL_ROOT=$FLOX_ENV`).

### Priority 5: Full build

```bash
ninja -C build && ctest --test-dir build
```

---

## Known Issues & Decisions Made

| Issue | Resolution |
|---|---|
| GCC 15 (nixpkgs-unstable) cannot build LLVM 16 | LLVM flake pins nixpkgs to `e6f23dc` (has llvmPackages_16 with GCC 13) |
| `mlir-python-sources` distribution target missing in LLVM 16 | Added `mlir_python_sources_install.diff` patch (extracted from upstream commit `9494bd84`) |
| OpenBLAS provides `libblas.so`, not `libblas.a` | Set `BLAS_LIBRARIES` directly in hook; skip `BLAS_INSTALL_PREFIX` lookup |
| cuQuantum needs `libcutensor.so.2` + `libnvidia-ml.so.1` | Use `flox-cuda/cudaPackages.cuquantum` which handles deps; `libnvidia-ml` is driver-only, runtime dep |
| `CURLConfig.cmake` not in `$FLOX_ENV/lib/cmake/CURL/` | CMakeLists.txt check will fail; workaround: pass `-DCURL_ROOT=$FLOX_ENV` at cmake time |
| pybind11 has custom patches in `tpls/customizations/pybind11/` | Left as submodule for now; CMake uses it via `add_subdirectory` |

---

## Dependency Classification Summary

| Dependency | Source | Status |
|---|---|---|
| cmake, ninja, gcc, git, python3, doxygen | nixpkgs catalog | ✅ working |
| zlib, openssl, curl | nixpkgs catalog (with dev outputs) | ✅ working |
| openblas (BLAS) | nixpkgs catalog | ✅ working (dynamic only) |
| cuquantum 25.09.0.7 | flox-cuda catalog | ✅ in manifest |
| libcutensor 2.3.1.0 | flox-cuda catalog | ✅ in manifest |
| aws-sdk-cpp | nixpkgs catalog | ✅ in manifest |
| QRMI 0.12.0 | custom flake (pre-built binary) | ✅ flake written, not yet activated |
| LLVM/MLIR/Clang 16 | custom flake (`llvmPackages_16.override`) | ⚠️ flake written, **not yet built** |
| MLIR Python bindings | custom flake (extends llvm-cudaq) | ⚠️ flake written, **not yet tested** |
| pybind11 | git submodule `tpls/pybind11` | ✅ handled by CMake |
| fmt, spdlog, eigen, etc. | git submodules in `tpls/` | ✅ handled by CMake |
