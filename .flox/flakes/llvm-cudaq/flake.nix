{
  description = "LLVM/MLIR/Clang build for CUDA Quantum with custom patches";

  inputs = {
    # Pin to nixpkgs that still has llvmPackages_16
    nixpkgs.url = "github:NixOS/nixpkgs/e6f23dc08d3624daab7094b701aa3954923c6bbb";
    flake-utils.url = "github:numtide/flake-utils";
    # Pin matches cuda-quantum tpls/llvm submodule
    llvm-project = {
      url = "github:llvm/llvm-project/7cbf1a2591520c2491aa35339f227775f4d3adf6";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, llvm-project }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Monorepo from flake input; apply cuda-quantum patches
        patchedMonorepo = pkgs.applyPatches {
          name = "llvm-project-patched";
          src = llvm-project;
          patches = [
            ./patches/CompilerRTUtils.cmake.diff
            ./patches/fix_region_simplification.diff
            ./patches/idempotent_option_category.diff
            ./patches/llvm_pr71968_mod.diff
            ./patches/mlir_python_sources_install.diff
          ];
        };

        # Reuse nixpkgs' llvmPackages_16 but with our patched source
        llvm16 = pkgs.llvmPackages_16.override {
          monorepoSrc = patchedMonorepo;
        };

      in {
        # Combined package: all LLVM components merged into a single prefix,
        # matching the layout CUDA-Q's CMakeLists.txt expects.
        packages.default = pkgs.symlinkJoin {
          name = "llvm-cudaq";
          paths = [
            llvm16.llvm
            llvm16.clang-unwrapped
            llvm16.mlir
            llvm16.lld
            llvm16.openmp
            llvm16.compiler-rt
            llvm16.libcxx
            llvm16.libunwind
          ];
        };

        # Individual components for granular use
        packages.llvm = llvm16.llvm;
        packages.clang = llvm16.clang-unwrapped;
        packages.mlir = llvm16.mlir;
        packages.lld = llvm16.lld;
        packages.openmp = llvm16.openmp;

        # Expose the patched monorepo source for downstream flakes
        packages.monorepoSrc = patchedMonorepo;
      }
    );
}
