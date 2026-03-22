{
  description = "MLIR Python bindings for CUDA Quantum (per Python version)";

  inputs = {
    llvm-cudaq.url = "path:../llvm-cudaq";
    # Use the same nixpkgs as llvm-cudaq to ensure ABI compatibility
    nixpkgs.follows = "llvm-cudaq/nixpkgs";
    flake-utils.follows = "llvm-cudaq/flake-utils";
  };

  outputs = { self, llvm-cudaq, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Reconstruct the patched llvmPackages_16 from the exposed monorepoSrc
        llvm16 = pkgs.llvmPackages_16.override {
          monorepoSrc = llvm-cudaq.packages.${system}.monorepoSrc;
        };

        # Supported Python versions for CUDA-Q wheels
        pythonVersions = {
          python310 = pkgs.python310;
          python311 = pkgs.python311;
          python312 = pkgs.python312;
          python313 = pkgs.python313;
        };

        # Build MLIR with Python bindings for a given Python interpreter
        mlirWithPython = python: llvm16.mlir.overrideAttrs (old: {
          pname = "mlir-python-cudaq-${python.pythonVersion}";

          nativeBuildInputs = old.nativeBuildInputs ++ [ python ];

          buildInputs = old.buildInputs ++ [
            python.pkgs.pybind11
            python.pkgs.numpy
          ];

          cmakeFlags = old.cmakeFlags ++ [
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
            "-DPython3_EXECUTABLE=${python.interpreter}"
          ];
        });

      in {
        packages = builtins.mapAttrs
          (_name: python: mlirWithPython python)
          pythonVersions
        // {
          default = mlirWithPython pythonVersions.python311;
        };
      }
    );
}
