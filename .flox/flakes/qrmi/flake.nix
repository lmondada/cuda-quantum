{
  description = "QRMI C library for Pasqal QRMI connector (pre-built binary)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };

        version = "0.12.0";

        qrmi = pkgs.stdenv.mkDerivation {
          pname = "qrmi";
          inherit version;

          src = pkgs.fetchurl {
            url = "https://github.com/qiskit-community/qrmi/releases/download/v${version}/libqrmi-${version}-el8-x86_64.tar.gz";
            hash = "sha256-KYYVDU9V4fZWa+8W2fs4l8oE3X6qaBhl9+8kTymKZ0Y=";
          };

          nativeBuildInputs = with pkgs; [
            autoPatchelfHook
          ];

          buildInputs = with pkgs; [
            stdenv.cc.cc.lib  # libstdc++
          ];

          dontBuild = true;
          dontConfigure = true;

          unpackPhase = ''
            tar xzf $src
          '';

          installPhase = ''
            mkdir -p $out/include $out/lib64
            cp libqrmi-${version}/qrmi.h $out/include/
            cp libqrmi-${version}/libqrmi.so $out/lib64/
          '';

          meta = with pkgs.lib; {
            description = "QRMI C library for Pasqal quantum processor connector";
            license = licenses.asl20;
            platforms = [ "x86_64-linux" ];
          };
        };
      in {
        packages.default = qrmi;
      }
    );
}
