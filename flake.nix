{
  description = "Ducc development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        py-pkgs = pkgs.python3Packages;
        src = ./.;

        ducc = py-pkgs.buildPythonPackage (finalAttrs: {
          pname = "ducc0";
          version = "0.42.0";
          pyproject = true;

          # stdenv = pkgs.clangStdenv;

          inherit src;
          postPatch = ''
            substituteInPlace pyproject.toml --replace-fail '"pybind11>=2.13.6",' ""
          '';

          env = {
            DUCC0_USE_NANOBIND = "";
            # multiarch compilation is only supported on x86_64
            DUCC0_OPTIMIZATION =
              if pkgs.stdenv.hostPlatform.isx86_64 && pkgs.stdenv.hostPlatform.isLinux then
                "multiarch-strip"
              else
                "portable";
          };

          # __FILE__ ends up in error messages; strip the build directory
          # (randomised per build on darwin) to keep the build reproducible.
          preBuild = ''
            export NIX_CFLAGS_COMPILE+=" -ffile-prefix-map=$PWD/="
          '';

          build-system = with py-pkgs; [
            pkgs.cmake
            nanobind
            ninja
            scikit-build-core
            setuptools-scm
          ];
          dontUseCmakeConfigure = true;
          dependencies = with py-pkgs; [ numpy ];

          nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.cc.isClang [
            pkgs.llvmPackages.llvm
          ];

          nativeCheckInputs = with py-pkgs; [
            pytest-xdist
            pytestCheckHook
            scipy
          ];

          pytestFlags = [ "python/test" ];
          pythonImportsCheck = [ "ducc0" ];

          # Not part of nixpkgs' ducc package
          preCheck = ''
            bash test/run.sh
          '';
          # /Not part of nixpkgs' ducc package

          postInstall = ''
            mkdir -p $out/include
            cp -r ${finalAttrs.src}/src/ducc0 $out/include
          '';
        });

      in {
        # Run `nix build .` to build the python ducc package and run the tests.
        packages.default = ducc;

        # Run `nix develop .` to enter the development shell. Then compile ducc
        # with, e.g., `pip3 install .`
        devShells.default = pkgs.mkShell {
          buildInputs = ducc.dependencies ++ ducc.build-system
            ++ (with py-pkgs; [ venvShellHook matplotlib pip pytest pybind11 ]);
          venvDir = ".nix-venv";

          shellHook = ''
            export PIP_PREFIX=$(pwd)/_build/pip_packages
            export PYTHONPATH="$PIP_PREFIX/${py-pkgs.python.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            unset SOURCE_DATE_EPOCH
          '';
        };

      });
}
