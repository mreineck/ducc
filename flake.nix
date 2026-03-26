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

        ducc = py-pkgs.buildPythonPackage {
          pname = "ducc0";
          version = "0.41.0";
          pyproject = true;

          # stdenv = pkgs.clangStdenv;

          inherit src;
          postPatch = ''
            substituteInPlace pyproject.toml --replace-fail '"pybind11>=2.13.6", ' ""
          '';

          DUCC0_USE_NANOBIND = "";
          DUCC0_OPTIMIZATION = "portable";
          build-system = with py-pkgs; [
            pkgs.cmake
            nanobind
            ninja
            scikit-build-core
            setuptools-scm
          ];
          dontUseCmakeConfigure = true;
          dependencies = with py-pkgs; [ numpy ];

          nativeCheckInputs = with py-pkgs; [
            pytestCheckHook
            scipy
            pytest-xdist
            hypothesis
          ];
          pytestFlagsArray = [ "python/test" ];
          pythonImportsCheck = [ "ducc0" ];

          postInstall = ''
            mkdir -p $out/include/ducc0
            cp -r $src/src/ducc0/* $out/include/ducc0
          '';
        };

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
