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
          version = "0.35.0";
          inherit src;
          pyproject = true;
          build-system = with py-pkgs; [
            pkgs.cmake
            pybind11
            nanobind
            ninja
            scikit-build-core
            setuptools-scm
          ];
          dontUseCmakeConfigure = true;

          dependencies = with py-pkgs; [ numpy scipy ];

          checkInputs = [ py-pkgs.pytestCheckHook ];
          pythonImportsCheck = [ "ducc0" ];

          # Uncomment to specify optimization levels. Note: need to pass
          # --impure to `nix build` to enable all optimizations
          # DUCC0_OPTIMIZATION = "none";

          # Uncomment the next line to enable build via nanobind
          # DUCC0_USE_NANOBIND = "";

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
            ++ (with py-pkgs; [ venvShellHook matplotlib ]);
          venvDir = ".nix-venv";
        };

      });
}
