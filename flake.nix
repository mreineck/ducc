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

        ducc = py-pkgs.buildPythonPackage {
          pname = "ducc0";
          version = "0.34.0";
          src = ./.;
          pyproject = true;
          build-system = with py-pkgs; [ setuptools ];
          dependencies = with py-pkgs; [ numpy scipy pybind11 ];

          checkInputs = [ py-pkgs.pytestCheckHook ];
          pythonImportsCheck = [ "ducc0" ];

          DUCC0_OPTIMIZATION = "portable-strip";
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
