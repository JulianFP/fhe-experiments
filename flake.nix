{
  description = "Experimenting with FHE";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    pre-commit-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    project-W-frontend = {
      url = "github:julianfp/project-W-frontend";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      nixpkgs,
      systems,
      project-W-frontend,
      ...
    }:
    let
      eachSystem = nixpkgs.lib.genAttrs (import systems);
      pkgsFor = eachSystem (
        system:
        import nixpkgs {
          inherit system;
        }
      );
    in
    {
      devShells = eachSystem (system: {
        default = import ./shell.nix {
          inherit inputs system;
          pkgs = pkgsFor.${system};
        };
      });
    };
}
