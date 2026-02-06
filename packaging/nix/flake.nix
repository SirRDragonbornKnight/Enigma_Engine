{
  description = "ForgeAI - Fully modular AI framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python311;
        
        forgeai = python.pkgs.buildPythonPackage rec {
          pname = "forgeai";
          version = "1.0.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
            pip
          ];

          propagatedBuildInputs = with python.pkgs; [
            torch
            numpy
            pyqt5
            pillow
            requests
            aiohttp
            flask
            pyyaml
            tiktoken
            tqdm
          ];

          checkInputs = with python.pkgs; [
            pytest
            pytest-cov
          ];

          doCheck = false;  # Skip tests in build

          meta = with pkgs.lib; {
            description = "Fully modular AI framework for local and cloud AI";
            homepage = "https://github.com/forgeai/forge_ai";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.all;
          };
        };

        # Development shell
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            python.pkgs.pip
            python.pkgs.virtualenv
            git
            curl
            portaudio
            ffmpeg
          ] ++ (with python.pkgs; [
            torch
            numpy
            pyqt5
            pillow
            requests
            aiohttp
            flask
            pytest
            black
            mypy
            ipython
          ]);

          shellHook = ''
            echo "ForgeAI development environment"
            echo "Python: $(python --version)"
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };

      in {
        packages = {
          default = forgeai;
          forgeai = forgeai;
        };

        devShells.default = devShell;

        apps = {
          default = {
            type = "app";
            program = "${forgeai}/bin/forgeai";
          };
          
          gui = {
            type = "app";
            program = "${forgeai}/bin/forgeai-gui";
          };
        };

        # NixOS module
        nixosModules.default = { config, lib, pkgs, ... }:
          with lib;
          let
            cfg = config.services.forgeai;
          in {
            options.services.forgeai = {
              enable = mkEnableOption "ForgeAI AI service";
              
              port = mkOption {
                type = types.port;
                default = 8000;
                description = "Port for the ForgeAI API server";
              };
              
              dataDir = mkOption {
                type = types.path;
                default = "/var/lib/forgeai";
                description = "Directory for ForgeAI data";
              };
              
              user = mkOption {
                type = types.str;
                default = "forgeai";
                description = "User to run ForgeAI as";
              };
              
              group = mkOption {
                type = types.str;
                default = "forgeai";
                description = "Group to run ForgeAI as";
              };
            };

            config = mkIf cfg.enable {
              users.users.${cfg.user} = {
                isSystemUser = true;
                group = cfg.group;
                home = cfg.dataDir;
                createHome = true;
              };
              
              users.groups.${cfg.group} = {};

              systemd.services.forgeai = {
                description = "ForgeAI AI Service";
                wantedBy = [ "multi-user.target" ];
                after = [ "network.target" ];

                serviceConfig = {
                  Type = "simple";
                  User = cfg.user;
                  Group = cfg.group;
                  WorkingDirectory = cfg.dataDir;
                  ExecStart = "${forgeai}/bin/forgeai --serve --port ${toString cfg.port}";
                  Restart = "on-failure";
                  RestartSec = "5s";
                  
                  # Security hardening
                  NoNewPrivileges = true;
                  ProtectSystem = "strict";
                  ProtectHome = true;
                  PrivateTmp = true;
                  ReadWritePaths = [ cfg.dataDir ];
                };
              };

              networking.firewall.allowedTCPPorts = mkIf config.networking.firewall.enable [ cfg.port ];
            };
          };
      }
    );
}
