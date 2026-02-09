"""
One-Click Deploy

Deployment scripts for Railway, Render, and Fly.io platforms.
Generates configuration files and handles deployment automation.

FILE: enigma_engine/deploy/one_click.py
TYPE: Deployment
MAIN CLASSES: DeployConfig, RailwayDeploy, RenderDeploy, FlyDeploy
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeployPlatform(Enum):
    """Supported deployment platforms."""
    RAILWAY = "railway"
    RENDER = "render"
    FLY_IO = "fly.io"
    DOCKER = "docker"
    LOCAL = "local"


@dataclass
class DeployConfig:
    """Deployment configuration."""
    name: str = "forge-ai"
    platform: DeployPlatform = DeployPlatform.DOCKER
    region: str = "us-east"
    port: int = 8080
    memory_mb: int = 512
    cpu_count: int = 1
    gpu_enabled: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 3


class DeployGenerator:
    """Base class for deployment generators."""
    
    def __init__(self, config: DeployConfig, output_dir: Path):
        self._config = config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> dict[str, Path]:
        """
        Generate deployment files.
        
        Returns:
            Dict mapping file name to path
        """
        raise NotImplementedError
    
    def deploy(self) -> bool:
        """
        Execute deployment.
        
        Returns:
            True if successful
        """
        raise NotImplementedError


class RailwayDeploy(DeployGenerator):
    """Railway.app deployment generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate Railway configuration files."""
        files = {}
        
        # railway.toml
        toml_content = f'''[build]
builder = "DOCKERFILE"
dockerfilePath = "./Dockerfile"

[deploy]
startCommand = "python run.py --serve"
healthcheckPath = "{self._config.health_check_path}"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[service]
internalPort = {self._config.port}
'''
        
        toml_path = self._output_dir / "railway.toml"
        toml_path.write_text(toml_content)
        files["railway.toml"] = toml_path
        
        # railway.json (legacy/alternative)
        json_config = {
            "$schema": "https://railway.app/railway.schema.json",
            "build": {
                "builder": "DOCKERFILE",
                "dockerfilePath": "./Dockerfile"
            },
            "deploy": {
                "startCommand": "python run.py --serve",
                "healthcheckPath": self._config.health_check_path,
                "restartPolicyType": "ON_FAILURE"
            }
        }
        
        json_path = self._output_dir / "railway.json"
        json_path.write_text(json.dumps(json_config, indent=2))
        files["railway.json"] = json_path
        
        # Environment template
        env_content = f'''# Railway Environment Variables
PORT={self._config.port}
FORGE_MODE=production
FORGE_LOG_LEVEL=info
'''
        for key, value in self._config.env_vars.items():
            env_content += f"{key}={value}\n"
        
        env_path = self._output_dir / ".env.railway"
        env_path.write_text(env_content)
        files[".env.railway"] = env_path
        
        return files
    
    def deploy(self) -> bool:
        """Deploy to Railway using CLI."""
        try:
            # Check if Railway CLI is installed
            result = subprocess.run(
                ["railway", "--version"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.error("Railway CLI not installed. Run: npm i -g @railway/cli")
                return False
            
            # Deploy
            result = subprocess.run(
                ["railway", "up"],
                cwd=self._output_dir.parent,
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                logger.info("Railway deployment successful")
                return True
            else:
                logger.error(f"Railway deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("Railway CLI not found")
            return False


class RenderDeploy(DeployGenerator):
    """Render.com deployment generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate Render configuration files."""
        files = {}
        
        # render.yaml (Infrastructure as Code)
        yaml_content = f'''services:
  - type: web
    name: {self._config.name}
    env: docker
    dockerfilePath: ./Dockerfile
    region: {self._config.region}
    plan: starter
    healthCheckPath: {self._config.health_check_path}
    autoDeploy: true
    
    envVars:
      - key: PORT
        value: "{self._config.port}"
      - key: FORGE_MODE
        value: production
'''
        
        # Add custom env vars
        for key, value in self._config.env_vars.items():
            yaml_content += f'''      - key: {key}
        value: "{value}"
'''
        
        # Auto-scaling if enabled
        if self._config.auto_scaling:
            yaml_content += f'''
    scaling:
      minInstances: {self._config.min_instances}
      maxInstances: {self._config.max_instances}
      targetMemoryPercent: 80
      targetCPUPercent: 80
'''
        
        yaml_path = self._output_dir / "render.yaml"
        yaml_path.write_text(yaml_content)
        files["render.yaml"] = yaml_path
        
        return files
    
    def deploy(self) -> bool:
        """Deploy to Render (via Git push or API)."""
        logger.info("Render deployment via render.yaml")
        logger.info("Push to connected Git repository to trigger deployment")
        logger.info("Or use Render Dashboard: https://dashboard.render.com")
        return True


class FlyDeploy(DeployGenerator):
    """Fly.io deployment generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate Fly.io configuration files."""
        files = {}
        
        # fly.toml
        toml_content = f'''app = "{self._config.name}"
primary_region = "{self._config.region}"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = {self._config.port}
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = {self._config.min_instances}

[checks]
  [checks.health]
    port = {self._config.port}
    type = "http"
    interval = "10s"
    timeout = "2s"
    grace_period = "5s"
    method = "GET"
    path = "{self._config.health_check_path}"

[[vm]]
  cpu_kind = "shared"
  cpus = {self._config.cpu_count}
  memory_mb = {self._config.memory_mb}
'''
        
        # GPU support
        if self._config.gpu_enabled:
            toml_content += '''
  gpu_kind = "a100-pcie-40gb"
'''
        
        toml_path = self._output_dir / "fly.toml"
        toml_path.write_text(toml_content)
        files["fly.toml"] = toml_path
        
        # Secrets script
        secrets_script = '''#!/bin/bash
# Set Fly.io secrets
'''
        for key, value in self._config.env_vars.items():
            secrets_script += f'fly secrets set {key}="{value}"\n'
        
        secrets_path = self._output_dir / "fly-secrets.sh"
        secrets_path.write_text(secrets_script)
        files["fly-secrets.sh"] = secrets_path
        
        return files
    
    def deploy(self) -> bool:
        """Deploy to Fly.io using CLI."""
        try:
            # Check if Fly CLI is installed
            result = subprocess.run(
                ["fly", "version"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.error("Fly CLI not installed. Run: curl -L https://fly.io/install.sh | sh")
                return False
            
            # Launch/deploy
            result = subprocess.run(
                ["fly", "deploy", "--config", str(self._output_dir / "fly.toml")],
                cwd=self._output_dir.parent,
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                logger.info("Fly.io deployment successful")
                return True
            else:
                logger.error(f"Fly.io deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("Fly CLI not found")
            return False


class DockerComposeDeploy(DeployGenerator):
    """Docker Compose deployment generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate Docker Compose production files."""
        files = {}
        
        # docker-compose.prod.yml
        compose_content = f'''version: '3.8'

services:
  forge-ai:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {self._config.name}
    restart: unless-stopped
    ports:
      - "{self._config.port}:{self._config.port}"
    environment:
      - PORT={self._config.port}
      - FORGE_MODE=production
'''
        
        for key, value in self._config.env_vars.items():
            compose_content += f"      - {key}={value}\n"
        
        compose_content += f'''    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self._config.port}{self._config.health_check_path}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '{self._config.cpu_count}'
          memory: {self._config.memory_mb}M
'''
        
        if self._config.gpu_enabled:
            compose_content += '''    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
'''
        
        compose_content += '''
volumes:
  forge_data:

networks:
  default:
    driver: bridge
'''
        
        compose_path = self._output_dir / "docker-compose.prod.yml"
        compose_path.write_text(compose_content)
        files["docker-compose.prod.yml"] = compose_path
        
        return files
    
    def deploy(self) -> bool:
        """Deploy using Docker Compose."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.prod.yml", "up", "-d", "--build"],
                cwd=self._output_dir,
                capture_output=True, text=True, timeout=1800
            )
            
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("Docker Compose not found")
            return False


def generate_all_platforms(config: DeployConfig, output_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Generate deployment files for all platforms.
    
    Args:
        config: Deployment configuration
        output_dir: Output directory
        
    Returns:
        Dict mapping platform to generated files
    """
    results = {}
    
    generators = {
        "railway": RailwayDeploy(config, output_dir / "railway"),
        "render": RenderDeploy(config, output_dir / "render"),
        "fly": FlyDeploy(config, output_dir / "fly"),
        "docker": DockerComposeDeploy(config, output_dir / "docker"),
    }
    
    for name, generator in generators.items():
        try:
            results[name] = generator.generate()
            logger.info(f"Generated {name} deployment files")
        except Exception as e:
            logger.error(f"Failed to generate {name}: {e}")
            results[name] = {}
    
    return results


def one_click_deploy(platform: DeployPlatform, config: DeployConfig = None, output_dir: Path = None) -> bool:
    """
    One-click deployment to specified platform.
    
    Args:
        platform: Target platform
        config: Deployment config (uses defaults if None)
        output_dir: Output directory for config files
        
    Returns:
        True if successful
    """
    config = config or DeployConfig(platform=platform)
    output_dir = output_dir or Path("deploy_configs")
    
    generators = {
        DeployPlatform.RAILWAY: RailwayDeploy,
        DeployPlatform.RENDER: RenderDeploy,
        DeployPlatform.FLY_IO: FlyDeploy,
        DeployPlatform.DOCKER: DockerComposeDeploy,
    }
    
    generator_class = generators.get(platform)
    if not generator_class:
        logger.error(f"Unsupported platform: {platform}")
        return False
    
    generator = generator_class(config, output_dir)
    generator.generate()
    return generator.deploy()


__all__ = [
    'DeployConfig',
    'DeployPlatform',
    'DeployGenerator',
    'RailwayDeploy',
    'RenderDeploy',
    'FlyDeploy',
    'DockerComposeDeploy',
    'generate_all_platforms',
    'one_click_deploy'
]
