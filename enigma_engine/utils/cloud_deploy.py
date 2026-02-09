"""
Cloud Deployment Helpers for Enigma AI Engine

Deploy models and services to cloud platforms.

Features:
- Docker container generation
- Cloud provider helpers (AWS, GCP, Azure)
- Serverless deployment configs
- Model serving endpoints
- Auto-scaling configurations

Usage:
    from enigma_engine.utils.cloud_deploy import CloudDeployer, get_deployer
    
    deployer = get_deployer()
    
    # Generate Dockerfile
    deployer.generate_dockerfile("models/my_model")
    
    # Generate cloud configs
    deployer.generate_aws_config(model_path="models/my_model")
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


class InstanceSize(Enum):
    """Instance size presets."""
    SMALL = "small"  # 2 CPU, 4GB RAM
    MEDIUM = "medium"  # 4 CPU, 8GB RAM
    LARGE = "large"  # 8 CPU, 16GB RAM
    GPU_SMALL = "gpu_small"  # T4 GPU
    GPU_LARGE = "gpu_large"  # A10/V100 GPU


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    name: str
    model_path: str
    
    # Resources
    instance_size: InstanceSize = InstanceSize.MEDIUM
    min_replicas: int = 1
    max_replicas: int = 3
    
    # Networking
    port: int = 8000
    health_check_path: str = "/health"
    
    # Options
    enable_gpu: bool = False
    enable_autoscaling: bool = True
    
    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # Tags
    tags: Dict[str, str] = field(default_factory=dict)


class DockerGenerator:
    """Generate Docker configurations."""
    
    BASE_IMAGE_CPU = "python:3.10-slim"
    BASE_IMAGE_GPU = "nvidia/cuda:11.8-runtime-ubuntu22.04"
    
    def generate_dockerfile(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> str:
        """Generate Dockerfile."""
        base_image = self.BASE_IMAGE_GPU if config.enable_gpu else self.BASE_IMAGE_CPU
        
        dockerfile = f'''FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY enigma_engine/ ./enigma_engine/
COPY {config.model_path}/ ./models/

# Copy server script
COPY server.py .

# Set environment variables
ENV PORT={config.port}
ENV MODEL_PATH=/app/models

# Expose port
EXPOSE {config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.port}{config.health_check_path} || exit 1

# Run server
CMD ["python", "server.py"]
'''
        
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        
        return dockerfile
    
    def generate_docker_compose(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> str:
        """Generate docker-compose.yaml."""
        gpu_config = ""
        if config.enable_gpu:
            gpu_config = '''
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]'''
        
        compose = f'''version: "3.8"

services:
  {config.name}:
    build: .
    ports:
      - "{config.port}:{config.port}"
    environment:
      - PORT={config.port}
      - MODEL_PATH=/app/models
'''
        
        for key, value in config.env_vars.items():
            compose += f"      - {key}={value}\n"
        
        compose += gpu_config
        compose += f'''
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config.port}{config.health_check_path}"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
        
        compose_path = output_path / "docker-compose.yaml"
        compose_path.write_text(compose)
        
        return compose
    
    def generate_server_script(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> str:
        """Generate server.py."""
        server_code = '''"""Auto-generated server for Enigma AI model."""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import Enigma engine
from enigma_engine.core.inference import EnigmaEngine

app = FastAPI(title="Enigma AI Server")

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
engine = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global engine
    print(f"Loading model from {MODEL_PATH}")
    engine = EnigmaEngine()
    engine.load(MODEL_PATH)
    print("Model loaded successfully")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": engine is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = engine.generate(
        request.prompt,
        max_gen=request.max_tokens,
        temperature=request.temperature
    )
    
    return GenerateResponse(
        text=result,
        tokens_used=len(result.split())
    )


@app.get("/")
async def root():
    return {"message": "Enigma AI Server", "endpoints": ["/health", "/generate"]}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        
        server_path = output_path / "server.py"
        server_path.write_text(server_code)
        
        return server_code


class AWSGenerator:
    """Generate AWS deployment configurations."""
    
    INSTANCE_TYPES = {
        InstanceSize.SMALL: "t3.small",
        InstanceSize.MEDIUM: "t3.medium",
        InstanceSize.LARGE: "t3.large",
        InstanceSize.GPU_SMALL: "g4dn.xlarge",
        InstanceSize.GPU_LARGE: "p3.2xlarge"
    }
    
    def generate_ecs_task_definition(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate ECS task definition."""
        gpu_config = []
        if config.enable_gpu:
            gpu_config = [{
                "type": "GPU",
                "value": "1"
            }]
        
        task_def = {
            "family": config.name,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "1024" if config.instance_size in [InstanceSize.SMALL, InstanceSize.MEDIUM] else "2048",
            "memory": "2048" if config.instance_size == InstanceSize.SMALL else "4096",
            "executionRoleArn": f"arn:aws:iam::ACCOUNT_ID:role/{config.name}-execution-role",
            "taskRoleArn": f"arn:aws:iam::ACCOUNT_ID:role/{config.name}-task-role",
            "containerDefinitions": [{
                "name": config.name,
                "image": f"ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/{config.name}:latest",
                "portMappings": [{
                    "containerPort": config.port,
                    "protocol": "tcp"
                }],
                "environment": [
                    {"name": k, "value": v}
                    for k, v in config.env_vars.items()
                ],
                "resourceRequirements": gpu_config,
                "healthCheck": {
                    "command": ["CMD-SHELL", f"curl -f http://localhost:{config.port}{config.health_check_path} || exit 1"],
                    "interval": 30,
                    "timeout": 5,
                    "retries": 3
                },
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/{config.name}",
                        "awslogs-region": "REGION",
                        "awslogs-stream-prefix": "ecs"
                    }
                }
            }],
            "tags": [{"key": k, "value": v} for k, v in config.tags.items()]
        }
        
        task_path = output_path / "ecs-task-definition.json"
        task_path.write_text(json.dumps(task_def, indent=2))
        
        return task_def
    
    def generate_cloudformation(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate CloudFormation template."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"Enigma AI deployment for {config.name}",
            "Parameters": {
                "VpcId": {
                    "Type": "AWS::EC2::VPC::Id",
                    "Description": "VPC for deployment"
                },
                "SubnetIds": {
                    "Type": "List<AWS::EC2::Subnet::Id>",
                    "Description": "Subnets for deployment"
                }
            },
            "Resources": {
                "ECSCluster": {
                    "Type": "AWS::ECS::Cluster",
                    "Properties": {
                        "ClusterName": config.name
                    }
                },
                "LoadBalancer": {
                    "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
                    "Properties": {
                        "Name": f"{config.name}-alb",
                        "Subnets": {"Ref": "SubnetIds"},
                        "SecurityGroups": [{"Ref": "ALBSecurityGroup"}],
                        "Type": "application"
                    }
                },
                "ALBSecurityGroup": {
                    "Type": "AWS::EC2::SecurityGroup",
                    "Properties": {
                        "GroupDescription": "ALB Security Group",
                        "VpcId": {"Ref": "VpcId"},
                        "SecurityGroupIngress": [{
                            "IpProtocol": "tcp",
                            "FromPort": 80,
                            "ToPort": 80,
                            "CidrIp": "0.0.0.0/0"
                        }]
                    }
                }
            },
            "Outputs": {
                "LoadBalancerDNS": {
                    "Value": {"Fn::GetAtt": ["LoadBalancer", "DNSName"]},
                    "Description": "Load Balancer DNS"
                }
            }
        }
        
        cf_path = output_path / "cloudformation.yaml"
        cf_path.write_text(json.dumps(template, indent=2))
        
        return template
    
    def generate_lambda_config(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate AWS Lambda configuration (for small models)."""
        sam_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Transform": "AWS::Serverless-2016-10-31",
            "Description": f"Serverless Enigma AI - {config.name}",
            "Globals": {
                "Function": {
                    "Timeout": 30,
                    "MemorySize": 3008
                }
            },
            "Resources": {
                "InferenceFunction": {
                    "Type": "AWS::Serverless::Function",
                    "Properties": {
                        "FunctionName": config.name,
                        "CodeUri": ".",
                        "Handler": "lambda_handler.handler",
                        "Runtime": "python3.10",
                        "Environment": {
                            "Variables": {
                                "MODEL_PATH": "/opt/models",
                                **config.env_vars
                            }
                        },
                        "Events": {
                            "Api": {
                                "Type": "Api",
                                "Properties": {
                                    "Path": "/generate",
                                    "Method": "post"
                                }
                            }
                        }
                    }
                }
            }
        }
        
        sam_path = output_path / "template.yaml"
        sam_path.write_text(json.dumps(sam_template, indent=2))
        
        return sam_template


class KubernetesGenerator:
    """Generate Kubernetes deployment configurations."""
    
    def generate_deployment(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate Kubernetes Deployment."""
        gpu_resources = {}
        if config.enable_gpu:
            gpu_resources = {
                "nvidia.com/gpu": "1"
            }
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "labels": {
                    "app": config.name
                }
            },
            "spec": {
                "replicas": config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": f"{config.name}:latest",
                            "ports": [{
                                "containerPort": config.port
                            }],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in config.env_vars.items()
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "2Gi",
                                    "cpu": "1"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2",
                                    **gpu_resources
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": config.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": config.port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        deploy_path = output_path / "deployment.yaml"
        deploy_path.write_text(json.dumps(deployment, indent=2))
        
        return deployment
    
    def generate_service(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate Kubernetes Service."""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-service"
            },
            "spec": {
                "selector": {
                    "app": config.name
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": config.port
                }],
                "type": "LoadBalancer"
            }
        }
        
        svc_path = output_path / "service.yaml"
        svc_path.write_text(json.dumps(service, indent=2))
        
        return service
    
    def generate_hpa(
        self,
        config: DeploymentConfig,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler."""
        if not config.enable_autoscaling:
            return {}
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{config.name}-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": config.name
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 70
                        }
                    }
                }]
            }
        }
        
        hpa_path = output_path / "hpa.yaml"
        hpa_path.write_text(json.dumps(hpa, indent=2))
        
        return hpa


class CloudDeployer:
    """High-level cloud deployment interface."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize deployer.
        
        Args:
            output_dir: Directory for generated files
        """
        self._output_dir = output_dir or Path("deploy")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._docker = DockerGenerator()
        self._aws = AWSGenerator()
        self._k8s = KubernetesGenerator()
    
    def generate_all(
        self,
        config: DeploymentConfig,
        providers: Optional[List[CloudProvider]] = None
    ) -> Dict[str, List[Path]]:
        """
        Generate all deployment configs.
        
        Args:
            config: Deployment configuration
            providers: List of providers to generate for
            
        Returns:
            Dict mapping provider to generated file paths
        """
        providers = providers or [CloudProvider.DOCKER, CloudProvider.KUBERNETES]
        
        output_path = self._output_dir / config.name
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated = {}
        
        if CloudProvider.DOCKER in providers:
            self._docker.generate_dockerfile(config, output_path)
            self._docker.generate_docker_compose(config, output_path)
            self._docker.generate_server_script(config, output_path)
            generated[CloudProvider.DOCKER.value] = [
                output_path / "Dockerfile",
                output_path / "docker-compose.yaml",
                output_path / "server.py"
            ]
        
        if CloudProvider.AWS in providers:
            aws_path = output_path / "aws"
            aws_path.mkdir(exist_ok=True)
            self._aws.generate_ecs_task_definition(config, aws_path)
            self._aws.generate_cloudformation(config, aws_path)
            self._aws.generate_lambda_config(config, aws_path)
            generated[CloudProvider.AWS.value] = list(aws_path.glob("*"))
        
        if CloudProvider.KUBERNETES in providers:
            k8s_path = output_path / "kubernetes"
            k8s_path.mkdir(exist_ok=True)
            self._k8s.generate_deployment(config, k8s_path)
            self._k8s.generate_service(config, k8s_path)
            self._k8s.generate_hpa(config, k8s_path)
            generated[CloudProvider.KUBERNETES.value] = list(k8s_path.glob("*"))
        
        logger.info(f"Generated deployment configs in {output_path}")
        return generated
    
    def generate_dockerfile(
        self,
        model_path: str,
        name: Optional[str] = None,
        enable_gpu: bool = False
    ) -> Path:
        """Quick Dockerfile generation."""
        name = name or Path(model_path).stem
        config = DeploymentConfig(
            name=name,
            model_path=model_path,
            enable_gpu=enable_gpu
        )
        
        output_path = self._output_dir / name
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._docker.generate_dockerfile(config, output_path)
        return output_path / "Dockerfile"
    
    def generate_aws_config(
        self,
        model_path: str,
        name: Optional[str] = None,
        instance_size: InstanceSize = InstanceSize.MEDIUM
    ) -> Path:
        """Quick AWS config generation."""
        name = name or Path(model_path).stem
        config = DeploymentConfig(
            name=name,
            model_path=model_path,
            instance_size=instance_size
        )
        
        output_path = self._output_dir / name / "aws"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._aws.generate_ecs_task_definition(config, output_path)
        self._aws.generate_cloudformation(config, output_path)
        
        return output_path
    
    def generate_k8s_config(
        self,
        model_path: str,
        name: Optional[str] = None,
        replicas: int = 2
    ) -> Path:
        """Quick Kubernetes config generation."""
        name = name or Path(model_path).stem
        config = DeploymentConfig(
            name=name,
            model_path=model_path,
            min_replicas=replicas
        )
        
        output_path = self._output_dir / name / "kubernetes"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._k8s.generate_deployment(config, output_path)
        self._k8s.generate_service(config, output_path)
        
        return output_path


# Global instance
_deployer: Optional[CloudDeployer] = None


def get_deployer() -> CloudDeployer:
    """Get or create global deployer."""
    global _deployer
    if _deployer is None:
        _deployer = CloudDeployer()
    return _deployer
