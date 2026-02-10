"""
Kubernetes Helm Chart

Helm chart templates for deploying Enigma AI Engine on Kubernetes.
Includes deployment, service, configmap, and autoscaling.

FILE: enigma_engine/deploy/helm/values.yaml (conceptual)
TYPE: Deployment Configuration
PURPOSE: Generate Helm chart for K8s deployment
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Container resource requirements."""
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "512Mi"
    memory_limit: str = "4Gi"
    gpu_request: int = 0
    gpu_limit: int = 0


@dataclass
class ServiceConfig:
    """Service configuration."""
    type: str = "ClusterIP"  # ClusterIP, NodePort, LoadBalancer
    port: int = 8080
    target_port: int = 8080
    node_port: Optional[int] = None


@dataclass
class IngressConfig:
    """Ingress configuration."""
    enabled: bool = False
    host: str = "forge.local"
    tls_enabled: bool = False
    tls_secret: str = "forge-tls"
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class AutoscalingConfig:
    """HPA configuration."""
    enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    scale_down_stabilization: int = 300  # seconds


@dataclass
class PersistenceConfig:
    """Persistent volume configuration."""
    enabled: bool = True
    storage_class: str = ""  # Use default if empty
    size: str = "10Gi"
    access_mode: str = "ReadWriteOnce"


@dataclass
class HelmChartConfig:
    """Complete Helm chart configuration."""
    # Basic
    name: str = "forge-ai"
    namespace: str = "default"
    replicas: int = 1
    
    # Image
    image_repository: str = "ghcr.io/forge-ai/forge-ai"
    image_tag: str = "latest"
    image_pull_policy: str = "IfNotPresent"
    image_pull_secrets: list[str] = field(default_factory=list)
    
    # Resources
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Service
    service: ServiceConfig = field(default_factory=ServiceConfig)
    
    # Ingress
    ingress: IngressConfig = field(default_factory=IngressConfig)
    
    # Autoscaling
    autoscaling: AutoscalingConfig = field(default_factory=AutoscalingConfig)
    
    # Persistence
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    
    # Config
    environment: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    
    # Node selection
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict] = field(default_factory=list)
    affinity: dict = field(default_factory=dict)


class HelmChartGenerator:
    """Generates Helm chart files."""
    
    def __init__(self, config: HelmChartConfig):
        """
        Initialize generator.
        
        Args:
            config: Chart configuration
        """
        self._config = config
    
    def generate(self, output_dir: Path):
        """
        Generate all Helm chart files.
        
        Args:
            output_dir: Directory for chart files
        """
        chart_dir = output_dir / self._config.name
        templates_dir = chart_dir / "templates"
        
        chart_dir.mkdir(parents=True, exist_ok=True)
        templates_dir.mkdir(exist_ok=True)
        
        # Generate files
        self._write_chart_yaml(chart_dir)
        self._write_values_yaml(chart_dir)
        self._write_deployment(templates_dir)
        self._write_service(templates_dir)
        self._write_configmap(templates_dir)
        self._write_secret(templates_dir)
        self._write_hpa(templates_dir)
        self._write_pvc(templates_dir)
        self._write_ingress(templates_dir)
        self._write_helpers(templates_dir)
        self._write_notes(templates_dir)
        
        logger.info(f"Helm chart generated at {chart_dir}")
    
    def _write_chart_yaml(self, chart_dir: Path):
        """Write Chart.yaml."""
        chart = {
            "apiVersion": "v2",
            "name": self._config.name,
            "description": "Enigma AI Engine - Modular AI Framework",
            "type": "application",
            "version": "0.1.0",
            "appVersion": "1.0.0",
            "keywords": ["ai", "llm", "inference", "training"],
            "home": "https://github.com/forge-ai/forge-ai",
            "maintainers": [
                {"name": "Enigma AI Engine Team", "email": "team@forge-ai.dev"}
            ]
        }
        
        with open(chart_dir / "Chart.yaml", 'w') as f:
            yaml.dump(chart, f, default_flow_style=False)
    
    def _write_values_yaml(self, chart_dir: Path):
        """Write values.yaml."""
        values = {
            "replicaCount": self._config.replicas,
            
            "image": {
                "repository": self._config.image_repository,
                "tag": self._config.image_tag,
                "pullPolicy": self._config.image_pull_policy
            },
            
            "imagePullSecrets": [{"name": s} for s in self._config.image_pull_secrets],
            
            "resources": {
                "requests": {
                    "cpu": self._config.resources.cpu_request,
                    "memory": self._config.resources.memory_request
                },
                "limits": {
                    "cpu": self._config.resources.cpu_limit,
                    "memory": self._config.resources.memory_limit
                }
            },
            
            "service": {
                "type": self._config.service.type,
                "port": self._config.service.port
            },
            
            "ingress": {
                "enabled": self._config.ingress.enabled,
                "host": self._config.ingress.host,
                "tls": {
                    "enabled": self._config.ingress.tls_enabled,
                    "secretName": self._config.ingress.tls_secret
                },
                "annotations": self._config.ingress.annotations
            },
            
            "autoscaling": {
                "enabled": self._config.autoscaling.enabled,
                "minReplicas": self._config.autoscaling.min_replicas,
                "maxReplicas": self._config.autoscaling.max_replicas,
                "targetCPUUtilizationPercentage": self._config.autoscaling.target_cpu_percent,
                "targetMemoryUtilizationPercentage": self._config.autoscaling.target_memory_percent
            },
            
            "persistence": {
                "enabled": self._config.persistence.enabled,
                "storageClass": self._config.persistence.storage_class,
                "size": self._config.persistence.size,
                "accessMode": self._config.persistence.access_mode
            },
            
            "env": self._config.environment,
            
            "nodeSelector": self._config.node_selector,
            "tolerations": self._config.tolerations,
            "affinity": self._config.affinity
        }
        
        # Add GPU if configured
        if self._config.resources.gpu_limit > 0:
            values["resources"]["limits"]["nvidia.com/gpu"] = self._config.resources.gpu_limit
        if self._config.resources.gpu_request > 0:
            values["resources"]["requests"]["nvidia.com/gpu"] = self._config.resources.gpu_request
        
        with open(chart_dir / "values.yaml", 'w') as f:
            yaml.dump(values, f, default_flow_style=False)
    
    def _write_deployment(self, templates_dir: Path):
        """Write deployment.yaml template."""
        deployment = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "forge-ai.fullname" . }}
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "forge-ai.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "forge-ai.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          envFrom:
            - configMapRef:
                name: {{ include "forge-ai.fullname" . }}-config
            - secretRef:
                name: {{ include "forge-ai.fullname" . }}-secret
                optional: true
          {{- if .Values.persistence.enabled }}
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: models
              mountPath: /app/models
          {{- end }}
      {{- if .Values.persistence.enabled }}
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: {{ include "forge-ai.fullname" . }}-data
        - name: models
          persistentVolumeClaim:
            claimName: {{ include "forge-ai.fullname" . }}-models
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
'''
        with open(templates_dir / "deployment.yaml", 'w') as f:
            f.write(deployment)
    
    def _write_service(self, templates_dir: Path):
        """Write service.yaml template."""
        service = '''apiVersion: v1
kind: Service
metadata:
  name: {{ include "forge-ai.fullname" . }}
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "forge-ai.selectorLabels" . | nindent 4 }}
'''
        with open(templates_dir / "service.yaml", 'w') as f:
            f.write(service)
    
    def _write_configmap(self, templates_dir: Path):
        """Write configmap.yaml template."""
        configmap = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "forge-ai.fullname" . }}-config
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
data:
  FORGE_DATA_DIR: "/app/data"
  FORGE_MODEL_DIR: "/app/models"
  FORGE_LOG_LEVEL: "INFO"
  FORGE_API_HOST: "0.0.0.0"
  FORGE_API_PORT: "8080"
'''
        with open(templates_dir / "configmap.yaml", 'w') as f:
            f.write(configmap)
    
    def _write_secret(self, templates_dir: Path):
        """Write secret.yaml template."""
        secret = '''{{- if .Values.secrets }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "forge-ai.fullname" . }}-secret
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
type: Opaque
data:
  {{- range $key, $value := .Values.secrets }}
  {{ $key }}: {{ $value | b64enc }}
  {{- end }}
{{- end }}
'''
        with open(templates_dir / "secret.yaml", 'w') as f:
            f.write(secret)
    
    def _write_hpa(self, templates_dir: Path):
        """Write hpa.yaml template."""
        hpa = '''{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "forge-ai.fullname" . }}
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "forge-ai.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
{{- end }}
'''
        with open(templates_dir / "hpa.yaml", 'w') as f:
            f.write(hpa)
    
    def _write_pvc(self, templates_dir: Path):
        """Write pvc.yaml template."""
        pvc = '''{{- if .Values.persistence.enabled }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "forge-ai.fullname" . }}-data
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.persistence.accessMode }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.persistence.size }}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "forge-ai.fullname" . }}-models
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.persistence.accessMode }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: 50Gi
{{- end }}
'''
        with open(templates_dir / "pvc.yaml", 'w') as f:
            f.write(pvc)
    
    def _write_ingress(self, templates_dir: Path):
        """Write ingress.yaml template."""
        ingress = '''{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "forge-ai.fullname" . }}
  labels:
    {{- include "forge-ai.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.tls.enabled }}
  tls:
    - hosts:
        - {{ .Values.ingress.host }}
      secretName: {{ .Values.ingress.tls.secretName }}
  {{- end }}
  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ include "forge-ai.fullname" . }}
                port:
                  number: {{ .Values.service.port }}
{{- end }}
'''
        with open(templates_dir / "ingress.yaml", 'w') as f:
            f.write(ingress)
    
    def _write_helpers(self, templates_dir: Path):
        """Write _helpers.tpl template."""
        helpers = '''{{/*
Expand the name of the chart.
*/}}
{{- define "forge-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "forge-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "forge-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "forge-ai.labels" -}}
helm.sh/chart: {{ include "forge-ai.chart" . }}
{{ include "forge-ai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "forge-ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "forge-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
'''
        with open(templates_dir / "_helpers.tpl", 'w') as f:
            f.write(helpers)
    
    def _write_notes(self, templates_dir: Path):
        """Write NOTES.txt template."""
        notes = '''Enigma AI Engine has been deployed!

1. Get the application URL by running:
{{- if .Values.ingress.enabled }}
  http{{ if .Values.ingress.tls.enabled }}s{{ end }}://{{ .Values.ingress.host }}
{{- else if contains "NodePort" .Values.service.type }}
  export NODE_PORT=$(kubectl get --namespace {{ .Release.Namespace }} -o jsonpath="{.spec.ports[0].nodePort}" services {{ include "forge-ai.fullname" . }})
  export NODE_IP=$(kubectl get nodes --namespace {{ .Release.Namespace }} -o jsonpath="{.items[0].status.addresses[0].address}")
  echo http://$NODE_IP:$NODE_PORT
{{- else if contains "LoadBalancer" .Values.service.type }}
  NOTE: It may take a few minutes for the LoadBalancer IP to be available.
  export SERVICE_IP=$(kubectl get svc --namespace {{ .Release.Namespace }} {{ include "forge-ai.fullname" . }} --template "{{"{{ range (index .status.loadBalancer.ingress 0) }}{{.}}{{ end }}"}}")
  echo http://$SERVICE_IP:{{ .Values.service.port }}
{{- else if contains "ClusterIP" .Values.service.type }}
  kubectl --namespace {{ .Release.Namespace }} port-forward svc/{{ include "forge-ai.fullname" . }} {{ .Values.service.port }}:{{ .Values.service.port }}
  Then visit http://localhost:{{ .Values.service.port }}
{{- end }}

2. Check the logs:
  kubectl logs -f deployment/{{ include "forge-ai.fullname" . }} --namespace {{ .Release.Namespace }}

3. API Documentation:
  http://localhost:{{ .Values.service.port }}/docs
'''
        with open(templates_dir / "NOTES.txt", 'w') as f:
            f.write(notes)


def generate_helm_chart(output_dir: Path, config: HelmChartConfig = None):
    """
    Generate a Helm chart for Enigma AI Engine.
    
    Args:
        output_dir: Output directory
        config: Chart configuration
    """
    config = config or HelmChartConfig()
    generator = HelmChartGenerator(config)
    generator.generate(output_dir)


__all__ = [
    'HelmChartConfig',
    'HelmChartGenerator',
    'ResourceRequirements',
    'ServiceConfig',
    'IngressConfig',
    'AutoscalingConfig',
    'PersistenceConfig',
    'generate_helm_chart'
]
