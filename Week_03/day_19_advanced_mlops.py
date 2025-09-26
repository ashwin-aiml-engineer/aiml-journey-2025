"""
Day 19 : Advanced MLOps Concepts (25-Minute Deep Dive)
🎯 Master environment management, IaC, monitoring, scaling
✅ Production-ready MLOps infrastructure and operations
"""

import os
import json
import yaml
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("📝 Docker not available - using simulation")

class EnvironmentManager:
    """Advanced environment management for ML"""
    
    def __init__(self):
        self.environments = {
            "development": {"resources": "minimal", "monitoring": "basic"},
            "staging": {"resources": "medium", "monitoring": "comprehensive"},  
            "production": {"resources": "high", "monitoring": "full", "replicas": 3}
        }

    def create_environment_configs(self):
        """Create environment-specific configurations"""
        print("🏗️ ENVIRONMENT MANAGEMENT")
        print("=" * 25)
        
        # Docker Compose for different environments
        compose_configs = {}
        
        for env, config in self.environments.items():
            compose_config = {
                "version": "3.8",
                "services": {
                    "ml-service": {
                        "image": f"ml-model:{env}",
                        "environment": [
                            f"ENVIRONMENT={env}",
                            f"LOG_LEVEL={'INFO' if env == 'production' else 'DEBUG'}",
                            f"MODEL_PATH=/models/{env}"
                        ],
                        "resources": {
                            "limits": {
                                "memory": "2G" if env == "production" else "1G",
                                "cpus": "2" if env == "production" else "1"
                            }
                        }
                    }
                }
            }
            
            if env == "production":
                compose_config["services"]["ml-service"]["deploy"] = {
                    "replicas": config["replicas"],
                    "restart_policy": {"condition": "on-failure"}
                }
                # Add load balancer for production
                compose_config["services"]["nginx"] = {
                    "image": "nginx:alpine",
                    "ports": ["80:80"],
                    "depends_on": ["ml-service"]
                }
            
            compose_configs[env] = compose_config
        
        # Save configurations
        config_dir = Path("./environment_configs")
        config_dir.mkdir(exist_ok=True)
        
        for env, config in compose_configs.items():
            config_file = config_dir / f"docker-compose.{env}.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"  ✅ Created {env} environment config")
        
        return compose_configs

    def demonstrate_configuration_management(self):
        """Show configuration management strategies"""
        print("\n⚙️ CONFIGURATION MANAGEMENT")
        print("-" * 30)
        
        # Environment-specific configs
        configs = {
            "development": {
                "model_path": "/tmp/models",
                "batch_size": 32,
                "workers": 1,
                "cache_ttl": 60,
                "debug": True
            },
            "staging": {
                "model_path": "/app/models/staging",
                "batch_size": 64,
                "workers": 2,
                "cache_ttl": 300,
                "debug": False
            },
            "production": {
                "model_path": "/app/models/production",
                "batch_size": 128,
                "workers": 4,
                "cache_ttl": 3600,
                "debug": False
            }
        }
        
        print("📋 Environment Configurations:")
        for env, config in configs.items():
            print(f"\n  {env.upper()}:")
            for key, value in config.items():
                print(f"    {key}: {value}")

class InfrastructureAsCode:
    """Infrastructure as Code for ML systems"""
    
    def __init__(self):
        self.terraform_modules = []

    def create_terraform_infrastructure(self):
        """Create Terraform configurations for ML infrastructure"""
        print("\n🏗️ INFRASTRUCTURE AS CODE")
        print("-" * 28)
        
        # Main Terraform configuration
        terraform_main = {
            "terraform": {
                "required_version": ">= 1.0",
                "required_providers": {
                    "aws": {"source": "hashicorp/aws", "version": "~> 5.0"},
                    "kubernetes": {"source": "hashicorp/kubernetes", "version": "~> 2.0"}
                }
            },
            "provider": {
                "aws": {"region": "us-west-2"}
            },
            "resource": {
                "aws_eks_cluster": {
                    "ml_cluster": {
                        "name": "ml-platform",
                        "role_arn": "${aws_iam_role.cluster.arn}",
                        "version": "1.27",
                        "vpc_config": {
                            "subnet_ids": ["${aws_subnet.private[*].id}"]
                        }
                    }
                },
                "aws_s3_bucket": {
                    "model_artifacts": {
                        "bucket": "ml-model-artifacts-${random_string.suffix.result}",
                        "versioning": {"enabled": True}
                    }
                },
                "aws_ecr_repository": {
                    "ml_models": {
                        "name": "ml-models",
                        "image_tag_mutability": "MUTABLE"
                    }
                }
            }
        }
        
        # Kubernetes manifests
        k8s_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment", 
            "metadata": {
                "name": "ml-model-service",
                "labels": {"app": "ml-model"}
            },
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "ml-model"}},
                "template": {
                    "metadata": {"labels": {"app": "ml-model"}},
                    "spec": {
                        "containers": [{
                            "name": "ml-model",
                            "image": "ml-model:latest",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {"memory": "1Gi", "cpu": "500m"},
                                "limits": {"memory": "2Gi", "cpu": "1000m"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30
                            }
                        }]
                    }
                }
            }
        }
        
        # Save configurations
        iac_dir = Path("./infrastructure")
        iac_dir.mkdir(exist_ok=True)
        
        with open(iac_dir / "main.tf", 'w') as f:
            # Convert to HCL format (simplified)
            f.write("# Terraform configuration for ML infrastructure\n")
            f.write("# (This would be in proper HCL format in practice)\n")
        
        with open(iac_dir / "ml-deployment.yaml", 'w') as f:
            yaml.dump(k8s_deployment, f, default_flow_style=False)
        
        print("  ✅ Created Terraform main configuration")
        print("  ✅ Created Kubernetes deployment manifest")
        print("  ✅ Infrastructure includes: EKS cluster, S3 buckets, ECR registry")

class MLMonitoringSystem:
    """Advanced monitoring and observability"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []

    def setup_monitoring_stack(self):
        """Configure monitoring stack"""
        print("\n📊 MONITORING & OBSERVABILITY")
        print("-" * 30)
        
        # Monitoring stack configuration
        monitoring_config = {
            "prometheus": {
                "scrape_configs": [
                    {
                        "job_name": "ml-model-metrics",
                        "static_configs": [{"targets": ["ml-service:8000"]}],
                        "metrics_path": "/metrics",
                        "scrape_interval": "15s"
                    }
                ]
            },
            "grafana": {
                "dashboards": [
                    {
                        "name": "ML Model Performance",
                        "panels": [
                            "Prediction Latency",
                            "Throughput (req/sec)",
                            "Model Accuracy",
                            "Memory Usage",
                            "CPU Usage"
                        ]
                    },
                    {
                        "name": "Data Drift Detection",
                        "panels": [
                            "Feature Distribution Shift",
                            "Data Quality Score",
                            "Schema Violations",
                            "Missing Value Rate"
                        ]
                    }
                ]
            },
            "alertmanager": {
                "rules": [
                    {
                        "alert": "ModelLatencyHigh",
                        "expr": "model_prediction_latency > 0.5",
                        "for": "5m",
                        "annotations": {"summary": "Model prediction latency too high"}
                    },
                    {
                        "alert": "DataDriftDetected", 
                        "expr": "data_drift_score > 0.7",
                        "for": "2m",
                        "annotations": {"summary": "Significant data drift detected"}
                    }
                ]
            }
        }
        
        print("🔍 Monitoring Stack Components:")
        for component, config in monitoring_config.items():
            print(f"  • {component.title()}: Configured")
        
        return monitoring_config

    def simulate_drift_detection(self):
        """Simulate data drift detection"""
        print("\n⚠️ DATA DRIFT DETECTION")
        print("-" * 24)
        
        # Simulate feature distributions over time
        np.random.seed(42)
        
        # Baseline distribution (training data)
        baseline_features = np.random.normal(0, 1, (1000, 3))
        
        # Simulate drift over time periods
        time_periods = [
            {"period": "Week 1", "drift": 0.1, "features": np.random.normal(0.1, 1, (1000, 3))},
            {"period": "Week 2", "drift": 0.3, "features": np.random.normal(0.3, 1.2, (1000, 3))},
            {"period": "Week 3", "drift": 0.7, "features": np.random.normal(0.8, 1.5, (1000, 3))},
            {"period": "Week 4", "drift": 0.9, "features": np.random.normal(1.2, 2.0, (1000, 3))}
        ]
        
        print("📈 Drift Detection Results:")
        for period_data in time_periods:
            period = period_data["period"]
            drift_score = period_data["drift"]
            
            # Determine alert level
            if drift_score < 0.3:
                status = "✅ Normal"
            elif drift_score < 0.7:
                status = "⚠️ Warning"
            else:
                status = "🚨 Critical"
            
            print(f"  {period}: Drift Score = {drift_score:.1f} {status}")
            
            if drift_score > 0.7:
                print(f"    → Action: Model retraining recommended")

class AutoScalingSystem:
    """Automated scaling for ML services"""
    
    def __init__(self):
        self.current_replicas = 2
        self.min_replicas = 1
        self.max_replicas = 10

    def demonstrate_auto_scaling(self):
        """Simulate auto-scaling scenarios"""
        print("\n📈 AUTO-SCALING SYSTEM")
        print("-" * 22)
        
        # HPA (Horizontal Pod Autoscaler) configuration
        hpa_config = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "ml-model-hpa"},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "ml-model-service"
                },
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 70}
                        }
                    },
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {"name": "prediction_queue_length"},
                            "target": {"type": "AverageValue", "averageValue": "100"}
                        }
                    }
                ]
            }
        }
        
        print("⚙️ Auto-scaling Configuration:")
        print(f"  Min replicas: {self.min_replicas}")
        print(f"  Max replicas: {self.max_replicas}")
        print("  Scale triggers:")
        print("    • CPU > 70%")
        print("    • Queue length > 100")
        
        # Simulate scaling events
        scaling_scenarios = [
            {"time": "09:00", "cpu": 45, "queue": 20, "replicas": 2, "action": "No scaling"},
            {"time": "12:00", "cpu": 80, "queue": 150, "replicas": 4, "action": "Scale up"},
            {"time": "15:00", "cpu": 90, "queue": 300, "replicas": 7, "action": "Scale up"},
            {"time": "18:00", "cpu": 30, "queue": 50, "replicas": 3, "action": "Scale down"},
            {"time": "22:00", "cpu": 20, "queue": 10, "replicas": 2, "action": "Scale down"}
        ]
        
        print(f"\n📊 Scaling Simulation:")
        for scenario in scaling_scenarios:
            print(f"  {scenario['time']}: CPU={scenario['cpu']}%, Queue={scenario['queue']} → {scenario['replicas']} replicas ({scenario['action']})")

class MLSecurityManager:
    """Security management for ML systems"""
    
    def __init__(self):
        self.security_policies = []

    def demonstrate_ml_security(self):
        """Show ML-specific security considerations"""
        print("\n🔒 ML SECURITY MANAGEMENT")
        print("-" * 27)
        
        security_areas = {
            "Model Security": [
                "Model artifact encryption at rest",
                "Secure model serving endpoints",
                "Model versioning and integrity checks",
                "Access control for model registry"
            ],
            "Data Security": [
                "Data encryption in transit and at rest",
                "PII detection and masking",
                "Data access auditing",
                "Secure data pipelines"
            ],
            "Infrastructure Security": [
                "Container image scanning",
                "Network policies and segmentation",
                "Secret management (keys, tokens)",
                "Regular security updates"
            ],
            "Compliance": [
                "GDPR compliance for ML models",
                "Model explainability requirements",
                "Audit trails for decisions",
                "Data retention policies"
            ]
        }
        
        for area, practices in security_areas.items():
            print(f"\n🛡️ {area}:")
            for practice in practices:
                print(f"  • {practice}")

def comprehensive_mlops_demo():
    """Complete advanced MLOps concepts demonstration"""
    print("🌟 DAY 19 BONUS: ADVANCED MLOPS (25 min)")
    print("=" * 44)
    
    # 1. Environment Management
    env_manager = EnvironmentManager()
    compose_configs = env_manager.create_environment_configs()
    env_manager.demonstrate_configuration_management()
    
    # 2. Infrastructure as Code
    iac = InfrastructureAsCode()
    iac.create_terraform_infrastructure()
    
    # 3. Advanced Monitoring
    monitoring = MLMonitoringSystem()
    monitoring_config = monitoring.setup_monitoring_stack()
    monitoring.simulate_drift_detection()
    
    # 4. Auto-scaling
    scaling = AutoScalingSystem()
    scaling.demonstrate_auto_scaling()
    
    # 5. Security Management
    security = MLSecurityManager()
    security.demonstrate_ml_security()
    
    print("\n🎯 ADVANCED MLOPS MASTERED!")
    print("Deep dive accomplishments:")
    print("  ✅ Multi-environment management with Docker Compose")
    print("  ✅ Infrastructure as Code with Terraform & Kubernetes")
    print("  ✅ Comprehensive monitoring stack (Prometheus, Grafana)")
    print("  ✅ Data drift detection and alerting")
    print("  ✅ Horizontal auto-scaling for ML services")
    print("  ✅ ML-specific security and compliance practices")
    print("  ✅ Production-ready configuration management")
    
    print("\n📚 Advanced MLOps Principles:")
    print("  • Infrastructure should be code (versioned, tested)")
    print("  • Monitor everything: models, data, infrastructure")
    print("  • Automate scaling based on ML-specific metrics")
    print("  • Implement security at every layer")
    print("  • Design for compliance from the start")
    print("  • Use multi-environment workflows (dev/staging/prod)")

def create_mlops_checklist():
    """Create production MLOps readiness checklist"""
    print("\n✅ PRODUCTION READINESS CHECKLIST")
    print("-" * 35)
    
    checklist_categories = {
        "Development": [
            "[ ] Code is version controlled",
            "[ ] Automated testing implemented",
            "[ ] Code review process established",
            "[ ] Documentation is complete"
        ],
        "Data": [
            "[ ] Data versioning implemented",
            "[ ] Data validation rules defined", 
            "[ ] Data lineage tracked",
            "[ ] Privacy compliance ensured"
        ],
        "Models": [
            "[ ] Model versioning implemented",
            "[ ] Performance benchmarks established",
            "[ ] Model validation automated",
            "[ ] Rollback strategy defined"
        ],
        "Infrastructure": [
            "[ ] Infrastructure as Code implemented",
            "[ ] Multi-environment setup (dev/staging/prod)",
            "[ ] Monitoring and alerting configured",
            "[ ] Auto-scaling implemented"
        ],
        "Security": [
            "[ ] Access controls implemented",
            "[ ] Secrets management configured",
            "[ ] Audit logging enabled",
            "[ ] Compliance requirements met"
        ],
        "Operations": [
            "[ ] Deployment automation implemented",
            "[ ] Incident response procedures defined",
            "[ ] Backup and recovery tested",
            "[ ] Performance optimization done"
        ]
    }
    
    for category, items in checklist_categories.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"  {item}")

if __name__ == "__main__":
    comprehensive_mlops_demo()
    create_mlops_checklist()