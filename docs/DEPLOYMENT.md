# Deployment Guide

## Overview

This guide covers deployment options for the UAP Analysis Suite in various environments, from local installations to enterprise deployments.

## Deployment Options

### 1. Local Development Deployment

#### Prerequisites
- Python 3.8+ installed
- Git for source management
- Sufficient disk space (10GB+)

#### Installation Steps
```bash
# Clone repository
git clone https://github.com/sanchez314c/UAP-Analysis.git
cd UAP-Analysis

# Create virtual environment
python -m venv uap_env
source uap_env/bin/activate  # Linux/macOS
# or
uap_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install platform-specific dependencies
pip install -r requirements-macos.txt  # macOS
# or
pip install -r requirements-linux.txt  # Linux

# Verify installation
python scripts/test_setup.py
```

### 2. Production Deployment

#### Single Machine Deployment

**System Requirements**
- OS: macOS 10.15+, Ubuntu 18.04+, Windows 10
- CPU: 8+ cores recommended
- RAM: 32GB+ for large video analysis
- GPU: NVIDIA GTX 1060+ or Apple Silicon M1+
- Storage: SSD with 100GB+ free space

**Deployment Steps**
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash uap_analyzer
sudo su - uap_analyzer

# Install in /opt directory
sudo mkdir -p /opt/uap_analysis
sudo chown uap_analyzer:uap_analyzer /opt/uap_analysis

# Deploy from source
cd /opt/uap_analysis
git clone https://github.com/sanchez314c/UAP-Analysis.git .
python -m venv production_env
source production_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-linux.txt  # or requirements-macos.txt

# Create systemd service (Linux)
sudo tee /etc/systemd/system/uap-analysis.service > /dev/null <<EOF
[Unit]
Description=UAP Analysis Service
After=network.target

[Service]
Type=simple
User=uap_analyzer
WorkingDirectory=/opt/uap_analysis
Environment=PATH=/opt/uap_analysis/production_env/bin
ExecStart=/opt/uap_analysis/production_env/bin/python scripts/uap_gui.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable uap-analysis
sudo systemctl start uap-analysis
```

#### Multi-Node Cluster Deployment

**Architecture**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Node 1    │    │   Node 2    │    │   Node N    │
│  (Master)    │    │ (Worker)     │    │ (Worker)     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                  ┌─────────────┐
                  │  Shared     │
                  │  Storage    │
                  └─────────────┘
```

**Setup Steps**
```bash
# On master node
export UAP_MASTER=true
export UAP_NODE_ID=master
export UAP_SHARED_STORAGE=/mnt/uap_data

# On worker nodes
export UAP_MASTER=false
export UAP_NODE_ID=worker1
export UAP_MASTER_IP=192.168.1.100
export UAP_SHARED_STORAGE=/mnt/uap_data

# Configure distributed processing
cat > configs/distributed_config.yaml <<EOF
distributed:
  enabled: true
  master_node: "192.168.1.100"
  shared_storage: "/mnt/uap_data"
  worker_nodes:
    - "192.168.1.101"
    - "192.168.1.102"
    - "192.168.1.103"
  
performance:
  parallel_workers: 8
  chunk_size: 1000
  memory_limit: "16GB"
EOF
```

### 3. Container Deployment

#### Docker Deployment

**Dockerfile**
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-linux.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 uap && chown -R uap:uap /app
USER uap

# Expose GUI port (if using web interface)
EXPOSE 8080

# Default command
CMD ["python", "scripts/uap_gui.py"]
```

**Docker Compose**
```yaml
version: '3.8'

services:
  uap-analysis:
    build: .
    container_name: uap-analysis
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./configs:/app/configs
    environment:
      - UAP_DATA_DIR=/app/data
      - UAP_RESULTS_DIR=/app/results
      - UAP_CONFIG_DIR=/app/configs
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Deployment Commands**
```bash
# Build image
docker build -t uap-analysis:latest .

# Run container
docker run -it \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  uap-analysis:latest

# Or with docker-compose
docker-compose up -d
```

#### Kubernetes Deployment

**Deployment Manifest**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uap-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: uap-analysis
  template:
    metadata:
      labels:
        app: uap-analysis
    spec:
      containers:
      - name: uap-analysis
        image: uap-analysis:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: results-volume
          mountPath: /app/results
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: uap-data-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: uap-results-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: uap-analysis-service
spec:
  selector:
    app: uap-analysis
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### 4. Cloud Deployment

#### AWS Deployment

**EC2 Instance Setup**
```bash
# Launch GPU-enabled instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge \
  --key-name uap-analysis-key \
  --security-group-ids sg-903004f8 \
  --subnet-id subnet-6e7f829e \
  --user-data file://cloud-init.txt

# cloud-init.txt
#!/bin/bash
yum update -y
yum install -y git python3 python3-pip
cd /opt
git clone https://github.com/sanchez314c/UAP-Analysis.git
cd UAP-Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-linux.txt
```

**S3 Integration**
```python
# Configure S3 for data storage
import boto3
from src.utils import CloudStorage

s3_client = boto3.client('s3')
storage = CloudStorage(
    provider='aws',
    bucket='uap-analysis-data',
    region='us-west-2'
)

# Upload results
storage.upload_file('analysis_results.json', 'results/analysis_results.json')
```

#### Azure Deployment

**VM Configuration**
```bash
# Create GPU-enabled VM
az vm create \
  --resource-group uap-analysis-rg \
  --name uap-analysis-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install dependencies
ssh azureuser@uap-analysis-vm
# Follow local deployment steps
```

#### GCP Deployment

**Compute Engine Setup**
```bash
# Create GPU instance
gcloud compute instances create uap-analysis-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-k80,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# Install and deploy
gcloud compute ssh uap-analysis-vm
# Follow local deployment steps
```

### 5. Enterprise Deployment

#### Active Directory Integration

```python
# LDAP/AD authentication
from src.auth import LDAPAuthenticator

auth = LDAPAuthenticator(
    server='ldap://corp.company.com',
    base_dn='dc=corp,dc=company,dc=com',
    user_group='uap_analysts'
)

# Integrate with GUI
gui = UAPAnalyzerGUI(authenticator=auth)
```

#### Network Configuration

```yaml
# enterprise_network_config.yaml
network:
  proxy:
    http_proxy: "http://proxy.company.com:8080"
    https_proxy: "http://proxy.company.com:8080"
    no_proxy: "localhost,127.0.0.1,.company.com"
  
  firewall:
    allowed_ports: [8080, 22]
    outbound_rules: strict
  
  vpn:
    required: true
    config_file: "/etc/openvpn/client.conf"
```

#### Monitoring and Logging

```yaml
# monitoring_config.yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"
  
  elk_stack:
    enabled: true
    logstash_host: "logstash.company.com"
    elasticsearch_url: "https://elasticsearch.company.com:9200"
  
  alerts:
    email: "admin@company.com"
    slack_webhook: "https://hooks.slack.com/..."
```

## Configuration Management

### Environment Variables

```bash
# Core configuration
export UAP_CONFIG_DIR="/etc/uap_analysis"
export UAP_DATA_DIR="/var/lib/uap_analysis"
export UAP_LOG_DIR="/var/log/uap_analysis"
export UAP_RESULTS_DIR="/var/uap_results"

# Performance tuning
export UAP_GPU_MEMORY_FRACTION="0.8"
export UAP_PARALLEL_WORKERS="8"
export UAP_CACHE_SIZE="4GB"

# Security
export UAP_ENABLE_AUDIT_LOG="true"
export UAP_ENCRYPTION_KEY_FILE="/etc/uap_analysis/key.pem"
```

### Configuration Files

```yaml
# /etc/uap_analysis/production_config.yaml
production:
  debug_mode: false
  log_level: "INFO"
  max_video_size: "10GB"
  
security:
  enable_authentication: true
  session_timeout: 3600
  audit_all_operations: true
  
performance:
  gpu_acceleration: true
  memory_limit: "32GB"
  parallel_processing: true
  
storage:
  backend: "filesystem"
  path: "/var/uap_results"
  backup_enabled: true
  backup_retention: 30
```

## Monitoring and Maintenance

### Health Checks

```python
# Health check endpoint
from src.utils import HealthChecker

health = HealthChecker()
status = health.check_all()

if not status['healthy']:
    # Send alert
    alert_manager.send_alert(
        severity='critical',
        message=f"UAP Analysis health check failed: {status}"
    )
```

### Performance Monitoring

```bash
# Monitor resource usage
watch -n 1 '
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

echo "=== Memory Usage ==="
free -h

echo "=== CPU Usage ==="
top -bn1 | grep "Cpu(s)" | awk "{print \$2}" | cut -d"%" -f1
'
```

### Log Management

```bash
# Log rotation configuration
cat > /etc/logrotate.d/uap-analysis <<EOF
/var/log/uap_analysis/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 uap_analyzer uap_analyzer
    postrotate
        systemctl reload uap-analysis
    endscript
}
EOF
```

## Backup and Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# backup_uap_data.sh

BACKUP_DIR="/backup/uap_analysis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" \
  /var/lib/uap_analysis \
  /var/uap_results \
  /etc/uap_analysis

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/data_$DATE.tar.gz" \
  s3://uap-analysis-backups/

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

```bash
# Recovery procedure
#!/bin/bash
# recover_uap.sh

# Stop services
sudo systemctl stop uap-analysis

# Restore from backup
aws s3 cp s3://uap-analysis-backups/data_20250131_120000.tar.gz ./
tar -xzf data_20250131_120000.tar.gz -C /

# Verify installation
python scripts/test_setup.py

# Start services
sudo systemctl start uap-analysis
```

## Security Considerations

### Network Security

```yaml
# security_hardening.yaml
network_security:
  firewall_rules:
    - port: 22
      source: "admin_network"
      action: "allow"
    - port: 8080
      source: "internal_network"
      action: "allow"
    - default: "deny"
  
  tls:
    enabled: true
    cert_file: "/etc/ssl/certs/uap-analysis.crt"
    key_file: "/etc/ssl/private/uap-analysis.key"
  
  authentication:
    method: "ldap"
    multi_factor: true
    session_encryption: true
```

### Data Encryption

```python
# Encrypt sensitive data
from src.utils import EncryptionManager

encryption = EncryptionManager(
    key_file='/etc/uap_analysis/encryption.key'
)

# Encrypt results
encrypted_results = encryption.encrypt_data(analysis_results)

# Store encrypted data
storage.store_encrypted(encrypted_results, 'results.enc')
```

## Troubleshooting

### Common Deployment Issues

1. **GPU Not Detected**
   ```bash
   # Check GPU availability
   nvidia-smi
   # Verify CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R uap_analyzer:uap_analyzer /var/lib/uap_analysis
   sudo chmod 755 /opt/uap_analysis
   ```

3. **Memory Issues**
   ```bash
   # Configure memory limits
   export UAP_MEMORY_LIMIT="16GB"
   # Or in config
   echo "memory_limit: 16GB" >> /etc/uap_analysis/config.yaml
   ```

4. **Network Connectivity**
   ```bash
   # Test connectivity
   ping -c 3 storage.company.com
   # Check proxy settings
   echo $http_proxy
   echo $https_proxy
   ```

---

For specific deployment scenarios or additional support, see the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide or contact the deployment team.