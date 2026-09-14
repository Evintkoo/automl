# 🐳 Docker Deployment Guide

This comprehensive guide covers deploying AutoML using Docker and Docker Compose with production-ready configurations, security features, and monitoring.

## 🎯 Quick Start (5 Minutes)

### Prerequisites
- **Docker Engine 20.10+** and **Docker Compose 2.0+**
- **8GB+ RAM** available for containers (4GB minimum)
- **20GB+ free disk space** (models + logs + monitoring data)

### Ultra-Fast Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/Evintkoo/automl.git
cd automl

# 2. Quick development setup
cp .env.example .env
docker-compose -f compose.yaml -f compose.dev.yaml up -d

# 3. Production setup (recommended)
./scripts/generate_secure_config.sh  # Linux/macOS
# OR
scripts\generate_secure_config.bat   # Windows

docker-compose up -d
```

### Verify Installation

```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Access web interface: http://localhost:7860
# Access API docs: http://localhost:8000/docs
```

## 📋 Available Services

| Service | Port | Purpose | Development | Production | Health Check |
|---------|------|---------|-------------|------------|--------------|
| **API Server** | 8000 | Main ML API | ✅ | ✅ | `/health` |
| **Web Interface** | 7860 | Gradio UI | ✅ | ✅ | Auto-check |
| **Redis** | 6379 | Caching & Sessions | ✅ | ✅ | `redis-cli ping` |
| **Nginx** | 80/443 | Reverse Proxy | ❌ | ✅ | Auto-configured |
| **Prometheus** | 9090 | Metrics Collection | ❌ | ✅ | `/-/healthy` |
| **Grafana** | 3000 | Monitoring Dashboard | ❌ | ✅ | `/api/health` |
| **PostgreSQL** | 5432 | Database (optional) | ✅ | Optional | `pg_isready` |

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "🌐 Frontend Layer"
        WEB[🖥️ Web Interface<br/>Port 7860]
        NGINX[🔧 Nginx<br/>Port 80/443]
    end

    subgraph "🔧 Application Layer"  
        API[🚀 AutoML API<br/>Port 8000]
        WORKER[👥 API Workers<br/>Scalable]
    end

    subgraph "💾 Data Layer"
        REDIS[⚡ Redis Cache<br/>Port 6379]
        POSTGRES[🗄️ PostgreSQL<br/>Port 5432]
        MODELS[(📦 Model Storage)]
    end

    subgraph "📊 Monitoring Layer"
        PROMETHEUS[📊 Prometheus<br/>Port 9090]
        GRAFANA[📈 Grafana<br/>Port 3000]
        LOKI[📝 Loki Logs<br/>Port 3100]
    end

    NGINX --> API
    NGINX --> WEB
    WEB --> API
    API --> REDIS
    API --> POSTGRES
    API --> MODELS
    
    PROMETHEUS --> API
    PROMETHEUS --> REDIS
    PROMETHEUS --> POSTGRES
    GRAFANA --> PROMETHEUS
    GRAFANA --> LOKI
    LOKI --> API

    classDef frontend fill:#e1f5fe,stroke:#01579b
    classDef app fill:#f3e5f5,stroke:#4a148c
    classDef data fill:#fff3e0,stroke:#e65100
    classDef monitoring fill:#e8f5e8,stroke:#1b5e20

    class WEB,NGINX frontend
    class API,WORKER app  
    class REDIS,POSTGRES,MODELS data
    class PROMETHEUS,GRAFANA,LOKI monitoring
```

## ⚙️ Environment Configuration

### Development Configuration

```bash
# Copy example environment
cp .env.example .env

# Key development settings in .env:
API_ENV=development
API_DEBUG=true
SECURITY_LEVEL=development
REQUIRE_API_KEY=false
ENABLE_CORS=true
```

### Production Configuration (Secure)

```bash
# Generate secure configuration automatically
./scripts/generate_secure_config.sh  # Linux/macOS
scripts\generate_secure_config.bat   # Windows

# Or manually set production values in .env:
API_ENV=production
API_DEBUG=false
SECURITY_LEVEL=production  
REQUIRE_API_KEY=true
ENFORCE_HTTPS=true
ENABLE_RATE_LIMITING=true
ENABLE_AUDIT_LOGGING=true
```

### Essential Environment Variables

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `API_ENV` | `development` | `production` | Environment mode |
| `API_DEBUG` | `true` | `false` | Debug logging |
| `REQUIRE_API_KEY` | `false` | `true` | API authentication |
| `SECURITY_LEVEL` | `development` | `production` | Security mode |
| `API_WORKERS` | `2` | `4-8` | Worker processes |
| `REDIS_PASSWORD` | Optional | **Required** | Redis authentication |
| `JWT_SECRET` | Default | **Generated** | JWT signing key |

## 🚀 Deployment Modes

### Development Mode

Perfect for local development with hot reload and debugging tools:

```bash
# Start development environment
docker-compose -f compose.yaml -f compose.dev.yaml up -d

# Development features:
# - Hot code reload
# - Debug logging
# - Development tools
# - No authentication required
# - Jupyter Lab included
```

### Production Mode  

Optimized for production with security, monitoring, and performance:

```bash
# Generate secure configuration
./scripts/generate_secure_config.sh

# Start production environment
docker-compose up -d

# Production features:
# - SSL/TLS encryption
# - API authentication
# - Rate limiting
# - Comprehensive monitoring
# - Security hardening
```

### Staging Mode

Balance between development and production:

```bash
# Start staging environment
docker-compose -f compose.yaml -f compose.staging.yaml up -d

# Staging features:
# - Production-like security
# - Debug endpoints enabled
# - Monitoring included
# - Test data persistence
```

## 🔒 Security Configuration

### SSL/TLS Setup

#### Option 1: Self-Signed Certificates (Development)

```bash
# Generate self-signed certificates
python generate_ssl.py

# Or manually with OpenSSL
mkdir -p certs
openssl req -x509 -newkey rsa:4096 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -days 365 -nodes
```

#### Option 2: Let's Encrypt (Production)

```bash
# Install certbot
sudo apt-get install certbot  # Ubuntu
brew install certbot          # macOS

# Generate certificates
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/server.crt
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/server.key
```

### API Security

```bash
# Generate secure API keys
python -c "
import secrets
keys = [f'automl_{secrets.token_urlsafe(32)}' for _ in range(3)]
print('API_KEYS=' + ','.join(keys))
"

# Generate JWT secret
python -c "
import secrets
print('JWT_SECRET=' + secrets.token_urlsafe(64))
"

# Set in .env file
echo "API_KEYS=automl_your_generated_keys_here" >> .env
echo "JWT_SECRET=your_generated_jwt_secret_here" >> .env
```

### Firewall Configuration

```bash
# Linux (UFW)
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS  
sudo ufw deny 8000/tcp   # Block direct API access
sudo ufw enable

# Windows Firewall (PowerShell as Admin)
New-NetFirewallRule -DisplayName "HTTP" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
New-NetFirewallRule -DisplayName "HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
New-NetFirewallRule -DisplayName "Block Direct API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Block
```

## 📊 Monitoring & Observability

### Access Monitoring Services

| Service | URL | Default Login | Purpose |
|---------|-----|---------------|---------|
| **Grafana** | http://localhost:3000 | admin/admin123 | Dashboards & Visualization |
| **Prometheus** | http://localhost:9090 | No auth | Metrics & Alerting |
| **API Metrics** | http://localhost:8000/metrics | No auth | Application metrics |

### Pre-Built Dashboards

**1. System Overview Dashboard**
- CPU, Memory, Disk usage
- Container health status
- Network throughput
- Database performance

**2. API Performance Dashboard**
- Request rates and response times
- Error rates by endpoint
- Authentication metrics
- Rate limiting events

**3. ML Model Dashboard**
- Model training metrics
- Batch processing performance
- Model accuracy trends
- Resource utilization

**4. Security Dashboard**
- Failed authentication attempts
- Rate limiting violations
- Suspicious request patterns
- Security audit events

### Custom Metrics

The API exposes custom metrics at `/metrics`:

```bash
# View available metrics
curl http://localhost:8000/metrics

# Key metrics include:
# - automl_api_requests_total
# - automl_model_training_duration_seconds  
# - automl_batch_processing_queue_size
# - automl_memory_usage_bytes
# - automl_active_models_count
```

## 🧪 Testing & Validation

### Automated Testing Suite

```bash
# Comprehensive Docker testing
python test_docker.py --mode production

# Quick development testing  
python test_docker.py --mode development --skip-build

# Configuration validation only
python validate_docker_config.py

# Production readiness check
python check_production_readiness.py
```

### Manual Testing

```bash
# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8000/api/health

# API functionality (with API key)
export API_KEY="your-api-key"
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/models

# Load testing
pip install locust
locust -f scripts/load_test.py --host http://localhost:8000
```

### Service Verification

```bash
# Check all services are running
docker-compose ps

# View service logs
docker-compose logs automl-api
docker-compose logs redis
docker-compose logs prometheus

# Resource usage
docker stats

# Network connectivity
docker network ls
docker network inspect automl-network
```

## 🔧 Scaling & Performance

### Horizontal Scaling

```bash
# Scale API containers
docker-compose up -d --scale automl-api=3

# Nginx automatically load balances requests
# Verify load balancing:
curl -H "X-Show-Instance: true" http://localhost/health
```

### Resource Optimization

```yaml
# In compose.yaml - adjust based on your hardware
services:
  automl-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  redis:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

### Performance Tuning

```bash
# Environment variables for performance
API_WORKERS=8                    # Scale based on CPU cores
BATCH_MAX_SIZE=128              # Larger batches for throughput
ENABLE_JIT_COMPILATION=true     # Performance optimization
ENABLE_MODEL_CACHING=true       # Cache models in memory
REDIS_MAXMEMORY=1gb             # Adjust cache size
```

## 🛠️ Maintenance & Operations

### Regular Maintenance Tasks

```bash
# View logs
docker-compose logs -f

# Update images
docker-compose pull
docker-compose up -d

# Backup data
docker-compose exec postgres pg_dump -U automl automl_prod > backup.sql

# Clean up resources
docker system prune -f
docker volume prune -f
```

### Backup Strategy

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/${DATE}"

mkdir -p ${BACKUP_DIR}
docker-compose exec -T postgres pg_dump -U automl automl_prod > ${BACKUP_DIR}/database.sql
cp -r models ${BACKUP_DIR}/
cp -r configs ${BACKUP_DIR}/
tar -czf backups/automl_backup_${DATE}.tar.gz -C backups ${DATE}
rm -rf ${BACKUP_DIR}
EOF

chmod +x backup.sh

# Schedule with cron (Linux/macOS)
(crontab -l; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

## ❓ Troubleshooting

### Common Issues

#### 1. **Containers Won't Start**

```bash
# Check logs
docker-compose logs

# Check disk space
df -h

# Check port conflicts
netstat -tulpn | grep :8000  # Linux
netstat -an | findstr :8000  # Windows

# Clean and rebuild
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

#### 2. **API Not Responding**

```bash
# Check API container
docker-compose logs automl-api

# Verify container is running
docker-compose ps automl-api

# Test internal connectivity
docker-compose exec automl-api curl localhost:8000/health

# Restart API service
docker-compose restart automl-api
```

#### 3. **Memory Issues**

```bash
# Check resource usage
docker stats

# Increase memory limits in compose.yaml
# Restart services
docker-compose down
docker-compose up -d
```

#### 4. **Database Connection Issues**

```bash
# Check PostgreSQL status
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U automl

# Reset database
docker-compose down
docker volume rm automl_postgres-data
docker-compose up -d
```

### Getting Help

For issues not covered here:

1. **Check logs**: `docker-compose logs [service]`
2. **Verify configuration**: `python validate_docker_config.py`
3. **Run diagnostics**: `python check_production_readiness.py`
4. **GitHub Issues**: Report bugs with logs and configuration
5. **Documentation**: Review related guides in this documentation

## 📚 Advanced Topics

### Custom Docker Images

```dockerfile
# Extend the base image
FROM automl:latest

# Add custom dependencies
COPY requirements-custom.txt .
RUN pip install -r requirements-custom.txt

# Add custom configurations
COPY custom-config.yaml /app/config/
```

### Integration with CI/CD

```yaml
# .github/workflows/docker.yml
name: Docker Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and test
        run: |
          docker-compose build
          docker-compose up -d
          python test_docker.py --mode production
          docker-compose down
```

### Kubernetes Deployment

```bash
# Generate Kubernetes manifests
docker-compose convert > automl-k8s.yaml

# Deploy to Kubernetes
kubectl apply -f automl-k8s.yaml

# Or use Helm chart (if available)
helm install automl ./helm-chart
```

## 🎉 Success! You're Production Ready

Once your deployment is running:

✅ **Scalable Architecture** - Load-balanced API with auto-scaling  
✅ **Enterprise Security** - Authentication, encryption, audit logging  
✅ **Comprehensive Monitoring** - Metrics, dashboards, alerting  
✅ **Automated Operations** - Health checks, backups, updates  
✅ **High Performance** - Optimized for speed and efficiency

### Access Your Services

- 🌐 **Web Interface**: http://localhost:7860
- 🔌 **API Documentation**: http://localhost:8000/docs  
- 📊 **Monitoring Dashboard**: http://localhost:3000
- 📈 **Metrics**: http://localhost:9090

### Next Steps

- 🔒 **[Security Hardening](security.md)** - Advanced security configuration
- 📊 **[Monitoring Setup](monitoring.md)** - Advanced observability
- ⚡ **[Performance Tuning](../technical/performance.md)** - Optimization guide
- 🔧 **[Production Guide](production.md)** - Production best practices

---

**Need help?** Check our [Troubleshooting Guide](troubleshooting.md) or [create an issue](https://github.com/Evintkoo/automl/issues) on GitHub.

*Docker Deployment Guide v1.0 | Last updated: January 2025*
