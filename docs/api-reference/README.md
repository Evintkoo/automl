# 📋 API Reference

Complete API documentation for AutoML - integrate machine learning capabilities into your applications.

## 🎯 Overview

The AutoML API provides comprehensive endpoints for:
- 🚂 **Model Training** - Train ML models with automated hyperparameter optimization
- ⚡ **Inference** - High-performance prediction services  
- 🔄 **Data Processing** - Advanced data preprocessing and transformation
- 📦 **Batch Operations** - Efficient bulk processing with priority queues
- 🗄️ **Model Management** - Secure model storage, versioning, and deployment
- 📊 **Monitoring** - Real-time metrics and system health

## 🚀 Quick Start

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### Interactive Documentation
- **Swagger UI**: `/docs` - Interactive API explorer
- **ReDoc**: `/redoc` - Alternative documentation interface
- **OpenAPI Spec**: `/openapi.json` - Machine-readable API specification

### Authentication

Most endpoints require API key authentication:

```bash
# Get your API key from .env file or administrator
export API_KEY="automl_your_api_key_here"

# Use in requests
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/models
```

## 📚 Table of Contents

1. [🔐 Authentication](#-authentication)
2. [🚂 Training Endpoints](#-training-endpoints)
3. [⚡ Inference Endpoints](#-inference-endpoints) 
4. [🔄 Data Processing](#-data-processing)
5. [📦 Batch Operations](#-batch-operations)
6. [🗄️ Model Management](#️-model-management)
7. [📊 System & Monitoring](#-system--monitoring)
8. [🔧 Configuration](#-configuration)
9. [❌ Error Handling](#-error-handling)
10. [📖 Examples](#-examples)

## 🔐 Authentication

### API Key Authentication

```bash
# Required header for most endpoints
X-API-Key: automl_your_api_key_here
```

### JWT Authentication (Advanced)

```bash
# Login to get JWT token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# Use JWT token
curl -H "Authorization: Bearer your_jwt_token" \
  http://localhost:8000/api/protected-endpoint
```

### Health Check (No Auth Required)

```bash
GET /health
```

```json
{
  "status": "healthy",
  "version": "0.1.4",
  "timestamp": "2025-01-15T12:00:00Z",
  "uptime": "2 hours, 15 minutes",
  "environment": "production"
}
```

## 🚂 Training Endpoints

### Start Model Training

Train a new machine learning model with automated optimization.

```bash
POST /api/train-engine/train
```

**Request Body:**
```json
{
  "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "target": [0, 1, 0],
  "task_type": "classification",
  "model_type": "random_forest",
  "optimization_strategy": "bayesian",
  "config": {
    "cv_folds": 5,
    "max_iter": 1000,
    "enable_automl": true,
    "test_size": 0.2
  }
}
```

**Response:**
```json
{
  "job_id": "train_job_123456",
  "status": "started",
  "message": "Model training initiated",
  "estimated_duration": "5-15 minutes",
  "config": {
    "task_type": "classification",
    "optimization_strategy": "bayesian",
    "cv_folds": 5
  }
}
```

### Training Status

Check the status of a training job.

```bash
GET /api/train-engine/status/{job_id}
```

**Response:**
```json
{
  "job_id": "train_job_123456",
  "status": "training",
  "progress": 65,
  "current_step": "hyperparameter_optimization",
  "best_score": 0.847,
  "trials_completed": 13,
  "estimated_remaining": "3 minutes",
  "metrics": {
    "accuracy": 0.847,
    "precision": 0.823,
    "recall": 0.867,
    "f1_score": 0.844
  }
}
```

### Get Training Results

Retrieve results from completed training job.

```bash
GET /api/train-engine/results/{job_id}
```

**Response:**
```json
{
  "job_id": "train_job_123456",
  "status": "completed",
  "model_id": "model_789abc",
  "best_model": "RandomForestClassifier",
  "best_params": {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5
  },
  "performance_metrics": {
    "accuracy": 0.892,
    "precision": 0.885,
    "recall": 0.897,
    "f1_score": 0.891,
    "cross_val_scores": [0.89, 0.91, 0.88, 0.90, 0.89]
  },
  "feature_importance": {
    "feature_0": 0.35,
    "feature_1": 0.28,
    "feature_2": 0.37
  },
  "training_time": "8 minutes 23 seconds"
}
```

### Upload Dataset for Training

Upload a dataset file for training.

```bash
POST /api/train-engine/upload-dataset
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -F "file=@dataset.csv" \
  -F "target_column=target" \
  -F "task_type=classification" \
  http://localhost:8000/api/train-engine/upload-dataset
```

**Response:**
```json
{
  "dataset_id": "dataset_456def",
  "filename": "dataset.csv",
  "rows": 1000,
  "columns": 15,
  "target_column": "target",
  "task_type": "classification",
  "preview": {
    "head": [...],
    "dtypes": {...},
    "missing_values": {...}
  }
}
```

## ⚡ Inference Endpoints

### Single Prediction

Make a prediction using a trained model.

```bash
POST /api/inference/predict/{model_id}
```

**Request Body:**
```json
{
  "data": [[1, 2, 3, 4, 5]],
  "return_probabilities": true,
  "return_confidence": true
}
```

**Response:**
```json
{
  "predictions": [1],
  "probabilities": [[0.23, 0.77]],
  "confidence_scores": [0.77],
  "model_id": "model_789abc",
  "prediction_time_ms": 5.2,
  "request_id": "pred_123xyz"
}
```

### Batch Prediction

Process multiple predictions efficiently.

```bash
POST /api/inference/predict-batch/{model_id}
```

**Request Body:**
```json
{
  "data": [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
  ],
  "batch_size": 32,
  "priority": "high"
}
```

**Response:**
```json
{
  "predictions": [1, 0, 1],
  "probabilities": [
    [0.23, 0.77],
    [0.82, 0.18], 
    [0.34, 0.66]
  ],
  "batch_stats": {
    "total_samples": 3,
    "processing_time_ms": 12.4,
    "throughput_per_second": 242,
    "batch_efficiency": 0.94
  }
}
```

### Async Prediction

Submit prediction job for asynchronous processing.

```bash
POST /api/inference/predict-async/{model_id}
```

**Request Body:**
```json
{
  "data": [[1, 2, 3, 4, 5]],
  "callback_url": "https://your-app.com/prediction-callback",
  "priority": "normal"
}
```

**Response:**
```json
{
  "job_id": "async_pred_456",
  "status": "queued",
  "estimated_completion": "30 seconds",
  "position_in_queue": 5
}
```

### Get Async Results

Retrieve results from asynchronous prediction.

```bash
GET /api/inference/results/{job_id}
```

**Response:**
```json
{
  "job_id": "async_pred_456", 
  "status": "completed",
  "predictions": [1],
  "probabilities": [[0.23, 0.77]],
  "completed_at": "2025-01-15T12:05:30Z",
  "processing_time_ms": 8.7
}
```

## 🔄 Data Processing

### Preprocess Data

Clean and prepare data for training or inference.

```bash
POST /api/data-processor/preprocess
```

**Request Body:**
```json
{
  "data": [[1, 2, null], [4, null, 6], [7, 8, 9]],
  "config": {
    "handle_missing": "impute",
    "normalize": "standard",
    "remove_outliers": true,
    "feature_selection": true
  }
}
```

**Response:**
```json
{
  "processed_data": [[0.1, 0.2, 0.0], [-0.5, 0.0, 0.8], [1.2, 1.1, 1.3]],
  "transformations_applied": [
    "missing_value_imputation",
    "standard_scaling", 
    "outlier_removal"
  ],
  "feature_info": {
    "original_features": 3,
    "selected_features": 3,
    "removed_outliers": 1
  },
  "processing_time_ms": 15.3
}
```

### Data Validation

Validate data quality and format.

```bash
POST /api/data-processor/validate
```

**Request Body:**
```json
{
  "data": [[1, 2, 3], [4, 5, 6]],
  "schema": {
    "type": "numerical",
    "required_columns": 3,
    "allow_missing": false
  }
}
```

**Response:**
```json
{
  "valid": true,
  "validation_results": {
    "schema_compliance": true,
    "missing_values": 0,
    "data_types_correct": true,
    "outliers_detected": 0
  },
  "data_quality_score": 0.95,
  "recommendations": []
}
```

## 📦 Batch Operations

### Submit Batch Job

Process large datasets efficiently with priority queues.

```bash
POST /api/batch/submit
```

**Request Body:**
```json
{
  "operation": "training",
  "data": "s3://bucket/large-dataset.csv",
  "config": {
    "batch_size": 1000,
    "parallel_workers": 4,
    "priority": "high"
  },
  "callback_url": "https://your-app.com/batch-callback"
}
```

**Response:**
```json
{
  "batch_job_id": "batch_789xyz",
  "status": "queued",
  "estimated_start_time": "2025-01-15T12:10:00Z",
  "queue_position": 2,
  "estimated_duration": "2 hours"
}
```

### Batch Job Status

Check status of batch processing job.

```bash
GET /api/batch/status/{batch_job_id}
```

**Response:**
```json
{
  "batch_job_id": "batch_789xyz",
  "status": "processing",
  "progress": {
    "completed_batches": 45,
    "total_batches": 120,
    "percentage": 37.5
  },
  "current_throughput": "1500 samples/second",
  "estimated_remaining": "1 hour 15 minutes",
  "resource_usage": {
    "cpu_percent": 78,
    "memory_mb": 2048,
    "active_workers": 4
  }
}
```

## 🗄️ Model Management

### List Models

Get all available models.

```bash
GET /api/models
```

**Query Parameters:**
- `limit`: Number of models to return (default: 50)
- `offset`: Pagination offset (default: 0)
- `task_type`: Filter by task type (classification, regression)
- `status`: Filter by status (active, archived)

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_789abc",
      "name": "iris_classifier",
      "task_type": "classification", 
      "algorithm": "RandomForestClassifier",
      "accuracy": 0.892,
      "created_at": "2025-01-15T10:30:00Z",
      "status": "active",
      "file_size_mb": 2.3
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get Model Details

Retrieve detailed information about a specific model.

```bash
GET /api/models/{model_id}
```

**Response:**
```json
{
  "model_id": "model_789abc",
  "name": "iris_classifier",
  "algorithm": "RandomForestClassifier",
  "task_type": "classification",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5
  },
  "performance_metrics": {
    "accuracy": 0.892,
    "precision": 0.885,
    "recall": 0.897,
    "f1_score": 0.891
  },
  "feature_info": {
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "feature_importance": [0.35, 0.15, 0.28, 0.22]
  },
  "metadata": {
    "created_at": "2025-01-15T10:30:00Z",
    "training_duration": "8 minutes 23 seconds", 
    "data_size": 150,
    "version": "1.0",
    "file_size_mb": 2.3
  }
}
```

### Upload Model

Upload a pre-trained model.

```bash
POST /api/models/upload
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -F "model_file=@model.pkl" \
  -F "name=my_custom_model" \
  -F "task_type=classification" \
  -F "algorithm=custom" \
  http://localhost:8000/api/models/upload
```

### Delete Model

Remove a model from the registry.

```bash
DELETE /api/models/{model_id}
```

**Response:**
```json
{
  "message": "Model model_789abc deleted successfully",
  "model_id": "model_789abc",
  "deleted_at": "2025-01-15T12:15:00Z"
}
```

## 📊 System & Monitoring

### System Health

Get comprehensive system health information.

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.4",
  "environment": "production",
  "uptime": "2 hours, 15 minutes",
  "timestamp": "2025-01-15T12:00:00Z",
  "services": {
    "api": "healthy",
    "redis": "healthy", 
    "database": "healthy",
    "model_storage": "healthy"
  },
  "system_metrics": {
    "cpu_percent": 45.2,
    "memory_percent": 68.7,
    "disk_usage_percent": 23.1,
    "active_connections": 12
  }
}
```

### Performance Metrics

Get detailed performance metrics.

```bash
GET /api/metrics
```

**Response:**
```json
{
  "api_metrics": {
    "total_requests": 15847,
    "requests_per_second": 23.4,
    "average_response_time_ms": 145.7,
    "error_rate_percent": 0.02
  },
  "ml_metrics": {
    "models_trained": 45,
    "predictions_made": 125847,
    "active_training_jobs": 2,
    "average_training_time_minutes": 12.5
  },
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage_mb": 2048,
    "disk_usage_gb": 15.7,
    "network_io_mbps": 12.3
  },
  "cache_metrics": {
    "hit_rate_percent": 87.3,
    "cache_size_mb": 512,
    "evictions_per_hour": 145
  }
}
```

### Prometheus Metrics

Machine-readable metrics for monitoring systems.

```bash
GET /metrics
```

**Response (Prometheus format):**
```
# HELP automl_api_requests_total Total API requests
# TYPE automl_api_requests_total counter
automl_api_requests_total{method="GET",endpoint="/health"} 1523
automl_api_requests_total{method="POST",endpoint="/api/train"} 45

# HELP automl_model_training_duration_seconds Model training duration
# TYPE automl_model_training_duration_seconds histogram
automl_model_training_duration_seconds_bucket{le="300"} 12
automl_model_training_duration_seconds_bucket{le="600"} 28
```

## 🔧 Configuration

### Get System Configuration

Retrieve current system configuration.

```bash
GET /api/config
```

**Response:**
```json
{
  "api_config": {
    "environment": "production",
    "debug": false,
    "workers": 4,
    "max_request_size_mb": 100
  },
  "ml_config": {
    "default_optimization_strategy": "bayesian",
    "max_training_time_hours": 24,
    "enable_automl": true,
    "default_cv_folds": 5
  },
  "security_config": {
    "require_api_key": true,
    "rate_limit_enabled": true,
    "max_requests_per_minute": 100,
    "jwt_enabled": true
  },
  "performance_config": {
    "enable_caching": true,
    "batch_size": 32,
    "enable_jit_compilation": true,
    "max_concurrent_jobs": 10
  }
}
```

### Update Configuration

Update system configuration (admin only).

```bash
PATCH /api/config
```

**Request Body:**
```json
{
  "ml_config": {
    "default_cv_folds": 10,
    "enable_automl": true
  },
  "performance_config": {
    "batch_size": 64,
    "enable_caching": true
  }
}
```

## ❌ Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request data is invalid",
    "details": "Missing required field: 'data'",
    "timestamp": "2025-01-15T12:00:00Z",
    "request_id": "req_123abc"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request data |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `MODEL_NOT_FOUND` | 404 | Specified model doesn't exist |
| `TRAINING_FAILED` | 500 | Model training error |
| `PREDICTION_FAILED` | 500 | Prediction error |

### Rate Limiting

API requests are subject to rate limiting:

- **Default Limit**: 100 requests per minute per API key
- **Headers**: Rate limit info in response headers
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests  
  - `X-RateLimit-Reset`: Reset time (Unix timestamp)

**Rate Limit Exceeded Response:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": "Maximum 100 requests per minute allowed",
    "retry_after": 60
  }
}
```

## 📖 Examples

### Complete Training Workflow

```python
import requests
import json

api_url = "http://localhost:8000"
headers = {"X-API-Key": "your-api-key"}

# 1. Upload dataset
with open("iris.csv", "rb") as f:
    upload_response = requests.post(
        f"{api_url}/api/train-engine/upload-dataset",
        headers=headers,
        files={"file": f},
        data={"target_column": "species", "task_type": "classification"}
    )
    
dataset_id = upload_response.json()["dataset_id"]

# 2. Start training
training_request = {
    "dataset_id": dataset_id,
    "task_type": "classification",
    "optimization_strategy": "bayesian",
    "config": {
        "cv_folds": 5,
        "enable_automl": true
    }
}

train_response = requests.post(
    f"{api_url}/api/train-engine/train",
    headers=headers,
    json=training_request
)

job_id = train_response.json()["job_id"]

# 3. Monitor training
while True:
    status_response = requests.get(
        f"{api_url}/api/train-engine/status/{job_id}",
        headers=headers
    )
    
    status_data = status_response.json()
    if status_data["status"] == "completed":
        break
    elif status_data["status"] == "failed":
        print(f"Training failed: {status_data['error']}")
        break
        
    print(f"Progress: {status_data['progress']}%")
    time.sleep(10)

# 4. Get results
results = requests.get(
    f"{api_url}/api/train-engine/results/{job_id}",
    headers=headers
).json()

model_id = results["model_id"]
print(f"Training completed! Model ID: {model_id}")
print(f"Accuracy: {results['performance_metrics']['accuracy']:.3f}")

# 5. Make predictions
prediction_request = {
    "data": [[5.1, 3.5, 1.4, 0.2]],
    "return_probabilities": true
}

prediction = requests.post(
    f"{api_url}/api/inference/predict/{model_id}",
    headers=headers,
    json=prediction_request
).json()

print(f"Prediction: {prediction['predictions'][0]}")
print(f"Confidence: {prediction['confidence_scores'][0]:.3f}")
```

---

## 🚀 Ready to Integrate?

This API reference covers all endpoints available in AutoML. For more specific examples:

- 🐍 **[Python Examples](examples/python.md)** - Complete Python integration examples
- 🌐 **[JavaScript Examples](examples/javascript.md)** - Node.js and browser examples  
- ⚡ **[cURL Examples](examples/curl.md)** - Command-line examples

Need help? Check our [User Guides](../user-guides/) or [create an issue](https://github.com/Evintkoo/automl/issues) on GitHub.

*API Reference v1.0 | Last updated: January 2025 | AutoML v0.1.4*
