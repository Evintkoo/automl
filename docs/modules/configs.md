# Configuration System (`modules/configs.py`)

## Overview

The configuration system provides type-safe configuration classes for all system components in AutoML. Built using Python dataclasses and enums, it ensures consistent and validated configuration across all modules.

## Features

- **Type-safe configurations** using dataclasses and enums
- **Serialization/deserialization** support with JSON compatibility
- **Configuration validation** with automatic type checking
- **Nested configuration** support for complex components
- **Default value management** with sensible defaults
- **Environment variable integration** for runtime configuration

## Configuration Classes

### Core Configuration Types

#### TaskType
```python
class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
```

#### OptimizationMode
```python
class OptimizationMode(Enum):
    PERFORMANCE = "performance"
    MEMORY = "memory"
    BALANCED = "balanced"
    ACCURACY = "accuracy"
```

### Quantization Configuration

#### QuantizationConfig
Complete configuration for model quantization:

```python
@dataclass
class QuantizationConfig:
    quantization_type: QuantizationType = QuantizationType.INT8
    quantization_mode: QuantizationMode = QuantizationMode.DYNAMIC_PER_BATCH
    per_channel: bool = False
    symmetric: bool = True
    enable_cache: bool = True
    cache_size: int = 256
    calibration_samples: int = 100
    calibration_method: str = "percentile"
    percentile: float = 99.99
    skip_layers: List[str] = field(default_factory=list)
```

**Parameters:**
- `quantization_type`: Type of quantization (INT8, UINT8, INT16, FLOAT16, MIXED, NONE)
- `quantization_mode`: Mode of quantization (STATIC, DYNAMIC, DYNAMIC_PER_BATCH, etc.)
- `per_channel`: Enable per-channel quantization
- `symmetric`: Use symmetric quantization
- `enable_cache`: Enable quantization cache
- `cache_size`: Size of quantization cache
- `calibration_samples`: Number of samples for calibration
- `calibration_method`: Method for calibration (percentile, minmax, etc.)
- `percentile`: Percentile value for calibration
- `skip_layers`: List of layer names to skip during quantization

### Batch Processing Configuration

#### BatchProcessorConfig
Configuration for batch processing operations:

```python
@dataclass
class BatchProcessorConfig:
    max_batch_size: int = 32
    timeout_seconds: float = 30.0
    max_queue_size: int = 1000
    enable_dynamic_batching: bool = True
    batch_timeout_ms: int = 50
    max_workers: int = 4
    enable_compression: bool = False
    compression_level: int = 6
    enable_metrics: bool = True
    metrics_interval: int = 60
```

### Data Preprocessing Configuration

#### PreprocessorConfig
Configuration for data preprocessing operations:

```python
@dataclass
class PreprocessorConfig:
    normalize_features: bool = True
    normalization_method: str = "standard"
    handle_missing_values: bool = True
    missing_value_strategy: str = "mean"
    remove_outliers: bool = False
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    feature_selection: bool = False
    feature_selection_method: str = "variance"
    feature_selection_threshold: float = 0.1
    enable_feature_engineering: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = False
```

### Inference Engine Configuration

#### InferenceEngineConfig
Configuration for inference engine operations:

```python
@dataclass
class InferenceEngineConfig:
    enable_batching: bool = True
    batch_size: int = 32
    enable_cache: bool = True
    cache_size: int = 1024
    enable_jit: bool = True
    enable_mixed_precision: bool = False
    enable_quantization: bool = False
    quantization_config: Optional[QuantizationConfig] = None
    max_workers: int = 4
    timeout_seconds: float = 30.0
    enable_metrics: bool = True
    enable_profiling: bool = False
```

### Training Engine Configuration

#### MLTrainingEngineConfig
Main configuration for the ML training engine:

```python
@dataclass
class MLTrainingEngineConfig:
    # Task configuration
    task_type: TaskType = TaskType.CLASSIFICATION
    enable_automl: bool = True
    
    # Training parameters
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    
    # Optimization
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
    enable_hyperparameter_tuning: bool = True
    hyperparameter_tuning_trials: int = 100
    
    # Performance optimizations
    enable_jit: bool = True
    enable_mixed_precision: bool = False
    enable_quantization: bool = False
    quantization_config: Optional[QuantizationConfig] = None
    
    # Resource management
    max_workers: int = 4
    memory_limit_gb: Optional[float] = None
    
    # Experiment tracking
    enable_experiment_tracking: bool = True
    experiment_name: Optional[str] = None
    
    # Model management
    enable_model_registry: bool = True
    model_registry_path: str = MODEL_REGISTRY_PATH
    enable_checkpoints: bool = True
    checkpoint_path: str = CHECKPOINT_PATH
```

## Usage Examples

### Basic Configuration
```python
from modules.configs import MLTrainingEngineConfig, TaskType, OptimizationMode

# Create basic configuration
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    enable_automl=True,
    cv_folds=5,
    max_iter=1000
)
```

### Advanced Configuration with Quantization
```python
from modules.configs import (
    MLTrainingEngineConfig, QuantizationConfig,
    TaskType, QuantizationType, QuantizationMode
)

# Create quantization config
quant_config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    quantization_mode=QuantizationMode.DYNAMIC_PER_BATCH,
    symmetric=True,
    calibration_samples=100
)

# Create training config with quantization
config = MLTrainingEngineConfig(
    task_type=TaskType.CLASSIFICATION,
    enable_automl=True,
    enable_quantization=True,
    quantization_config=quant_config,
    optimization_mode=OptimizationMode.PERFORMANCE
)
```

### Configuration Serialization
```python
import json
from modules.configs import MLTrainingEngineConfig

# Create configuration
config = MLTrainingEngineConfig()

# Serialize to dictionary
config_dict = asdict(config)

# Serialize to JSON
config_json = json.dumps(config_dict, indent=2)

# Deserialize from dictionary
new_config = MLTrainingEngineConfig(**config_dict)
```

### Environment Variable Integration
```python
import os
from modules.configs import MLTrainingEngineConfig

# Override defaults with environment variables
config = MLTrainingEngineConfig(
    max_workers=int(os.getenv('ML_MAX_WORKERS', '4')),
    memory_limit_gb=float(os.getenv('ML_MEMORY_LIMIT', '8.0')),
    enable_automl=os.getenv('ML_ENABLE_AUTOML', 'true').lower() == 'true'
)
```

## Configuration Validation

The configuration system includes automatic validation:

```python
from modules.configs import MLTrainingEngineConfig, TaskType

try:
    config = MLTrainingEngineConfig(
        task_type=TaskType.CLASSIFICATION,
        cv_folds=5,  # Valid
        test_size=0.2,  # Valid
        max_iter=1000  # Valid
    )
except (ValueError, TypeError) as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Use Type Hints
Always use the provided enum types for type safety:
```python
# Good
config.task_type = TaskType.CLASSIFICATION

# Avoid
config.task_type = "classification"
```

### 2. Validate Configurations
Always validate configurations before use:
```python
def validate_config(config: MLTrainingEngineConfig) -> bool:
    if config.cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    return True
```

### 3. Use Configuration Templates
Create configuration templates for common use cases:
```python
# Performance-optimized configuration
PERFORMANCE_CONFIG = MLTrainingEngineConfig(
    optimization_mode=OptimizationMode.PERFORMANCE,
    enable_jit=True,
    enable_mixed_precision=True,
    max_workers=8
)

# Memory-optimized configuration
MEMORY_CONFIG = MLTrainingEngineConfig(
    optimization_mode=OptimizationMode.MEMORY,
    enable_quantization=True,
    quantization_config=QuantizationConfig(
        quantization_type=QuantizationType.INT8
    ),
    max_workers=2
)
```

## Integration with Other Modules

The configuration system integrates seamlessly with all other modules:

```python
# Training Engine
from modules.engine.train_engine import MLTrainingEngine
engine = MLTrainingEngine(config)

# Inference Engine
from modules.engine.inference_engine import InferenceEngine
inference = InferenceEngine(config.to_inference_config())

# Data Preprocessor
from modules.engine.data_preprocessor import DataPreprocessor
preprocessor = DataPreprocessor(config.to_preprocessor_config())
```

## Configuration Migration

When upgrading between versions, use configuration migration utilities:

```python
def migrate_config_v1_to_v2(old_config_dict: dict) -> dict:
    """Migrate configuration from v1 to v2 format"""
    new_config = old_config_dict.copy()
    
    # Add new fields with defaults
    new_config.setdefault('enable_quantization', False)
    new_config.setdefault('quantization_config', None)
    
    # Rename fields if necessary
    if 'optimization_type' in new_config:
        new_config['optimization_mode'] = new_config.pop('optimization_type')
    
    return new_config
```

## Error Handling

Common configuration errors and solutions:

```python
from modules.configs import MLTrainingEngineConfig, ConfigurationError

try:
    config = MLTrainingEngineConfig.from_dict(config_dict)
except ConfigurationError as e:
    # Handle configuration-specific errors
    logging.error(f"Configuration error: {e}")
except ValueError as e:
    # Handle value errors
    logging.error(f"Invalid configuration value: {e}")
except TypeError as e:
    # Handle type errors
    logging.error(f"Invalid configuration type: {e}")
```

## Related Documentation

- [Training Engine Documentation](engine/train_engine.md)
- [Inference Engine Documentation](engine/inference_engine.md)
- [Data Preprocessor Documentation](engine/data_preprocessor.md)
- [Quantizer Documentation](engine/quantizer.md)
- [Device Optimizer Documentation](device_optimizer.md)
