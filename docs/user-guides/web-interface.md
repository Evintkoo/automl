# 🌐 Web Interface Guide

The AutoML web interface provides a user-friendly way to interact with the platform without coding. Built with Gradio, it offers an intuitive experience for training models, making predictions, and managing your ML workflows.

## 🚀 Accessing the Web Interface

### Starting the Interface

```bash
# Start the web interface
python app.py

# Or use the start script
python start_api.py
```

The interface will be available at:
- **Local**: `http://localhost:7860`
- **Network**: `http://your-ip:7860` (if enabled)

## 🏠 Dashboard Overview

### Main Navigation

```
┌─ 📊 Dashboard ─────────────────────────────────────┐
├─ 📁 Data Upload      │ Upload and manage datasets  │
├─ 🚂 Model Training   │ Train new models            │
├─ ⚡ Predictions       │ Make predictions            │
├─ 📈 Model Gallery    │ View trained models         │
├─ ⚙️ Settings         │ Configure system            │
└─ 📊 Analytics        │ Performance metrics         │
```

### Status Panel
- **System Health**: Real-time system status
- **Active Jobs**: Currently running training/prediction tasks
- **Resource Usage**: CPU, Memory, GPU utilization
- **Recent Activity**: Latest operations and results

## 📁 Data Upload & Management

### Supported File Formats

| Format | Description | Max Size |
|--------|-------------|----------|
| CSV | Comma-separated values | 100 MB |
| JSON | JavaScript Object Notation | 50 MB |
| Excel | .xlsx, .xls files | 100 MB |
| Parquet | Apache Parquet format | 200 MB |

### Upload Process

1. **Select File**: Click "Choose File" or drag & drop
2. **Preview Data**: Review first 10 rows
3. **Configure Options**:
   - Delimiter (for CSV)
   - Header row detection
   - Data types
4. **Upload**: Click "Upload Dataset"

### Data Preview Features

```python
# Example data preview information
Dataset Info:
├─ Shape: (1000, 15)
├─ Missing Values: 23 (1.5%)
├─ Numeric Columns: 8
├─ Categorical Columns: 7
└─ Target Column: Detected automatically
```

## 🚂 Model Training Interface

### Training Configuration

#### Basic Settings
- **Target Column**: Select the column to predict
- **Problem Type**: Classification or Regression (auto-detected)
- **Train/Test Split**: Default 80/20, customizable
- **Cross-validation**: Number of folds (default: 5)

#### Advanced Options
- **Algorithm Selection**:
  - Auto (recommended)
  - Specific algorithms: XGBoost, LightGBM, CatBoost, etc.
- **Hyperparameter Tuning**:
  - Basic: Quick optimization
  - Advanced: Extensive search
  - Custom: Manual parameter specification

### Training Process

```mermaid
graph LR
    A[Data Upload] --> B[Target Selection]
    B --> C[Algorithm Choice]
    C --> D[Parameter Tuning]
    D --> E[Cross-validation]
    E --> F[Model Training]
    F --> G[Evaluation]
    G --> H[Model Saved]
```

### Real-time Training Progress

The interface shows:
- **Progress Bar**: Training completion percentage
- **Current Algorithm**: Which algorithm is being trained
- **Best Score**: Current best validation score
- **ETA**: Estimated time to completion
- **Live Logs**: Real-time training logs

### Training Results

After training completes:
- **Model Metrics**: Accuracy, F1, RMSE, etc.
- **Feature Importance**: Top contributing features
- **Confusion Matrix**: For classification problems
- **Learning Curves**: Training vs validation performance
- **Model Comparison**: If multiple algorithms were tested

## ⚡ Making Predictions

### Single Prediction

1. **Select Model**: Choose from trained models
2. **Input Features**: 
   - Manual entry via form fields
   - Copy/paste values
   - JSON input for complex data
3. **Get Prediction**: Click "Predict"
4. **View Results**: Prediction with confidence scores

### Batch Predictions

1. **Upload Data**: New dataset for predictions
2. **Select Model**: Choose appropriate model
3. **Map Columns**: Ensure feature alignment
4. **Process**: Start batch prediction
5. **Download Results**: Get predictions as CSV/JSON

### Prediction Interface Features

- **Feature Validation**: Automatic data type checking
- **Missing Value Handling**: Options for incomplete data
- **Confidence Intervals**: For regression predictions
- **Probability Scores**: For classification predictions

## 📈 Model Gallery

### Model Information Display

For each trained model:
- **Model Name**: Auto-generated or custom name
- **Creation Date**: When the model was trained
- **Algorithm**: Primary algorithm used
- **Performance Metrics**: Key evaluation scores
- **Dataset Info**: Training data summary
- **Status**: Active, Archived, Failed

### Model Actions

- **🔍 View Details**: Detailed model information
- **⚡ Make Prediction**: Quick prediction interface
- **📊 Performance**: Detailed evaluation metrics
- **📁 Download**: Export model files
- **🗑️ Delete**: Remove model (with confirmation)
- **📋 Clone**: Create new model with same settings

## ⚙️ Settings & Configuration

### System Settings
- **Resource Limits**: CPU/Memory allocation
- **Storage Paths**: Data and model storage locations
- **Logging Level**: Debug, Info, Warning, Error
- **Auto-cleanup**: Automatic old file removal

### Interface Preferences
- **Theme**: Light/Dark mode
- **Language**: UI language selection
- **Timezone**: For timestamps
- **Notifications**: Email/browser notifications

### Security Settings
- **API Keys**: Generate and manage access keys
- **User Management**: Add/remove users
- **Access Controls**: Feature-based permissions
- **Audit Logging**: Activity tracking

## 📊 Analytics Dashboard

### Training Analytics
- **Model Performance Trends**: Historical accuracy
- **Training Time Analysis**: Efficiency metrics
- **Resource Usage**: Computational costs
- **Algorithm Comparison**: Performance across algorithms

### Usage Analytics
- **User Activity**: Login patterns, feature usage
- **Data Processing**: Upload volumes, processing times
- **Prediction Volumes**: API usage statistics
- **Error Rates**: System health monitoring

## 🔧 Troubleshooting Common Issues

### Upload Problems
```
❌ File too large → Use data compression or split files
❌ Invalid format → Check supported formats
❌ Encoding issues → Save as UTF-8
❌ Memory error → Reduce dataset size
```

### Training Issues
```
❌ Training stuck → Check resource availability
❌ Low accuracy → Review data quality
❌ Out of memory → Reduce batch size
❌ Timeout error → Increase time limits
```

### Prediction Problems
```
❌ Feature mismatch → Verify column names/types
❌ Model not found → Check model status
❌ Invalid input → Review data format
❌ Slow predictions → Check system load
```

## 🎯 Best Practices

### Data Management
- Keep datasets under recommended size limits
- Use consistent column naming conventions
- Clean data before uploading
- Validate data types and formats

### Training Workflow
- Start with auto settings for initial models
- Use cross-validation for reliable metrics
- Compare multiple algorithms when possible
- Save successful configurations

### Prediction Workflow
- Validate input data format matches training data
- Use batch predictions for large datasets
- Monitor prediction confidence scores
- Archive old models when not needed

## 🆘 Getting Help

- **Tooltips**: Hover over interface elements for quick help
- **Help Button**: Context-sensitive help sections
- **Documentation Links**: Direct links to relevant guides
- **Error Messages**: Detailed error descriptions with solutions

---

*Need more advanced features? Check out our [API Reference](../api-reference/) for programmatic access to all functionality.*
