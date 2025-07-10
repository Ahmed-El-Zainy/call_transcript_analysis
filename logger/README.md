# Universal Logger

A comprehensive, feature-rich logging solution that can be used across any Python project type - from web applications to machine learning pipelines, microservices, and more.

## 🚀 Key Features

### Core Features
- **Universal Compatibility**: Works with any Python project type
- **Hierarchical Structure**: Automatically organizes logs by date (year/month/day)
- **Multiple Format Support**: Simple, Standard, Detailed, and JSON formats
- **Module-Specific Configuration**: Different settings for different modules
- **Thread-Safe**: Safe for multi-threaded applications

### Advanced Features
- **Log Rotation**: Size-based and time-based rotation with compression
- **Performance Logging**: Built-in performance monitoring and decorators
- **Context Management**: Temporary configuration changes
- **Audit Logging**: Structured audit trails
- **Error Context**: Rich error logging with context information
- **Custom Handlers**: Easy integration with external services
- **Global Instance**: Singleton pattern for project-wide access

### Project-Specific Features
- **Auto-Setup Script**: Quickly configure for any project type
- **Pre-configured Templates**: Ready-to-use configs for common project types
- **Child Loggers**: Hierarchical logger relationships
- **Custom Fields**: Add project-specific metadata to all logs

## 📦 Installation

1. **Copy the core files to your project:**
   ```bash
   # Copy these files to your project directory
   - enhanced_logger.py
   - logging_config.yaml
   - project_setup.py
   ```

2. **Install dependencies:**
   ```bash
   pip install PyYAML python-json-logger
   ```

3. **Quick setup for your project:**
   ```bash
   python project_setup.py my_project --type web --output ./my_project
   ```

## 🏗️ Project Types Supported

| Project Type | Description | Default Modules |
|-------------|-------------|-----------------|
| `web` | Web applications (FastAPI, Django, Flask) | main, fastapi, django, flask, auth, database, api |
| `ml` | Machine Learning projects | main, data_processing, tensorflow, pytorch, sklearn, model_training |
| `microservices` | Microservices architecture | main, service_a, service_b, gateway, discovery, auth |
| `etl` | ETL pipelines | main, extract, transform, load, validation, scheduler |
| `iot` | IoT and real-time systems | main, sensor_data, actuator_control, network, device_manager |
| `ecommerce` | E-commerce platforms | main, user_management, product_catalog, order_processing, payment_gateway, inventory |
| `devops` | DevOps and infrastructure | main, server_monitoring, deployment, backup_system, metrics |
| `batch` | Batch processing systems | main, job_scheduler, worker, queue_manager, notification |
| `custom` | Custom project | You specify modules |

## 🔧 Quick Start

### Basic Usage

```python
from enhanced_logger import UniversalLogger

# Initialize logger for your project
logger_tracker = UniversalLogger(
    project_name="my_app",
    config_path="logging_config.yaml"
)

# Get module-specific loggers
main_logger = logger_tracker.get_logger("main")
api_logger = logger_tracker.get_logger("api")

# Basic logging
main_logger.info("Application started")
api_logger.debug("API request received", extra={"endpoint": "/users"})
```

### Using Project Wrapper (Recommended)

After running the setup script, use the generated wrapper:

```python
from my_project_logger import get_logger, log_performance_decorator

# Get loggers
main_logger = get_logger("main")
db_logger = get_logger("database")

# Log messages
main_logger.info("Application started")
db_logger.info("Database connection established")

# Performance logging
@log_performance_decorator("database")
def query_users():
    # Your database query code
    return users

users = query_users()  # Automatically logs execution time
```

## 📊 Log Formats

### Standard Format
```
2024-01-15 10:30:45,123 - main - INFO - Application started
```

### Detailed Format
```
2024-01-15 10:30:45,123 - main - INFO - /path/to/file.py:25 - main_function - Application started
```

### JSON Format
```json
{
  "timestamp": "2024-01-15T10:30:45.123",
  "level": "INFO",
  "logger": "main",
  "message": "Application started",
  "module": "main",
  "filename": "app.py",
  "line_number": 25,
  "function": "main_function"
}
```

## 🎯 Advanced Usage Examples

### Web Application Example

```python
from enhanced_logger import UniversalLogger

logger_tracker = UniversalLogger(project_name="web_app")

# Different module loggers
app_logger = logger_tracker.get_logger("main")
auth_logger = logger_tracker.get_logger("auth")
db_logger = logger_tracker.get_logger("database")

# FastAPI request logging
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    api_logger = logger_tracker.get_logger("fastapi")
    
    api_logger.info("User request", extra={
        "user_id": user_id,
        "endpoint": "/users",
        "method": "GET"
    })
    
    try:
        user = await get_user_from_db(user_id)
        api_logger.info("User retrieved successfully", extra={"user_id": user_id})
        return user
    except Exception as e:
        logger_tracker.log_error_with_context("fastapi", e, {"user_id": user_id})
        raise
```

### Machine Learning Pipeline

```python
from enhanced_logger import UniversalLogger, log_performance

logger_tracker = UniversalLogger(project_name="ml_pipeline")

# Data processing with performance logging
@log_performance(logger_tracker, "data_processing")
def preprocess_data(data):
    data_logger = logger_tracker.get_logger("data_processing")
    
    data_logger.info("Starting data preprocessing", extra={
        "rows": len(data),
        "columns": len(data.columns)
    })
    
    # Your preprocessing code
    processed_data = data.fillna(0)
    
    data_logger.info("Data preprocessing completed", extra={
        "processed_rows": len(processed_data)
    })
    
    return processed_data

# Model training
def train_model(X, y):
    model_logger = logger_tracker.get_logger("tensorflow")
    
    model_logger.info("Model training started", extra={
        "samples": len(X),
        "features": X.shape[1]
    })
    
    # Training code
    model = train_tensorflow_model(X, y)
    
    model_logger.info("Model training completed", extra={
        "accuracy": 0.95,
        "loss": 0.05
    })
    
    return model
```

### Microservices Architecture

```python
# Service A
service_a_logger = UniversalLogger(project_name="user_service")
user_logger = service_a_logger.get_logger("user_management")

# Service B
service_b_logger = UniversalLogger(project_name="order_service")
order_logger = service_b_logger.get_logger("order_processing")

# Distributed tracing
trace_id = generate_trace_id()

user_logger.info("User created", extra={
    "user_id": "usr_123",
    "service": "user_service",
    "trace_id": trace_id
})

order_logger.info("Order created", extra={
    "order_id": "ord_456",
    "user_id": "usr