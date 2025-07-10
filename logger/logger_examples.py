"""
Universal Logger Usage Examples
This file demonstrates how to use the enhanced logger across different project types.
"""

from custom_logger import UniversalLogger, get_global_logger, LoggingContext, log_performance
import time
import json
from datetime import datetime

# Example 1: Basic Web Application Setup
def web_app_example():
    """Example for web applications (FastAPI, Django, Flask)"""
    print("=== Web Application Example ===")
    
    # Initialize logger for web project
    logger_tracker = UniversalLogger(
        project_name="web_app",
        config_path="enhanced_config.yaml"
    )
    
    # Different module loggers
    app_logger = logger_tracker.get_logger("main")
    api_logger = logger_tracker.get_logger("fastapi")
    auth_logger = logger_tracker.get_logger("auth")
    db_logger = logger_tracker.get_logger("database")
    
    # Log application startup
    app_logger.info("Web application starting up")
    
    # Log API request
    api_logger.debug("Processing API request", extra={
        "endpoint": "/api/users",
        "method": "GET",
        "user_id": "12345"
    })
    
    # Log authentication
    auth_logger.info("User authentication successful", extra={
        "user_id": "12345",
        "ip_address": "192.168.1.100"
    })
    
    # Log database query
    db_logger.info("Database query executed", extra={
        "query": "SELECT * FROM users",
        "execution_time": 0.045
    })
    
    logger_tracker.close_all_loggers()

# Example 2: Data Science/ML Project
def ml_project_example():
    """Example for Machine Learning projects"""
    print("=== Machine Learning Project Example ===")
    
    # Initialize logger for ML project
    logger_tracker = UniversalLogger(
        project_name="ml_project",
        config_path="enhanced_config.yaml"
    )
    
    # Different module loggers
    main_logger = logger_tracker.get_logger("main")
    data_logger = logger_tracker.get_logger("data_processing")
    model_logger = logger_tracker.get_logger("tensorflow")
    
    # Log data processing
    main_logger.info("Starting ML pipeline")
    
    # Log data preprocessing
    data_logger.debug("Loading dataset", extra={
        "dataset": "customer_data.csv",
        "rows": 10000,
        "columns": 25
    })
    
    # Performance logging with decorator
    @log_performance(logger_tracker, "data_processing")
    def preprocess_data():
        time.sleep(0.1)  # Simulate processing
        return "preprocessed_data"
    
    result = preprocess_data()
    
    # Log model training
    model_logger.info("Model training started", extra={
        "model_type": "RandomForest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10}
    })
    
    # Log model metrics
    model_logger.info("Model training completed", extra={
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.94,
        "f1_score": 0.93
    })
    
    logger_tracker.close_all_loggers()

# Example 3: Microservices Architecture
def microservices_example():
    """Example for microservices architecture"""
    print("=== Microservices Example ===")
    
    # Service A
    service_a_logger = UniversalLogger(
        project_name="service_a",
        config_path="enhanced_config.yaml"
    )
    
    # Service B
    service_b_logger = UniversalLogger(
        project_name="service_b",
        config_path="enhanced_config.yaml"
    )
    
    # Service A logging
    user_service_logger = service_a_logger.get_logger("user_management")
    user_service_logger.info("User created", extra={
        "user_id": "usr_123",
        "service": "user_service",
        "trace_id": "trace_456"
    })
    
    # Service B logging
    order_service_logger = service_b_logger.get_logger("order_processing")
    order_service_logger.info("Order created", extra={
        "order_id": "ord_789",
        "user_id": "usr_123",
        "service": "order_service",
        "trace_id": "trace_456"
    })
    
    # Cross-service error logging
    try:
        raise ConnectionError("Service B unreachable")
    except ConnectionError as e:
        service_a_logger.log_error_with_context(
            "user_management", 
            e, 
            {
                "calling_service": "user_service",
                "target_service": "order_service",
                "trace_id": "trace_456"
            }
        )
    
    service_a_logger.close_all_loggers()
    service_b_logger.close_all_loggers()

# Example 4: ETL Pipeline
def etl_pipeline_example():
    """Example for ETL (Extract, Transform, Load) pipelines"""
    print("=== ETL Pipeline Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="etl_pipeline",
        config_path="enhanced_config.yaml"
    )
    
    extract_logger = logger_tracker.get_logger("extract")
    transform_logger = logger_tracker.get_logger("transform")
    load_logger = logger_tracker.get_logger("load")
    
    # Extract phase
    extract_logger.info("Starting data extraction", extra={
        "source": "postgresql",
        "table": "raw_events",
        "batch_id": "batch_001"
    })
    
    # Transform phase
    transform_logger.info("Data transformation started", extra={
        "input_records": 10000,
        "batch_id": "batch_001"
    })
    
    # Log transformation metrics
    logger_tracker.log_performance(
        "transform", 
        "data_cleaning", 
        0.234,
        records_processed=10000,
        records_valid=9850,
        records_invalid=150
    )
    
    # Load phase
    load_logger.info("Data loading completed", extra={
        "target": "data_warehouse",
        "records_loaded": 9850,
        "batch_id": "batch_001"
    })
    
    logger_tracker.close_all_loggers()

# Example 5: Using Global Logger
def global_logger_example():
    """Example using global logger instance"""
    print("=== Global Logger Example ===")
    
    # Initialize global logger
    global_logger = get_global_logger("shared_project", "enhanced_config.yaml")
    
    # Use from different modules
    module1_logger = global_logger.get_logger("module1")
    module2_logger = global_logger.get_logger("module2")
    
    module1_logger.info("Module 1 operation")
    module2_logger.info("Module 2 operation")
    
    # Get stats
    stats = global_logger.get_log_stats()
    print(f"Global logger stats: {stats}")

# Example 6: Audit Logging
def audit_logging_example():
    """Example for audit logging"""
    print("=== Audit Logging Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="financial_app",
        config_path="enhanced_config.yaml"
    )
    
    audit_logger = logger_tracker.create_audit_logger()
    
    # Log user actions
    audit_logger.info("User login", extra={
        "user_id": "user_123",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0...",
        "timestamp": datetime.now().isoformat()
    })
    
    audit_logger.info("Financial transaction", extra={
        "user_id": "user_123",
        "transaction_id": "txn_456",
        "amount": 1000.00,
        "currency": "USD",
        "type": "transfer"
    })
    
    audit_logger.info("User logout", extra={
        "user_id": "user_123",
        "session_duration": 1800
    })
    
    logger_tracker.close_all_loggers()

# Example 7: Context Manager Usage
def context_manager_example():
    """Example using context manager for temporary config changes"""
    print("=== Context Manager Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="context_test",
        config_path="enhanced_config.yaml"
    )
    
    logger = logger_tracker.get_logger("test")
    
    # Normal logging
    logger.info("Normal log message")
    
    # Temporary configuration change
    temp_config = {
        'console_output': False,
        'format': 'JSON'
    }
    
    with LoggingContext(logger_tracker, temp_config):
        # This will use the temporary configuration
        temp_logger = logger_tracker.get_logger("temp_test")
        temp_logger.info("Temporary config log message")
    
    # Back to normal configuration
    logger.info("Back to normal logging")
    
    logger_tracker.close_all_loggers()

# Example 8: Child Logger Usage
def child_logger_example():
    """Example using child loggers for hierarchical logging"""
    print("=== Child Logger Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="hierarchical_app",
        config_path="enhanced_config.yaml"
    )
    
    # Parent logger
    parent_logger = logger_tracker.get_logger("payment_gateway")
    
    # Child loggers
    stripe_logger = logger_tracker.create_child_logger("payment_gateway", "stripe")
    paypal_logger = logger_tracker.create_child_logger("payment_gateway", "paypal")
    
    parent_logger.info("Payment processing started")
    stripe_logger.info("Stripe payment initiated", extra={"amount": 100.00})
    paypal_logger.info("PayPal payment initiated", extra={"amount": 50.00})
    
    logger_tracker.close_all_loggers()

# Example 9: Custom Handler Integration
def custom_handler_example():
    """Example adding custom handlers"""
    print("=== Custom Handler Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="custom_handler_app",
        config_path="enhanced_config.yaml"
    )
    
    # Add a custom handler (e.g., sending logs to external service)
    class CustomHandler(logging.Handler):
        def emit(self, record):
            # In real implementation, this could send to Slack, email, etc.
            if record.levelno >= logging.ERROR:
                print(f"ALERT: {self.format(record)}")
    
    custom_handler = CustomHandler()
    custom_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger_tracker.add_custom_handler("main", custom_handler)
    
    main_logger = logger_tracker.get_logger("main")
    main_logger.info("This is an info message")
    main_logger.error("This is an error message")  # Will trigger custom handler
    
    logger_tracker.close_all_loggers()

# Example 10: IoT/Real-time System
def iot_example():
    """Example for IoT and real-time systems"""
    print("=== IoT System Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="iot_system",
        config_path="enhanced_config.yaml"
    )
    
    # Different device loggers
    sensor_logger = logger_tracker.get_logger("sensor_data")
    actuator_logger = logger_tracker.get_logger("actuator_control")
    network_logger = logger_tracker.get_logger("network")
    
    # Log sensor data
    sensor_logger.info("Temperature reading", extra={
        "sensor_id": "temp_001",
        "value": 23.5,
        "unit": "celsius",
        "timestamp": datetime.now().isoformat(),
        "location": "room_a"
    })
    
    # Log actuator control
    actuator_logger.info("Fan speed adjusted", extra={
        "actuator_id": "fan_001",
        "speed": 75,
        "trigger": "temperature_threshold"
    })
    
    # Log network issues
    network_logger.warning("Network latency high", extra={
        "latency_ms": 250,
        "gateway": "192.168.1.1"
    })
    
    logger_tracker.close_all_loggers()

# Example 11: E-commerce Platform
def ecommerce_example():
    """Example for e-commerce platform"""
    print("=== E-commerce Platform Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="ecommerce_platform",
        config_path="enhanced_config.yaml"
    )
    
    # Different module loggers
    user_logger = logger_tracker.get_logger("user_management")
    product_logger = logger_tracker.get_logger("product_catalog")
    order_logger = logger_tracker.get_logger("order_processing")
    payment_logger = logger_tracker.get_logger("payment_gateway")
    inventory_logger = logger_tracker.get_logger("inventory")
    
    # User registration
    user_logger.info("User registered", extra={
        "user_id": "usr_12345",
        "email": "user@example.com",
        "registration_source": "web"
    })
    
    # Product search
    product_logger.debug("Product search", extra={
        "query": "laptop",
        "results_count": 45,
        "user_id": "usr_12345"
    })
    
    # Order creation
    order_logger.info("Order created", extra={
        "order_id": "ord_67890",
        "user_id": "usr_12345",
        "total_amount": 1299.99,
        "items_count": 3
    })
    
    # Payment processing
    payment_logger.info("Payment processed", extra={
        "order_id": "ord_67890",
        "payment_method": "credit_card",
        "amount": 1299.99,
        "transaction_id": "txn_abc123"
    })
    
    # Inventory update
    inventory_logger.info("Inventory updated", extra={
        "product_id": "prod_001",
        "quantity_change": -1,
        "new_quantity": 24
    })
    
    logger_tracker.close_all_loggers()

# Example 12: DevOps and Infrastructure
def devops_example():
    """Example for DevOps and infrastructure monitoring"""
    print("=== DevOps Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="infrastructure",
        config_path="enhanced_config.yaml"
    )
    
    # Different infrastructure loggers
    server_logger = logger_tracker.get_logger("server_monitoring")
    deploy_logger = logger_tracker.get_logger("deployment")
    backup_logger = logger_tracker.get_logger("backup_system")
    
    # Server monitoring
    server_logger.info("Server health check", extra={
        "server_id": "srv_001",
        "cpu_usage": 65.2,
        "memory_usage": 78.5,
        "disk_usage": 45.0,
        "status": "healthy"
    })
    
    # Deployment logging
    deploy_logger.info("Deployment started", extra={
        "version": "v2.1.0",
        "environment": "production",
        "rollback_plan": "available"
    })
    
    # Backup system
    backup_logger.info("Backup completed", extra={
        "backup_id": "backup_001",
        "size_mb": 2048,
        "duration_seconds": 300,
        "success": True
    })
    
    logger_tracker.close_all_loggers()

# Example 13: Batch Processing System
def batch_processing_example():
    """Example for batch processing systems"""
    print("=== Batch Processing Example ===")
    
    logger_tracker = UniversalLogger(
        project_name="batch_processor",
        config_path="enhanced_config.yaml"
    )
    
    job_logger = logger_tracker.get_logger("job_scheduler")
    worker_logger = logger_tracker.get_logger("worker")
    
    # Job scheduling
    job_logger.info("Batch job scheduled", extra={
        "job_id": "job_001",
        "job_type": "data_processing",
        "scheduled_time": "2024-01-15T02:00:00Z",
        "priority": "high"
    })
    
    # Worker processing
    worker_logger.info("Job processing started", extra={
        "job_id": "job_001",
        "worker_id": "worker_001",
        "input_records": 100000
    })
    
    # Progress logging
    for i in range(0, 101, 20):
        worker_logger.info("Job progress", extra={
            "job_id": "job_001",
            "progress_percent": i,
            "records_processed": i * 1000
        })
        time.sleep(0.1)  # Simulate processing
    
    worker_logger.info("Job completed successfully", extra={
        "job_id": "job_001",
        "total_records": 100000,
        "success_records": 99950,
        "failed_records": 50
    })
    
    logger_tracker.close_all_loggers()

# Main execution
if __name__ == "__main__":
    # Run all examples
    examples = [
        web_app_example,
        ml_project_example,
        microservices_example,
        etl_pipeline_example,
        global_logger_example,
        audit_logging_example,
        context_manager_example,
        child_logger_example,
        custom_handler_example,
        iot_example,
        ecommerce_example,
        devops_example,
        batch_processing_example
    ]
    
    for example in examples:
        try:
            example()
            print()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("All examples completed!")